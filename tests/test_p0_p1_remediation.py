import os
import hmac
import json
import uuid
import hashlib
import tempfile
import unittest
import concurrent.futures
from starlette.testclient import TestClient

from backend.app.main import app
from backend.app.policy_service import PolicyService, get_policy_service
from backend.app.customer_guard import CustomerGuard
from backend.app.net_value_engine import calculate_expected_net_value
from backend.app.idempotency_store import IdempotencyStore, compute_payload_hash
from backend.app.runtime_control import get_runtime_control, RuntimeMode
from backend.app.webhook_verifier import verify_razorpay_signature, WebhookVerificationError
from backend.app.drift_monitor_service import DriftMonitorService
from backend.app.segment_guard import SegmentGuard
from backend.app.experiment_router import ExperimentRouter
from backend.app.serializer import is_forbidden_key, sanitize_production_response

class TestP0P1RemediationSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_p0_1_no_oracle_fields_in_production_api_responses(self):
        """P0-1: Verify that no production API response contains oracle or post-treatment fields."""
        endpoints = [
            "/api/policy/summary",
            "/api/transactions?limit=10",
            "/api/transactions/RECLAIM-V2-000001",
            "/policy/metadata",
            "/readiness"
        ]

        def check_no_oracle(data):
            if isinstance(data, dict):
                for k, v in data.items():
                    self.assertFalse(is_forbidden_key(k), f"Forbidden key '{k}' found in production API response!")
                    check_no_oracle(v)
            elif isinstance(data, list):
                for item in data:
                    check_no_oracle(item)

        for endpoint in endpoints:
            res = self.client.get(endpoint)
            self.assertEqual(res.status_code, 200, f"Endpoint {endpoint} failed with {res.status_code}")
            check_no_oracle(res.json())

    def test_p0_2_missing_customer_id_unsafe_fallback(self):
        """P0-2: Verify missing/null/empty/whitespace customer_id forces safe RETRY_ALL fallback."""
        invalid_cids = [None, "", "   ", "\t\n"]

        for invalid_cid in invalid_cids:
            # 1. /decision/evaluate endpoint
            res1 = self.client.post("/decision/evaluate", json={
                "transaction_id": "RECLAIM-V2-000001",
                "customer_id": invalid_cid
            })
            self.assertEqual(res1.status_code, 200)
            data1 = res1.json()
            self.assertEqual(data1["selected_policy"], "RETRY_ALL")
            self.assertTrue(data1["fallback_applied"])
            self.assertEqual(data1["fallback_reason"], "CUSTOMER_ID_REQUIRED")

            # 2. /events/payment-failed endpoint
            res2 = self.client.post("/events/payment-failed", json={
                "event_id": f"evt-{uuid.uuid4()}",
                "transaction_id": "RECLAIM-V2-000001",
                "customer_id": invalid_cid,
                "amount": 500.0,
                "payment_method": "CARD"
            })
            self.assertEqual(res2.status_code, 200)
            data2 = res2.json()
            self.assertEqual(data2["execution_decision"], "SUPPRESS")
            self.assertEqual(data2["suppression_reason"], "CUSTOMER_ID_REQUIRED")

    def test_p0_3_atomic_customer_guard_concurrency(self):
        """P0-3: Multi-threaded test asserting atomic customer guard allows EXACTLY 1 action for a customer within 24h."""
        unique_cid = f"TEST_CUST_CONCURRENCY_{uuid.uuid4()}"
        guard = CustomerGuard()
        results = []

        def worker(tx_num):
            tx_id = f"RECLAIM-V2-{tx_num:06d}"
            res = guard.evaluate_execution_decision(
                customer_id=unique_cid,
                transaction_id=tx_id,
                policy_decision="TARGET"
            )
            results.append(res)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(1, 11)]
            concurrent.futures.wait(futures)

        executes = [r for r in results if r["execution_decision"] == "EXECUTE"]
        suppresses = [r for r in results if r["execution_decision"] == "SUPPRESS"]

        self.assertEqual(len(executes), 1, "Exactly ONE concurrent request must reserve the action slot!")
        self.assertEqual(len(suppresses), 9, "Remaining 9 concurrent requests must be suppressed!")

    def test_p0_4_negative_or_unknown_net_value_blocks_execution(self):
        """P0-4: Verify negative, zero, missing, or invalid net value suppresses execution."""
        # 1. Direct net value engine test
        res_negative = calculate_expected_net_value(expected_recovery_value=0.10, action_type="SUPPORT_CALL") # cost $15
        self.assertFalse(res_negative["is_profitable"])
        self.assertFalse(res_negative["action_executable"])

        res_zero = calculate_expected_net_value(expected_recovery_value=0.00, action_type="SMART_RETRY")
        self.assertFalse(res_zero["is_profitable"])

        res_invalid = calculate_expected_net_value(expected_recovery_value="INVALID_AMOUNT")
        self.assertFalse(res_invalid["is_profitable"])

        res_currency = calculate_expected_net_value(expected_recovery_value=100.0, currency="INVALID_CURRENCY")
        self.assertFalse(res_currency["is_profitable"])

    def test_p0_5_policy_hash_mismatch_forces_fallback_runtime_mode(self):
        """P0-5: Verify SHA256 mismatch forces runtime mode FALLBACK and turns all decisions into RETRY_ALL."""
        runtime = get_runtime_control()
        try:
            # Initialize PolicyService with non-existent registry path to force hash mismatch
            svc = PolicyService(registry_path="non_existent_fake_registry.json")
            
            self.assertFalse(svc.hash_validated)
            self.assertEqual(runtime.get_config().get("mode"), "FALLBACK")

            decision = svc.get_decision("RECLAIM-V2-000001")
            self.assertEqual(decision["selected_policy"], "RETRY_ALL")
            self.assertTrue(decision["fallback_applied"])
            self.assertEqual(decision["fallback_reason"], "POLICY_HASH_MISMATCH")
        finally:
            # Restore runtime mode to ACTIVE for remaining tests
            runtime.set_mode(RuntimeMode.ACTIVE, updated_by="TEST_RESTORE", reason="Restore after test_p0_5")

    def test_p0_6_razorpay_webhook_signature_verification(self):
        """P0-6: Verify HMAC-SHA256 signature verification over raw request body."""
        secret = "test_razorpay_webhook_secret_12345"
        raw_body = b'{"event":"payment.failed","amount":500}'
        
        computed_sig = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()

        # 1. Valid signature
        self.assertTrue(verify_razorpay_signature(raw_body, computed_sig, secret=secret))

        # 2. Invalid signature
        with self.assertRaises(WebhookVerificationError):
            verify_razorpay_signature(raw_body, "invalid_signature_string", secret=secret)

        # 3. Altered body
        altered_body = b'{"event":"payment.failed","amount":9999}'
        with self.assertRaises(WebhookVerificationError):
            verify_razorpay_signature(altered_body, computed_sig, secret=secret)

    def test_p0_7_idempotency_concurrent_race_and_payload_hash(self):
        """P0-7: Test 20 concurrent threads using same idempotency key and payload hash matching."""
        unique_key = f"P0_7_KEY_{uuid.uuid4()}"
        store = IdempotencyStore()
        payload = {"event_id": "evt-1", "amount": 100}

        results = []
        def worker():
            res = store.check_and_start(unique_key, "evt-1", "RECLAIM-V2-000001", payload_data=payload)
            results.append(res)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            concurrent.futures.wait(futures)

        starts = [r for r in results if r is None] # None means proceed with processing
        duplicates = [r for r in results if r is not None and r.get("is_duplicate_processing")]

        self.assertEqual(len(starts), 1, "Exactly ONE thread must start processing!")
        self.assertEqual(len(duplicates), 19, "Remaining 19 threads must be flagged as duplicate processing!")

        # Payload conflict test: same key, different payload
        different_payload = {"event_id": "evt-1", "amount": 99999}
        conflict_res = store.check_and_start(unique_key, "evt-1", "RECLAIM-V2-000001", payload_data=different_payload)
        self.assertIsNotNone(conflict_res)
        self.assertTrue(conflict_res.get("is_payload_conflict"))

    def test_p0_9_no_unmasked_customer_ids_in_api_responses(self):
        """P0-9: Recursively inspect API responses ensuring raw customer IDs (CUST_0001) are masked."""
        res = self.client.get("/api/transactions?limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        for tx in data["transactions"]:
            cid = tx.get("customer_id")
            if cid:
                self.assertIn("****", cid, f"Raw unmasked customer ID '{cid}' was exposed in API output!")
                self.assertFalse(cid.startswith("CUST_"), f"Unmasked customer ID '{cid}' exposed!")

    def test_p1_10_continuous_drift_monitoring_fallback_trigger(self):
        """P1-10: Verify DriftMonitorService triggers FALLBACK runtime mode when PSI > 0.25."""
        drift_svc = DriftMonitorService()
        # Mock high PSI score
        drift_svc.get_latest_drift_report = lambda: {"psi_score": 0.35, "drift_alert_level": "CRITICAL"}
        
        eval_res = drift_svc.evaluate_runtime_drift_safety()
        self.assertEqual(eval_res["drift_alert_level"], "CRITICAL")
        self.assertTrue(eval_res["fallback_triggered"])
        self.assertEqual(eval_res["runtime_mode"], "FALLBACK")

        # Restore runtime mode
        get_runtime_control().set_mode(RuntimeMode.ACTIVE, updated_by="TEST", reason="Restore")

    def test_p1_11_segment_concentration_safeguard(self):
        """P1-11: Verify SegmentGuard suppresses action when single segment concentration exceeds cap."""
        seg_guard = SegmentGuard(max_segment_concentration=0.40)
        
        # Populate segment history
        for _ in range(30):
            seg_guard.record_target_segment(payment_method="UPI", failure_reason="BANK_DECLINE", amount=500.0)

        # Check safety for UPI segment
        eval_res = seg_guard.evaluate_segment_safety(payment_method="UPI", failure_reason="BANK_DECLINE", amount=500.0)
        self.assertFalse(eval_res["is_safe"])
        self.assertEqual(eval_res["suppression_reason"], "SEGMENT_CONCENTRATION_CAP_EXCEEDED")

    def test_p1_13_runtime_experiment_assignment(self):
        """P1-13: Verify deterministic A/B experiment assignment (Control vs Treatment)."""
        os.environ["EXPERIMENT_ENABLED"] = "true"
        router = ExperimentRouter(experiment_id="EXP_TEST_001")
        
        res1 = router.assign_group("CUST_0001", "RECLAIM-V2-000001")
        res2 = router.assign_group("CUST_0001", "RECLAIM-V2-000001")
        
        self.assertTrue(res1["experiment_enabled"])
        self.assertEqual(res1["assigned_group"], res2["assigned_group"], "Group assignment must be deterministic!")
        
        os.environ["EXPERIMENT_ENABLED"] = "false"

if __name__ == "__main__":
    unittest.main()
