import unittest
import os
import json
import uuid
from starlette.testclient import TestClient

from backend.app.main import app
from backend.app.policy_service import PolicyService, PolicyValidationError, get_policy_service
from backend.app.runtime_control import get_runtime_control, RuntimeMode
from backend.app.idempotency_store import get_idempotency_store
from backend.app.customer_guard import get_customer_guard
from backend.app.net_value_engine import calculate_expected_net_value
from backend.app.action_provider import get_action_provider
from backend.app.privacy import mask_customer_id, mask_transaction_id, sanitize_log_dict

class TestReclaimRedTeamAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # Flag 13 & 1: Policy Registry Hash & Row Counts
    def test_01_policy_registry_and_hash_verification(self):
        svc = get_policy_service()
        summary = svc.get_summary()
        self.assertEqual(summary["policy_version"], "V10.2")
        self.assertEqual(summary["total_transactions"], 10000)
        self.assertEqual(summary["selected_transactions"], 500)
        self.assertAlmostEqual(summary["targeting_rate"], 0.05, places=4)
        self.assertTrue(svc.hash_validated, "Policy CSV SHA256 hash must match policy_registry.json")

    # Flag 17: Unknown Transaction Safe Fallback
    def test_02_unknown_transaction_fallback(self):
        svc = get_policy_service()
        decision = svc.get_decision("NON_EXISTENT_TXN_999999")
        self.assertEqual(decision["selected_policy"], "RETRY_ALL")
        self.assertTrue(decision["fallback_applied"])
        self.assertEqual(decision["fallback_reason"], "TRANSACTION_NOT_FOUND")

    # Flag 9: Runtime Modes & Emergency Kill Switch
    def test_03_kill_switch_and_runtime_modes(self):
        runtime = get_runtime_control()
        svc = get_policy_service()
        target_tx_id = "RECLAIM-V2-000010"

        # Trigger Kill Switch
        runtime.trigger_kill_switch(reason="TEST_EMERGENCY")
        decision = svc.get_decision(target_tx_id)
        self.assertEqual(decision["selected_policy"], "RETRY_ALL")
        self.assertTrue(decision["fallback_applied"])

        # Restore ACTIVE mode
        runtime.set_mode(RuntimeMode.ACTIVE, reason="TEST_RESTORE")
        decision_active = svc.get_decision(target_tx_id)
        self.assertEqual(decision_active["selected_policy"], "TARGET")

    # Flag 4: Customer Cooldown & Opt-Out Safeguards
    def test_04_customer_cooldown_and_suppression(self):
        guard = get_customer_guard()
        cust_id = f"TEST_CUST_COOLDOWN_{uuid.uuid4().hex[:6]}"
        tx_id = "RECLAIM-V2-000010"

        # First evaluation -> EXECUTE
        res1 = guard.evaluate_execution_decision(cust_id, tx_id, "TARGET")
        self.assertEqual(res1["execution_decision"], "EXECUTE")

        # Record action execution
        guard.record_customer_action(cust_id, tx_id, "EXECUTE")

        # Second evaluation within 24h -> SUPPRESS due to CUSTOMER_COOLDOWN
        res2 = guard.evaluate_execution_decision(cust_id, "RECLAIM-V2-000039", "TARGET")
        self.assertEqual(res2["execution_decision"], "SUPPRESS")
        self.assertEqual(res2["suppression_reason"], "CUSTOMER_COOLDOWN")

        # Opt-Out Test
        opt_cust = f"TEST_CUST_OPTOUT_{uuid.uuid4().hex[:6]}"
        guard.opt_out_customer(opt_cust)
        res_opt = guard.evaluate_execution_decision(opt_cust, "RECLAIM-V2-000010", "TARGET")
        self.assertEqual(res_opt["execution_decision"], "SUPPRESS")
        self.assertEqual(res_opt["suppression_reason"], "CUSTOMER_OPT_OUT")

    # Flag 7: Currency-Safe Decimal Net Value Calculation
    def test_05_decimal_net_value_calculation(self):
        res = calculate_expected_net_value(expected_recovery_value=100.0, action_type="SMART_RETRY")
        self.assertEqual(res["communication_cost"], 0.50)
        self.assertEqual(res["expected_net_value"], 99.50)
        self.assertTrue(res["is_profitable"])

        # Unprofitable case
        unprof = calculate_expected_net_value(expected_recovery_value=0.20, action_type="SUPPORT_CALL")
        self.assertFalse(unprof["is_profitable"])
        self.assertEqual(unprof["net_value_decision"], "REJECT_UNPROFITABLE")

    # Flag 8: Idempotency Replay Protection
    def test_06_idempotency_replay_protection(self):
        idempotency_key = f"IDEM-TEST-KEY-{uuid.uuid4().hex[:8]}"
        event_payload = {
            "event_id": f"EVT-TEST-{uuid.uuid4().hex[:6]}",
            "transaction_id": "RECLAIM-V2-000010",
            "customer_id": "C00819",
            "amount": 878.10,
            "idempotency_key": idempotency_key
        }

        # First Call
        resp1 = self.client.post("/events/payment-failed", json=event_payload)
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.json()
        self.assertEqual(data1["policy_decision"], "TARGET")
        self.assertFalse(data1.get("is_replayed_event", False))

        # Replayed Call (Same Idempotency Key)
        resp2 = self.client.post("/events/payment-failed", json=event_payload)
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertTrue(data2.get("is_replayed_event", True))
        self.assertEqual(data2["idempotency_status"], "REPLAYED")

    # Flag 19: Action Provider Simulation Lock
    def test_07_action_provider_simulation_lock(self):
        provider = get_action_provider()
        preview = provider.preview({"transaction_id": "RECLAIM-V2-000010"})
        self.assertTrue(preview["simulation_mode"])
        self.assertFalse(preview["will_execute_real_payment"])

        exec_res = provider.execute({"transaction_id": "RECLAIM-V2-000010"})
        self.assertTrue(exec_res["simulation_mode"])
        self.assertFalse(exec_res["is_real_execution"])

    # Flag 16: Identifier Masking & Privacy
    def test_08_identifier_masking_and_sanitization(self):
        masked_cust = mask_customer_id("C00233")
        self.assertEqual(masked_cust, "C00****33")

        masked_tx = mask_transaction_id("RECLAIM-V2-000001")
        self.assertEqual(masked_tx, "RECLAIM-****0001")

        log_data = {"customer_id": "C00233", "card_number": "4111222233334444", "status": "OK"}
        sanitized = sanitize_log_dict(log_data)
        self.assertEqual(sanitized["card_number"], "[REDACTED]")
        self.assertEqual(sanitized["customer_id"], "C00****33")

    # Flag 5 & 20: Event Routes, Readiness & Data Mode Tagging
    def test_09_readiness_and_data_mode_tagging(self):
        readiness_resp = self.client.get("/readiness")
        self.assertEqual(readiness_resp.status_code, 200)
        self.assertEqual(readiness_resp.json()["status"], "READY")

        metadata_resp = self.client.get("/policy/metadata")
        self.assertEqual(metadata_resp.status_code, 200)
        self.assertEqual(metadata_resp.json()["data_mode"], "PRODUCTION_POLICY")

if __name__ == "__main__":
    unittest.main()
