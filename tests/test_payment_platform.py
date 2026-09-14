"""
Unit and Integration Tests for RECLAIM Payment Platform Expansion
===================================================================
Tests payment lifecycle state transitions, database persistence, gateway simulator,
HMAC webhook signature validation, idempotency, outcome reconciliation, CSAT feedback,
and experiment evaluation APIs.
"""

import unittest
import os
import tempfile
import json
from fastapi.testclient import TestClient

from backend.app.domain.models import PaymentState, FailureReason, ReconciliationStatus
from backend.app.payment_service import PaymentService
from backend.app.gateways.simulator import SimulatorGateway
from backend.app.reconciliation_service import ReconciliationService
from backend.app.feedback_service import FeedbackService
from backend.app.webhook_verifier import WebhookVerificationError
from backend.app.main import app

class TestPaymentPlatformDomain(unittest.TestCase):
    def setUp(self):
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.tmp_db.name
        self.tmp_db.close()
        self.service = PaymentService(db_path=self.db_path)

    def tearDown(self):
        if hasattr(self, 'service'):
            del self.service
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_payment_creation_and_retrieval(self):
        payment = self.service.create_payment(
            customer_id="CUST_100",
            amount=5000.0,
            currency="INR"
        )
        self.assertIn("id", payment)
        self.assertEqual(payment["status"], "CREATED")
        self.assertEqual(payment["amount"], 5000.0)

        retrieved = self.service.get_payment_by_id(payment["id"])
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["customer_id"], "CUST_100")

    def test_valid_state_transitions(self):
        payment = self.service.create_payment("CUST_101", 1200.0)
        pay_id = payment["id"]
        
        # Transition CREATED -> FAILED
        p1 = self.service.transition_payment_state(
            payment_id=pay_id, 
            new_state=PaymentState.FAILED, 
            event_type="PAYMENT_FAILED",
            reason="Initial attempt failed",
            failure_reason=FailureReason.INSUFFICIENT_FUNDS.value
        )
        self.assertEqual(p1["status"], "FAILED")

        # Transition FAILED -> RETRY_SCHEDULED
        p2 = self.service.transition_payment_state(
            payment_id=pay_id,
            new_state=PaymentState.RETRY_SCHEDULED,
            event_type="RECOVERY_POLICY_EVALUATED",
            reason="Targeted V10.2"
        )
        self.assertEqual(p2["status"], "RETRY_SCHEDULED")

        # Transition RETRY_SCHEDULED -> RETRY_PROCESSING
        p3 = self.service.transition_payment_state(
            payment_id=pay_id,
            new_state=PaymentState.RETRY_PROCESSING,
            event_type="RECOVERY_ATTEMPT_SUBMITTED",
            reason="Gateway retry launched"
        )
        self.assertEqual(p3["status"], "RETRY_PROCESSING")

        # Transition RETRY_PROCESSING -> RECOVERED
        p4 = self.service.transition_payment_state(
            payment_id=pay_id,
            new_state=PaymentState.RECOVERED,
            event_type="RECOVERY_WEBHOOK_RECEIVED",
            reason="Retry successful via simulator"
        )
        self.assertEqual(p4["status"], "RECOVERED")

    def test_invalid_state_transition_raises_error(self):
        payment = self.service.create_payment("CUST_102", 2500.0)
        # Invalid direct jump: CREATED -> RECOVERED
        with self.assertRaises(Exception):
            self.service.transition_payment_state(
                payment_id=payment["id"],
                new_state=PaymentState.RECOVERED,
                event_type="ILLEGAL_TRANSITION"
            )

    def test_attempt_and_recovery_recording(self):
        payment = self.service.create_payment("CUST_103", 3000.0)
        attempt = self.service.record_payment_attempt(
            payment_id=payment["id"],
            attempt_type="RETRY",
            status="FAILED",
            amount=3000.0,
            gateway_reference="pay_sim_123",
            failure_reason="BAD_PIN"
        )
        self.assertEqual(attempt["attempt_number"], 2)

        action = self.service.record_recovery_action(
            payment_id=payment["id"],
            customer_id="CUST_103",
            decision="TARGET",
            decision_reason="Uplift policy targeted",
            policy_version="V10.2",
            model_score=0.85,
            action_type="TARGET",
            action_status="EXECUTED",
            attempt_id=attempt["id"]
        )
        self.assertEqual(action["action_type"], "TARGET")
        self.assertEqual(action["policy_version"], "V10.2")


class TestGatewaySimulatorAndHMAC(unittest.TestCase):
    def setUp(self):
        self.gateway = SimulatorGateway(webhook_secret="test_secret_123")

    def test_create_and_process_payment(self):
        result = self.gateway.create_payment("CUST_SIM_1", 1500.0, "INR")
        self.assertIn("gateway_reference", result)
        self.assertIn("pay_sim_", result["gateway_reference"])

    def test_webhook_hmac_generation_and_verification(self):
        webhook_info = self.gateway.generate_signed_webhook_payload(
            event_type="payment.captured",
            payment_id="pay_test_001",
            external_id="pay_sim_001",
            amount=1500.0,
            status="CAPTURED"
        )
        
        raw_bytes = webhook_info["raw_body"].encode("utf-8")
        sig = webhook_info["signature"]
        
        self.assertTrue(self.gateway.verify_webhook_signature(raw_bytes, sig))
        
        # Test tampered signature raises WebhookVerificationError
        with self.assertRaises(WebhookVerificationError):
            self.gateway.verify_webhook_signature(raw_bytes, "invalid_sig_xyz")


class TestReconciliationAndFeedback(unittest.TestCase):
    def setUp(self):
        self.payment_service = PaymentService()
        self.recon_service = ReconciliationService()
        self.feedback_service = FeedbackService()

    def test_reconciliation_matching(self):
        payment = self.payment_service.create_payment("CUST_200", 4500.0)
        pay_id = payment["id"]

        self.payment_service.transition_payment_state(
            payment_id=pay_id,
            new_state=PaymentState.FAILED,
            event_type="PAYMENT_FAILED"
        )
        self.payment_service.transition_payment_state(
            payment_id=pay_id,
            new_state=PaymentState.RETRY_SCHEDULED,
            event_type="RECOVERY_POLICY_EVALUATED"
        )
        self.payment_service.transition_payment_state(
            payment_id=pay_id,
            new_state=PaymentState.RETRY_PROCESSING,
            event_type="RECOVERY_ATTEMPT_SUBMITTED"
        )
        self.payment_service.transition_payment_state(
            payment_id=pay_id,
            new_state=PaymentState.RECOVERED,
            event_type="RECOVERY_WEBHOOK_RECEIVED"
        )

        # Reconcile payment
        recon = self.recon_service.reconcile_payment_outcome(pay_id, webhook_status="CAPTURED", webhook_amount=4500.0)
        self.assertEqual(recon["reconciliation_status"], "MATCHED")

        summary = self.recon_service.list_reconciliations()
        self.assertGreaterEqual(summary["total"], 1)

    def test_customer_feedback_csat(self):
        fb = self.feedback_service.submit_customer_feedback(
            transaction_id="TX_REC_001",
            rating=5,
            comment="Instant recovery!",
            feedback_category="SURVEY_CSAT"
        )
        self.assertEqual(fb["rating"], 5)

        summary = self.feedback_service.get_feedback_analytics()
        self.assertGreaterEqual(summary["customer_feedback_count"], 1)
        self.assertEqual(summary["average_customer_rating"], 5.0)


class TestPaymentPlatformAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_create_and_get_payment_endpoint(self):
        res = self.client.post("/api/payments", json={
            "customer_id": "CUST_API_10",
            "amount": 2999.0,
            "currency": "INR",
            "payment_method": "CARD"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue("CUS" in data["customer_id"])
        payment_id = data["id"]

        # Get timeline
        timeline_res = self.client.get(f"/api/payments/{payment_id}/timeline")
        self.assertEqual(timeline_res.status_code, 200)

    def test_simulator_webhook_endpoint(self):
        gateway = SimulatorGateway(webhook_secret="sim_webhook_secret_razorpay_reclaim")
        webhook_info = gateway.generate_signed_webhook_payload(
            event_type="payment.captured",
            payment_id="pay_sim_test_web",
            external_id="pay_sim_ext_web",
            amount=2999.0,
            status="CAPTURED"
        )

        res = self.client.post(
            "/api/webhooks/simulator",
            content=webhook_info["raw_body"].encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Simulator-Signature": webhook_info["signature"]
            }
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ACCEPTED")

    def test_experiments_endpoint(self):
        res = self.client.get("/api/recovery/experiments")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("groups", data)
        self.assertEqual(data["policy_version"], "V10.2")

    def test_outcomes_endpoint(self):
        res = self.client.get("/api/recovery/outcomes")
        self.assertEqual(res.status_code, 200)
        self.assertIn("reconciliations", res.json())

if __name__ == "__main__":
    unittest.main()
