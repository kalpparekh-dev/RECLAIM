import unittest
import uuid
from backend.app.policy_service import PolicyService
from backend.app.net_value_engine import calculate_expected_net_value
from backend.app.customer_guard import CustomerGuard
from backend.app.runtime_control import get_runtime_control, RuntimeMode
from backend.app.idempotency_store import IdempotencyStore, EventStatus

class TestSecondPassAdversarialVerification(unittest.TestCase):
    
    def test_01_uncertainty_status_explicitly_not_available(self):
        """Red Flag 18: Verify system explicitly reports uncertainty_status = NOT_AVAILABLE."""
        policy_svc = PolicyService()
        decision = policy_svc.get_decision("RECLAIM-V2-000001")
        self.assertEqual(decision.get("uncertainty_status"), "NOT_AVAILABLE")

    def test_02_sha256_hash_mismatch_does_not_halt_service(self):
        """Red Flag 9: Verify SHA256 mismatch forces runtime mode FALLBACK and safe RETRY_ALL."""
        svc = PolicyService(registry_path="D:\\RECLAIM\\data\\non_existent_registry.json")
        self.assertFalse(svc.hash_validated)
        decision = svc.get_decision("RECLAIM-V2-000001")
        self.assertEqual(decision["selected_policy"], "RETRY_ALL")
        self.assertTrue(decision["fallback_applied"])
        self.assertEqual(decision["fallback_reason"], "POLICY_HASH_MISMATCH")
        get_runtime_control().set_mode(RuntimeMode.ACTIVE, updated_by="TEST", reason="Restore")

    def test_03_negative_net_value_calculation_structure(self):
        """Red Flag 8: Verify net value engine flags negative profit (PARTIALLY_SOLVED)."""
        res = calculate_expected_net_value(
            expected_recovery_value=0.10,
            action_type="SUPPORT_CALL", # Cost $15.00
            incentive_cost=5.00
        )
        self.assertFalse(res["is_profitable"])
        self.assertEqual(res["net_value_decision"], "REJECT_UNPROFITABLE")
        self.assertLess(res["expected_net_value"], 0.0)

    def test_04_runtime_mode_fallback_enforcement(self):
        """Red Flag 12: Verify PAUSED runtime mode forces RETRY_ALL decision (SOLVED/PARTIALLY_SOLVED)."""
        runtime = get_runtime_control()
        initial_mode = runtime.get_config().get("mode")
        
        runtime.set_mode(RuntimeMode.PAUSED, updated_by="TEST", reason="Adversarial Test")
        policy_svc = PolicyService()
        decision = policy_svc.get_decision("RECLAIM-V2-000001")
        
        self.assertEqual(decision["selected_policy"], "RETRY_ALL")
        self.assertTrue(decision["fallback_applied"])
        self.assertIn("RUNTIME_MODE_PAUSED", decision["fallback_reason"])
        
        # Restore mode
        runtime.set_mode(RuntimeMode(initial_mode), updated_by="TEST", reason="Restore")

    def test_05_idempotency_replay_detection(self):
        """Red Flag 11: Verify idempotency store detects duplicate completed keys."""
        unique_key = f"test-key-{uuid.uuid4()}"
        store = IdempotencyStore()
        
        # First check returns None (new)
        res1 = store.check_and_start(unique_key, "evt-1", "RECLAIM-V2-000001")
        self.assertIsNone(res1)
        
        # Complete
        store.complete(unique_key, "SMART_RETRY", {"status": "SUCCESS"})
        
        # Second check returns replayed payload
        res2 = store.check_and_start(unique_key, "evt-1", "RECLAIM-V2-000001")
        self.assertIsNotNone(res2)
        self.assertTrue(res2.get("is_replayed_event"))
        self.assertEqual(res2.get("idempotency_status"), EventStatus.REPLAYED.value)

if __name__ == "__main__":
    unittest.main()
