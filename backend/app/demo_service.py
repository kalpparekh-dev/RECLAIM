import os
import json
import uuid
import datetime
from typing import Dict, Any, List, Optional
from backend.app.policy_service import get_policy_service

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMO_STORE_PATH = os.path.join(BASE_DIR, "data", "demo_execution_store.json")

class DemoService:
    def __init__(self, store_path: str = DEMO_STORE_PATH):
        self.store_path = store_path
        self._ensure_store_exists()

    def _ensure_store_exists(self):
        dir_path = os.path.dirname(self.store_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(self.store_path):
            with open(self.store_path, "w") as f:
                json.dump([], f)

    def _read_store(self) -> List[Dict[str, Any]]:
        try:
            with open(self.store_path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_store(self, events: List[Dict[str, Any]]):
        with open(self.store_path, "w") as f:
            json.dump(events, f, indent=2)

    def execute_demo_action(
        self,
        transaction_id: str,
        operator_action: str = "EXECUTE_RECOMMENDED",
        operator_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        policy_svc = get_policy_service()
        tx = policy_svc.get_transaction(transaction_id)
        if not tx:
            return {
                "error": f"Transaction ID '{transaction_id}' not found in V10.2 production policy.",
                "executed": False
            }

        # Calculate simulated outcome based on treatment vs control probabilities
        p_treatment = float(tx["predicted_treatment_probability"])
        p_control = float(tx["predicted_control_probability"])
        p_uplift = float(tx["predicted_probability_uplift"])
        is_selected = int(tx["intervention_selected"]) == 1

        simulated_success = (p_treatment > 0.5) if is_selected else (p_control > 0.5)
        
        event_id = f"DEMO-EVT-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        event = {
            "event_id": event_id,
            "timestamp": timestamp,
            "transaction_id": tx["transaction_id"],
            "customer_id": tx["customer_id"],
            "production_policy": tx["selected_policy"],
            "intervention_selected": tx["intervention_selected"],
            "operator_action": operator_action,
            "operator_notes": operator_notes or "Executed via RECLAIM Demo Recovery Console",
            "simulated_treatment_probability": p_treatment,
            "simulated_control_probability": p_control,
            "simulated_uplift": p_uplift,
            "simulated_result": "SIMULATED_RECOVERY_SUCCESS" if simulated_success else "SIMULATED_RECOVERY_PENDING",
            "data_nature": "DEMO_SIMULATION_ONLY",
            "disclaimer": "SIMULATED — NOT AN OBSERVED RECOVERY. Does not alter production policy or claim actual historical revenue."
        }

        events = self._read_store()
        events.insert(0, event) # Most recent first
        self._write_store(events)

        return {
            "executed": True,
            "demo_event": event
        }

    def list_demo_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        events = self._read_store()
        return events[:limit]

_demo_service_instance = None

def get_demo_service() -> DemoService:
    global _demo_service_instance
    if _demo_service_instance is None:
        _demo_service_instance = DemoService()
    return _demo_service_instance
