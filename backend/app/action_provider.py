import uuid
import datetime
from typing import Dict, Any

class ActionProviderInterface:
    def preview(self, action_request: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def execute(self, action_request: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class MockActionProvider(ActionProviderInterface):
    """
    Default simulation provider that prevents accidental real payment or message execution.
    """
    def __init__(self, simulation_mode: bool = True, allow_real_actions: bool = False):
        self.simulation_mode = simulation_mode
        self.allow_real_actions = allow_real_actions

    def preview(self, action_request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "preview_id": f"PREVIEW-{uuid.uuid4().hex[:8].upper()}",
            "transaction_id": action_request.get("transaction_id"),
            "action_type": action_request.get("action_type", "SMART_RETRY"),
            "simulation_mode": self.simulation_mode,
            "will_execute_real_payment": False,
            "estimated_cost": action_request.get("estimated_cost", 0.50),
            "preview_status": "READY_FOR_SIMULATION"
        }

    def execute(self, action_request: Dict[str, Any]) -> Dict[str, Any]:
        if not self.allow_real_actions:
            return {
                "action_id": f"ACT-SIM-{uuid.uuid4().hex[:8].upper()}",
                "transaction_id": action_request.get("transaction_id"),
                "action_type": action_request.get("action_type", "SMART_RETRY"),
                "execution_status": "SIMULATED_SUCCESS",
                "is_real_execution": False,
                "simulation_mode": True,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "disclaimer": "SIMULATED — NO REAL PAYMENT OR MESSAGE EXECUTED"
            }
        raise RuntimeError("Real action execution is disabled in safety configuration.")

_action_provider_instance = None

def get_action_provider() -> ActionProviderInterface:
    global _action_provider_instance
    if _action_provider_instance is None:
        _action_provider_instance = MockActionProvider(simulation_mode=True, allow_real_actions=False)
    return _action_provider_instance
