import os
import hashlib
import datetime
from typing import Dict, Any, Optional

class ExperimentRouter:
    def __init__(self, experiment_id: str = "EXP_RECLAIM_V10_2_001"):
        self.experiment_id = experiment_id
        self.enabled = os.getenv("EXPERIMENT_ENABLED", "false").lower() in ("true", "1", "yes")

    def assign_group(self, customer_id: str, transaction_id: str) -> Dict[str, Any]:
        """
        Deterministically hashes customer_id + transaction_id to assign 50% Control / 50% Treatment.
        Returns:
        {
          "experiment_enabled": bool,
          "experiment_id": str,
          "assigned_group": "TREATMENT" | "CONTROL",
          "forced_action": None | "RETRY_ALL",
          "hash_value": int
        }
        """
        if not self.enabled:
            return {
                "experiment_enabled": False,
                "experiment_id": self.experiment_id,
                "assigned_group": "TREATMENT",
                "forced_action": None,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }

        seed = f"{self.experiment_id}:{customer_id}:{transaction_id}"
        hash_digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        hash_val = int(hash_digest[:8], 16)

        # 50% split threshold
        is_treatment = (hash_val % 100) < 50
        group = "TREATMENT" if is_treatment else "CONTROL"

        return {
            "experiment_enabled": True,
            "experiment_id": self.experiment_id,
            "assigned_group": group,
            "forced_action": None if is_treatment else "RETRY_ALL",
            "hash_value": hash_val,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

_experiment_router_instance = None

def get_experiment_router() -> ExperimentRouter:
    global _experiment_router_instance
    if _experiment_router_instance is None:
        _experiment_router_instance = ExperimentRouter()
    return _experiment_router_instance
