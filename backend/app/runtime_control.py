import os
import json
import datetime
from enum import Enum
from typing import Dict, Any, Optional

class RuntimeMode(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    FALLBACK = "FALLBACK"
    MAINTENANCE = "MAINTENANCE"

CONFIG_FILE_PATH = r"D:\RECLAIM\data\runtime_config.json"

class RuntimeControlManager:
    def __init__(self, config_path: str = CONFIG_FILE_PATH):
        self.config_path = config_path
        self._ensure_config()

    def _ensure_config(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        if not os.path.exists(self.config_path):
            initial_config = {
                "mode": RuntimeMode.ACTIVE.value,
                "fallback_decision": "RETRY_ALL",
                "require_policy_validation": True,
                "allow_action_execution": False, # Locked in simulation mode
                "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "updated_by": "SYSTEM_INIT",
                "audit_log": []
            }
            self._save_config(initial_config)

    def get_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception:
            return {
                "mode": RuntimeMode.FALLBACK.value,
                "fallback_decision": "RETRY_ALL",
                "require_policy_validation": True,
                "allow_action_execution": False
            }

    def _save_config(self, config: Dict[str, Any]):
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def set_mode(self, mode: RuntimeMode, updated_by: str = "ADMIN_OPERATOR", reason: str = "Manual toggle") -> Dict[str, Any]:
        config = self.get_config()
        old_mode = config.get("mode")
        config["mode"] = mode.value
        config["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        config["updated_by"] = updated_by

        audit_entry = {
            "timestamp": config["last_updated"],
            "old_mode": old_mode,
            "new_mode": mode.value,
            "updated_by": updated_by,
            "reason": reason
        }
        if "audit_log" not in config or not isinstance(config["audit_log"], list):
            config["audit_log"] = []
        config["audit_log"].insert(0, audit_entry)

        self._save_config(config)
        return config

    def trigger_kill_switch(self, reason: str = "EMERGENCY_KILL_SWITCH_ACTIVATED", updated_by: str = "SYSTEM_SAFETY") -> Dict[str, Any]:
        return self.set_mode(RuntimeMode.FALLBACK, updated_by=updated_by, reason=reason)

    def is_active(self) -> bool:
        return self.get_config().get("mode") == RuntimeMode.ACTIVE.value

_runtime_control_instance = None

def get_runtime_control() -> RuntimeControlManager:
    global _runtime_control_instance
    if _runtime_control_instance is None:
        _runtime_control_instance = RuntimeControlManager()
    return _runtime_control_instance
