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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_FILE_PATH = os.path.join(BASE_DIR, "data", "runtime_config.json")

class RuntimeControlManager:
    def __init__(self, config_path: str = CONFIG_FILE_PATH):
        self.config_path = config_path
        self._ensure_config()

    def _ensure_config(self):
        dir_path = os.path.dirname(self.config_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
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

    def is_active(self) -> bool:
        config = self.get_config()
        return config.get("mode") == RuntimeMode.ACTIVE.value

    def _save_config(self, config: Dict[str, Any]):
        dir_path = os.path.dirname(self.config_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def set_mode(self, mode: RuntimeMode, updated_by: str = "ADMIN_OPERATOR", reason: str = "Manual toggle") -> Dict[str, Any]:
        config = self.get_config()
        old_mode = config.get("mode", RuntimeMode.ACTIVE.value)
        config["mode"] = mode.value
        config["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        config["updated_by"] = updated_by

        audit_entry = {
            "timestamp": config["last_updated"],
            "previous_mode": old_mode,
            "new_mode": mode.value,
            "updated_by": updated_by,
            "reason": reason
        }
        config.setdefault("audit_log", []).append(audit_entry)
        self._save_config(config)
        return config

    def trigger_kill_switch(self, updated_by: str = "EMERGENCY_OPERATOR", reason: str = "EMERGENCY_KILL_SWITCH") -> Dict[str, Any]:
        return self.set_mode(RuntimeMode.PAUSED, updated_by=updated_by, reason=reason)

_runtime_control_instance = None

def get_runtime_control() -> RuntimeControlManager:
    global _runtime_control_instance
    if _runtime_control_instance is None:
        _runtime_control_instance = RuntimeControlManager()
    return _runtime_control_instance
