import os
import json
import datetime
from typing import Dict, Any, Optional
from backend.app.runtime_control import get_runtime_control, RuntimeMode

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRIFT_REPORT_PATH = os.path.join(BASE_DIR, "data", "generated", "drift_monitoring_report.json")

class DriftMonitorService:
    def __init__(self, report_path: str = DRIFT_REPORT_PATH):
        self.report_path = report_path

    def get_latest_drift_report(self) -> Dict[str, Any]:
        if not os.path.exists(self.report_path):
            return {
                "status": "NO_DRIFT_DATA",
                "psi_score": 0.0,
                "drift_alert_level": "NORMAL",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        try:
            with open(self.report_path, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"status": "ERROR_READING_REPORT", "detail": str(e)}

    def evaluate_runtime_drift_safety(self) -> Dict[str, Any]:
        report = self.get_latest_drift_report()
        psi = report.get("psi_score", 0.0)
        runtime = get_runtime_control()

        if psi > 0.25: # Critical drift threshold
            alert_level = "CRITICAL"
            runtime.set_mode(
                RuntimeMode.FALLBACK,
                updated_by="DRIFT_MONITOR",
                reason=f"CRITICAL_MODEL_DRIFT (PSI={psi:.4f} > 0.25)"
            )
        elif psi > 0.10: # Warning drift threshold
            alert_level = "WARNING"
        else:
            alert_level = "NORMAL"

        return {
            "psi_score": psi,
            "drift_alert_level": alert_level,
            "runtime_mode": runtime.get_config().get("mode"),
            "fallback_triggered": alert_level == "CRITICAL",
            "report": report,
            "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

_drift_monitor_instance = None

def get_drift_monitor_service() -> DriftMonitorService:
    global _drift_monitor_instance
    if _drift_monitor_instance is None:
        _drift_monitor_instance = DriftMonitorService()
    return _drift_monitor_instance
