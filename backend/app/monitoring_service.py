import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MONITORING_CSV_PATH = os.path.join(BASE_DIR, "data", "generated", "reclaim_production_v21_monitoring.csv")
METRICS_CSV_PATH = os.path.join(BASE_DIR, "data", "generated", "reclaim_production_v21_metrics.csv")
DATA_QUALITY_CSV_PATH = os.path.join(BASE_DIR, "data", "generated", "reclaim_production_v21_data_quality.csv")

class MonitoringService:
    def __init__(
        self,
        monitoring_path: str = MONITORING_CSV_PATH,
        metrics_path: str = METRICS_CSV_PATH,
        data_quality_path: str = DATA_QUALITY_CSV_PATH
    ):
        self.monitoring_path = monitoring_path
        self.metrics_path = metrics_path
        self.data_quality_path = data_quality_path

    def get_monitoring_summary(self) -> Dict[str, Any]:
        if not os.path.exists(self.monitoring_path):
            return {"error": "Monitoring CSV not found", "status": "UNKNOWN"}

        df = pd.read_csv(self.monitoring_path)
        total_rows = len(df)
        unique_txs = int(df["transaction_id"].nunique())
        
        selected_count = int((df["selected_policy"] == "TARGET").sum())
        retry_all_count = int((df["selected_policy"] == "RETRY_ALL").sum())
        targeting_rate = selected_count / total_rows if total_rows > 0 else 0.0

        positive_uplift_count = int((df["monitoring_positive_uplift"] == True).sum())
        zero_negative_uplift_count = total_rows - positive_uplift_count

        passed_count = int((df["monitoring_status"] == "PASS").sum())
        status = "PASS" if passed_count == total_rows else "WARN"

        uplift_mean = float(df["predicted_probability_uplift"].mean())
        uplift_std = float(df["predicted_probability_uplift"].std())
        uplift_min = float(df["predicted_probability_uplift"].min())
        uplift_max = float(df["predicted_probability_uplift"].max())

        return {
            "version": "V21",
            "source_version": "V10.2",
            "total_rows": total_rows,
            "unique_transactions": unique_txs,
            "targeting_rate": targeting_rate,
            "selected_count": selected_count,
            "retry_all_count": retry_all_count,
            "positive_uplift_count": positive_uplift_count,
            "zero_negative_uplift_count": zero_negative_uplift_count,
            "monitoring_status": status,
            "prediction_distribution_summary": {
                "mean": round(uplift_mean, 6),
                "std": round(uplift_std, 6),
                "min": round(uplift_min, 6),
                "max": round(uplift_max, 6)
            }
        }

    def get_metrics(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.metrics_path):
            return []
        df = pd.read_csv(self.metrics_path)
        df = df.replace({np.nan: None})
        records = df.to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if pd.isna(v):
                    rec[k] = None
        return records

    def get_data_quality(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.data_quality_path):
            return []
        df = pd.read_csv(self.data_quality_path)
        df = df.replace({np.nan: None})
        records = df.to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if pd.isna(v):
                    rec[k] = None
        return records

_monitoring_service_instance = None

def get_monitoring_service() -> MonitoringService:
    global _monitoring_service_instance
    if _monitoring_service_instance is None:
        _monitoring_service_instance = MonitoringService()
    return _monitoring_service_instance
