import os
import pandas as pd
from typing import Dict, Any

OUTCOMES_CSV_PATH = r"D:\RECLAIM\data\generated\recovery_outcomes.csv"

class OutcomesService:
    def __init__(self, csv_path: str = OUTCOMES_CSV_PATH):
        self.csv_path = csv_path

    def get_summary(self) -> Dict[str, Any]:
        if not os.path.exists(self.csv_path):
            return {
                "error": "Historical recovery outcomes CSV not found",
                "data_nature": "HISTORICAL_RESEARCH_ONLY"
            }

        df = pd.read_csv(self.csv_path)
        total_records = len(df)
        unique_customers = int(df["customer_id"].nunique())
        
        recovered_df = df[df["recovered"] == 1]
        recovered_count = len(recovered_df)
        recovery_rate = recovered_count / total_records if total_records > 0 else 0.0

        total_amount = float(df["amount"].sum())
        total_recovered_amount = float(df["recovered_amount"].sum())
        total_net_recovered_value = float(df["net_recovered_value"].sum())

        avg_recoverability_score = float(df["recoverability_score"].mean()) if "recoverability_score" in df else 0.0

        method_breakdown = df["payment_method"].value_counts().to_dict() if "payment_method" in df else {}
        failure_reason_breakdown = df["failure_reason"].value_counts().to_dict() if "failure_reason" in df else {}

        return {
            "data_nature": "HISTORICAL_RESEARCH_DATA",
            "disclaimer": "This dataset represents historical payment failure recovery research records. It is NOT linked to V10.2 production policy transaction IDs.",
            "total_records": total_records,
            "unique_customers": unique_customers,
            "recovered_count": recovered_count,
            "unrecovered_count": total_records - recovered_count,
            "historical_recovery_rate": round(recovery_rate, 4),
            "total_failed_volume": round(total_amount, 2),
            "total_recovered_amount": round(total_recovered_amount, 2),
            "total_net_recovered_value": round(total_net_recovered_value, 2),
            "avg_recoverability_score": round(avg_recoverability_score, 2),
            "payment_method_breakdown": method_breakdown,
            "failure_reason_breakdown": failure_reason_breakdown
        }

_outcomes_service_instance = None

def get_outcomes_service() -> OutcomesService:
    global _outcomes_service_instance
    if _outcomes_service_instance is None:
        _outcomes_service_instance = OutcomesService()
    return _outcomes_service_instance
