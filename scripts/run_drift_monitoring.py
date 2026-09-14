import os
import json
import hashlib
import numpy as np
import pandas as pd

POLICY_CSV = r"D:\RECLAIM\data\generated\causal_uplift_v10_2_predictions.csv"
REGISTRY_JSON = r"D:\RECLAIM\data\generated\policy_registry.json"
OUTPUT_REPORT = r"D:\RECLAIM\data\generated\drift_monitoring_report.json"

def run_drift_monitoring():
    if not os.path.exists(POLICY_CSV):
        print(f"Error: Policy CSV missing at {POLICY_CSV}")
        return

    df = pd.read_csv(POLICY_CSV)
    total_tx = len(df)

    # 1. Policy Hash & Integrity Audit
    with open(POLICY_CSV, "rb") as f:
        current_hash = hashlib.sha256(f.read()).hexdigest()

    registry_hash = None
    if os.path.exists(REGISTRY_JSON):
        with open(REGISTRY_JSON, "r") as f:
            reg = json.load(f)
            registry_hash = reg.get("policy_sha256")

    hash_match = (current_hash == registry_hash)

    # 2. Prediction Distribution Drift Checks
    treatment_probs = df["predicted_treatment_probability"]
    control_probs = df["predicted_control_probability"]
    uplifts = df["predicted_probability_uplift"]

    selected_count = int((df["intervention_selected"] == 1).sum())
    target_rate = selected_count / total_tx

    drift_alerts = []

    if not hash_match:
        drift_alerts.append({"level": "CRITICAL", "type": "POLICY_HASH_MISMATCH", "message": "Policy CSV hash does not match registry SHA256."})

    if total_tx != 10000:
        drift_alerts.append({"level": "CRITICAL", "type": "ROW_COUNT_DRIFT", "message": f"Expected 10,000 rows, got {total_tx}."})

    if selected_count != 500:
        drift_alerts.append({"level": "CRITICAL", "type": "TARGET_COUNT_DRIFT", "message": f"Expected 500 target rows, got {selected_count}."})

    if abs(target_rate - 0.05) > 1e-6:
        drift_alerts.append({"level": "CRITICAL", "type": "TARGET_RATE_DRIFT", "message": f"Target rate drifted to {target_rate:.4f}."})

    # Summary Stats
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "policy_version": "V10.2",
        "current_sha256": current_hash,
        "registry_sha256": registry_hash,
        "hash_match": hash_match,
        "total_transactions": total_tx,
        "selected_transactions": selected_count,
        "targeting_rate": target_rate,
        "prediction_drift_metrics": {
            "treatment_prob_mean": round(float(treatment_probs.mean()), 4),
            "control_prob_mean": round(float(control_probs.mean()), 4),
            "uplift_mean": round(float(uplifts.mean()), 6),
            "uplift_std": round(float(uplifts.std()), 6),
            "uplift_p95": round(float(uplifts.quantile(0.95)), 6),
            "uplift_p99": round(float(uplifts.quantile(0.99)), 6)
        },
        "drift_alerts": drift_alerts,
        "monitoring_status": "PASS" if len(drift_alerts) == 0 else "CRITICAL_DRIFT_DETECTED"
    }

    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[RECLAIM] Drift monitoring complete. Status: {summary['monitoring_status']}. Report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    run_drift_monitoring()
