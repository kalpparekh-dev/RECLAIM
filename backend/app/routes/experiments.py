from typing import Dict, Any
from fastapi import APIRouter
from backend.app.experiment_router import get_experiment_router
from backend.app.payment_service import get_payment_service
from backend.app.serializer import sanitize_production_response

router = APIRouter(prefix="/api/recovery", tags=["Experiments"])

@router.get("/experiments")
def get_experiment_evaluation_metrics():
    exp_router = get_experiment_router()
    payment_svc = get_payment_service()

    conn = payment_svc._get_connection()
    try:
        cursor = conn.cursor()
        
        # Treatment vs Control counts
        cursor.execute("SELECT experiment_group, COUNT(*) as cnt, SUM(CASE WHEN action_status = 'EXECUTED' THEN 1 ELSE 0 END) as executed_cnt FROM recovery_actions GROUP BY experiment_group")
        group_rows = cursor.fetchall()
        groups_summary = {r["experiment_group"]: {"total_decisions": r["cnt"], "executed_actions": r["executed_cnt"]} for r in group_rows}

        # Reconciled recoveries per group
        cursor.execute("""
            SELECT ra.experiment_group, COUNT(*) as rec_cnt, SUM(r.recovered_amount) as rec_amount
            FROM outcome_reconciliations r
            JOIN recovery_actions ra ON r.payment_id = ra.payment_id
            WHERE r.reconciliation_status = 'MATCHED' AND r.final_status = 'RECOVERED'
            GROUP BY ra.experiment_group
        """)
        rec_rows = cursor.fetchall()
        for r in rec_rows:
            grp = r["experiment_group"]
            if grp in groups_summary:
                groups_summary[grp]["recovered_count"] = r["rec_cnt"]
                groups_summary[grp]["recovered_amount"] = round(r["rec_amount"] or 0.0, 2)

        return sanitize_production_response({
            "experiment_id": exp_router.experiment_id,
            "experiment_enabled": exp_router.enabled,
            "policy_version": "V10.2",
            "groups": groups_summary,
            "score_semantics": "MODEL_RANKING_SCORE",
            "calibration_status": "NOT_CALIBRATED",
            "note": "Evaluation metrics compare Control (50% Retry All) vs Treatment (50% RECLAIM Uplift Policy)."
        })
    finally:
        conn.close()
