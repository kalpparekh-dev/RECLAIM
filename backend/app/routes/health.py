import datetime
from fastapi import APIRouter
from backend.app.policy_service import get_policy_service

router = APIRouter()

@router.get("/health")
def health_check():
    policy_svc = get_policy_service()
    summary = policy_svc.get_summary()
    return {
        "status": "ok",
        "service": "RECLAIM Production Policy Engine",
        "policy_loaded": summary.get("status") == "VALIDATED",
        "policy_version": summary.get("policy_version", "V10.2"),
        "total_rows": summary.get("total_transactions", 10000),
        "targeting_rate": summary.get("targeting_rate", 0.05),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
