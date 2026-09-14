from fastapi import APIRouter
from backend.app.outcomes_service import get_outcomes_service

router = APIRouter(prefix="/api/outcomes")

@router.get("/summary")
def get_outcomes_summary():
    out_svc = get_outcomes_service()
    return out_svc.get_summary()
