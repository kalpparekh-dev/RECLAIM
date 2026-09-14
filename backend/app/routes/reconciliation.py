from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from backend.app.reconciliation_service import get_reconciliation_service
from backend.app.serializer import sanitize_production_response

router = APIRouter(prefix="/api/recovery", tags=["Reconciliation"])

class ReconcileOutcomeRequest(BaseModel):
    payment_id: str
    webhook_status: str = "CAPTURED"
    webhook_amount: float

@router.get("/outcomes")
def list_outcome_reconciliations(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100)
):
    rec_svc = get_reconciliation_service()
    res = rec_svc.list_reconciliations(page=page, limit=limit)
    return sanitize_production_response(res, mask_ids=True)

@router.post("/reconcile")
def trigger_manual_reconciliation(req: ReconcileOutcomeRequest):
    rec_svc = get_reconciliation_service()
    res = rec_svc.reconcile_payment_outcome(
        payment_id=req.payment_id,
        webhook_status=req.webhook_status,
        webhook_amount=req.webhook_amount
    )
    return sanitize_production_response(res, mask_ids=True)
