from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from backend.app.policy_service import get_policy_service
from backend.app.serializer import sanitize_production_response

router = APIRouter(prefix="/api")

class DecisionRequest(BaseModel):
    transaction_id: str

@router.get("/policy/summary")
def get_policy_summary():
    policy_svc = get_policy_service()
    return sanitize_production_response(policy_svc.get_summary())

@router.get("/transactions")
def list_transactions(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search transaction or customer ID"),
    customer_id: Optional[str] = Query(None, description="Filter by customer ID"),
    selected: Optional[bool] = Query(None, description="Filter by intervention selected (True/False)"),
    action: Optional[str] = Query(None, description="Filter by action (TARGET or RETRY_ALL)"),
    sort_by: Optional[str] = Query(None, description="Sort field"),
    order: str = Query("asc", pattern="^(asc|desc)$")
):
    policy_svc = get_policy_service()
    res = policy_svc.list_transactions(
        page=page,
        limit=limit,
        search=search,
        customer_id=customer_id,
        selected=selected,
        action=action,
        sort_by=sort_by,
        order=order
    )
    return sanitize_production_response(res, mask_ids=True)

@router.get("/transactions/{transaction_id}")
def get_transaction_detail(transaction_id: str):
    policy_svc = get_policy_service()
    tx = policy_svc.get_transaction(transaction_id)
    if not tx:
        raise HTTPException(status_code=404, detail=f"Transaction '{transaction_id}' not found in production policy.")
    return sanitize_production_response(tx, mask_ids=True)

@router.post("/decision")
def get_production_decision(req: DecisionRequest):
    policy_svc = get_policy_service()
    decision = policy_svc.get_decision(req.transaction_id)
    if not decision.get("found"):
        raise HTTPException(status_code=404, detail=f"Transaction ID '{req.transaction_id}' not found.")
    return sanitize_production_response(decision, mask_ids=True)
