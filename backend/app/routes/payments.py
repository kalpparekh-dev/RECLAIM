from typing import Optional
from fastapi import APIRouter, Query, HTTPException, Header
from backend.app.domain.models import (
    CreatePaymentRequest,
    SimulatePaymentOutcomeRequest,
    InitiateRetryRequest,
    PaymentState,
    FailureReason
)
from backend.app.payment_service import get_payment_service, PaymentServiceError
from backend.app.gateways.factory import get_default_gateway
from backend.app.queue_worker import get_queue_worker
from backend.app.serializer import sanitize_production_response

router = APIRouter(prefix="/api/payments", tags=["Payments"])

@router.post("")
def create_payment(
    req: CreatePaymentRequest,
    x_idempotency_key: Optional[str] = Header(None)
):
    payment_svc = get_payment_service()
    gateway = get_default_gateway()

    idem_key = x_idempotency_key or req.idempotency_key
    try:
        payment = payment_svc.create_payment(
            customer_id=req.customer_id,
            amount=req.amount,
            currency=req.currency,
            payment_method=req.payment_method,
            order_id=req.order_id,
            idempotency_key=idem_key,
            actor="API"
        )
        return sanitize_production_response(payment)
    except PaymentServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("")
def list_payments(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by PaymentState"),
    payment_method: Optional[str] = Query(None, description="Filter by CARD, UPI, NETBANKING"),
    search: Optional[str] = Query(None, description="Search ID, Order ID, Customer ID")
):
    payment_svc = get_payment_service()
    res = payment_svc.list_payments(
        page=page,
        limit=limit,
        status=status,
        payment_method=payment_method,
        search=search
    )
    return sanitize_production_response(res, mask_ids=True)

@router.get("/{payment_id}")
def get_payment_detail(payment_id: str):
    payment_svc = get_payment_service()
    payment = payment_svc.get_payment_by_id(payment_id)
    if not payment:
        raise HTTPException(status_code=404, detail=f"Payment '{payment_id}' not found.")
    return sanitize_production_response(payment, mask_ids=True)

@router.get("/{payment_id}/timeline")
def get_payment_timeline(payment_id: str):
    payment_svc = get_payment_service()
    payment = payment_svc.get_payment_by_id(payment_id)
    if not payment:
        raise HTTPException(status_code=404, detail=f"Payment '{payment_id}' not found.")

    timeline = []
    # Payment creation
    timeline.append({
        "timestamp": payment["created_at"],
        "stage": "PAYMENT_CREATED",
        "title": "Payment Created",
        "status": "CREATED",
        "detail": f"Amount: {payment['currency']} {payment['amount']:.2f} via {payment['payment_method']}"
    })

    # State history
    for h in payment.get("state_history", []):
        timeline.append({
            "timestamp": h["created_at"],
            "stage": h["event_type"],
            "title": f"State: {h['previous_state']} -> {h['new_state']}",
            "status": h["new_state"],
            "detail": h.get("reason") or f"Actor: {h['actor']}"
        })

    # Attempts
    for a in payment.get("attempts", []):
        timeline.append({
            "timestamp": a["started_at"],
            "stage": f"ATTEMPT_{a['attempt_number']}",
            "title": f"Attempt #{a['attempt_number']} ({a['attempt_type']})",
            "status": a["status"],
            "detail": f"Failure reason: {a.get('failure_reason')}" if a.get("failure_reason") else "Attempt initiated"
        })

    # Recovery Action
    rec_act = payment.get("recovery_action")
    if rec_act:
        timeline.append({
            "timestamp": rec_act["created_at"],
            "stage": "RECLAIM_DECISION",
            "title": f"RECLAIM Decision: {rec_act['decision']}",
            "status": rec_act["action_status"],
            "detail": f"Action: {rec_act['action_type']} | Score: {rec_act['model_score']:.4f} ({rec_act['score_semantics']})"
        })

    # Reconciliation
    rec = payment.get("reconciliation")
    if rec:
        timeline.append({
            "timestamp": rec["reconciled_at"],
            "stage": "OUTCOME_RECONCILIATION",
            "title": f"Reconciliation: {rec['reconciliation_status']}",
            "status": rec["final_status"],
            "detail": f"Recovered Amount: {rec['recovered_amount']:.2f} | Time: {rec.get('time_to_recovery_seconds')}s"
        })

    # Sort by timestamp
    timeline.sort(key=lambda x: x["timestamp"])

    return sanitize_production_response({
        "payment_id": payment_id,
        "external_id": payment["external_id"],
        "status": payment["status"],
        "timeline": timeline
    })

@router.post("/{payment_id}/fail")
def fail_payment(
    payment_id: str,
    reason: Optional[str] = "BANK_SERVER_ERROR"
):
    payment_svc = get_payment_service()
    try:
        payment = payment_svc.transition_payment_state(
            payment_id=payment_id,
            new_state=PaymentState.FAILED,
            event_type="MANUAL_FAILURE_SIMULATED",
            reason=f"Payment failed: {reason}",
            actor="SIMULATOR",
            failure_reason=reason
        )

        # Enqueue background failure processing
        worker = get_queue_worker()
        worker.enqueue_event({
            "payment_id": payment_id,
            "event_type": "payment.failed",
            "failure_reason": reason,
            "amount": payment["amount"]
        })

        return sanitize_production_response(payment)
    except PaymentServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{payment_id}/retry")
def initiate_manual_retry(
    payment_id: str,
    req: Optional[InitiateRetryRequest] = None
):
    payment_svc = get_payment_service()
    payment = payment_svc.get_payment_by_id(payment_id)
    if not payment:
        raise HTTPException(status_code=404, detail=f"Payment '{payment_id}' not found.")

    try:
        # Transition to RETRY_PROCESSING
        payment_svc.transition_payment_state(
            payment_id=payment_id,
            new_state=PaymentState.RETRY_PROCESSING,
            event_type="RETRY_INITIATED",
            reason="Manual payment retry initiated",
            actor="OPERATOR"
        )

        # Record attempt
        att = payment_svc.record_payment_attempt(
            payment_id=payment_id,
            attempt_type="RETRY",
            status="SUCCESS",
            amount=payment["amount"],
            gateway_reference=payment["external_id"]
        )

        # Auto-reconcile success
        payment_svc.transition_payment_state(
            payment_id=payment_id,
            new_state=PaymentState.RECOVERED,
            event_type="RETRY_SUCCESSFUL",
            reason="Payment retry succeeded on gateway",
            actor="GATEWAY"
        )

        return sanitize_production_response(payment_svc.get_payment_by_id(payment_id))
    except PaymentServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
