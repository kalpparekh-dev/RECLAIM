import json
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, HTTPException, Header
from pydantic import BaseModel

from backend.app.gateways.factory import get_simulator_gateway
from backend.app.payment_service import get_payment_service, PaymentServiceError
from backend.app.queue_worker import get_queue_worker
from backend.app.serializer import sanitize_production_response
from backend.app.webhook_verifier import verify_razorpay_signature, WebhookVerificationError
from backend.app.domain.models import PaymentState

router = APIRouter(prefix="/api", tags=["Gateway & Sandbox"])

class SandboxCreatePaymentRequest(BaseModel):
    customer_id: str
    amount: float
    currency: str = "INR"
    payment_method: str = "CARD"
    initial_outcome: str = "FAILURE" # SUCCESS, FAILURE, PENDING
    failure_reason: str = "BANK_SERVER_ERROR"

class SandboxWebhookSimulateRequest(BaseModel):
    payment_id: str
    event_type: str = "payment.failed" # payment.failed, payment.captured
    tamper_signature: bool = False
    alter_body: bool = False

@router.post("/webhooks/simulator")
@router.post("/webhooks/razorpay")
async def ingest_webhook_event(
    request: Request,
    x_razorpay_signature: Optional[str] = Header(None, alias="X-Razorpay-Signature")
):
    raw_body = await request.body()
    try:
        verify_razorpay_signature(raw_body, x_razorpay_signature)
    except WebhookVerificationError as ve:
        raise HTTPException(status_code=400, detail=f"Webhook signature verification failed: {str(ve)}")

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed JSON payload in webhook body.")

    event_id = payload.get("event_id") or payload.get("id") or f"evt_{hash(raw_body) & 0xffffffff}"
    event_type = payload.get("event", "payment.failed")
    
    # Extract payment entity
    entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
    reclaim_pay_id = entity.get("reclaim_payment_id") or entity.get("id")
    amount = float(entity.get("amount", 0)) / 100.0 if entity.get("amount") else 100.0
    failure_reason = entity.get("error_code")

    # Enqueue background worker for processing
    worker = get_queue_worker()
    worker.enqueue_event({
        "event_id": event_id,
        "payment_id": reclaim_pay_id,
        "event_type": event_type,
        "amount": amount,
        "failure_reason": failure_reason,
        "payload": payload
    })

    return {
        "status": "ACCEPTED",
        "event_id": event_id,
        "event_type": event_type,
        "reclaim_payment_id": reclaim_pay_id,
        "message": "Webhook payload validated and enqueued for background worker processing."
    }

@router.post("/gateway/sandbox/create-payment")
def sandbox_create_payment(req: SandboxCreatePaymentRequest):
    sim = get_simulator_gateway()
    payment_svc = get_payment_service()

    payment = payment_svc.create_payment(
        customer_id=req.customer_id,
        amount=req.amount,
        currency=req.currency,
        payment_method=req.payment_method,
        actor="SANDBOX"
    )

    pay_id = payment["id"]
    ext_id = payment["external_id"]

    if req.initial_outcome == "FAILURE":
        payment_svc.transition_payment_state(
            payment_id=pay_id,
            new_state=PaymentState.FAILED,
            event_type="SANDBOX_INITIAL_FAILURE",
            reason=f"Sandbox injected failure: {req.failure_reason}",
            actor="SANDBOX",
            failure_reason=req.failure_reason
        )
        # Enqueue background worker processing
        worker = get_queue_worker()
        worker.enqueue_event({
            "payment_id": pay_id,
            "event_type": "payment.failed",
            "failure_reason": req.failure_reason,
            "amount": req.amount
        })
    elif req.initial_outcome == "SUCCESS":
        payment_svc.transition_payment_state(
            payment_id=pay_id,
            new_state=PaymentState.CAPTURED,
            event_type="SANDBOX_INITIAL_SUCCESS",
            reason="Sandbox payment captured",
            actor="SANDBOX"
        )
    elif req.initial_outcome == "PENDING":
        payment_svc.transition_payment_state(
            payment_id=pay_id,
            new_state=PaymentState.PENDING,
            event_type="SANDBOX_INITIAL_PENDING",
            reason="Sandbox payment pending gateway confirmation",
            actor="SANDBOX"
        )

    updated_payment = payment_svc.get_payment_by_id(pay_id)
    return sanitize_production_response(updated_payment)

@router.post("/gateway/sandbox/send-signed-webhook")
def sandbox_send_signed_webhook(req: SandboxWebhookSimulateRequest):
    sim = get_simulator_gateway()
    payment_svc = get_payment_service()

    payment = payment_svc.get_payment_by_id(req.payment_id)
    if not payment:
        raise HTTPException(status_code=404, detail=f"Payment '{req.payment_id}' not found.")

    secret = "invalid_tampered_secret" if req.tamper_signature else sim.webhook_secret
    
    webhook_pkg = sim.generate_signed_webhook_payload(
        event_type=req.event_type,
        payment_id=payment["id"],
        external_id=payment["external_id"],
        amount=payment["amount"],
        status="failed" if req.event_type == "payment.failed" else "captured",
        failure_reason=payment.get("failure_reason", "BANK_SERVER_ERROR"),
        custom_secret=secret
    )

    if req.alter_body:
        webhook_pkg["payload_dict"]["payload"]["payment"]["entity"]["amount"] = 999999
        webhook_pkg["raw_body"] = json.dumps(webhook_pkg["payload_dict"], sort_keys=True)

    return {
        "simulator_package": webhook_pkg,
        "instructions": "Send POST /api/webhooks/simulator with raw_body and X-Razorpay-Signature header to test ingestion."
    }
