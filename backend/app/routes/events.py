import uuid
import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Header, Request
from pydantic import BaseModel, Field

from backend.app.policy_service import get_policy_service
from backend.app.runtime_control import get_runtime_control, RuntimeMode
from backend.app.idempotency_store import get_idempotency_store
from backend.app.customer_guard import get_customer_guard
from backend.app.action_recommendation import get_action_recommendation_service
from backend.app.net_value_engine import calculate_expected_net_value
from backend.app.action_provider import get_action_provider
from backend.app.privacy import mask_customer_id, mask_transaction_id
from backend.app.serializer import sanitize_production_response
from backend.app.webhook_verifier import verify_razorpay_signature, WebhookVerificationError
from backend.app.segment_guard import get_segment_guard
from backend.app.experiment_router import get_experiment_router
from backend.app.drift_monitor_service import get_drift_monitor_service

router = APIRouter()

class PaymentFailedEventRequest(BaseModel):
    event_id: str
    transaction_id: str
    customer_id: Optional[str] = None
    amount: float = Field(gt=0)
    payment_method: Optional[str] = "CARD"
    failure_reason: Optional[str] = "BANK_DECLINE"
    currency: Optional[str] = "INR"
    idempotency_key: Optional[str] = None

class EvaluateDecisionRequest(BaseModel):
    transaction_id: str
    customer_id: Optional[str] = None

class ActionPreviewRequest(BaseModel):
    transaction_id: str
    action_type: str = "SMART_RETRY"
    estimated_cost: float = 0.50

class ModeChangeRequest(BaseModel):
    mode: str
    reason: Optional[str] = "Operator configuration change"

@router.get("/readiness")
def readiness_check():
    policy_svc = get_policy_service()
    runtime = get_runtime_control()
    summary = policy_svc.get_summary()

    # P0-5: If hash verification failed or runtime mode is not active, system is NOT_READY
    is_ready = policy_svc.hash_validated and summary.get("status") == "VALIDATED" and runtime.is_active()

    return {
        "status": "READY" if is_ready else "NOT_READY",
        "runtime_mode": runtime.get_config().get("mode"),
        "policy_loaded": summary.get("status") is not None,
        "hash_validated": policy_svc.hash_validated,
        "policy_version": summary.get("policy_version", "V10.2"),
        "total_transactions": summary.get("total_transactions", 10000),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

@router.get("/policy/metadata")
def get_policy_metadata():
    policy_svc = get_policy_service()
    return sanitize_production_response({
        "policy_summary": policy_svc.get_summary(),
        "registry_metadata": policy_svc.registry_metadata,
        "data_mode": "PRODUCTION_POLICY",
        "is_production": True,
        "score_semantics": "MODEL_RANKING_SCORE",
        "calibration_status": "NOT_CALIBRATED"
    })

@router.post("/events/payment-failed")
async def process_payment_failed_event(
    request: Request,
    event: PaymentFailedEventRequest,
    x_idempotency_key: Optional[str] = Header(None),
    x_razorpay_signature: Optional[str] = Header(None, alias="X-Razorpay-Signature")
):
    # P0-6: Webhook Signature Verification
    try:
        raw_body = await request.body()
        verify_razorpay_signature(raw_body, x_razorpay_signature)
    except WebhookVerificationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    idempotency_key = x_idempotency_key or event.idempotency_key or event.event_id
    idempotency_store = get_idempotency_store()

    # P0-7: Idempotency Check with payload hash & collision handling
    existing_resp = idempotency_store.check_and_start(
        idempotency_key=idempotency_key,
        event_id=event.event_id,
        transaction_id=event.transaction_id,
        payload_data=event.model_dump()
    )

    if existing_resp:
        if existing_resp.get("is_payload_conflict"):
            raise HTTPException(status_code=409, detail=existing_resp.get("message", "Idempotency key payload conflict"))
        if existing_resp.get("is_duplicate_processing"):
            return {
                "event_id": event.event_id,
                "status": "DUPLICATE_PROCESSING_REJECTED",
                "message": existing_resp.get("message", "Event is currently being processed by another thread.")
            }
        return sanitize_production_response(existing_resp)

    # P0-8: Global safe fallback exception handler
    try:
        policy_svc = get_policy_service()
        runtime = get_runtime_control()
        customer_guard = get_customer_guard()
        rec_svc = get_action_recommendation_service()
        segment_guard = get_segment_guard()
        exp_router = get_experiment_router()

        # P0-2: Make missing customer ID unsafe
        if not event.customer_id or not str(event.customer_id).strip():
            resp = sanitize_production_response({
                "event_id": event.event_id,
                "transaction_id": event.transaction_id,
                "policy_version": "V10.2",
                "policy_decision": "RETRY_ALL",
                "execution_decision": "SUPPRESS",
                "suppression_reason": "CUSTOMER_ID_REQUIRED",
                "customer_id_available": False,
                "fallback_applied": True,
                "action_recommendation": {"action_type": "RETRY_ALL"},
                "data_mode": "PRODUCTION_POLICY",
                "is_production": True,
                "simulation_mode": True,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })
            idempotency_store.complete(idempotency_key, "RETRY_ALL", resp)
            return resp

        customer_id_clean = event.customer_id.strip()

        # P1-13: A/B Experiment assignment check
        exp_res = exp_router.assign_group(customer_id_clean, event.transaction_id)
        
        # 1. Get Production Decision
        decision_data = policy_svc.get_decision(event.transaction_id)
        policy_decision = decision_data.get("selected_policy", "RETRY_ALL")

        if exp_res.get("forced_action") == "RETRY_ALL":
            policy_decision = "RETRY_ALL"

        # 2. Evaluate Customer Safeguards (P0-3)
        guard_res = customer_guard.evaluate_execution_decision(
            customer_id=customer_id_clean,
            transaction_id=event.transaction_id,
            policy_decision=policy_decision
        )

        execution_decision = guard_res["execution_decision"]
        suppression_reason = guard_res["suppression_reason"]

        # 3. P1-11: Segment Safety Guard
        if execution_decision == "EXECUTE":
            seg_res = segment_guard.evaluate_segment_safety(
                payment_method=event.payment_method,
                failure_reason=event.failure_reason,
                amount=event.amount
            )
            if not seg_res["is_safe"]:
                execution_decision = "SUPPRESS"
                suppression_reason = seg_res["suppression_reason"]

        # 4. Action Recommendation
        action_rec = rec_svc.recommend_action(
            transaction_id=event.transaction_id,
            policy_decision=policy_decision,
            failure_reason=event.failure_reason,
            payment_method=event.payment_method,
            amount=event.amount
        )

        # 5. P0-4: Net Value Calculation & Strict Non-Negative Profit Check
        net_val_res = calculate_expected_net_value(
            expected_recovery_value=event.amount * decision_data.get("predicted_treatment_probability", 0.5),
            action_type=action_rec["action_type"],
            currency=event.currency or "INR"
        )

        if not net_val_res["is_profitable"] or net_val_res["expected_net_value"] <= 0:
            if execution_decision == "EXECUTE":
                execution_decision = "SUPPRESS"
                suppression_reason = net_val_res.get("reason") or "NEGATIVE_NET_VALUE"

        # 6. Record Customer Action & Segment history
        if execution_decision == "EXECUTE":
            segment_guard.record_target_segment(event.payment_method, event.failure_reason, event.amount)

        raw_response_payload = {
            "event_id": event.event_id,
            "transaction_id": event.transaction_id,
            "customer_id": customer_id_clean,
            "policy_version": "V10.2",
            "policy_decision": policy_decision,
            "execution_decision": execution_decision,
            "suppression_reason": suppression_reason,
            "experiment_group": exp_res.get("assigned_group"),
            "action_recommendation": action_rec,
            "net_value_analysis": net_val_res,
            "score_semantics": "MODEL_RANKING_SCORE",
            "calibration_status": "NOT_CALIBRATED",
            "data_mode": "PRODUCTION_POLICY",
            "is_production": True,
            "simulation_mode": True,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        # P0-1 & P0-9: Sanitize payload to strip oracle fields and raw customer IDs
        sanitized_response = sanitize_production_response(raw_response_payload)
        idempotency_store.complete(idempotency_key, action_rec.get("action_type"), sanitized_response)
        return sanitized_response

    except Exception as e:
        idempotency_store.fail(idempotency_key, str(e))
        # P0-8: Return safe fallback response instead of unhandled HTTP 500 error
        fallback_resp = sanitize_production_response({
            "event_id": event.event_id,
            "transaction_id": event.transaction_id,
            "customer_id": event.customer_id or "UNKNOWN",
            "policy_version": "V10.2",
            "policy_decision": "RETRY_ALL",
            "execution_decision": "SUPPRESS",
            "fallback_applied": True,
            "fallback_reason": f"INTERNAL_SAFETY_FALLBACK: {str(e)}",
            "action_recommendation": {"action_type": "RETRY_ALL"},
            "data_mode": "PRODUCTION_POLICY",
            "is_production": True,
            "simulation_mode": True,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })
        return fallback_resp

@router.post("/decision/evaluate")
def evaluate_decision(req: EvaluateDecisionRequest):
    # P0-2: Require valid customer ID
    if not req.customer_id or not str(req.customer_id).strip():
        return sanitize_production_response({
            "found": False,
            "transaction_id": req.transaction_id,
            "selected_policy": "RETRY_ALL",
            "intervention_selected": 0,
            "action": "RETRY_ALL",
            "fallback_applied": True,
            "fallback_reason": "CUSTOMER_ID_REQUIRED",
            "customer_id_available": False,
            "decision_rationale": "Missing or empty customer_id; safe RETRY_ALL fallback applied."
        })

    customer_id_clean = req.customer_id.strip()
    policy_svc = get_policy_service()
    decision = policy_svc.get_decision(req.transaction_id)
    
    customer_guard = get_customer_guard()
    guard_res = customer_guard.evaluate_execution_decision(
        customer_id=customer_id_clean,
        transaction_id=req.transaction_id,
        policy_decision=decision.get("selected_policy", "RETRY_ALL")
    )
    decision["customer_safeguards"] = guard_res
    return sanitize_production_response(decision)

@router.post("/action/preview")
def preview_action(req: ActionPreviewRequest):
    provider = get_action_provider()
    res = provider.preview({
        "transaction_id": req.transaction_id,
        "action_type": req.action_type,
        "estimated_cost": req.estimated_cost
    })
    return sanitize_production_response(res)

@router.post("/runtime/mode")
def change_runtime_mode(req: ModeChangeRequest):
    runtime = get_runtime_control()
    try:
        new_mode = RuntimeMode(req.mode.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid runtime mode '{req.mode}'. Allowed: ACTIVE, PAUSED, FALLBACK, MAINTENANCE")

    return sanitize_production_response(runtime.set_mode(new_mode, updated_by="ADMIN_API", reason=req.reason))
