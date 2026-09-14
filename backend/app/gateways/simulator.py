import os
import hmac
import hashlib
import json
import uuid
import datetime
from typing import Dict, Any, Optional

from backend.app.gateways.base import PaymentGateway
from backend.app.webhook_verifier import verify_razorpay_signature

class SimulatorGateway(PaymentGateway):
    """
    Sandbox Payment Gateway Simulator.
    Simulates payment creation, failure injection, retries, and webhook emissions.
    Used for local testing and interactive demonstrations.
    """

    def __init__(self, webhook_secret: str = "sim_webhook_secret_razorpay_reclaim"):
        self.webhook_secret = webhook_secret

    def create_payment(
        self,
        customer_id: str,
        amount: float,
        currency: str = "INR",
        payment_method: str = "CARD",
        order_id: Optional[str] = None,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        ref_id = f"pay_sim_{uuid.uuid4().hex[:12]}"
        ord_id = order_id or f"order_{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return {
            "gateway_reference": ref_id,
            "order_id": ord_id,
            "customer_id": customer_id,
            "amount": amount,
            "currency": currency.upper(),
            "payment_method": payment_method.upper(),
            "gateway_status": "CREATED",
            "environment": "SIMULATOR_SANDBOX",
            "created_at": now
        }

    def fetch_payment(self, gateway_reference: str) -> Dict[str, Any]:
        return {
            "gateway_reference": gateway_reference,
            "gateway_status": "CAPTURED",
            "environment": "SIMULATOR_SANDBOX",
            "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    def initiate_retry(
        self,
        gateway_reference: str,
        action_type: str = "SMART_RETRY",
        target_outcome: Optional[str] = None
    ) -> Dict[str, Any]:
        attempt_ref = f"retry_sim_{uuid.uuid4().hex[:12]}"
        outcome = target_outcome or "SUCCESS"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return {
            "retry_reference": attempt_ref,
            "original_gateway_reference": gateway_reference,
            "action_type": action_type,
            "retry_status": "SUCCESS" if outcome == "SUCCESS" else "FAILED",
            "simulated_outcome": outcome,
            "environment": "SIMULATOR_SANDBOX",
            "timestamp": now
        }

    def fetch_payment_status(self, gateway_reference: str) -> str:
        return "CAPTURED"

    def verify_webhook_signature(
        self,
        raw_body: bytes,
        signature: Optional[str],
        secret: Optional[str] = None
    ) -> bool:
        sec = secret or self.webhook_secret
        return verify_razorpay_signature(raw_body, signature, secret=sec)

    def generate_signed_webhook_payload(
        self,
        event_type: str,
        payment_id: str,
        external_id: str,
        amount: float,
        status: str,
        failure_reason: Optional[str] = None,
        custom_secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """Utility for Sandbox page to generate signed Razorpay webhook payloads."""
        event_id = f"evt_sim_{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        payload = {
            "entity": "event",
            "event_id": event_id,
            "event": event_type, # payment.failed, payment.captured, payment.authorized
            "created_at": int(datetime.datetime.now().timestamp()),
            "payload": {
                "payment": {
                    "entity": {
                        "id": external_id,
                        "reclaim_payment_id": payment_id,
                        "amount": int(amount * 100), # Razorpay uses paise
                        "currency": "INR",
                        "status": status.lower(),
                        "error_code": failure_reason,
                        "error_description": f"Simulated failure: {failure_reason}" if failure_reason else None,
                        "created_at": now
                    }
                }
            }
        }

        raw_body = json.dumps(payload, sort_keys=True).encode("utf-8")
        sec = custom_secret or self.webhook_secret
        sig = hmac.new(sec.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()

        return {
            "event_id": event_id,
            "raw_body": raw_body.decode("utf-8"),
            "signature": sig,
            "payload_dict": payload
        }

    def handle_refund_or_reversal(
        self,
        gateway_reference: str,
        amount: Optional[float] = None,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        refund_id = f"rfnd_sim_{uuid.uuid4().hex[:12]}"
        return {
            "refund_id": refund_id,
            "gateway_reference": gateway_reference,
            "amount": amount,
            "status": "REFUNDED",
            "reason": reason or "Customer reversal",
            "environment": "SIMULATOR_SANDBOX"
        }
