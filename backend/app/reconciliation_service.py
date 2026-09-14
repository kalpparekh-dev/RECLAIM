import uuid
import datetime
from typing import Dict, Any, Optional, List
from backend.app.domain.models import ReconciliationStatus, PaymentState
from backend.app.payment_service import get_payment_service

class ReconciliationService:
    def reconcile_payment_outcome(
        self,
        payment_id: str,
        webhook_status: str,
        webhook_amount: float,
        recovery_action_id: Optional[str] = None,
        attempt_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reconciles payment state, recovery action, and gateway outcome.
        Assigns ReconciliationStatus: MATCHED, MISMATCHED, PENDING, UNKNOWN, MANUAL_REVIEW.
        """
        payment_svc = get_payment_service()
        payment = payment_svc.get_payment_by_id(payment_id)

        if not payment:
            return {
                "id": f"rec_{uuid.uuid4().hex[:12]}",
                "payment_id": payment_id,
                "reconciliation_status": ReconciliationStatus.UNKNOWN.value,
                "recovered_amount": 0.0,
                "reason": "Payment record not found in system."
            }

        orig_status = payment["status"]
        norm_webhook_status = webhook_status.upper()

        if norm_webhook_status in ("CAPTURED", "AUTHORIZED", "RECOVERED", "SUCCESS"):
            final_status = PaymentState.RECOVERED.value
            recovered_amount = webhook_amount

            # Check amount match
            if abs(webhook_amount - payment["amount"]) > 0.01:
                rec_status = ReconciliationStatus.MISMATCHED.value
                reason = f"Amount mismatch: Expected {payment['amount']}, Webhook got {webhook_amount}"
            else:
                rec_status = ReconciliationStatus.MATCHED.value
                reason = "Payment successfully recovered and reconciled with gateway outcome."

        elif norm_webhook_status in ("FAILED", "RECOVERY_FAILED"):
            final_status = PaymentState.RECOVERY_FAILED.value
            recovered_amount = 0.0
            rec_status = ReconciliationStatus.MATCHED.value
            reason = "Gateway outcome confirmed recovery attempt failed."

        elif norm_webhook_status in ("PENDING", "PROCESSING"):
            final_status = PaymentState.PENDING.value
            recovered_amount = 0.0
            rec_status = ReconciliationStatus.PENDING.value
            reason = "Payment outcome still pending gateway confirmation."

        else:
            final_status = orig_status
            recovered_amount = 0.0
            rec_status = ReconciliationStatus.MANUAL_REVIEW.value
            reason = f"Unrecognized gateway status '{webhook_status}' requires manual operator review."

        # Calculate time to recovery
        created_dt = datetime.datetime.fromisoformat(payment["created_at"])
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        time_to_rec = (now_dt - created_dt).total_seconds() if rec_status == ReconciliationStatus.MATCHED.value else None

        rec_id = f"rec_{uuid.uuid4().hex[:12]}"
        now_str = now_dt.isoformat()

        # Update DB record
        conn = payment_svc._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO outcome_reconciliations (id, payment_id, recovery_action_id, attempt_id, original_status, final_status, reconciliation_status, recovered_amount, time_to_recovery_seconds, reconciled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (rec_id, payment_id, recovery_action_id, attempt_id, orig_status, final_status, rec_status, recovered_amount, time_to_rec, now_str))
            conn.commit()
        finally:
            conn.close()

        # Transition payment state if recovered or failed
        if rec_status == ReconciliationStatus.MATCHED.value:
            if final_status == PaymentState.RECOVERED.value:
                payment_svc.transition_payment_state(
                    payment_id=payment_id,
                    new_state=PaymentState.RECOVERED,
                    event_type="RECONCILIATION_SUCCESS",
                    reason=reason,
                    actor="RECONCILER"
                )
            elif final_status == PaymentState.RECOVERY_FAILED.value:
                payment_svc.transition_payment_state(
                    payment_id=payment_id,
                    new_state=PaymentState.RECOVERY_FAILED,
                    event_type="RECONCILIATION_FAILED",
                    reason=reason,
                    actor="RECONCILER"
                )

        return {
            "id": rec_id,
            "payment_id": payment_id,
            "recovery_action_id": recovery_action_id,
            "attempt_id": attempt_id,
            "original_status": orig_status,
            "final_status": final_status,
            "reconciliation_status": rec_status,
            "recovered_amount": recovered_amount,
            "time_to_recovery_seconds": time_to_rec,
            "reconciled_at": now_str,
            "reason": reason
        }

    def list_reconciliations(self, page: int = 1, limit: int = 20) -> Dict[str, Any]:
        payment_svc = get_payment_service()
        conn = payment_svc._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM outcome_reconciliations")
            total = cursor.fetchone()["total"]

            offset = (page - 1) * limit
            cursor.execute("""
                SELECT * FROM outcome_reconciliations
                ORDER BY reconciled_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

            records = [dict(r) for r in cursor.fetchall()]
            return {
                "reconciliations": records,
                "total": total,
                "page": page,
                "limit": limit
            }
        finally:
            conn.close()

_reconciliation_service_instance = None

def get_reconciliation_service() -> ReconciliationService:
    global _reconciliation_service_instance
    if _reconciliation_service_instance is None:
        _reconciliation_service_instance = ReconciliationService()
    return _reconciliation_service_instance
