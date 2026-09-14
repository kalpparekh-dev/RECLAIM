import time
import queue
import threading
import datetime
from typing import Dict, Any, Optional

from backend.app.payment_service import get_payment_service
from backend.app.policy_service import get_policy_service
from backend.app.customer_guard import get_customer_guard
from backend.app.action_recommendation import get_action_recommendation_service
from backend.app.net_value_engine import calculate_expected_net_value
from backend.app.reconciliation_service import get_reconciliation_service
from backend.app.domain.models import PaymentState

class BackgroundQueueWorker:
    """
    In-Memory Event Queue & Background Outbox Worker Simulator.
    Processes webhook failure events asynchronously, triggers RECLAIM policy decisioning,
    schedules retries, and reconciles outcomes.
    """

    def __init__(self):
        self.event_queue = queue.Queue()
        self.processed_count = 0
        self.failed_count = 0
        self.dead_letter_queue = []
        self.worker_thread = None
        self.running = False
        self._start_worker()

    def _start_worker(self):
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()

    def _worker_loop(self):
        while self.running:
            try:
                event = self.event_queue.get(timeout=1.0)
                self._process_event(event)
                self.event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.failed_count += 1
                print(f"[RECLAIM QUEUE WORKER ERROR] {str(e)}")

    def enqueue_event(self, event_data: Dict[str, Any]):
        """Enqueues an event payload for background worker processing."""
        self.event_queue.put(event_data)

    def _process_event(self, event: Dict[str, Any]):
        payment_id = event.get("payment_id")
        event_type = event.get("event_type", "payment.failed")
        amount = event.get("amount", 0.0)

        if not payment_id:
            self.dead_letter_queue.append({"event": event, "reason": "Missing payment_id"})
            self.failed_count += 1
            return

        payment_svc = get_payment_service()
        payment = payment_svc.get_payment_by_id(payment_id)
        if not payment:
            self.dead_letter_queue.append({"event": event, "reason": f"Payment {payment_id} not found"})
            self.failed_count += 1
            return

        policy_svc = get_policy_service()
        customer_guard = get_customer_guard()
        rec_svc = get_action_recommendation_service()
        reconciler = get_reconciliation_service()

        if event_type == "payment.failed":
            # 1. Transition payment state to FAILED & RECOVERY_ELIGIBLE
            try:
                payment_svc.transition_payment_state(
                    payment_id=payment_id,
                    new_state=PaymentState.FAILED,
                    event_type="GATEWAY_PAYMENT_FAILED",
                    reason=event.get("failure_reason", "Gateway decline"),
                    actor="GATEWAY_WEBHOOK",
                    failure_reason=event.get("failure_reason")
                )
                payment_svc.transition_payment_state(
                    payment_id=payment_id,
                    new_state=PaymentState.RECOVERY_ELIGIBLE,
                    event_type="RECOVERY_EVALUATION_STARTED",
                    reason="Payment eligible for causal recovery policy evaluation",
                    actor="WORKER"
                )
            except Exception:
                pass # Already transitioned or processed

            # 2. Query Policy Decision Engine
            tx_id = payment.get("external_id", f"RECLAIM-V2-000001")
            decision_data = policy_svc.get_decision(tx_id)
            policy_decision = decision_data.get("selected_policy", "RETRY_ALL")

            # 3. Customer Safeguard Check
            guard_res = customer_guard.evaluate_execution_decision(
                customer_id=payment["customer_id"],
                transaction_id=tx_id,
                policy_decision=policy_decision
            )

            execution_decision = guard_res["execution_decision"]
            suppression_reason = guard_res["suppression_reason"]

            # 4. Action Recommendation & Net Value
            action_rec = rec_svc.recommend_action(
                transaction_id=tx_id,
                policy_decision=policy_decision,
                failure_reason=payment.get("failure_reason"),
                payment_method=payment.get("payment_method"),
                amount=payment["amount"]
            )

            net_val_res = calculate_expected_net_value(
                expected_recovery_value=payment["amount"] * decision_data.get("predicted_treatment_probability", 0.5),
                action_type=action_rec["action_type"]
            )

            if not net_val_res["is_profitable"] or net_val_res["expected_net_value"] <= 0:
                if execution_decision == "EXECUTE":
                    execution_decision = "SUPPRESS"
                    suppression_reason = net_val_res.get("reason") or "NEGATIVE_NET_VALUE"

            # 5. Record Recovery Action
            act_status = "EXECUTED" if execution_decision == "EXECUTE" else "SUPPRESSED"
            rec_act = payment_svc.record_recovery_action(
                payment_id=payment_id,
                customer_id=payment["customer_id"],
                decision=policy_decision,
                decision_reason=decision_data.get("decision_rationale", "Policy evaluation"),
                policy_version="V10.2",
                model_score=decision_data.get("predicted_probability_uplift", 0.0),
                action_type=action_rec["action_type"],
                action_status=act_status
            )

            # 6. If Executed, schedule RETRY_SCHEDULED state
            if execution_decision == "EXECUTE":
                payment_svc.transition_payment_state(
                    payment_id=payment_id,
                    new_state=PaymentState.RETRY_SCHEDULED,
                    event_type="RECOVERY_ACTION_EXECUTED",
                    reason=f"Recovery action {action_rec['action_type']} scheduled",
                    actor="RECLAIM_ENGINE"
                )
            else:
                payment_svc.transition_payment_state(
                    payment_id=payment_id,
                    new_state=PaymentState.SUPPRESSED,
                    event_type="RECOVERY_ACTION_SUPPRESSED",
                    reason=f"Recovery suppressed: {suppression_reason}",
                    actor="RECLAIM_ENGINE"
                )

            self.processed_count += 1

        elif event_type in ("payment.captured", "payment.authorized"):
            # Reconcile successful payment outcome
            rec_res = reconciler.reconcile_payment_outcome(
                payment_id=payment_id,
                webhook_status="CAPTURED",
                webhook_amount=amount
            )
            self.processed_count += 1

    def get_worker_status(self) -> Dict[str, Any]:
        return {
            "worker_status": "RUNNING" if self.running else "STOPPED",
            "queue_depth": self.event_queue.qsize(),
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "dead_letter_count": len(self.dead_letter_queue),
            "dead_letter_events": self.dead_letter_queue[:10],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

_queue_worker_instance = None

def get_queue_worker() -> BackgroundQueueWorker:
    global _queue_worker_instance
    if _queue_worker_instance is None:
        _queue_worker_instance = BackgroundQueueWorker()
    return _queue_worker_instance
