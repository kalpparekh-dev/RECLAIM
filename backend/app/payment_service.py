import os
import json
import sqlite3
import uuid
import datetime
from typing import Dict, Any, Optional, List, Tuple
from backend.app.domain.models import (
    PaymentState,
    VALID_STATE_TRANSITIONS,
    FailureReason,
    ReconciliationStatus
)
from backend.app.privacy import mask_customer_id

DB_PATH = r"D:\RECLAIM\data\payments.db"

class PaymentServiceError(ValueError):
    """Raised when payment state transition or domain validation fails."""
    pass

class PaymentService:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS payments (
                    id TEXT PRIMARY KEY,
                    external_id TEXT UNIQUE NOT NULL,
                    order_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT NOT NULL,
                    payment_method TEXT NOT NULL,
                    status TEXT NOT NULL,
                    failure_reason TEXT,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS payment_state_history (
                    id TEXT PRIMARY KEY,
                    payment_id TEXT NOT NULL,
                    previous_state TEXT NOT NULL,
                    new_state TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    reason TEXT,
                    correlation_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    FOREIGN KEY (payment_id) REFERENCES payments (id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS payment_attempts (
                    id TEXT PRIMARY KEY,
                    payment_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    attempt_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    amount REAL NOT NULL,
                    gateway_reference TEXT,
                    failure_reason TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    correlation_id TEXT NOT NULL,
                    FOREIGN KEY (payment_id) REFERENCES payments (id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recovery_actions (
                    id TEXT PRIMARY KEY,
                    payment_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    decision_reason TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    model_score REAL NOT NULL,
                    score_semantics TEXT NOT NULL,
                    calibration_status TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    action_status TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    experiment_group TEXT NOT NULL,
                    attempt_id TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (payment_id) REFERENCES payments (id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS webhook_events (
                    event_id TEXT PRIMARY KEY,
                    payment_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    signature_valid INTEGER NOT NULL,
                    processing_status TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    processed_at TEXT,
                    failure_reason TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS outcome_reconciliations (
                    id TEXT PRIMARY KEY,
                    payment_id TEXT NOT NULL,
                    recovery_action_id TEXT,
                    attempt_id TEXT,
                    original_status TEXT NOT NULL,
                    final_status TEXT NOT NULL,
                    reconciliation_status TEXT NOT NULL,
                    recovered_amount REAL NOT NULL,
                    time_to_recovery_seconds REAL,
                    reconciled_at TEXT NOT NULL,
                    FOREIGN KEY (payment_id) REFERENCES payments (id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customer_feedback (
                    id TEXT PRIMARY KEY,
                    transaction_id TEXT NOT NULL,
                    payment_id TEXT,
                    rating INTEGER NOT NULL,
                    comment TEXT,
                    feedback_category TEXT NOT NULL,
                    submitted_at TEXT NOT NULL
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pay_cust ON payments(customer_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pay_stat ON payments(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pay_ext ON payments(external_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_pay ON payment_state_history(payment_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_att_pay ON payment_attempts(payment_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_pay ON recovery_actions(payment_id)")

            conn.commit()

    def create_payment(
        self,
        customer_id: str,
        amount: float,
        currency: str = "INR",
        payment_method: str = "CARD",
        order_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        actor: str = "API"
    ) -> Dict[str, Any]:
        """Creates a new payment record in CREATED state with initial attempt."""
        if not customer_id or not customer_id.strip():
            raise PaymentServiceError("Valid customer_id is required to create a payment.")
        if amount <= 0:
            raise PaymentServiceError("Payment amount must be greater than zero.")

        cid = customer_id.strip()
        pay_id = f"pay_{uuid.uuid4().hex[:16]}"
        ext_id = f"pay_sim_{uuid.uuid4().hex[:12]}"
        ord_id = order_id or f"order_{uuid.uuid4().hex[:12]}"
        idem_key = idempotency_key or f"idem_{pay_id}"
        corr_id = f"corr_{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.cursor()

            # Check existing idempotency_key
            cursor.execute("SELECT id FROM payments WHERE idempotency_key = ?", (idem_key,))
            existing = cursor.fetchone()
            if existing:
                conn.rollback()
                conn.close()
                return self.get_payment_by_id(existing["id"])

            cursor.execute("""
                INSERT INTO payments (id, external_id, order_id, customer_id, amount, currency, payment_method, status, failure_reason, idempotency_key, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, 1, ?, ?)
            """, (pay_id, ext_id, ord_id, cid, amount, currency.upper(), payment_method.upper(), PaymentState.CREATED.value, idem_key, now, now))

            # Initial State History
            hist_id = f"hist_{uuid.uuid4().hex[:12]}"
            cursor.execute("""
                INSERT INTO payment_state_history (id, payment_id, previous_state, new_state, event_type, reason, correlation_id, created_at, actor)
                VALUES (?, ?, 'NONE', ?, 'PAYMENT_CREATED', 'Initial payment creation', ?, ?, ?)
            """, (hist_id, pay_id, PaymentState.CREATED.value, corr_id, now, actor))

            # Initial Attempt
            att_id = f"att_{uuid.uuid4().hex[:12]}"
            cursor.execute("""
                INSERT INTO payment_attempts (id, payment_id, attempt_number, attempt_type, status, amount, gateway_reference, failure_reason, started_at, completed_at, correlation_id)
                VALUES (?, ?, 1, 'INITIAL', 'STARTED', ?, ?, NULL, ?, NULL, ?)
            """, (att_id, pay_id, amount, ext_id, now, corr_id))

            conn.commit()
            conn.close()
            return self.get_payment_by_id(pay_id)
        except Exception as e:
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
            raise PaymentServiceError(f"Failed creating payment: {str(e)}")

    def transition_payment_state(
        self,
        payment_id: str,
        new_state: PaymentState,
        event_type: str,
        reason: Optional[str] = None,
        actor: str = "WORKER",
        failure_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Atomically transitions payment state validating state machine constraints."""
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.cursor()

            cursor.execute("SELECT status, version FROM payments WHERE id = ?", (payment_id,))
            row = cursor.fetchone()
            if not row:
                conn.rollback()
                conn.close()
                raise PaymentServiceError(f"Payment '{payment_id}' not found.")

            curr_state = PaymentState(row["status"])
            curr_version = row["version"]

            # Validate transition if not same state
            if curr_state != new_state:
                allowed_next = VALID_STATE_TRANSITIONS.get(curr_state, [])
                if new_state not in allowed_next:
                    conn.rollback()
                    conn.close()
                    raise PaymentServiceError(f"Invalid state transition from {curr_state.value} to {new_state.value}.")

            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            corr_id = f"corr_{uuid.uuid4().hex[:12]}"

            cursor.execute("""
                UPDATE payments
                SET status = ?, failure_reason = ?, version = version + 1, updated_at = ?
                WHERE id = ? AND version = ?
            """, (new_state.value, failure_reason, now, payment_id, curr_version))

            if cursor.rowcount == 0:
                conn.rollback()
                conn.close()
                raise PaymentServiceError("Concurrent modification error on payment version update.")

            # Record State History
            hist_id = f"hist_{uuid.uuid4().hex[:12]}"
            cursor.execute("""
                INSERT INTO payment_state_history (id, payment_id, previous_state, new_state, event_type, reason, correlation_id, created_at, actor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (hist_id, payment_id, curr_state.value, new_state.value, event_type, reason, corr_id, now, actor))

            conn.commit()
            conn.close()
            return self.get_payment_by_id(payment_id)
        except Exception as e:
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
            raise PaymentServiceError(f"Failed transitioning payment state: {str(e)}")

    def record_payment_attempt(
        self,
        payment_id: str,
        attempt_type: str,
        status: str,
        amount: float,
        gateway_reference: Optional[str] = None,
        failure_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Records a new payment attempt (Initial or Retry)."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM payment_attempts WHERE payment_id = ?", (payment_id,))
            att_num = cursor.fetchone()["cnt"] + 1

            att_id = f"att_{uuid.uuid4().hex[:12]}"
            corr_id = f"corr_{uuid.uuid4().hex[:12]}"
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()

            cursor.execute("""
                INSERT INTO payment_attempts (id, payment_id, attempt_number, attempt_type, status, amount, gateway_reference, failure_reason, started_at, completed_at, correlation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (att_id, payment_id, att_num, attempt_type, status, amount, gateway_reference, failure_reason, now, now, corr_id))

            conn.commit()
            conn.close()
            return {
                "id": att_id,
                "payment_id": payment_id,
                "attempt_number": att_num,
                "attempt_type": attempt_type,
                "status": status,
                "amount": amount,
                "gateway_reference": gateway_reference,
                "failure_reason": failure_reason,
                "started_at": now,
                "completed_at": now,
                "correlation_id": corr_id
            }
        finally:
            pass

    def record_recovery_action(
        self,
        payment_id: str,
        customer_id: str,
        decision: str,
        decision_reason: str,
        policy_version: str,
        model_score: float,
        action_type: str,
        action_status: str, # SCHEDULED, EXECUTED, SUPPRESSED, FAILED
        experiment_id: str = "EXP_RECLAIM_V10_2",
        experiment_group: str = "TREATMENT",
        attempt_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Records a recovery action distinguishing decision generation from execution."""
        act_id = f"act_{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO recovery_actions (id, payment_id, customer_id, decision, decision_reason, policy_version, model_score, score_semantics, calibration_status, action_type, action_status, experiment_id, experiment_group, attempt_id, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'MODEL_RANKING_SCORE', 'NOT_CALIBRATED', ?, ?, ?, ?, ?, ?, ?)
            """, (act_id, payment_id, customer_id, decision, decision_reason, policy_version, model_score, action_type, action_status, experiment_id, experiment_group, attempt_id, now, now if action_status in ("EXECUTED", "SUPPRESSED") else None))
            conn.commit()

            return {
                "id": act_id,
                "payment_id": payment_id,
                "customer_id": customer_id,
                "decision": decision,
                "decision_reason": decision_reason,
                "policy_version": policy_version,
                "model_score": model_score,
                "score_semantics": "MODEL_RANKING_SCORE",
                "calibration_status": "NOT_CALIBRATED",
                "action_type": action_type,
                "action_status": action_status,
                "experiment_id": experiment_id,
                "experiment_group": experiment_group,
                "attempt_id": attempt_id,
                "created_at": now,
                "completed_at": now if action_status in ("EXECUTED", "SUPPRESSED") else None
            }
        finally:
            conn.close()

    def get_payment_by_id(self, payment_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM payments WHERE id = ? OR external_id = ?", (payment_id, payment_id))
            row = cursor.fetchone()
            if not row:
                return None

            pay_dict = dict(row)
            pay_dict["masked_customer_id"] = mask_customer_id(pay_dict["customer_id"])

            # Load Attempts
            cursor.execute("SELECT * FROM payment_attempts WHERE payment_id = ? ORDER BY attempt_number ASC", (pay_dict["id"],))
            pay_dict["attempts"] = [dict(r) for r in cursor.fetchall()]

            # Load State History
            cursor.execute("SELECT * FROM payment_state_history WHERE payment_id = ? ORDER BY created_at ASC", (pay_dict["id"],))
            pay_dict["state_history"] = [dict(r) for r in cursor.fetchall()]

            # Load Recovery Action
            cursor.execute("SELECT * FROM recovery_actions WHERE payment_id = ? ORDER BY created_at DESC LIMIT 1", (pay_dict["id"],))
            act_row = cursor.fetchone()
            pay_dict["recovery_action"] = dict(act_row) if act_row else None

            # Load Reconciliation
            cursor.execute("SELECT * FROM outcome_reconciliations WHERE payment_id = ? ORDER BY reconciled_at DESC LIMIT 1", (pay_dict["id"],))
            rec_row = cursor.fetchone()
            pay_dict["reconciliation"] = dict(rec_row) if rec_row else None

            return pay_dict
        finally:
            conn.close()

    def list_payments(
        self,
        page: int = 1,
        limit: int = 20,
        status: Optional[str] = None,
        payment_method: Optional[str] = None,
        search: Optional[str] = None
    ) -> Dict[str, Any]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            where_clauses = []
            params = []

            if status:
                where_clauses.append("status = ?")
                params.append(status.upper())

            if payment_method:
                where_clauses.append("payment_method = ?")
                params.append(payment_method.upper())

            if search:
                where_clauses.append("(id LIKE ? OR external_id LIKE ? OR order_id LIKE ? OR customer_id LIKE ?)")
                search_term = f"%{search.strip()}%"
                params.extend([search_term, search_term, search_term, search_term])

            where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            cursor.execute(f"SELECT COUNT(*) as total FROM payments {where_str}", params)
            total = cursor.fetchone()["total"]

            total_pages = max(1, (total + limit - 1) // limit)
            page = max(1, min(page, total_pages))
            offset = (page - 1) * limit

            cursor.execute(f"""
                SELECT * FROM payments {where_str}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])

            rows = cursor.fetchall()
            payments = []
            for r in rows:
                p = dict(r)
                p["masked_customer_id"] = mask_customer_id(p["customer_id"])
                payments.append(p)

            return {
                "payments": payments,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total_matching": total,
                    "total_pages": total_pages
                }
            }
        finally:
            conn.close()

_payment_service_instance = None

def get_payment_service() -> PaymentService:
    global _payment_service_instance
    if _payment_service_instance is None:
        _payment_service_instance = PaymentService()
    return _payment_service_instance
