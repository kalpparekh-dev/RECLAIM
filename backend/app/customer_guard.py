import os
import sqlite3
import datetime
from typing import Dict, Any, Optional

DB_PATH = r"D:\RECLAIM\data\customer_safeguards.db"

class CustomerGuard:
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
                CREATE TABLE IF NOT EXISTS customer_action_limits (
                    customer_id TEXT PRIMARY KEY,
                    last_action_at TEXT,
                    daily_count INTEGER NOT NULL DEFAULT 0,
                    weekly_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customer_action_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    transaction_id TEXT NOT NULL,
                    execution_decision TEXT NOT NULL,
                    suppression_reason TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cust_hist ON customer_action_history(customer_id, timestamp)")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customer_opt_outs (
                    customer_id TEXT PRIMARY KEY,
                    opted_out_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def opt_out_customer(self, customer_id: str):
        if not customer_id or not isinstance(customer_id, str) or not customer_id.strip():
            return
        cid = customer_id.strip()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO customer_opt_outs (customer_id, opted_out_at) VALUES (?, ?)", (cid, now))
            conn.commit()

    def evaluate_execution_decision(
        self,
        customer_id: Optional[str],
        transaction_id: str,
        policy_decision: str,
        cooldown_hours: int = 24,
        max_actions_per_day: int = 1,
        max_actions_per_week: int = 3
    ) -> Dict[str, Any]:
        """
        Atomically evaluates customer safeguards and reserves action slot if allowed.
        Requires valid customer_id. Missing/empty customer_id triggers safe SUPPRESS.
        """
        # P0-2: Make missing customer ID unsafe
        if not customer_id or not isinstance(customer_id, str) or not customer_id.strip():
            return {
                "policy_decision": policy_decision,
                "execution_decision": "SUPPRESS",
                "suppression_reason": "CUSTOMER_ID_REQUIRED",
                "customer_id_available": False,
                "fallback_applied": True,
                "action_executable": False
            }

        cid = customer_id.strip()

        if policy_decision != "TARGET":
            return {
                "policy_decision": policy_decision,
                "execution_decision": "EXECUTE_FALLBACK",
                "suppression_reason": None,
                "customer_id_available": True,
                "action_executable": True
            }

        now_dt = datetime.datetime.now(datetime.timezone.utc)
        now_str = now_dt.isoformat()

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.cursor()

            # 1. Opt-out check
            cursor.execute("SELECT 1 FROM customer_opt_outs WHERE customer_id = ?", (cid,))
            if cursor.fetchone():
                conn.rollback()
                conn.close()
                return {
                    "policy_decision": "TARGET",
                    "execution_decision": "SUPPRESS",
                    "suppression_reason": "CUSTOMER_OPT_OUT",
                    "customer_id_available": True,
                    "action_executable": False
                }

            # 2. History check
            cursor.execute("""
                SELECT timestamp, execution_decision FROM customer_action_history
                WHERE customer_id = ? AND execution_decision = 'EXECUTE'
                ORDER BY timestamp DESC
            """, (cid,))
            rows = cursor.fetchall()

            parsed_history = []
            for r in rows:
                try:
                    dt = datetime.datetime.fromisoformat(r["timestamp"])
                    parsed_history.append(dt)
                except Exception:
                    pass

            cooldown_thresh = now_dt - datetime.timedelta(hours=cooldown_hours)
            recent_cooldown = [dt for dt in parsed_history if dt > cooldown_thresh]
            if recent_cooldown:
                conn.rollback()
                conn.close()
                return {
                    "policy_decision": "TARGET",
                    "execution_decision": "SUPPRESS",
                    "suppression_reason": "CUSTOMER_COOLDOWN",
                    "customer_id_available": True,
                    "action_executable": False
                }

            day_thresh = now_dt - datetime.timedelta(days=1)
            daily_actions = [dt for dt in parsed_history if dt > day_thresh]
            if len(daily_actions) >= max_actions_per_day:
                conn.rollback()
                conn.close()
                return {
                    "policy_decision": "TARGET",
                    "execution_decision": "SUPPRESS",
                    "suppression_reason": "DAILY_LIMIT_EXCEEDED",
                    "customer_id_available": True,
                    "action_executable": False
                }

            week_thresh = now_dt - datetime.timedelta(days=7)
            weekly_actions = [dt for dt in parsed_history if dt > week_thresh]
            if len(weekly_actions) >= max_actions_per_week:
                conn.rollback()
                conn.close()
                return {
                    "policy_decision": "TARGET",
                    "execution_decision": "SUPPRESS",
                    "suppression_reason": "WEEKLY_LIMIT_EXCEEDED",
                    "customer_id_available": True,
                    "action_executable": False
                }

            # If all checks pass, commit reservation atomically!
            cursor.execute("""
                INSERT INTO customer_action_history (customer_id, transaction_id, execution_decision, suppression_reason, timestamp)
                VALUES (?, ?, 'EXECUTE', NULL, ?)
            """, (cid, transaction_id, now_str))

            cursor.execute("""
                INSERT INTO customer_action_limits (customer_id, last_action_at, daily_count, weekly_count, updated_at)
                VALUES (?, ?, 1, 1, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    last_action_at = excluded.last_action_at,
                    daily_count = daily_count + 1,
                    weekly_count = weekly_count + 1,
                    updated_at = excluded.updated_at
            """, (cid, now_str, now_str))

            conn.commit()
            conn.close()

            return {
                "policy_decision": "TARGET",
                "execution_decision": "EXECUTE",
                "suppression_reason": None,
                "customer_id_available": True,
                "action_executable": True
            }

        except Exception as e:
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
            # Database lock/error produces safe fallback
            return {
                "policy_decision": "TARGET",
                "execution_decision": "SUPPRESS",
                "suppression_reason": f"SAFEGUARD_DB_ERROR: {str(e)}",
                "customer_id_available": True,
                "action_executable": False
            }

    def record_customer_action(self, customer_id: str, transaction_id: str, execution_decision: str, suppression_reason: Optional[str] = None):
        """Helper to record suppressed decisions in history."""
        if not customer_id or not isinstance(customer_id, str):
            return
        cid = customer_id.strip()
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO customer_action_history (customer_id, transaction_id, execution_decision, suppression_reason, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (cid, transaction_id, execution_decision, suppression_reason, now_str))
            conn.commit()

_customer_guard_instance = None

def get_customer_guard() -> CustomerGuard:
    global _customer_guard_instance
    if _customer_guard_instance is None:
        _customer_guard_instance = CustomerGuard()
    return _customer_guard_instance
