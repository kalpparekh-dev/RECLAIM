import os
import json
import sqlite3
import hashlib
import datetime
from enum import Enum
from typing import Dict, Any, Optional

class EventStatus(str, Enum):
    RECEIVED = "RECEIVED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REPLAYED = "REPLAYED"
    CONFLICT = "CONFLICT"

DB_PATH = r"D:\RECLAIM\data\idempotency.db"

def compute_payload_hash(payload: Any) -> str:
    if isinstance(payload, str):
        data_bytes = payload.encode("utf-8")
    else:
        try:
            data_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        except Exception:
            data_bytes = str(payload).encode("utf-8")
    return hashlib.sha256(data_bytes).hexdigest()

class IdempotencyStore:
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
                CREATE TABLE IF NOT EXISTS idempotency_records (
                    idempotency_key TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    transaction_id TEXT NOT NULL,
                    action_id TEXT,
                    payload_hash TEXT,
                    status TEXT NOT NULL,
                    decision_json TEXT,
                    response_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            # Migration check for existing DB without payload_hash column
            cursor.execute("PRAGMA table_info(idempotency_records)")
            columns = [col["name"] for col in cursor.fetchall()]
            if "payload_hash" not in columns:
                cursor.execute("ALTER TABLE idempotency_records ADD COLUMN payload_hash TEXT")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_id ON idempotency_records(transaction_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evt_id ON idempotency_records(event_id)")
            conn.commit()

    def check_and_start(
        self,
        idempotency_key: str,
        event_id: str,
        transaction_id: str,
        payload_data: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Atomically checks idempotency status using SQLite IMMEDIATE transactions.
        Detects duplicate concurrent requests safely and validates payload hash.
        """
        current_hash = compute_payload_hash(payload_data) if payload_data is not None else None
        conn = self._get_connection()

        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.cursor()

            cursor.execute(
                "SELECT status, payload_hash, response_json FROM idempotency_records WHERE idempotency_key = ?",
                (idempotency_key,)
            )
            row = cursor.fetchone()

            if row:
                status, existing_hash, response_json = row["status"], row["payload_hash"], row["response_json"]
                
                # Check payload conflict
                if current_hash and existing_hash and current_hash != existing_hash:
                    conn.rollback()
                    conn.close()
                    return {
                        "is_payload_conflict": True,
                        "status": EventStatus.CONFLICT.value,
                        "message": "Idempotency key reused with different payload data."
                    }

                if status == EventStatus.COMPLETED.value and response_json:
                    resp = json.loads(response_json)
                    resp["idempotency_status"] = EventStatus.REPLAYED.value
                    resp["is_replayed_event"] = True
                    conn.rollback()
                    conn.close()
                    return resp

                conn.rollback()
                conn.close()
                return {
                    "is_duplicate_processing": True,
                    "status": status,
                    "message": "Event with this key is currently being processed or failed."
                }

            # Insert new record
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            cursor.execute("""
                INSERT INTO idempotency_records (idempotency_key, event_id, transaction_id, payload_hash, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (idempotency_key, event_id, transaction_id, current_hash, EventStatus.RECEIVED.value, now, now))
            
            conn.commit()
            conn.close()
            return None

        except sqlite3.IntegrityError:
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
            return {
                "is_duplicate_processing": True,
                "status": EventStatus.PROCESSING.value,
                "message": "Concurrent duplicate request rejected safely."
            }
        except Exception as e:
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
            return {
                "is_duplicate_processing": True,
                "status": EventStatus.FAILED.value,
                "message": f"Idempotency DB error: {str(e)}"
            }

    def complete(self, idempotency_key: str, action_id: str, response_data: Dict[str, Any]):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            response_json = json.dumps(response_data)
            cursor.execute("""
                UPDATE idempotency_records
                SET status = ?, action_id = ?, response_json = ?, updated_at = ?
                WHERE idempotency_key = ?
            """, (EventStatus.COMPLETED.value, action_id, response_json, now, idempotency_key))
            conn.commit()
        finally:
            conn.close()

    def fail(self, idempotency_key: str, error_msg: str):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            err_json = json.dumps({"error": error_msg})
            cursor.execute("""
                UPDATE idempotency_records
                SET status = ?, response_json = ?, updated_at = ?
                WHERE idempotency_key = ?
            """, (EventStatus.FAILED.value, err_json, now, idempotency_key))
            conn.commit()
        finally:
            conn.close()

_idempotency_store_instance = None

def get_idempotency_store() -> IdempotencyStore:
    global _idempotency_store_instance
    if _idempotency_store_instance is None:
        _idempotency_store_instance = IdempotencyStore()
    return _idempotency_store_instance
