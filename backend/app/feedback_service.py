import uuid
import datetime
from typing import Dict, Any, Optional, List
from backend.app.domain.models import CustomerFeedbackCategory
from backend.app.payment_service import get_payment_service

class FeedbackService:
    def submit_customer_feedback(
        self,
        transaction_id: str,
        rating: int,
        comment: Optional[str] = None,
        feedback_category: str = "PAYMENT_EXPERIENCE",
        payment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submits optional customer experience feedback (rating scale 1 to 5)."""
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be an integer between 1 and 5.")

        fb_id = f"fb_{uuid.uuid4().hex[:12]}"
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cat_clean = (feedback_category or "PAYMENT_EXPERIENCE").upper()

        payment_svc = get_payment_service()
        conn = payment_svc._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO customer_feedback (id, transaction_id, payment_id, rating, comment, feedback_category, submitted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (fb_id, transaction_id, payment_id, rating, comment, cat_clean, now_str))
            conn.commit()

            return {
                "id": fb_id,
                "transaction_id": transaction_id,
                "payment_id": payment_id,
                "rating": rating,
                "comment": comment,
                "feedback_category": cat_clean,
                "submitted_at": now_str,
                "status": "RECORDED"
            }
        finally:
            conn.close()

    def get_feedback_analytics(self) -> Dict[str, Any]:
        """Calculates feedback & recovery metrics summary."""
        payment_svc = get_payment_service()
        conn = payment_svc._get_connection()
        try:
            cursor = conn.cursor()
            
            # Customer feedback stats
            cursor.execute("SELECT COUNT(*) as count, AVG(rating) as avg_rating FROM customer_feedback")
            fb_row = cursor.fetchone()
            fb_count = fb_row["count"] or 0
            avg_rating = round(fb_row["avg_rating"] or 0.0, 2)

            # Category distribution
            cursor.execute("SELECT feedback_category, COUNT(*) as cnt FROM customer_feedback GROUP BY feedback_category")
            cat_rows = cursor.fetchall()
            cat_dist = {r["feedback_category"]: r["cnt"] for r in cat_rows}

            # Reconciliation & Recovery stats
            cursor.execute("SELECT COUNT(*) as total, SUM(recovered_amount) as total_recovered, AVG(time_to_recovery_seconds) as avg_time FROM outcome_reconciliations WHERE reconciliation_status = 'MATCHED'")
            rec_row = cursor.fetchone()
            total_reconciled = rec_row["total"] or 0
            total_recovered = round(rec_row["total_recovered"] or 0.0, 2)
            avg_time = round(rec_row["avg_time"] or 0.0, 1)

            return {
                "customer_feedback_count": fb_count,
                "average_customer_rating": avg_rating,
                "category_distribution": cat_dist,
                "total_reconciled_recoveries": total_reconciled,
                "total_recovered_amount": total_recovered,
                "average_time_to_recovery_seconds": avg_time,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        finally:
            conn.close()

    def list_customer_feedback(self, page: int = 1, limit: int = 20) -> Dict[str, Any]:
        payment_svc = get_payment_service()
        conn = payment_svc._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM customer_feedback")
            total = cursor.fetchone()["total"]

            offset = (page - 1) * limit
            cursor.execute("""
                SELECT * FROM customer_feedback
                ORDER BY submitted_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

            records = [dict(r) for r in cursor.fetchall()]
            return {
                "feedback": records,
                "total": total,
                "page": page,
                "limit": limit
            }
        finally:
            conn.close()

_feedback_service_instance = None

def get_feedback_service() -> FeedbackService:
    global _feedback_service_instance
    if _feedback_service_instance is None:
        _feedback_service_instance = FeedbackService()
    return _feedback_service_instance
