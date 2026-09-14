from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from backend.app.domain.models import CustomerFeedbackRequest
from backend.app.feedback_service import get_feedback_service
from backend.app.serializer import sanitize_production_response

router = APIRouter(prefix="/api/recovery", tags=["Feedback"])

@router.post("/feedback/customer")
def submit_customer_feedback(req: CustomerFeedbackRequest):
    fb_svc = get_feedback_service()
    try:
        res = fb_svc.submit_customer_feedback(
            transaction_id=req.transaction_id,
            rating=req.rating,
            comment=req.comment,
            feedback_category=req.feedback_category.value,
            payment_id=req.payment_id
        )
        return sanitize_production_response(res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

@router.get("/feedback")
def get_feedback_and_analytics(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100)
):
    fb_svc = get_feedback_service()
    analytics = fb_svc.get_feedback_analytics()
    feedback_list = fb_svc.list_customer_feedback(page=page, limit=limit)

    return sanitize_production_response({
        "analytics": analytics,
        "feedback_records": feedback_list["feedback"],
        "pagination": {
            "page": page,
            "limit": limit,
            "total": feedback_list["total"]
        }
    }, mask_ids=True)
