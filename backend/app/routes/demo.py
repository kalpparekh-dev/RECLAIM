from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from backend.app.demo_service import get_demo_service

router = APIRouter(prefix="/api/demo")

class DemoExecuteRequest(BaseModel):
    transaction_id: str
    operator_action: Optional[str] = "EXECUTE_RECOMMENDED"
    operator_notes: Optional[str] = None

@router.post("/execute")
def execute_demo_action(req: DemoExecuteRequest):
    demo_svc = get_demo_service()
    res = demo_svc.execute_demo_action(
        transaction_id=req.transaction_id,
        operator_action=req.operator_action or "EXECUTE_RECOMMENDED",
        operator_notes=req.operator_notes
    )
    if "error" in res:
        raise HTTPException(status_code=404, detail=res["error"])
    return res

@router.get("/events")
def list_demo_events(limit: int = Query(50, ge=1, le=200)):
    demo_svc = get_demo_service()
    return {
        "events": demo_svc.list_demo_events(limit=limit),
        "data_nature": "DEMO_SIMULATION_ONLY",
        "disclaimer": "SIMULATED — NOT AN OBSERVED RECOVERY"
    }
