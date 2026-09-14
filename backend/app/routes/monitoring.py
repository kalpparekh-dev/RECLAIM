from fastapi import APIRouter
from backend.app.monitoring_service import get_monitoring_service
from backend.app.drift_monitor_service import get_drift_monitor_service
from backend.app.serializer import sanitize_production_response

router = APIRouter(prefix="/api/monitoring")

@router.get("/summary")
def get_monitoring_summary():
    mon_svc = get_monitoring_service()
    return sanitize_production_response(mon_svc.get_monitoring_summary())

@router.get("/metrics")
def get_monitoring_metrics():
    mon_svc = get_monitoring_service()
    return sanitize_production_response(mon_svc.get_metrics())

@router.get("/data-quality")
def get_data_quality():
    mon_svc = get_monitoring_service()
    return sanitize_production_response(mon_svc.get_data_quality())

@router.get("/drift")
def get_drift_monitoring():
    drift_svc = get_drift_monitor_service()
    return sanitize_production_response(drift_svc.evaluate_runtime_drift_safety())
