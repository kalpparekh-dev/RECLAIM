import os
import sys
import uuid
import logging
import datetime

# Ensure root workspace directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.app.routes import health, policy, monitoring, outcomes, demo, events, payments, gateway, reconciliation, feedback, experiments
from backend.app.policy_service import get_policy_service
from backend.app.queue_worker import get_queue_worker
from backend.app.serializer import sanitize_production_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reclaim")

app = FastAPI(
    title="RECLAIM Payment Recovery & Decisioning Platform",
    description="Enterprise payment recovery decisioning, state machine, gateway sandbox, reconciliation, and feedback platform",
    version="2.0.0"
)

# P0-8: Global Exception Handler for decisioning safety
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())
    logger.error(f"[RECLAIM ERROR] [ReqID: {request_id}] Unhandled error on {request.method} {request.url.path}: {str(exc)}", exc_info=True)

    # For decisioning endpoints, return safe RETRY_ALL fallback payload instead of unhandled 500
    if request.url.path in ("/events/payment-failed", "/decision/evaluate", "/api/decision"):
        return JSONResponse(
            status_code=200,
            content=sanitize_production_response({
                "selected_policy": "RETRY_ALL",
                "execution_decision": "SUPPRESS",
                "fallback_applied": True,
                "fallback_reason": f"INTERNAL_SAFETY_FALLBACK: {type(exc).__name__}",
                "request_id": request_id,
                "policy_version": "V10.2",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })
        )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id}
    )

# Enable CORS for local development and frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health.router)
app.include_router(policy.router)
app.include_router(monitoring.router)
app.include_router(outcomes.router)
app.include_router(demo.router)
app.include_router(events.router)
app.include_router(payments.router)
app.include_router(gateway.router)
app.include_router(reconciliation.router)
app.include_router(feedback.router)
app.include_router(experiments.router)

# Mount static frontend files if built
FRONTEND_DIST_DIR = os.path.join(BASE_DIR, "frontend", "dist")
ASSETS_DIR = os.path.join(FRONTEND_DIST_DIR, "assets")

if os.path.exists(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="static")

if os.path.exists(FRONTEND_DIST_DIR):
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path and not full_path.startswith("api") and full_path not in ["health", "readiness", "events", "decision", "action", "runtime", "policy"]:
            target_path = os.path.join(FRONTEND_DIST_DIR, full_path)
            if os.path.exists(target_path) and os.path.isfile(target_path):
                return FileResponse(target_path)
        index_path = os.path.join(FRONTEND_DIST_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"error": "Frontend build index.html not found"}

@app.on_event("startup")
def startup_event():
    print("[RECLAIM] Initializing RECLAIM Production Policy Engine V10.2 & Payment Recovery Platform...")
    svc = get_policy_service()
    summary = svc.get_summary()
    print(f"[RECLAIM] Production Policy V10.2 Loaded: {summary['total_transactions']} transactions, {summary['selected_transactions']} targeted ({summary['targeting_rate']*100:.1f}%)")
    worker = get_queue_worker()
    print(f"[RECLAIM] Background Event Queue Worker initialized (Status: {worker.running})")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)
