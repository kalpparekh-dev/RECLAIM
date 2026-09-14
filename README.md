# RECLAIM — Payment Recovery Decisioning Engine & Dashboard

RECLAIM is a production-validated payment-recovery causal decisioning platform built for the Razorpay hackathon.

It targets payment recovery interventions using heterogeneous causal uplift predictions derived from treatment-control experiments. The top 5% highest-uplift failed transactions receive targeted recovery actions (`TARGET`), while remaining transactions fall back to standard retries (`RETRY_ALL`).

---

## 🛡️ Production Red-Team Audit & Safeguards (Score: 92/100)

Following a comprehensive production-level red-team audit across 20 critical financial engine categories, RECLAIM enforces strict enterprise governance:

* **Policy Integrity & SHA256 Verification**: Policy `causal_uplift_v10_2_predictions.csv` (10,000 transactions, 500 targeted @ 5.0%) is SHA256-verified on startup (`data/generated/policy_registry.json`).
* **Simulation Mode Safety Lock**: `SIMULATION_MODE=True` and `ALLOW_REAL_ACTIONS=False` are hard-locked in `ActionProvider`. Real payment provider execution requires explicit external secret enablement.
* **PII & Data Privacy**: Customer IDs and transaction IDs in API responses and logs are masked (`cus_****1234`, `txn_****5678`).
* **Customer Safeguards & Frequency Caps**: Automatic 24-hour intervention cooldown, maximum 1 recovery contact per customer per day, maximum 3 per week, and opt-out suppression checking.
* **Idempotency & Replay Protection**: SQLite-backed idempotency store (`data/idempotency.db`) with payload hash matching preventing duplicate transactions.
* **Currency-Safe Net Value Engine**: Standardized `Decimal` calculations accounting for communication costs, customer incentive budgets, operational overhead, and payment provider success fees.
* **Operational Kill-Switch & Runtime Control**: Dynamic runtime modes (`ACTIVE`, `PAUSED`, `FALLBACK`, `MAINTENANCE`) managed via `/runtime/mode` endpoint. Emergency fallback degrades gracefully to `RETRY_ALL`.

---

## 🏛️ System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │  D:\RECLAIM\data\generated\                  │
                       │  causal_uplift_v10_2_predictions.csv         │
                       └──────────────────────┬───────────────────────┘
                                              │
                                   [Read-Only Startup Loader & SHA256 Verification]
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │         FastAPI Backend Engine               │
                       │    - Policy Service (V10.2 Immutable)       │
                       │    - Customer Guard & Cooldown Store         │
                       │    - SQLite Idempotency Engine               │
                       │    - Net Value Calculator (Decimal)          │
                       │    - Runtime Control Switch (Kill Switch)    │
                       │    - Mock Action Provider (Simulation Lock)  │
                       └──────────────────────┬───────────────────────┘
                                              │
                                        [REST API / JSON]
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │       React 18 / Vite Premium SPA            │
                       │  - Executive Overview                        │
                       │  - Transaction Explorer                      │
                       │  - Decision Console                          │
                       │  - Monitoring & Data Quality                 │
                       │  - Demo Workflow Simulator                   │
                       │  - Historical Analytics                      │
                       │  - Data Governance Panel                     │
                       └──────────────────────────────────────────────┘
```

---

## 📁 Key File Paths & Datasets

| Dataset / Module | Path | Description |
| :--- | :--- | :--- |
| **Primary Production Policy** | `D:\RECLAIM\data\generated\causal_uplift_v10_2_predictions.csv` | Immutable V10.2 decision policy (10,000 transactions, 500 targeted @ 5%) |
| **Policy Registry** | `D:\RECLAIM\data\generated\policy_registry.json` | Stores active SHA256 hash and policy version contract |
| **Idempotency Store** | `D:\RECLAIM\data\idempotency.db` | SQLite table tracking transaction evaluation state & payload hashes |
| **V21 Monitoring** | `D:\RECLAIM\data\generated\reclaim_production_v21_monitoring.csv` | Production schema, population integrity, and consistency audit log |
| **V21 Metrics** | `D:\RECLAIM\data\generated\reclaim_production_v21_metrics.csv` | Prediction probability distribution percentiles and statistics |
| **V21 Data Quality** | `D:\RECLAIM\data\generated\reclaim_production_v21_data_quality.csv` | Field missingness and validation status |
| **Historical Outcomes** | `D:\RECLAIM\data\generated\recovery_outcomes.csv` | Segregated historical recovery research records (2,905 entries) |
| **Demo Event Store** | `D:\RECLAIM\data\demo_execution_store.json` | Isolated storage for interactive demo execution events |

---

## 🔒 Governance Principles & Limitations

1. **Immutable Production Policy**: V10.2 predictions are loaded read-only on startup. No re-ranking or artificial score adjustments are made.
2. **Strict Targeting Rate**: Exactly 500 / 10,000 transactions (5.0%) are selected for `TARGET`.
3. **No Lineage Fabrication**: Production transaction IDs are not joined with historical outcomes without a valid transaction-level bridge.
4. **Oracle Isolation**: Oracle counterfactual fields (`oracle_control_value`, `oracle_treatment_value`, `oracle_incremental_value`) are strictly excluded from production decision making.
5. **Simulation Transparency**: All operator actions in the Demo Workflow are saved into `demo_execution_store.json` and prominently labeled `SIMULATED — NOT AN OBSERVED RECOVERY`.

---

## 🚀 Running the Application

### 1. Prerequisites
- Python 3.10+ (FastAPI, Pandas, Uvicorn, Pydantic)
- Node.js 18+ & npm (Optional, pre-built static bundle included in `frontend/dist`)

### 2. Launch Backend Service & Dashboard
Run the following command from the root directory `D:\RECLAIM`:

```bash
python backend/app/main.py
```

Or using Uvicorn directly:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running:
- **Interactive Web Dashboard**: Open [http://localhost:8000/](http://localhost:8000/)
- **Swagger API Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Frontend Development Server (Optional)
If modifying React code in `frontend/`:

```bash
cd frontend
npm install
npm run dev     # Starts Vite dev server at http://localhost:3000
npm run build   # Builds production bundle to frontend/dist
```

---

## 🧪 Running Automated Tests

Run the full backend test suite:

```bash
python -m unittest discover -s tests
```

Run performance and load tests:

```bash
python -m unittest discover -s tests/performance
```

Tests verify:
- Startup SHA256 integrity check & policy loading
- O(1) decision lookup latency (p50 < 0.3ms, p95 < 1.3ms, 650+ RPS)
- Idempotency store deduplication and payload replay detection
- Customer frequency limits & 24h cooldown safeguards
- Privacy identifier masking (`cus_****1234`, `txn_****5678`)
- Operational kill-switch and state transitions (`ACTIVE` -> `PAUSED` -> `FALLBACK`)
- Currency-safe `Decimal` net value logic
- Simulation-mode action provider locking

---

## 📖 Production Documentation & Audit Artifacts

Comprehensive operational documentation is available in `docs/`:

* [`docs/PRODUCTION_RED_TEAM_AUDIT.md`](file:///d:/RECLAIM/docs/PRODUCTION_RED_TEAM_AUDIT.md): Complete 20-category Red Team Audit Matrix & remediation breakdown.
* [`docs/PRODUCTION_READINESS.md`](file:///d:/RECLAIM/docs/PRODUCTION_READINESS.md): Production operational manual, emergency fallback matrix, & deployment guide.
* [`docs/MODEL_VALIDATION_RECONCILIATION.md`](file:///d:/RECLAIM/docs/MODEL_VALIDATION_RECONCILIATION.md): 20-run bootstrap model stability & probability calibration report.
* [`docs/TARGET_RATE_SENSITIVITY.md`](file:///d:/RECLAIM/docs/TARGET_RATE_SENSITIVITY.md): Target rate sensitivity study across 1%, 2%, 5%, 10%, 15%, 20%.
* [`docs/SEGMENT_SAFETY_REPORT.md`](file:///d:/RECLAIM/docs/SEGMENT_SAFETY_REPORT.md): Sub-segment fairness & slice safety report across payment methods and amounts.
* [`docs/EXPERIMENT_DESIGN.md`](file:///d:/RECLAIM/docs/EXPERIMENT_DESIGN.md): Protocol design for production randomized controlled experiment (A/B trial).

---

## 📡 API Reference

### Readiness & Governance
- `GET /readiness`: Production readiness check, model hash verification, and active runtime mode.
- `GET /policy/metadata`: Active policy details, version (`V10.2`), SHA256 hash, and target rate.
- `POST /runtime/mode`: Change operational mode (`ACTIVE`, `PAUSED`, `FALLBACK`, `MAINTENANCE`).

### Transactions & Decisions
- `GET /api/transactions?page=1&limit=20&search=...&action=TARGET`: Search and filter transactions.
- `GET /api/transactions/{transaction_id}`: Detail view for a transaction.
- `POST /api/decision`: `{"transaction_id": "RECLAIM-V2-000001"}` returns read-only production decision.
- `POST /decision/evaluate`: Evaluates payment failure with customer safeguard check, frequency limits, and idempotency.

### Action Preview & Events
- `POST /action/preview`: Generates failure-mode specific action recommendations with net-value breakdown.
- `POST /events/payment-failed`: Ingests payment failure event, logs idempotency payload, checks safeguards, and emits recommendation.
