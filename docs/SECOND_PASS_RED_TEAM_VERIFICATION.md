# RECLAIM — Second-Pass Adversarial Red-Team Verification Report

**Audit Date**: September 13, 2026  
**Auditor**: Antigravity Red-Team Verification System  
**Target Repository**: `D:\RECLAIM`  
**Immutable Production Policy File**: `D:\RECLAIM\data\generated\causal_uplift_v10_2_predictions.csv` (10,000 transactions, SHA256 `034bf6682905a1ee8e4b65d3247713a4b07efaa9871eb5de0991db55fdb45e55`)

---

## Executive Summary & Score Adjustment

The previous audit report (`D:\RECLAIM\docs\PRODUCTION_RED_TEAM_AUDIT.md`) claimed that all 20 Red Flag categories were **SOLVED** and assigned RECLAIM a unified production readiness score of **92/100**. 

A second-pass adversarial code execution trace, boundary test suite, and architectural inspection revealed that **the previous report's claims were too strong**. While significant engineering safeguards, simulation modules, and governance documentation were created, claiming a single 92/100 live production readiness score conflates demo/research capability with real-world financial gateway execution.

### Re-Evaluated Readiness Metrics (Four-Axis Scorecard)

| Assessment Axis | Score | Status Summary |
| :--- | :---: | :--- |
| **1. Research & Demo Readiness** | **95 / 100** | **PRODUCTION READY (DEMO/HACKATHON)**: Immutable policy, pre-computed uplift ranking, interactive simulator, data governance visualizer, and offline sensitivity reports are fully operational. |
| **2. Engineering Safety Readiness** | **72 / 100** | **PARTIALLY HARDENED**: Runtime control kill switch, O(1) in-memory lookup, customer safeguards, and SQLite idempotency exist, but require concurrency locks, SHA256 startup hard-blocking, and net-value suppression integration. |
| **3. Live Payment Gateway Readiness** | **25 / 100** | **BLOCKED BY EXTERNAL INTEGRATION**: Missing Razorpay/Stripe webhook HMAC signature verification, missing real payment gateway execution connectors, and missing production DB. |
| **4. Causal Evaluation Readiness** | **30 / 100** | **BLOCKED BY EXTERNAL INTEGRATION**: Missing live transaction-level outcome attribution feedback loop and active in-production randomized A/B experiment. |

---

## 🔍 Detailed 20-Point Adversarial Verification Matrix

### Red Flag 1 — Missing Validated Production Outcome Linkage
* **Actual Status**: `BLOCKED_BY_EXTERNAL_INTEGRATION`
* **Previous Claim**: Solved via JSON schema validation (`outcome_attribution_schema.json`).
* **Adversarial Findings**: An outcome attribution JSON schema exists at `data/schemas/outcome_attribution_schema.json`, but **no live production transaction-level bridge or payment gateway webhook feed exists**. Historical records in `recovery_outcomes.csv` (2,905 rows) represent segregated research datasets. The system cannot observe real payment recovery outcomes post-intervention.
* **Verified Test**: `tests/test_reclaim.py::TestReclaimAPI.test_06_historical_outcomes_segregation` verifies outcomes are research-isolated.
* **Remaining Gap**: Requires real payment gateway webhook receivers (e.g., Razorpay `payment.captured` webhooks) to populate production transaction outcomes.

---

### Red Flag 2 — Data Leakage and Counterfactual Mixing
* **Actual Status**: `PARTIALLY_SOLVED`
* **Previous Claim**: Solved via policy service column segregation.
* **Adversarial Findings**: In `backend/app/policy_service.py`, `get_decision(tx_id)` retrieves decision dictionaries that omit `oracle_*` fields (`oracle_control_value`, `oracle_treatment_value`, `oracle_incremental_value`). Furthermore, `/events/payment-failed` ignores any user-supplied body fields matching oracle names. However, `list_transactions()` in `policy_service.py` returns raw CSV DataFrame records containing `oracle_*` columns in the `GET /api/transactions` list endpoint.
* **Verified Test**: `tests/test_reclaim.py::TestReclaimAPI.test_02_decision_lookup_immutability` confirms `get_decision` returns decision fields without oracle exposure.
* **Remaining Gap**: Strip `oracle_*` columns from `list_transactions()` API output so research counterfactuals are unreachable across all REST endpoints.

---

### Red Flag 3 — Uncalibrated Uplift Scores
* **Actual Status**: `PARTIALLY_SOLVED`
* **Previous Claim**: Solved via model validation reconciliation document.
* **Adversarial Findings**: In `policy_service.py`, `uncertainty_status` is explicitly set to `"NOT_AVAILABLE"`, preventing raw uplift scores from being misrepresented as statistical confidence intervals. However, in `backend/app/net_value_engine.py`, `expected_recovery_value = amount * predicted_treatment_probability` uses uncalibrated model probabilities directly without runtime Platt scaling or isotonic regression calibration.
* **Verified Test**: `tests/test_second_pass.py::TestSecondPassAdversarialVerification.test_01_uncertainty_status_explicitly_not_available`.
* **Remaining Gap**: Implement an active runtime probability calibration layer (e.g. isotonic regression) using observed production recovery rates.

---

### Red Flag 4 — Arbitrary 5% Target Rate
* **Actual Status**: `PARTIALLY_SOLVED`
* **Previous Claim**: Solved via offline target rate sensitivity script.
* **Adversarial Findings**: `scripts/analyze_target_rate_sensitivity.py` generates `docs/TARGET_RATE_SENSITIVITY.md` comparing 1%, 2%, 5%, 10%, 15%, 20% rates. The analysis is reproducible and strictly preserves V10.2 policy immutability. However, because real production recovery outcome feedback is unavailable, the production rate justification relies on historical research data.
* **Verified Test**: Manual execution of `python scripts/analyze_target_rate_sensitivity.py`.
* **Remaining Gap**: Dynamic rate optimization requires a live feedback loop measuring marginal action cost vs. marginal recovery yield.

---

### Red Flag 5 — Missing Out-of-Sample and Drift Validation
* **Actual Status**: `PARTIALLY_SOLVED`
* **Previous Claim**: Solved via drift monitoring report script.
* **Adversarial Findings**: `scripts/run_drift_monitoring.py` exists as a standalone batch script generating `data/generated/drift_monitoring_report.json`. However, there is **no automated background scheduler (e.g. Celery/APScheduler)**, **no active alerting integration (e.g. PagerDuty/Slack)**, and **no automated trigger that switches policy runtime to `FALLBACK` mode upon drift detection**.
* **Verified Test**: `scripts/run_drift_monitoring.py` generates JSON drift report correctly.
* **Remaining Gap**: Wire drift detection output directly into `RuntimeControlManager` to automatically trigger `FALLBACK` mode when PSI > 0.25.

---

### Red Flag 6 — Customer Over-Intervention
* **Actual Status**: `PARTIALLY_SOLVED`
* **Previous Claim**: Solved via CustomerGuard module.
* **Adversarial Findings**: `backend/app/customer_guard.py` enforces a 24-hour cooldown, daily cap (1), weekly cap (3), and opt-out suppression in `/events/payment-failed`. However, `customer_safeguards_store.json` relies on synchronous file reads/writes without file locking or atomic database transaction control, creating race conditions under high concurrent traffic. Additionally, `/decision/evaluate` skips safeguard checks if `customer_id` is omitted in the request.
* **Verified Test**: `tests/test_reclaim.py::TestReclaimAPI.test_07_customer_safeguards_and_cooldown`.
* **Remaining Gap**: Migrate customer safeguard history to an atomic database table (e.g., PostgreSQL / Redis) with row-level locks and mandate `customer_id` on all evaluation endpoints.

---

### Red Flag 7 — Action Capability and Optimization
* **Actual Status**: `SIMULATION_ONLY`
* **Previous Claim**: Solved via failure-reason action recommendation layer.
* **Adversarial Findings**: `ActionRecommendationEngine` maps failure reasons and payment methods to recommended action types (`SMART_RETRY`, `WHATSAPP_NUDGE`, `FEE_WAIVER_OFFER`). However, `ActionProvider` (`action_provider.py`) is hard-locked in `SIMULATION_MODE=True` and `ALLOW_REAL_ACTIONS=False`. No actual payment gateway API (e.g. Razorpay, Stripe) calls or WhatsApp communications are dispatched.
* **Verified Test**: `tests/test_reclaim.py::TestReclaimAPI.test_09_action_provider_simulation_lock`.
* **Remaining Gap**: Develop production action execution adapters with real payment gateway API credentials.

---

### Red Flag 8 — Financial Net-Value Calculation
* **Actual Status**: `PARTIALLY_SOLVED`
* **Previous Claim**: Solved via Decimal net-value calculator.
* **Adversarial Findings**: `backend/app/net_value_engine.py` implements currency-safe `Decimal` arithmetic ($\text{Net Value} = \text{Recovery} - \text{Comm Cost} - \text{Incentive} - \text{Ops Cost} - \text{Provider Fees}$). However, in `events.py` (`POST /events/payment-failed`), when `net_val_res["is_profitable"] == False` (negative expected net value), **the event endpoint still returns `"execution_decision": "EXECUTE"` for TARGET transactions!** The net value calculation is purely informative and does not suppress unprofitable interventions.
* **Verified Test**: `tests/test_second_pass.py::TestSecondPassAdversarialVerification.test_03_negative_net_value_calculation_structure`.
* **Remaining Gap**: Modify `process_payment_failed_event` to force `execution_decision = "SUPPRESS"` with `suppression_reason = "UNPROFITABLE_NET_VALUE"` whenever `is_profitable` is False.

---

### Red Flag 9 — Model and Policy Versioning
* **Actual Status**: `PARTIALLY_SOLVED`
* **Previous Claim**: Solved via SHA256 registry hash verification.
* **Adversarial Findings**: `policy_registry.json` contains SHA256 hash `034bf6682905a1ee8e4b65d3247713a4b07efaa9871eb5de0991db55fdb45e55` matching `causal_uplift_v10_2_predictions.csv`. In `policy_service.py`, if the hash check fails on startup, the system prints `[RECLAIM WARNING]` and sets `status = "WARNING_HASH_UNVERIFIED"`. **Crucially, it does NOT raise an exception or block startup!** Furthermore, `/readiness` continues to report `READY` even when hash verification fails.
* **Verified Test**: `tests/test_second_pass.py::TestSecondPassAdversarialVerification.test_02_sha256_hash_mismatch_does_not_halt_service`.
* **Remaining Gap**: Update `_load_and_validate()` to raise `PolicyValidationError` or force `RuntimeMode.FALLBACK` on SHA256 mismatch.

---

### Red Flag 10 — Real-Time Event Processing
* **Actual Status**: `BLOCKED_BY_EXTERNAL_INTEGRATION`
* **Previous Claim**: Solved via `/events/payment-failed` REST endpoint.
* **Adversarial Findings**: `POST /events/payment-failed` accepts JSON payloads and executes decisioning. However, it **lacks webhook signature verification (e.g. Razorpay HMAC SHA256 headers)**, lacks API key authentication, and has no live webhook listener connected to a payment gateway.
* **Verified Test**: `tests/test_reclaim.py::TestReclaimAPI.test_08_events_and_decision_endpoint`.
* **Remaining Gap**: Implement Razorpay HMAC signature verification middleware and secret key validation.

---

### Red Flag 11 — Idempotency and Replay Protection
* **Actual Status**: `PARTIALLY_SOLVED`
* **Previous Claim**: Solved via SQLite idempotency store.
* **Adversarial Findings**: `backend/app/idempotency_store.py` provides SQLite-backed deduplication tracking `idempotency_key`, `status`, and `response_json`. However, `check_and_start()` uses un-locked `SELECT` followed by `INSERT`. Concurrent duplicate requests with the exact same key trigger `sqlite3.IntegrityError: UNIQUE constraint failed`, causing unhandled HTTP 500 exceptions instead of returning a clean duplicate rejection response. Additionally, local SQLite is not scalable for multi-node deployments.
* **Verified Test**: `tests/test_second_pass.py::TestSecondPassAdversarialVerification.test_05_idempotency_replay_detection`.
* **Remaining Gap**: Catch `sqlite3.IntegrityError` in `check_and_start()` to return `is_duplicate_processing: True`, and migrate to Redis for multi-instance deployments.

---

### Red Flag 12 — Kill Switch and Safe Fallback
* **Actual Status**: `PARTIALLY_SOLVED`
* **Previous Claim**: Solved via RuntimeControlManager.
* **Adversarial Findings**: `RuntimeControlManager` supports `ACTIVE`, `PAUSED`, `FALLBACK`, `MAINTENANCE`. When set to `PAUSED` or `FALLBACK`, `policy_service.get_decision()` correctly returns `selected_policy: "RETRY_ALL"` with `fallback_applied: True`. However, unhandled exceptions inside routes (such as DB failures or missing fields) raise standard FastAPI HTTP 500 exceptions rather than catching all runtime errors to return a safe `RETRY_ALL` fallback payload.
* **Verified Test**: `tests/test_second_pass.py::TestSecondPassAdversarialVerification.test_04_runtime_mode_fallback_enforcement`.
* **Remaining Gap**: Wrap route handlers in global exception middleware that converts unhandled errors into safe `RETRY_ALL` decision responses with `fallback_applied: True`.

---

### Red Flag 13 — Privacy and PII Leakage
* **Actual Status**: `PARTIALLY_SOLVED`
* **Previous Claim**: Solved via privacy identifier masking.
* **Adversarial Findings**: `backend/app/privacy.py` provides `mask_customer_id` (`cus_****1234`) and `mask_transaction_id` (`txn_****5678`). In `/events/payment-failed`, masked strings are included in response JSON. However, **raw unmasked `customer_id` and `transaction_id` are ALSO returned in the same JSON object**, and `GET /api/transactions` returns full raw customer IDs.
* **Verified Test**: `tests/test_reclaim.py::TestReclaimAPI.test_08_events_and_decision_endpoint`.
* **Remaining Gap**: Configure environment-driven PII suppression where production responses strip raw customer identifiers entirely.

---

### Red Flag 14 — Segment Safety and Fairness
* **Actual Status**: `PARTIALLY_SOLVED`
* **Previous Claim**: Solved via segment safety report.
* **Adversarial Findings**: `scripts/analyze_segment_safety.py` and `docs/SEGMENT_SAFETY_REPORT.md` evaluate targeting distribution across payment methods (card, UPI, netbanking) and transaction amount tiers. The documentation appropriately notes that sensitive demographic attributes were not assessed. However, no dynamic runtime fairness constraints exist to monitor or cap segment concentration live.
* **Verified Test**: Manual run of `python scripts/analyze_segment_safety.py`.
* **Remaining Gap**: Implement real-time slice monitoring to flag sudden spikes in payment-method targeting concentration.

---

### Red Flag 15 — Production Recovery Claims
* **Actual Status**: `PARTIALLY_SOLVED`
* **Previous Claim**: Solved via simulator transparency labeling.
* **Adversarial Findings**: Operator actions in the Demo Simulator are recorded into `demo_execution_store.json` and explicitly labeled `SIMULATED — NOT AN OBSERVED RECOVERY`. However, historical views in the UI display recovery metrics derived from segregated research data (`recovery_outcomes.csv`), which could be confused with actual production performance if read without context.
* **Verified Test**: `tests/test_reclaim.py::TestReclaimAPI.test_06_historical_outcomes_segregation`.
* **Remaining Gap**: Add an explicit, permanent UI banner on historical screens stating: *"Production outcome attribution is not currently available."*

---

### Red Flag 16 — Experimentation Framework
* **Actual Status**: `DOCUMENTED_ONLY`
* **Previous Claim**: Solved via experiment design document.
* **Adversarial Findings**: `docs/EXPERIMENT_DESIGN.md` documents an A/B experimental protocol. However, **no randomized assignment split (50/50 treatment/control hashing)**, **no group assignment persistence**, and **no live experiment evaluation engine** exists in `backend/app/`.
* **Verified Test**: Inspection of `backend/app/` (No experimental router module exists).
* **Remaining Gap**: Build a runtime `ExperimentRouter` module that deterministically hashes transaction IDs into Treatment (Uplift Policy) vs Control (Retry All) arms.

---

### Red Flag 17 — Model Instability Reconciled
* **Actual Status**: `SOLVED`
* **Previous Claim**: Solved via bootstrap stability script.
* **Adversarial Findings**: Bootstrap stability analysis (`scripts/validate_model_stability.py`) on `causal_uplift_v10_2_predictions.csv` demonstrates an **84.6% mean Jaccard overlap** across 20 runs on 80% subsamples. 
* **Reconciliation Explanation**: Earlier research iterations exhibited instability because they evaluated uncalibrated, dynamically re-trained uplift models where minor score variations near zero caused rank flipping. V10.2 pre-computes and locks uplift scores on 10,000 transactions, yielding a stable top 5% targeting cutoff.
* **Verified Test**: Execution of `python scripts/validate_model_stability.py`.
* **Remaining Gap**: None for fixed policy V10.2.

---

### Red Flag 18 — Uncertainty-Aware Decisioning
* **Actual Status**: `SOLVED`
* **Previous Claim**: Solved via explicit uncertainty status flag.
* **Adversarial Findings**: `policy_service.py` sets `"uncertainty_status": "NOT_AVAILABLE"` on all decision responses. The API does not fabricate confidence intervals or claim fake standard deviations on raw uplift predictions.
* **Verified Test**: `tests/test_second_pass.py::TestSecondPassAdversarialVerification.test_01_uncertainty_status_explicitly_not_available`.
* **Remaining Gap**: None.

---

### Red Flag 19 — Load and Latency Testing
* **Actual Status**: `PARTIALLY_SOLVED`
* **Previous Claim**: Solved via load test suite.
* **Adversarial Findings**: `tests/performance/test_load.py` measures in-memory policy lookup latency (`p50 = 0.243ms`, `p95 = 0.488ms`, `p99 = 1.666ms`) and thread pool concurrency (`2,503.4 RPS`). However, full HTTP network latency over ASGI/FastAPI with SQLite disk writes was not benchmarked under external distributed load tools (e.g., Locust or k6).
* **Verified Test**: `python -m unittest discover -s tests/performance`.
* **Remaining Gap**: Execute end-to-end HTTP load testing using an external HTTP benchmark tool against a running Uvicorn instance.

---

### Red Flag 20 — Operational Readiness
* **Actual Status**: `SOLVED`
* **Previous Claim**: Solved via operational runbook.
* **Adversarial Findings**: `docs/PRODUCTION_READINESS.md` and `README.md` contain verified runbook instructions, health check endpoints (`/health`, `/readiness`), runtime mode switches (`POST /runtime/mode`), and Swagger documentation paths (`http://localhost:8000/docs`). All startup and test commands run cleanly.
* **Verified Test**: Successful clean execution of `python backend/app/main.py` and unit test discovery suite.
* **Remaining Gap**: None.

---

## 📊 Comprehensive Final Classification Table

| ID | Red Flag Risk | Actual Status | Evidence Path | Verified Test | Remaining Gap |
| -- | ------------- | ------------- | ------------- | ------------- | ------------- |
| **1** | Production Outcome Linkage | `BLOCKED_BY_EXTERNAL_INTEGRATION` | `data/schemas/outcome_attribution_schema.json` | `test_06_historical_outcomes_segregation` | Missing live payment provider outcome webhooks |
| **2** | Data Leakage & Oracle Fields | `PARTIALLY_SOLVED` | `backend/app/policy_service.py` | `test_02_decision_lookup_immutability` | `list_transactions` exposes `oracle_*` columns in GET list API |
| **3** | Uncalibrated Uplift Scores | `PARTIALLY_SOLVED` | `backend/app/net_value_engine.py` | `test_01_uncertainty_status_explicitly_not_available` | Net value engine uses uncalibrated probabilities directly |
| **4** | Arbitrary 5% Target Rate | `PARTIALLY_SOLVED` | `scripts/analyze_target_rate_sensitivity.py` | Script execution | Dynamic rate tuning relies on research outcomes |
| **5** | Out-of-Sample Drift Monitoring | `PARTIALLY_SOLVED` | `scripts/run_drift_monitoring.py` | Script execution | Lacks automated background scheduler & alert integration |
| **6** | Customer Over-Intervention | `PARTIALLY_SOLVED` | `backend/app/customer_guard.py` | `test_07_customer_safeguards_and_cooldown` | JSON store non-atomic under concurrency; `/decision/evaluate` allows missing customer ID |
| **7** | Action Capability & Optimization | `SIMULATION_ONLY` | `backend/app/action_provider.py` | `test_09_action_provider_simulation_lock` | Execution adapter locked in `SIMULATION_MODE=True` |
| **8** | Financial Net-Value Engine | `PARTIALLY_SOLVED` | `backend/app/net_value_engine.py` | `test_03_negative_net_value_calculation_structure` | Negative net value does not auto-suppress `execution_decision` |
| **9** | Model Versioning & Hash Lock | `PARTIALLY_SOLVED` | `backend/app/policy_service.py` | `test_02_sha256_hash_mismatch_does_not_halt_service` | Hash mismatch logs warning but does not halt startup |
| **10** | Real-Time Event Processing | `BLOCKED_BY_EXTERNAL_INTEGRATION` | `backend/app/routes/events.py` | `test_08_events_and_decision_endpoint` | Lacks Razorpay HMAC signature verification |
| **11** | Idempotency & Replay Protection | `PARTIALLY_SOLVED` | `backend/app/idempotency_store.py` | `test_05_idempotency_replay_detection` | SQLite un-locked `SELECT+INSERT` throws HTTP 500 under concurrent race |
| **12** | Kill Switch & Safe Fallback | `PARTIALLY_SOLVED` | `backend/app/runtime_control.py` | `test_04_runtime_mode_fallback_enforcement` | Unhandled route exceptions return HTTP 500 instead of safe `RETRY_ALL` |
| **13** | Privacy & PII Leakage | `PARTIALLY_SOLVED` | `backend/app/privacy.py` | `test_08_events_and_decision_endpoint` | Raw unmasked customer IDs returned alongside masked IDs |
| **14** | Segment Safety & Fairness | `PARTIALLY_SOLVED` | `scripts/analyze_segment_safety.py` | Script execution | No live segment concentration monitoring or capping |
| **15** | Production Recovery Claims | `PARTIALLY_SOLVED` | `data/demo_execution_store.json` | `test_06_historical_outcomes_segregation` | UI historical views show research dataset metrics |
| **16** | Experimentation Framework | `DOCUMENTED_ONLY` | `docs/EXPERIMENT_DESIGN.md` | Code inspection | No runtime A/B experiment router implemented in code |
| **17** | Model Instability Reconciled | `SOLVED` | `scripts/validate_model_stability.py` | Script execution | Reconciled: V10.2 baseline pre-computed & locked |
| **18** | Uncertainty-Aware Decisioning | `SOLVED` | `backend/app/policy_service.py` | `test_01_uncertainty_status_explicitly_not_available` | Returns `uncertainty_status = NOT_AVAILABLE` |
| **19** | Load and Latency Testing | `PARTIALLY_SOLVED` | `tests/performance/test_load.py` | Performance test suite | In-memory tested; full HTTP network latency not tested |
| **20** | Operational Readiness | `SOLVED` | `docs/PRODUCTION_READINESS.md` | Server startup & test suite | Verified operational runbook & startup commands |

---

## 🚨 The Five Most Serious Remaining Production Risks

1. **Missing Webhook HMAC Signature Verification (Red Flag 10)**: Anyone can send fake HTTP POST requests to `/events/payment-failed` without authentication or payload verification.
2. **Non-Blocking Net Value & Customer Safeguards (Red Flags 6 & 8)**: Negative net-value calculations do not automatically block action execution in `/events/payment-failed`, and JSON-based customer safeguard history is non-atomic under concurrent traffic.
3. **Non-Blocking Policy Hash Mismatch (Red Flag 9)**: Tampering with `causal_uplift_v10_2_predictions.csv` prints a console warning but allows the engine to boot and serve unauthorized policies.
4. **Unhandled Exception Leakage (Red Flag 12)**: System errors in underlying DB or guard services trigger standard FastAPI HTTP 500 responses rather than catching exceptions to safely return `selected_policy: RETRY_ALL`.
5. **Absence of Real Production Outcome Feedback (Red Flag 1)**: Without real payment gateway outcome ingestion, actual recovery performance and uplift attribution cannot be validated in live production.

---

## 🔒 Mandatory Production Policy Confirmation Statement

> **CONFIRMATION**: The V10.2 production policy file located at `D:\RECLAIM\data\generated\causal_uplift_v10_2_predictions.csv` remains **100% IMMUTABLE AND UNCHANGED**. It contains exactly 10,000 transaction rows, 425 unique customers, exactly 500 `TARGET` transactions (targeting rate of exactly 5.0%), and 9,500 `RETRY_ALL` transactions. All decisioning logic, hash verifications, and API routes serve this immutable policy file strictly without re-ranking or artificial adjustment.
