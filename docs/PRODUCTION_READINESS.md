# RECLAIM Production Readiness & Architecture Manual

## Executive Summary

RECLAIM is a payment-recovery decisioning system built on an immutable V10.2 production policy dataset (`D:\RECLAIM\data\generated\causal_uplift_v10_2_predictions.csv`).

This manual details system readiness, fallback behavior, data mode classification, and operational procedures.

---

## 🏛️ System Architecture & Fallback Matrix

| Failure Scenario | Fallback Decision | Trigger Condition | Status Code |
| :--- | :--- | :--- | :--- |
| **Policy CSV Missing** | `RETRY_ALL` | File not found at startup | 500 / Fallback |
| **Policy Hash Mismatch** | `RETRY_ALL` | SHA256 does not match `policy_registry.json` | 500 / Warning |
| **Transaction ID Not Found** | `RETRY_ALL` | Query ID absent from 10,000 production index | 200 (Fallback) |
| **Runtime Mode Paused/Fallback**| `RETRY_ALL` | Kill switch or runtime state set to `PAUSED` | 200 (Fallback) |
| **Customer Cooldown Limit** | `SUPPRESS` | Customer targeted within last 24 hours | 200 (Suppressed) |
| **Negative Net Value** | `RETRY_ALL` | Calculated net recovery value < $0.00 | 200 (Fallback) |

---

## 🏷️ Data Mode Classification

All API responses and dashboard screens enforce explicit `data_mode` tags:
- `PRODUCTION_POLICY`: Immutable V10.2 production decision data.
- `HISTORICAL_RESEARCH`: Segregated historical research records (`recovery_outcomes.csv`).
- `SYNTHETIC_SIMULATION`: Offline rate sensitivity & stability simulations.
- `DEMO_EXECUTION`: Interactive operator demo events (`demo_execution_store.json`).
