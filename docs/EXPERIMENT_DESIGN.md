# Controlled Randomized Experiment Protocol (A/B Test Design)

## Executive Summary

This document specifies the controlled randomized experiment framework for evaluating future production recovery interventions without fabricating causal attribution claims on current historical data.

---

## 🧪 Experiment Groups & Random Assignment

1. **Control Group (`CONTROL`)**:
   - Treatment: Standard fallback retry path (`RETRY_ALL`).
   - Allocation: 50% of incoming failed transactions.
2. **Treatment Group (`TREATMENT`)**:
   - Treatment: RECLAIM Causal Decision Engine (`V10.2` Top-5% TARGET intervention).
   - Allocation: 50% of incoming failed transactions.

---

## 🔒 Customer-Level Deterministic Assignment

- **Assignment Hash**: `hash(customer_id + experiment_salt) % 100`
- **Customer Stability**: A customer is deterministically assigned to either `CONTROL` or `TREATMENT` for the entire duration of a 30-day experiment window.
- **No Crossover**: No customer may receive both `CONTROL` and `TREATMENT` treatments during the same active experiment.

---

## ⏱️ Delayed Outcome Attribution Rules

- **Decision Timestamp**: \(T_0\)
- **Action Execution Timestamp**: \(T_0 + \text{delay}\)
- **Attribution Window**: 72 hours (\(T_0 + 72\text{h}\))
- **Cutoff Rule**: Recoveries occurring after 72 hours are classified as `OUT_OF_WINDOW` and excluded from primary causal lift calculations.
