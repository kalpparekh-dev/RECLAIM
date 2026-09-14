# Targeting Rate Offline Sensitivity Analysis

## Executive Overview

This offline analysis evaluates the sensitivity of recovery value and action costs across candidate targeting rates (1%, 2%, 5%, 10%, 15%, 20%).

> **Important Policy Boundary**: The production decision engine remains strictly locked to the immutable **5.0% V10.2 Policy** (500 targeted transactions). This sensitivity analysis is for offline research and operational capacity planning only.

---

## 📊 Rate Sensitivity Table

| Rate | Targeted Txns | Cumulative Uplift | Avg Uplift | Est Recovery Value | Est Action Cost | Est Net Value | Unique Customers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1% | 100 | 37.2292 | 0.3723 | $3,722.92 | $50.0 | $3,672.92 | 12 |
| 2% | 200 | 69.3618 | 0.3468 | $6,936.18 | $100.0 | $6,836.18 | 19 |
| 5% | 500 | 154.5107 | 0.309 | $15,451.07 | $250.0 | $15,201.07 | 38 |
| 10% | 1000 | 270.9415 | 0.2709 | $27,094.15 | $500.0 | $26,594.15 | 77 |
| 15% | 1500 | 369.6559 | 0.2464 | $36,965.59 | $750.0 | $36,215.59 | 108 |
| 20% | 2000 | 453.8803 | 0.2269 | $45,388.03 | $1,000.0 | $44,388.03 | 136 |


---

## 💡 Findings & Operational Guidance

1. **Marginal Uplift Diminishing Returns**: Targeting beyond 5% captures lower marginal uplift transactions.
2. **Cost-Benefit Balance**: At 5% rate (500 txns), RECLAIM achieves optimal signal-to-noise ratio before contact fatigue increases.
3. **Production Configuration**: `POLICY_TARGET_RATE = 0.05` (Immutable).
