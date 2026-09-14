# Segment-Level Safety & Fairness Evaluation

## Executive Summary

This evaluation analyzes segment-level concentration, targeting distribution, and customer fairness across the V10.2 production policy dataset.

---

## 📊 Customer Concentration & Targeting Distribution

- **Total Production Transactions**: 10,000
- **Unique Customers**: 425
- **Max Transactions for Single Customer (Total)**: 75
- **Max Targeted Transactions for Single Customer (Un-guarded Policy)**: 33
- **Customer Safeguard Enforcement**: Required to prevent over-contacting the same customer. With `CustomerGuard` active, maximum targeted actions per customer is restricted to **1 action per 24 hours** and **max 3 per week**.

---

## 💳 Historical Payment Method Recovery Performance

| Payment Method | Total Historical Txns | Recovered Txns | Historical Recovery Rate |
| :--- | :--- | :--- | :--- |
| CARD | 914 | 596 | 65.2% |
| NETBANKING | 354 | 243 | 68.6% |
| UPI | 1416 | 998 | 70.5% |
| WALLET | 220 | 155 | 70.5% |


---

## 🛡️ Segment Safeguards & Operational Rules

1. **Small Sample Suppression**: Any customer segment with fewer than 10 transactions receives safe fallback `RETRY_ALL`.
2. **Extreme Concentration Cap**: No single customer may receive more than 1 live intervention in a 24-hour window.
3. **No Unvalidated Outcome Claims**: Outcome metrics are derived exclusively from historical research data and are not attributed to V10.2 production transaction IDs.
