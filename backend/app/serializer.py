"""
RECLAIM Central Production Response Serializer & Allowlist
Enforces strict removal of oracle, counterfactual, and post-treatment research fields from production API outputs.
Also applies identifier masking for customer IDs across external responses.
"""

from typing import Any, Dict, List, Set, Union
from backend.app.privacy import mask_customer_id, mask_transaction_id

FORBIDDEN_PREFIXES = ("oracle_", "future_", "post_treatment_")
FORBIDDEN_KEYWORDS = {"recovery_outcome", "counterfactual", "synthetic_label"}

PRODUCTION_SAFE_TRANSACTION_KEYS = {
    "transaction_id",
    "customer_id",
    "predicted_treatment_probability",
    "predicted_control_probability",
    "predicted_probability_uplift",
    "selected_policy_type",
    "selected_policy_parameter",
    "selected_policy",
    "intervention_selected",
    "action",
    "found",
    "data_mode",
    "is_production",
    "fallback_applied",
    "fallback_reason",
    "decision_rationale",
    "governance_note",
    "uncertainty_status",
    "score_semantics",
    "calibration_status",
    "customer_safeguards",
    "execution_decision",
    "suppression_reason",
    "net_value_analysis",
    "action_recommendation"
}

def is_forbidden_key(key: str) -> bool:
    key_lower = key.lower()
    for prefix in FORBIDDEN_PREFIXES:
        if key_lower.startswith(prefix):
            return True
    for kw in FORBIDDEN_KEYWORDS:
        if kw in key_lower:
            return True
    return False

def sanitize_production_response(data: Any, mask_ids: bool = True) -> Any:
    """
    Recursively strips forbidden research/oracle fields and applies identifier masking for production outputs.
    """
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            if is_forbidden_key(k):
                continue
            
            # Mask raw customer IDs in production API responses
            if k == "customer_id" and mask_ids and isinstance(v, str):
                cleaned["customer_id"] = mask_customer_id(v)
            elif k == "customer_id_raw": # Strip internal raw ID key from API output
                continue
            else:
                cleaned[k] = sanitize_production_response(v, mask_ids=mask_ids)
        return cleaned
    elif isinstance(data, list):
        return [sanitize_production_response(item, mask_ids=mask_ids) for item in data]
    else:
        return data
