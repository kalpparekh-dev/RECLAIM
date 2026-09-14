import re
from typing import Dict, Any

def mask_customer_id(customer_id: str) -> str:
    if not customer_id:
        return "cus_****"
    if len(customer_id) <= 4:
        return customer_id[0] + "****"
    return customer_id[:3] + "****" + customer_id[-2:]

def mask_transaction_id(transaction_id: str) -> str:
    if not transaction_id:
        return "txn_****"
    if len(transaction_id) <= 8:
        return transaction_id[:3] + "****"
    # E.g. RECLAIM-V2-000001 -> RECLAIM-****0001
    parts = transaction_id.split("-")
    if len(parts) >= 3:
        return f"{parts[0]}-****{parts[-1][-4:]}"
    return transaction_id[:7] + "-****" + transaction_id[-4:]

def sanitize_log_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    sensitive_keys = {"card_number", "cvv", "pan", "upi_pin", "password", "auth_token", "secret"}
    sanitized = {}
    for k, v in data.items():
        if k.lower() in sensitive_keys:
            sanitized[k] = "[REDACTED]"
        elif k == "customer_id" and isinstance(v, str):
            sanitized[k] = mask_customer_id(v)
        elif k == "transaction_id" and isinstance(v, str):
            sanitized[k] = mask_transaction_id(v)
        elif isinstance(v, dict):
            sanitized[k] = sanitize_log_dict(v)
        else:
            sanitized[k] = v
    return sanitized
