import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Dict, Any, Optional

DEFAULT_ACTION_COSTS = {
    "EMAIL": Decimal("0.10"),
    "SMS": Decimal("0.50"),
    "WHATSAPP": Decimal("0.30"),
    "SUPPORT_CALL": Decimal("15.00"),
    "INCENTIVE": Decimal("0.00"),
    "GATEWAY_REROUTE": Decimal("1.50"),
    "SMART_RETRY": Decimal("0.50")
}

ALLOWED_CURRENCIES = {"INR", "USD", "EUR", "GBP"}

def safe_to_decimal(val: Any) -> Optional[Decimal]:
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    try:
        s = str(val).strip()
        if not s or s.lower() in ("nan", "inf", "-inf", "null", "none"):
            return None
        d = Decimal(s)
        if d.is_nan() or d.is_infinite():
            return None
        return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError, TypeError):
        return None

def calculate_expected_net_value(
    expected_recovery_value: Any,
    action_type: str = "SMART_RETRY",
    incentive_cost: Any = 0.0,
    operational_cost: Any = 0.0,
    provider_cost: Any = 0.0,
    currency: str = "INR",
    min_net_value_threshold: Decimal = Decimal("0.01") # Require strictly positive net value (>0)
) -> Dict[str, Any]:
    """
    Calculates expected net value using currency-safe Decimal arithmetic.
    Blocks negative, zero, non-finite, missing, or invalid values.
    """
    if currency.upper() not in ALLOWED_CURRENCIES:
        return {
            "expected_recovery_value": 0.0,
            "total_action_cost": 0.0,
            "expected_net_value": 0.0,
            "is_profitable": False,
            "action_executable": False,
            "net_value_decision": "REJECT_INVALID_CURRENCY",
            "reason": f"Unsupported or missing currency '{currency}'"
        }

    recovery_dec = safe_to_decimal(expected_recovery_value)
    inc_cost_dec = safe_to_decimal(incentive_cost) or Decimal("0.00")
    ops_cost_dec = safe_to_decimal(operational_cost) or Decimal("0.00")
    prov_cost_dec = safe_to_decimal(provider_cost) or Decimal("0.00")

    if recovery_dec is None or recovery_dec < Decimal("0.00"):
        return {
            "expected_recovery_value": 0.0,
            "total_action_cost": 0.0,
            "expected_net_value": 0.0,
            "is_profitable": False,
            "action_executable": False,
            "net_value_decision": "REJECT_INVALID_RECOVERY_VALUE",
            "reason": "Missing, non-finite, or negative expected recovery value"
        }

    action_key = action_type.upper() if action_type else "SMART_RETRY"
    comm_cost_dec = DEFAULT_ACTION_COSTS.get(action_key, Decimal("0.50"))

    total_cost_dec = comm_cost_dec + inc_cost_dec + ops_cost_dec + prov_cost_dec
    net_value_dec = recovery_dec - total_cost_dec

    is_profitable = net_value_dec >= min_net_value_threshold

    return {
        "expected_recovery_value": float(recovery_dec),
        "total_action_cost": float(total_cost_dec),
        "communication_cost": float(comm_cost_dec),
        "incentive_cost": float(inc_cost_dec),
        "operational_cost": float(ops_cost_dec),
        "provider_cost": float(prov_cost_dec),
        "expected_net_value": float(net_value_dec),
        "is_profitable": is_profitable,
        "action_executable": is_profitable,
        "net_value_decision": "APPROVE" if is_profitable else "REJECT_UNPROFITABLE",
        "reason": None if is_profitable else ("NEGATIVE_NET_VALUE" if net_value_dec < Decimal("0.00") else "ZERO_OR_SUBTHRESHOLD_NET_VALUE")
    }
