import datetime
from typing import Dict, Any, Optional

class SegmentGuard:
    def __init__(self, max_segment_concentration: float = 0.45):
        self.max_concentration = max_segment_concentration
        self.segment_counts: Dict[str, int] = {}
        self.total_target_evaluations: int = 0

    def evaluate_segment_safety(
        self,
        payment_method: Optional[str] = "CARD",
        failure_reason: Optional[str] = "BANK_DECLINE",
        amount: float = 0.0
    ) -> Dict[str, Any]:
        pm = (payment_method or "CARD").upper()
        fr = (failure_reason or "BANK_DECLINE").upper()

        if amount > 5000:
            amount_tier = "HIGH"
        elif amount > 1000:
            amount_tier = "MEDIUM"
        else:
            amount_tier = "LOW"

        segment_key = f"{pm}:{fr}:{amount_tier}"

        if self.total_target_evaluations > 20: # Require baseline sample size
            current_count = self.segment_counts.get(segment_key, 0)
            share = current_count / self.total_target_evaluations
            if share > self.max_concentration:
                return {
                    "segment_key": segment_key,
                    "is_safe": False,
                    "concentration_share": float(share),
                    "max_allowed_share": self.max_concentration,
                    "suppression_reason": "SEGMENT_CONCENTRATION_CAP_EXCEEDED"
                }

        return {
            "segment_key": segment_key,
            "is_safe": True,
            "concentration_share": 0.0 if self.total_target_evaluations == 0 else self.segment_counts.get(segment_key, 0) / self.total_target_evaluations,
            "max_allowed_share": self.max_concentration,
            "suppression_reason": None
        }

    def record_target_segment(self, payment_method: Optional[str], failure_reason: Optional[str], amount: float):
        pm = (payment_method or "CARD").upper()
        fr = (failure_reason or "BANK_DECLINE").upper()
        amount_tier = "HIGH" if amount > 5000 else ("MEDIUM" if amount > 1000 else "LOW")
        segment_key = f"{pm}:{fr}:{amount_tier}"

        self.segment_counts[segment_key] = self.segment_counts.get(segment_key, 0) + 1
        self.total_target_evaluations += 1

_segment_guard_instance = None

def get_segment_guard() -> SegmentGuard:
    global _segment_guard_instance
    if _segment_guard_instance is None:
        _segment_guard_instance = SegmentGuard()
    return _segment_guard_instance
