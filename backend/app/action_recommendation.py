from typing import Dict, Any, Optional

class ActionRecommendationService:
    def recommend_action(
        self,
        transaction_id: str,
        policy_decision: str,
        failure_reason: Optional[str] = None,
        payment_method: Optional[str] = None,
        amount: float = 0.0
    ) -> Dict[str, Any]:
        """
        Maps failure reasons & payment methods to recovery actions without changing policy decision.
        """
        if policy_decision != "TARGET":
            return {
                "action_type": "RETRY_ALL",
                "action_reason": "Standard automated fallback retry path",
                "priority": "LOW",
                "requires_approval": False,
                "estimated_cost": 0.50,
                "cooldown_hours": 0,
                "provider": "STANDARD_GATEWAY",
                "simulation_mode": True
            }

        reason_str = (failure_reason or "").upper()
        method_str = (payment_method or "").upper()

        if "BANK" in reason_str or "DECLINE" in reason_str:
            action_type = "SCHEDULE_INTELLIGENT_RETRY"
            reason = "Temporary bank decline detected; scheduling off-peak retry window"
            cost = 0.50
            cooldown = 12
        elif "LIMIT" in reason_str:
            action_type = "SEND_PAYMENT_REMINDER"
            reason = "Limit exceeded; sending customer limit reset reminder"
            cost = 0.30
            cooldown = 24
        elif "NETWORK" in reason_str or "TIMEOUT" in reason_str:
            action_type = "GATEWAY_FALLBACK"
            reason = "Network error; rerouting through secondary gateway"
            cost = 1.50
            cooldown = 6
        elif "EXPIRED" in reason_str or "CARD" in method_str:
            action_type = "OFFER_ALTERNATE_METHOD"
            reason = "Card issue; prompting customer for alternate payment method"
            cost = 0.30
            cooldown = 24
        else:
            action_type = "SEND_PAYMENT_REMINDER"
            reason = "Targeted recovery intervention recommended by causal model"
            cost = 0.30
            cooldown = 24

        priority = "HIGH" if amount > 5000 else "MEDIUM"

        return {
            "action_type": action_type,
            "action_reason": reason,
            "priority": priority,
            "requires_approval": amount > 10000,
            "estimated_cost": cost,
            "cooldown_hours": cooldown,
            "provider": "MOCK_GATEWAY_PROVIDER",
            "simulation_mode": True
        }

_action_recommendation_instance = None

def get_action_recommendation_service() -> ActionRecommendationService:
    global _action_recommendation_instance
    if _action_recommendation_instance is None:
        _action_recommendation_instance = ActionRecommendationService()
    return _action_recommendation_instance
