from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class PaymentGatewayError(Exception):
    """Raised when an error occurs communicating with a payment gateway."""
    pass

class PaymentGateway(ABC):
    """
    Abstract Payment Gateway Interface.
    Implemented by SimulatorGateway (Sandbox) and RazorpayGatewayAdapter (Production / Test Mode).
    """

    @abstractmethod
    def create_payment(
        self,
        customer_id: str,
        amount: float,
        currency: str = "INR",
        payment_method: str = "CARD",
        order_id: Optional[str] = None,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a new payment/order on the gateway."""
        pass

    @abstractmethod
    def fetch_payment(self, gateway_reference: str) -> Dict[str, Any]:
        """Fetches payment state directly from gateway."""
        pass

    @abstractmethod
    def initiate_retry(
        self,
        gateway_reference: str,
        action_type: str = "SMART_RETRY",
        target_outcome: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initiates a payment retry intervention on the gateway."""
        pass

    @abstractmethod
    def fetch_payment_status(self, gateway_reference: str) -> str:
        """Returns the current status string on gateway."""
        pass

    @abstractmethod
    def verify_webhook_signature(
        self,
        raw_body: bytes,
        signature: Optional[str],
        secret: Optional[str] = None
    ) -> bool:
        """Verifies HMAC webhook signature."""
        pass

    @abstractmethod
    def handle_refund_or_reversal(
        self,
        gateway_reference: str,
        amount: Optional[float] = None,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initiates a refund or reversal on the gateway."""
        pass
