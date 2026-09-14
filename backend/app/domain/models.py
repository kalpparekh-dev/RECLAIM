import enum
import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class PaymentState(str, enum.Enum):
    CREATED = "CREATED"
    PROCESSING = "PROCESSING"
    AUTHORIZED = "AUTHORIZED"
    CAPTURED = "CAPTURED"
    FAILED = "FAILED"
    PENDING = "PENDING"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    REFUNDED = "REFUNDED"
    RECOVERY_ELIGIBLE = "RECOVERY_ELIGIBLE"
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    RETRY_PROCESSING = "RETRY_PROCESSING"
    RECOVERED = "RECOVERED"
    RECOVERY_FAILED = "RECOVERY_FAILED"
    SUPPRESSED = "SUPPRESSED"

VALID_STATE_TRANSITIONS: Dict[PaymentState, List[PaymentState]] = {
    PaymentState.CREATED: [PaymentState.PROCESSING, PaymentState.FAILED, PaymentState.CANCELLED],
    PaymentState.PROCESSING: [PaymentState.AUTHORIZED, PaymentState.CAPTURED, PaymentState.FAILED, PaymentState.PENDING, PaymentState.CANCELLED],
    PaymentState.AUTHORIZED: [PaymentState.CAPTURED, PaymentState.REFUNDED, PaymentState.CANCELLED],
    PaymentState.CAPTURED: [PaymentState.REFUNDED],
    PaymentState.FAILED: [PaymentState.RECOVERY_ELIGIBLE, PaymentState.RETRY_SCHEDULED, PaymentState.SUPPRESSED, PaymentState.CANCELLED],
    PaymentState.PENDING: [PaymentState.CAPTURED, PaymentState.FAILED, PaymentState.CANCELLED, PaymentState.EXPIRED],
    PaymentState.CANCELLED: [],
    PaymentState.EXPIRED: [],
    PaymentState.REFUNDED: [],
    PaymentState.RECOVERY_ELIGIBLE: [PaymentState.RETRY_SCHEDULED, PaymentState.SUPPRESSED, PaymentState.RECOVERY_FAILED],
    PaymentState.RETRY_SCHEDULED: [PaymentState.RETRY_PROCESSING, PaymentState.CANCELLED, PaymentState.SUPPRESSED],
    PaymentState.RETRY_PROCESSING: [PaymentState.RECOVERED, PaymentState.RECOVERY_FAILED, PaymentState.FAILED],
    PaymentState.RECOVERED: [PaymentState.REFUNDED],
    PaymentState.RECOVERY_FAILED: [PaymentState.RETRY_SCHEDULED, PaymentState.SUPPRESSED, PaymentState.CANCELLED],
    PaymentState.SUPPRESSED: [PaymentState.RETRY_SCHEDULED] # Operator manual override
}

class FailureReason(str, enum.Enum):
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    BANK_SERVER_ERROR = "BANK_SERVER_ERROR"
    GATEWAY_TIMEOUT = "GATEWAY_TIMEOUT"
    UPI_LIMIT_EXCEEDED = "UPI_LIMIT_EXCEEDED"
    NETWORK_ERROR = "NETWORK_ERROR"
    CUSTOMER_CANCELLATION = "CUSTOMER_CANCELLATION"
    EXPIRED_CARD = "EXPIRED_CARD"
    UNKNOWN_GATEWAY_ERROR = "UNKNOWN_GATEWAY_ERROR"

class ReconciliationStatus(str, enum.Enum):
    MATCHED = "MATCHED"
    PENDING = "PENDING"
    MISMATCHED = "MISMATCHED"
    UNKNOWN = "UNKNOWN"
    MANUAL_REVIEW = "MANUAL_REVIEW"

class CustomerFeedbackCategory(str, enum.Enum):
    RETRY_WORKED = "RETRY_WORKED"
    RETRY_DID_NOT_WORK = "RETRY_DID_NOT_WORK"
    TOO_MANY_REMINDERS = "TOO_MANY_REMINDERS"
    PAYMENT_EXPERIENCE = "PAYMENT_EXPERIENCE"
    OTHER = "OTHER"

# Request Schemas
class CreatePaymentRequest(BaseModel):
    amount: float = Field(gt=0, description="Payment amount")
    currency: str = Field(default="INR", description="Currency code")
    customer_id: str = Field(description="Customer ID")
    payment_method: str = Field(default="CARD", description="Payment method: CARD, UPI, NETBANKING")
    order_id: Optional[str] = None
    idempotency_key: Optional[str] = None

class SimulatePaymentOutcomeRequest(BaseModel):
    payment_id: str
    target_status: PaymentState # FAILED, CAPTURED, PENDING
    failure_reason: Optional[FailureReason] = FailureReason.BANK_SERVER_ERROR

class InitiateRetryRequest(BaseModel):
    payment_id: str
    action_type: str = "SMART_RETRY"

class CustomerFeedbackRequest(BaseModel):
    transaction_id: str
    payment_id: Optional[str] = None
    rating: int = Field(ge=1, le=5, description="1-5 rating scale")
    comment: Optional[str] = None
    feedback_category: CustomerFeedbackCategory = CustomerFeedbackCategory.PAYMENT_EXPERIENCE

# Model Representation DTOs
class PaymentStateHistoryDTO(BaseModel):
    id: str
    payment_id: str
    previous_state: str
    new_state: str
    event_type: str
    reason: Optional[str] = None
    correlation_id: str
    created_at: str
    actor: str

class PaymentAttemptDTO(BaseModel):
    id: str
    payment_id: str
    attempt_number: int
    attempt_type: str # INITIAL, RETRY
    status: str
    amount: float
    gateway_reference: Optional[str] = None
    failure_reason: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None
    correlation_id: str

class RecoveryActionDTO(BaseModel):
    id: str
    payment_id: str
    customer_id: str
    decision: str
    decision_reason: str
    policy_version: str
    model_score: float
    score_semantics: str
    calibration_status: str
    action_type: str
    action_status: str # SCHEDULED, EXECUTED, SUPPRESSED, FAILED
    experiment_id: str
    experiment_group: str
    attempt_id: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

class OutcomeReconciliationDTO(BaseModel):
    id: str
    payment_id: str
    recovery_action_id: Optional[str] = None
    attempt_id: Optional[str] = None
    original_status: str
    final_status: str
    reconciliation_status: str
    recovered_amount: float
    time_to_recovery_seconds: Optional[float] = None
    reconciled_at: str

class PaymentDTO(BaseModel):
    id: str
    external_id: str
    order_id: str
    customer_id: str
    masked_customer_id: str
    amount: float
    currency: str
    payment_method: str
    status: str
    failure_reason: Optional[str] = None
    idempotency_key: str
    version: int
    created_at: str
    updated_at: str
    attempts: List[PaymentAttemptDTO] = []
    state_history: List[PaymentStateHistoryDTO] = []
    recovery_action: Optional[RecoveryActionDTO] = None
    reconciliation: Optional[OutcomeReconciliationDTO] = None
