import os
import hmac
import hashlib
from typing import Optional

class WebhookVerificationError(ValueError):
    """Raised when webhook signature validation fails."""
    pass

def verify_razorpay_signature(
    raw_body: bytes,
    signature: Optional[str],
    secret: Optional[str] = None
) -> bool:
    """
    Verifies Razorpay webhook HMAC-SHA256 signature over raw request body in constant time.
    """
    effective_secret = secret or os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
    require_sig = os.getenv("REQUIRE_WEBHOOK_SIGNATURE", "false").lower() in ("true", "1", "yes")

    # If signature requirement is disabled and no signature provided, allow in test/simulation mode
    if not require_sig and not signature:
        return True

    if not effective_secret:
        if require_sig:
            raise WebhookVerificationError("RAZORPAY_WEBHOOK_SECRET environment variable is not configured.")
        return True

    if not signature:
        raise WebhookVerificationError("Missing 'X-Razorpay-Signature' header.")

    computed = hmac.new(
        effective_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(computed.lower(), signature.strip().lower()):
        raise WebhookVerificationError("Invalid webhook signature.")

    return True
