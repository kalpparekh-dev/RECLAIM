from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

CUSTOMER_PATH = Path(
    "data/generated/customers.csv"
)

TRANSACTION_PATH = Path(
    "data/generated/transactions.csv"
)

OUTPUT_PATH = Path(
    "data/generated/transaction_outcomes.csv"
)

DEFAULT_SEED = 20260822


# ============================================================
# Failure reasons
# ============================================================

FAILURE_REASONS = [
    "INSUFFICIENT_FUNDS",
    "BANK_DECLINE",
    "NETWORK_ERROR",
    "LIMIT_EXCEEDED",
    "PAYMENT_TIMEOUT",
]


# ============================================================
# Utility
# ============================================================

def sigmoid(value: float) -> float:
    """Convert a score into a probability."""

    # Prevent numerical overflow.
    value = np.clip(value, -20, 20)

    return float(
        1.0 / (1.0 + np.exp(-value))
    )


def choose_failure_reason(
    payment_method: str,
    rng: np.random.Generator,
) -> str:
    """
    Choose a failure reason using payment-method-dependent
    probabilities.
    """

    if payment_method == "UPI":

        reasons = [
            "NETWORK_ERROR",
            "BANK_DECLINE",
            "INSUFFICIENT_FUNDS",
            "PAYMENT_TIMEOUT",
            "LIMIT_EXCEEDED",
        ]

        probabilities = [
            0.25,
            0.25,
            0.30,
            0.15,
            0.05,
        ]

    elif payment_method == "CARD":

        reasons = [
            "INSUFFICIENT_FUNDS",
            "BANK_DECLINE",
            "LIMIT_EXCEEDED",
            "NETWORK_ERROR",
            "PAYMENT_TIMEOUT",
        ]

        probabilities = [
            0.30,
            0.30,
            0.20,
            0.12,
            0.08,
        ]

    elif payment_method == "NETBANKING":

        reasons = [
            "BANK_DECLINE",
            "NETWORK_ERROR",
            "PAYMENT_TIMEOUT",
            "INSUFFICIENT_FUNDS",
            "LIMIT_EXCEEDED",
        ]

        probabilities = [
            0.30,
            0.30,
            0.20,
            0.15,
            0.05,
        ]

    else:  # WALLET

        reasons = [
            "INSUFFICIENT_FUNDS",
            "LIMIT_EXCEEDED",
            "NETWORK_ERROR",
            "BANK_DECLINE",
            "PAYMENT_TIMEOUT",
        ]

        probabilities = [
            0.30,
            0.20,
            0.25,
            0.15,
            0.10,
        ]

    return str(
        rng.choice(
            reasons,
            p=probabilities,
        )
    )


# ============================================================
# Failure probability
# ============================================================

def calculate_failure_probability(
    transaction: pd.Series,
    customer: pd.Series,
) -> float:
    """
    Calculate the probability that a payment attempt fails.

    IMPORTANT:
    Hidden customer traits are used only by the simulator.
    They will NOT be exposed to the future RECLAIM agent.
    """

    amount = float(
        transaction["amount"]
    )

    payment_method = (
        transaction["payment_method"]
    )

    reliability = float(
        customer["reliability"]
    )

    financial_stability = float(
        customer["financial_stability"]
    )

    activity_level = float(
        customer["activity_level"]
    )

    # --------------------------------------------------------
    # Base failure rate
    # --------------------------------------------------------

    score = -2.55

    # --------------------------------------------------------
    # Customer reliability
    #
    # Higher reliability → lower failure probability.
    # --------------------------------------------------------

    score += (
        1.25
        * (0.55 - reliability)
    )

    # --------------------------------------------------------
    # Financial stability
    #
    # Lower financial stability → more likely to fail.
    # --------------------------------------------------------

    score += (
        1.10
        * (0.55 - financial_stability)
    )

    # --------------------------------------------------------
    # Transaction amount
    #
    # Larger payments carry somewhat more failure risk.
    # Logarithm prevents large transactions from completely
    # dominating the model.
    # --------------------------------------------------------

    amount_effect = np.log1p(
        amount / 1000
    )

    score += (
        0.25
        * (amount_effect - 0.7)
    )

    # --------------------------------------------------------
    # Activity level
    #
    # Very high activity creates slightly more opportunity
    # for payment friction.
    # --------------------------------------------------------

    score += (
        0.15
        * (activity_level - 0.5)
    )

    # --------------------------------------------------------
    # Payment-method effect
    # --------------------------------------------------------

    payment_effects = {
        "UPI": 0.00,
        "CARD": 0.05,
        "NETBANKING": 0.10,
        "WALLET": 0.03,
    }

    score += payment_effects.get(
        payment_method,
        0.0,
    )

    probability = sigmoid(score)

    # Keep the probability inside a sensible range.
    return float(
        np.clip(
            probability,
            0.01,
            0.35,
        )
    )


# ============================================================
# Main simulation
# ============================================================

def generate_outcomes(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Join customer simulator state
    # --------------------------------------------------------

    customer_state_columns = [
        "customer_id",
        "reliability",
        "financial_stability",
        "responsiveness",
        "retry_sensitivity",
        "reminder_sensitivity",
        "activity_level",
    ]

    customer_state = customers[
        customer_state_columns
    ]

    merged = transactions.merge(
        customer_state,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    if merged[
        "reliability"
    ].isna().any():

        raise ValueError(
            "Some transactions have no matching customer."
        )

    # --------------------------------------------------------
    # Generate outcomes
    # --------------------------------------------------------

    statuses = []
    failure_reasons = []
    probabilities = []

    for _, row in merged.iterrows():

        probability = calculate_failure_probability(
            transaction=row,
            customer=row,
        )

        random_value = rng.random()

        failed = (
            random_value < probability
        )

        if failed:

            status = "FAILED"

            reason = choose_failure_reason(
                payment_method=row[
                    "payment_method"
                ],
                rng=rng,
            )

        else:

            status = "SUCCESS"
            reason = None

        statuses.append(status)
        failure_reasons.append(reason)
        probabilities.append(probability)

    # --------------------------------------------------------
    # Add observable outcome information
    # --------------------------------------------------------

    outcomes = transactions.copy()

    outcomes["status"] = statuses

    outcomes["failure_reason"] = (
        failure_reasons
    )

    # This is simulator metadata.
    # It will NOT be exposed to the agent.
    outcomes["_simulated_failure_probability"] = (
        probabilities
    )

    return outcomes


# ============================================================
# Validation
# ============================================================

def validate_outcomes(
    outcomes: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print("FAILURE MODEL SANITY CHECK")
    print("=" * 70)

    total = len(outcomes)

    failed = (
        outcomes["status"] == "FAILED"
    ).sum()

    successful = (
        outcomes["status"] == "SUCCESS"
    ).sum()

    failure_rate = (
        failed / total
        if total > 0
        else 0
    )

    print(
        f"Total transactions: {total:,}"
    )

    print(
        f"Successful:         {successful:,}"
    )

    print(
        f"Failed:             {failed:,}"
    )

    print(
        f"Failure rate:       "
        f"{failure_rate * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Status validation
    # --------------------------------------------------------

    valid_statuses = (
        set(
            outcomes["status"].unique()
        )
        <= {
            "SUCCESS",
            "FAILED",
        }
    )

    print(
        f"\n[{'PASS' if valid_statuses else 'FAIL'}] "
        "Valid transaction statuses"
    )

    # --------------------------------------------------------
    # Failed transactions must have reasons
    # --------------------------------------------------------

    failed_transactions = outcomes[
        outcomes["status"] == "FAILED"
    ]

    reasons_present = (
        failed_transactions[
            "failure_reason"
        ]
        .notna()
        .all()
    )

    print(
        f"[{'PASS' if reasons_present else 'FAIL'}] "
        "Failed transactions have failure reasons"
    )

    # --------------------------------------------------------
    # Successful transactions should not have reasons
    # --------------------------------------------------------

    successful_transactions = outcomes[
        outcomes["status"] == "SUCCESS"
    ]

    no_success_reasons = (
        successful_transactions[
            "failure_reason"
        ]
        .isna()
        .all()
    )

    print(
        f"[{'PASS' if no_success_reasons else 'FAIL'}] "
        "Successful transactions have no failure reason"
    )

    # --------------------------------------------------------
    # Failure reason distribution
    # --------------------------------------------------------

    print(
        "\nFailure reason distribution:"
    )

    if len(failed_transactions) > 0:

        reason_distribution = (
            failed_transactions[
                "failure_reason"
            ]
            .value_counts(
                normalize=True
            )
            .mul(100)
            .round(2)
        )

        for reason, percentage in (
            reason_distribution.items()
        ):

            print(
                f"{reason:<22}: "
                f"{percentage:>6.2f}%"
            )

    # --------------------------------------------------------
    # Payment method failure rates
    # --------------------------------------------------------

    print(
        "\nFailure rate by payment method:"
    )

    method_rates = (
        outcomes
        .groupby("payment_method")
        .apply(
            lambda group:
            (
                group["status"] == "FAILED"
            ).mean(),
            include_groups=False,
        )
        .mul(100)
        .round(2)
    )

    for method, rate in method_rates.items():

        print(
            f"{method:<12}: "
            f"{rate:>6.2f}%"
        )

    # --------------------------------------------------------
    # Amount comparison
    # --------------------------------------------------------

    print(
        "\nAverage transaction amount:"
    )

    successful_mean = (
        successful_transactions[
            "amount"
        ].mean()
    )

    failed_mean = (
        failed_transactions[
            "amount"
        ].mean()
    )

    print(
        f"Successful: ₹{successful_mean:,.2f}"
    )

    print(
        f"Failed:     ₹{failed_mean:,.2f}"
    )

    # --------------------------------------------------------
    # Simulated probability range
    # --------------------------------------------------------

    print(
        "\nSimulated failure probability:"
    )

    print(
        outcomes[
            "_simulated_failure_probability"
        ]
        .describe()
        .round(4)
        .to_string()
    )


# ============================================================
# Save
# ============================================================

def save_outcomes(
    outcomes: pd.DataFrame,
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    outcomes.to_csv(
        OUTPUT_PATH,
        index=False,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("RECLAIM FAILURE MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not CUSTOMER_PATH.exists():

        raise FileNotFoundError(
            f"Customer dataset not found: "
            f"{CUSTOMER_PATH}"
        )

    if not TRANSACTION_PATH.exists():

        raise FileNotFoundError(
            f"Transaction dataset not found: "
            f"{TRANSACTION_PATH}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    customers = pd.read_csv(
        CUSTOMER_PATH
    )

    transactions = pd.read_csv(
        TRANSACTION_PATH,
        parse_dates=["timestamp"],
    )

    print(
        f"Customers loaded:    "
        f"{len(customers):,}"
    )

    print(
        f"Transactions loaded: "
        f"{len(transactions):,}"
    )

    # --------------------------------------------------------
    # Generate outcomes
    # --------------------------------------------------------

    print(
        "\nSimulating payment outcomes..."
    )

    outcomes = generate_outcomes(
        transactions=transactions,
        customers=customers,
        seed=DEFAULT_SEED,
    )

    print(
        "Outcome simulation complete."
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_outcomes(
        outcomes
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outcomes(
        outcomes
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()