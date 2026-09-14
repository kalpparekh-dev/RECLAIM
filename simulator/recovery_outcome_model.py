from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path(
    "data/generated/recovery_opportunities.csv"
)

OUTPUT_PATH = Path(
    "data/generated/recovery_outcomes.csv"
)

RANDOM_SEED = 42


# ============================================================
# ACTION EFFECTIVENESS
# ============================================================
#
# Synthetic simulation assumptions.
#
# These are NOT real Razorpay statistics.
#
# The important thing is that the assumptions are explicit,
# reproducible and used consistently across the simulation.
# ============================================================

ACTION_BASE_PROBABILITY = {
    "RETRY_NOW": 0.72,
    "DELAYED_RETRY": 0.66,
    "REMINDER": 0.48,
    "ESCALATE": 0.35,
    "NO_ACTION": 0.00,
}


# ============================================================
# ACTION COST
# ============================================================

ACTION_COST = {
    "RETRY_NOW": 1.50,
    "DELAYED_RETRY": 1.00,
    "REMINDER": 0.50,
    "ESCALATE": 2.00,
    "NO_ACTION": 0.00,
}


# ============================================================
# CUSTOMER FRICTION
# ============================================================
#
# Repeated interventions are not free from a customer
# experience perspective.
#
# This is represented as a small synthetic cost.
# ============================================================

ACTION_FRICTION = {
    "RETRY_NOW": 0.25,
    "DELAYED_RETRY": 0.15,
    "REMINDER": 0.10,
    "ESCALATE": 0.40,
    "NO_ACTION": 0.00,
}


# ============================================================
# HELPERS
# ============================================================

def print_section(title: str) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# PROBABILITY MODEL
# ============================================================

def calculate_actual_recovery_probability(
    row: pd.Series,
) -> float:

    action = str(
        row["recommended_action"]
    )

    if action == "NO_ACTION":
        return 0.0

    # --------------------------------------------------------
    # Base action probability
    # --------------------------------------------------------

    probability = ACTION_BASE_PROBABILITY[
        action
    ]

    # --------------------------------------------------------
    # Historical reliability
    # --------------------------------------------------------

    historical_success = float(
        row[
            "historical_success_rate"
        ]
    )

    probability *= (
        0.65
        +
        0.35 * historical_success
    )

    # --------------------------------------------------------
    # Recent behaviour
    # --------------------------------------------------------

    recent_failure_rate = float(
        row[
            "failure_rate_last_7d"
        ]
    )

    recent_stability = (
        1.0
        -
        recent_failure_rate
    )

    probability *= (
        0.75
        +
        0.25 * recent_stability
    )

    # --------------------------------------------------------
    # Previous transaction outcome
    # --------------------------------------------------------

    previous_success = int(
        row[
            "previous_transaction_success"
        ]
    )

    if previous_success == 1:

        probability += 0.06

    elif previous_success == -1:

        probability -= 0.04

    # --------------------------------------------------------
    # Failure streak
    # --------------------------------------------------------

    failure_streak = int(
        row[
            "failure_streak"
        ]
    )

    if failure_streak >= 2:

        probability -= (
            0.08
            *
            min(
                failure_streak - 1,
                2,
            )
        )

    # --------------------------------------------------------
    # Failure reason
    # --------------------------------------------------------

    failure_reason = str(
        row["failure_reason"]
    )

    if (
        action == "RETRY_NOW"
        and failure_reason
        in {
            "NETWORK_ERROR",
            "PAYMENT_TIMEOUT",
        }
    ):

        probability += 0.08

    if (
        action == "DELAYED_RETRY"
        and failure_reason
        == "INSUFFICIENT_FUNDS"
    ):

        probability += 0.10

    if (
        action == "REMINDER"
        and failure_reason
        == "INSUFFICIENT_FUNDS"
    ):

        probability += 0.04

    if (
        action == "RETRY_NOW"
        and failure_reason
        == "LIMIT_EXCEEDED"
    ):

        probability -= 0.12

    if (
        action == "RETRY_NOW"
        and failure_reason
        == "BANK_DECLINE"
    ):

        probability -= 0.08

    # --------------------------------------------------------
    # Amount sensitivity
    # --------------------------------------------------------
    #
    # Very large payments are slightly harder to recover
    # immediately.
    #
    # This prevents huge transactions from automatically
    # dominating the simulation.
    # --------------------------------------------------------

    amount = float(
        row["amount"]
    )

    if amount >= 10000:

        probability -= 0.05

    elif amount >= 5000:

        probability -= 0.02

    # --------------------------------------------------------
    # Confidence adjustment
    # --------------------------------------------------------

    confidence = float(
        row[
            "model_confidence"
        ]
    ) / 100.0

    probability *= (
        0.85
        +
        0.15 * confidence
    )

    # --------------------------------------------------------
    # Clamp
    # --------------------------------------------------------

    probability = float(
        np.clip(
            probability,
            0.01,
            0.95,
        )
    )

    return probability


# ============================================================
# RECOVERY SIMULATION
# ============================================================

def simulate_recovery(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    result = df.copy()

    print(
        "Simulating customer responses..."
    )

    # --------------------------------------------------------
    # Actual probability
    # --------------------------------------------------------

    result[
        "actual_recovery_probability"
    ] = result.apply(
        calculate_actual_recovery_probability,
        axis=1,
    )

    # --------------------------------------------------------
    # Random customer outcome
    # --------------------------------------------------------
    #
    # A payment is not recovered simply because the model
    # predicted it.
    #
    # We sample the actual outcome from the simulated
    # customer response probability.
    # --------------------------------------------------------

    random_values = rng.random(
        len(result)
    )

    result[
        "recovered"
    ] = (
        random_values
        <
        result[
            "actual_recovery_probability"
        ]
    )

    result[
        "recovered"
    ] = result[
        "recovered"
    ].astype(int)

    # --------------------------------------------------------
    # Recovered amount
    # --------------------------------------------------------

    result[
        "recovered_amount"
    ] = np.where(
        result["recovered"] == 1,
        result["amount"],
        0.0,
    )

    # --------------------------------------------------------
    # Intervention cost
    # --------------------------------------------------------

    result[
        "intervention_cost"
    ] = result[
        "recommended_action"
    ].map(
        ACTION_COST
    )

    # --------------------------------------------------------
    # Customer friction
    # --------------------------------------------------------

    result[
        "customer_friction_cost"
    ] = result[
        "recommended_action"
    ].map(
        ACTION_FRICTION
    )

    # --------------------------------------------------------
    # Net recovered value
    # --------------------------------------------------------

    result[
        "net_recovered_value"
    ] = (
        result[
            "recovered_amount"
        ]
        -
        result[
            "intervention_cost"
        ]
        -
        result[
            "customer_friction_cost"
        ]
    )

    result[
        "net_recovered_value"
    ] = (
        result[
            "net_recovered_value"
        ]
        .clip(lower=0)
        .round(2)
    )

    # --------------------------------------------------------
    # Outcome label
    # --------------------------------------------------------

    result[
        "recovery_status"
    ] = np.where(
        result["recovered"] == 1,
        "RECOVERED",
        np.where(
            result[
                "recommended_action"
            ]
            == "NO_ACTION",
            "ABSTAINED",
            "NOT_RECOVERED",
        ),
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_outcomes(
    result: pd.DataFrame,
) -> None:

    print_section(
        "RECOVERY OUTCOME VALIDATION"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = {
        "actual_recovery_probability",
        "recovered",
        "recovered_amount",
        "intervention_cost",
        "customer_friction_cost",
        "net_recovered_value",
        "recovery_status",
    }

    missing = (
        required_columns
        -
        set(result.columns)
    )

    if missing:

        print(
            "[FAIL] Missing outcome columns:"
        )

        for column in sorted(missing):

            print(
                f"  - {column}"
            )

        raise ValueError(
            "Outcome validation failed."
        )

    print(
        "[PASS] Required outcome columns"
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability_valid = (
        result[
            "actual_recovery_probability"
        ]
        .between(0, 1)
        .all()
    )

    print(
        f"[{'PASS' if probability_valid else 'FAIL'}] "
        "Actual recovery probabilities are between 0 and 1"
    )

    # --------------------------------------------------------
    # Recovered flag
    # --------------------------------------------------------

    recovered_valid = (
        set(
            result[
                "recovered"
            ].unique()
        )
        <= {0, 1}
    )

    print(
        f"[{'PASS' if recovered_valid else 'FAIL'}] "
        "Recovery flags are binary"
    )

    # --------------------------------------------------------
    # Recovered amount
    # --------------------------------------------------------

    amount_valid = (
        (
            result[
                "recovered_amount"
            ]
            >= 0
        )
        &
        (
            result[
                "recovered_amount"
            ]
            <= result[
                "amount"
            ]
        )
    ).all()

    print(
        f"[{'PASS' if amount_valid else 'FAIL'}] "
        "Recovered amounts are valid"
    )

    # --------------------------------------------------------
    # Outcome consistency
    # --------------------------------------------------------

    recovered_consistency = (
        (
            result["recovered"] == 1
        )
        ==
        (
            result[
                "recovery_status"
            ]
            == "RECOVERED"
        )
    )

    # NO_ACTION must never be marked recovered.

    no_action_consistency = (
        result.loc[
            result[
                "recommended_action"
            ]
            == "NO_ACTION",
            "recovered",
        ]
        == 0
    ).all()

    print(
        f"[{'PASS' if recovered_consistency.all() else 'FAIL'}] "
        "Recovery status matches recovery flag"
    )

    print(
        f"[{'PASS' if no_action_consistency else 'FAIL'}] "
        "Abstained opportunities are not recovered"
    )

    # --------------------------------------------------------
    # No missing values
    # --------------------------------------------------------

    no_missing = (
        result[
            list(required_columns)
        ]
        .isna()
        .sum()
        .sum()
        == 0
    )

    print(
        f"[{'PASS' if no_missing else 'FAIL'}] "
        "No missing outcome values"
    )


# ============================================================
# BUSINESS METRICS
# ============================================================

def calculate_business_metrics(
    result: pd.DataFrame,
) -> dict:

    total_failed_value = (
        result["amount"]
        .sum()
    )

    attempted = (
        result[
            "recommended_action"
        ]
        != "NO_ACTION"
    )

    intervention_count = (
        attempted.sum()
    )

    recovered_count = (
        result[
            "recovered"
        ]
        .sum()
    )

    recovered_value = (
        result[
            "recovered_amount"
        ]
        .sum()
    )

    total_intervention_cost = (
        result[
            "intervention_cost"
        ]
        .sum()
    )

    total_friction_cost = (
        result[
            "customer_friction_cost"
        ]
        .sum()
    )

    net_recovered = (
        result[
            "net_recovered_value"
        ]
        .sum()
    )

    intervention_rate = (
        intervention_count
        /
        len(result)
        *
        100
    )

    recovery_rate = (
        recovered_value
        /
        total_failed_value
        *
        100
    )

    recovery_rate_count = (
        recovered_count
        /
        len(result)
        *
        100
    )

    recovery_per_intervention = (
        recovered_value
        /
        intervention_count
        if intervention_count > 0
        else 0
    )

    return {
        "total_failed_value":
            total_failed_value,

        "intervention_count":
            intervention_count,

        "intervention_rate":
            intervention_rate,

        "recovered_count":
            recovered_count,

        "recovered_value":
            recovered_value,

        "recovery_rate":
            recovery_rate,

        "recovery_rate_count":
            recovery_rate_count,

        "intervention_cost":
            total_intervention_cost,

        "friction_cost":
            total_friction_cost,

        "net_recovered":
            net_recovered,

        "recovery_per_intervention":
            recovery_per_intervention,
    }


# ============================================================
# REPORT
# ============================================================

def print_report(
    result: pd.DataFrame,
) -> None:

    print_section(
        "RECLAIM MEASURED RECOVERY REPORT"
    )

    metrics = calculate_business_metrics(
        result
    )

    print(
        f"Failed payment value:       "
        f"₹{metrics['total_failed_value']:,.2f}"
    )

    print(
        f"Interventions:              "
        f"{metrics['intervention_count']:,}"
    )

    print(
        f"Intervention rate:          "
        f"{metrics['intervention_rate']:.2f}%"
    )

    print(
        f"Recovered payments:         "
        f"{metrics['recovered_count']:,}"
    )

    print(
        f"Recovered value:            "
        f"₹{metrics['recovered_value']:,.2f}"
    )

    print(
        f"Recovery by value:          "
        f"{metrics['recovery_rate']:.2f}%"
    )

    print(
        f"Recovery by transaction:    "
        f"{metrics['recovery_rate_count']:.2f}%"
    )

    print(
        f"Intervention cost:          "
        f"₹{metrics['intervention_cost']:,.2f}"
    )

    print(
        f"Customer friction cost:     "
        f"₹{metrics['friction_cost']:,.2f}"
    )

    print(
        f"Net recovered value:        "
        f"₹{metrics['net_recovered']:,.2f}"
    )

    print(
        f"Recovery / intervention:    "
        f"₹{metrics['recovery_per_intervention']:,.2f}"
    )

    # --------------------------------------------------------
    # Status distribution
    # --------------------------------------------------------

    print(
        "\nOutcome distribution:"
    )

    distribution = (
        result[
            "recovery_status"
        ]
        .value_counts()
    )

    for status, count in (
        distribution.items()
    ):

        percentage = (
            count
            /
            len(result)
            *
            100
        )

        print(
            f"{status:<18}: "
            f"{count:>5,} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Action effectiveness
    # --------------------------------------------------------

    print(
        "\nMeasured action performance:"
    )

    action_table = (
        result
        .groupby(
            "recommended_action",
            observed=True,
        )
        .agg(
            interventions=(
                "transaction_id",
                "count",
            ),
            recovered=(
                "recovered",
                "sum",
            ),
            attempted_value=(
                "amount",
                "sum",
            ),
            recovered_value=(
                "recovered_amount",
                "sum",
            ),
        )
        .reset_index()
    )

    action_table[
        "recovery_rate"
    ] = (
        action_table[
            "recovered"
        ]
        /
        action_table[
            "interventions"
        ]
        *
        100
    )

    print(
        action_table.to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

def save_result(
    result: pd.DataFrame,
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("RECLAIM RECOVERY OUTCOME MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Recovery opportunities not found: "
            f"{INPUT_PATH}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Recovery opportunities loaded: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Simulate actual outcomes
    # --------------------------------------------------------

    result = simulate_recovery(
        df
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_outcomes(
        result
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_report(
        result
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_result(
        result
    )

    print(
        "\nRECOVERY OUTCOME STATUS: READY"
    )


if __name__ == "__main__":
    main()