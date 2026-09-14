from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path(
    "data/generated/recovery_context.csv"
)

OUTPUT_PATH = Path(
    "data/generated/recovery_opportunities.csv"
)


# ============================================================
# INTERVENTION ECONOMICS
# ============================================================
#
# These are synthetic simulation assumptions.
#
# They represent the estimated effectiveness of each action,
# NOT real Razorpay statistics.
#
# We keep them explicit so the entire decision can be audited.
# ============================================================

ACTION_EFFECTIVENESS = {
    "RETRY_NOW": 0.72,
    "DELAYED_RETRY": 0.66,
    "REMINDER": 0.48,
    "ESCALATE": 0.35,
    "NO_ACTION": 0.00,
}


ACTION_COST = {
    "RETRY_NOW": 1.50,
    "DELAYED_RETRY": 1.00,
    "REMINDER": 0.50,
    "ESCALATE": 2.00,
    "NO_ACTION": 0.00,
}


# Minimum expected recovery value required before
# RECLAIM recommends an intervention.
MIN_EXPECTED_VALUE = 50.0


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_section(title: str) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(
    series: pd.Series,
    minimum: float | None = None,
    maximum: float | None = None,
) -> pd.Series:

    if minimum is None:
        minimum = series.min()

    if maximum is None:
        maximum = series.max()

    denominator = maximum - minimum

    if denominator == 0:

        return pd.Series(
            0.5,
            index=series.index,
        )

    result = (
        (series - minimum)
        /
        denominator
    )

    return result.clip(0, 1)


# ============================================================
# RECOVERABILITY SCORE
# ============================================================

def calculate_recoverability_score(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # --------------------------------------------------------
    # 1. Historical reliability
    # --------------------------------------------------------
    #
    # A customer with a strong historical success rate is
    # generally more promising than one with repeated failures.
    #

    historical_reliability = (
        result[
            "historical_success_rate"
        ]
        .clip(0, 1)
    )

    # --------------------------------------------------------
    # 2. Recent stability
    # --------------------------------------------------------
    #
    # Low recent failure rate is positive.
    #

    recent_stability = (
        1
        -
        result[
            "failure_rate_last_7d"
        ]
        .clip(0, 1)
    )

    # --------------------------------------------------------
    # 3. Recent engagement
    # --------------------------------------------------------
    #
    # Customers actively transacting recently give us more
    # behavioral evidence.
    #

    recent_activity = normalize(
        result[
            "transactions_last_7d"
        ]
    )

    # --------------------------------------------------------
    # 4. Previous-success signal
    # --------------------------------------------------------
    #
    # If the previous transaction succeeded, the customer
    # demonstrates recent ability to complete payment.
    #

    previous_success = np.where(
        result[
            "previous_transaction_success"
        ] == 1,
        1.0,
        np.where(
            result[
                "previous_transaction_success"
            ] == -1,
            0.5,
            0.0,
        ),
    )

    previous_success = pd.Series(
        previous_success,
        index=result.index,
    )

    # --------------------------------------------------------
    # 5. Failure streak penalty
    # --------------------------------------------------------

    streak_penalty = (
        result[
            "failure_streak"
        ]
        .clip(0, 3)
        /
        3
    )

    streak_signal = (
        1 - streak_penalty
    )

    # --------------------------------------------------------
    # 6. Historical transaction evidence
    # --------------------------------------------------------
    #
    # More historical observations mean stronger evidence.
    #

    evidence_strength = (
        result[
            "historical_transaction_count"
        ]
        /
        40
    ).clip(0, 1)

    # --------------------------------------------------------
    # Weighted recoverability
    # --------------------------------------------------------

    score = (
        historical_reliability * 0.30
        +
        recent_stability * 0.20
        +
        recent_activity * 0.10
        +
        previous_success * 0.20
        +
        streak_signal * 0.15
        +
        evidence_strength * 0.05
    )

    result[
        "recoverability_score"
    ] = (
        score * 100
    ).round(2)

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------
    #
    # Confidence depends primarily on how much observable
    # history we have.
    #

    confidence = (
        evidence_strength * 0.60
        +
        recent_activity * 0.20
        +
        (
            1
            -
            (
                result[
                    "failure_rate_last_7d"
                ]
                -
                0.5
            ).abs()
        )
        * 0.20
    )

    result[
        "model_confidence"
    ] = (
        confidence
        .clip(0, 1)
        * 100
    ).round(2)

    return result


# ============================================================
# BASE RECOVERY PROBABILITY
# ============================================================

def calculate_base_recovery_probability(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    score_probability = (
        result[
            "recoverability_score"
        ]
        / 100
    )

    # Convert score into a probability-like estimate.
    #
    # This is a simulation model, not a calibrated production
    # probability model.
    #

    probability = (
        0.10
        +
        (
            score_probability
            * 0.80
        )
    )

    # Repeated failures reduce expected recovery.

    probability -= (
        result[
            "failure_streak"
        ]
        .clip(0, 3)
        * 0.06
    )

    # Strong recent history gets a small positive adjustment.

    probability += (
        result[
            "historical_success_rate"
        ]
        * 0.05
    )

    result[
        "base_recovery_probability"
    ] = (
        probability
        .clip(0.02, 0.95)
        .round(4)
    )

    return result


# ============================================================
# ACTION SCORING
# ============================================================

def calculate_action_values(
    row: pd.Series,
) -> dict:

    amount = float(
        row["amount"]
    )

    base_probability = float(
        row[
            "base_recovery_probability"
        ]
    )

    recoverability = (
        float(
            row[
                "recoverability_score"
            ]
        )
        /
        100
    )

    failure_reason = str(
        row["failure_reason"]
    )

    values = {}

    for action, effectiveness in (
        ACTION_EFFECTIVENESS.items()
    ):

        cost = ACTION_COST[
            action
        ]

        # ----------------------------------------------------
        # Action-specific adjustment
        # ----------------------------------------------------

        probability = (
            base_probability
            *
            effectiveness
        )

        # ----------------------------------------------------
        # Failure reason adjustments
        # ----------------------------------------------------
        #
        # These are intentionally modest.
        # We don't want failure reason alone to determine
        # the decision.
        #

        if (
            action == "RETRY_NOW"
            and failure_reason
            in {
                "NETWORK_ERROR",
                "PAYMENT_TIMEOUT",
            }
        ):

            probability *= 1.12

        if (
            action == "DELAYED_RETRY"
            and failure_reason
            == "INSUFFICIENT_FUNDS"
        ):

            probability *= 1.15

        if (
            action == "REMINDER"
            and failure_reason
            == "BANK_DECLINE"
        ):

            probability *= 1.05

        if (
            action == "ESCALATE"
            and recoverability < 0.35
        ):

            probability *= 0.70

        probability = min(
            probability,
            0.95,
        )

        expected_recovery = (
            amount
            *
            probability
        )

        net_expected_value = (
            expected_recovery
            -
            cost
        )

        values[action] = {
            "probability": probability,
            "expected_recovery": expected_recovery,
            "cost": cost,
            "net_expected_value": (
                net_expected_value
            ),
        }

    return values


# ============================================================
# SELECT BEST ACTION
# ============================================================

def select_best_action(
    row: pd.Series,
) -> pd.Series:

    action_values = (
        calculate_action_values(row)
    )

    best_action = max(
        action_values,
        key=lambda action:
            action_values[action][
                "net_expected_value"
            ],
    )

    best_result = (
        action_values[
            best_action
        ]
    )

    # --------------------------------------------------------
    # Apply stopping rule
    # --------------------------------------------------------

    if (
        best_action == "NO_ACTION"
        or
        best_result[
            "net_expected_value"
        ]
        < MIN_EXPECTED_VALUE
    ):

        final_action = "NO_ACTION"

        final_probability = 0.0

        final_expected_recovery = 0.0

        final_net_value = 0.0

    else:

        final_action = best_action

        final_probability = (
            best_result[
                "probability"
            ]
        )

        final_expected_recovery = (
            best_result[
                "expected_recovery"
            ]
        )

        final_net_value = (
            best_result[
                "net_expected_value"
            ]
        )

    return pd.Series(
        {
            "recommended_action":
                final_action,

            "action_recovery_probability":
                round(
                    final_probability,
                    4,
                ),

            "expected_recovery":
                round(
                    final_expected_recovery,
                    2,
                ),

            "intervention_cost":
                round(
                    (
                        ACTION_COST[
                            final_action
                        ]
                    ),
                    2,
                ),

            "expected_net_recovery":
                round(
                    final_net_value,
                    2,
                ),
        }
    )


# ============================================================
# EXPLANATION GENERATOR
# ============================================================

def generate_reason(
    row: pd.Series,
) -> str:

    reasons = []

    success_rate = float(
        row[
            "historical_success_rate"
        ]
    )

    recent_failure_rate = float(
        row[
            "failure_rate_last_7d"
        ]
    )

    failure_streak = int(
        row[
            "failure_streak"
        ]
    )

    previous_success = int(
        row[
            "previous_transaction_success"
        ]
    )

    if success_rate >= 0.90:

        reasons.append(
            "strong historical payment reliability"
        )

    elif success_rate < 0.70:

        reasons.append(
            "weak historical payment reliability"
        )

    if recent_failure_rate <= 0.05:

        reasons.append(
            "low recent failure rate"
        )

    elif recent_failure_rate >= 0.25:

        reasons.append(
            "elevated recent failure rate"
        )

    if previous_success == 1:

        reasons.append(
            "previous payment succeeded"
        )

    if failure_streak >= 2:

        reasons.append(
            "repeated consecutive failures"
        )

    if not reasons:

        reasons.append(
            "mixed historical and recent payment signals"
        )

    return "; ".join(
        reasons
    )


# ============================================================
# MAIN INTELLIGENCE PIPELINE
# ============================================================

def build_intelligence(
    context: pd.DataFrame,
) -> pd.DataFrame:

    print_section(
        "RECLAIM RECOVERABILITY ENGINE"
    )

    result = calculate_recoverability_score(
        context
    )

    result = calculate_base_recovery_probability(
        result
    )

    print(
        "Calculating optimal intervention..."
    )

    action_results = result.apply(
        select_best_action,
        axis=1,
    )

    result = pd.concat(
        [
            result,
            action_results,
        ],
        axis=1,
    )

    result[
        "decision_reason"
    ] = result.apply(
        generate_reason,
        axis=1,
    )

    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

    result[
        "priority"
    ] = pd.cut(
        result[
            "expected_net_recovery"
        ],
        bins=[
            -np.inf,
            0,
            250,
            1000,
            np.inf,
        ],
        labels=[
            "NONE",
            "LOW",
            "MEDIUM",
            "HIGH",
        ],
    )

    # --------------------------------------------------------
    # Opportunity ranking
    # --------------------------------------------------------

    result[
        "opportunity_rank"
    ] = (
        result[
            "expected_net_recovery"
        ]
        .rank(
            ascending=False,
            method="first",
        )
        .astype(int)
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_intelligence(
    result: pd.DataFrame,
) -> None:

    print_section(
        "RECOVERY INTELLIGENCE VALIDATION"
    )

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    required_columns = {
        "recoverability_score",
        "model_confidence",
        "base_recovery_probability",
        "recommended_action",
        "action_recovery_probability",
        "expected_recovery",
        "intervention_cost",
        "expected_net_recovery",
        "decision_reason",
        "priority",
        "opportunity_rank",
    }

    missing = (
        required_columns
        -
        set(result.columns)
    )

    if missing:

        print(
            "[FAIL] Intelligence columns missing:"
        )

        for column in sorted(missing):

            print(
                f"  - {column}"
            )

        raise ValueError(
            "Recovery intelligence validation failed."
        )

    print(
        "[PASS] Intelligence output columns"
    )

    # --------------------------------------------------------
    # Score ranges
    # --------------------------------------------------------

    score_valid = (
        result[
            "recoverability_score"
        ]
        .between(0, 100)
        .all()
    )

    confidence_valid = (
        result[
            "model_confidence"
        ]
        .between(0, 100)
        .all()
    )

    probability_valid = (
        result[
            "base_recovery_probability"
        ]
        .between(0, 1)
        .all()
    )

    print(
        f"[{'PASS' if score_valid else 'FAIL'}] "
        "Recoverability score range"
    )

    print(
        f"[{'PASS' if confidence_valid else 'FAIL'}] "
        "Confidence range"
    )

    print(
        f"[{'PASS' if probability_valid else 'FAIL'}] "
        "Recovery probability range"
    )

    # --------------------------------------------------------
    # Financial sanity
    # --------------------------------------------------------

    expected_recovery_valid = (
        result[
            "expected_recovery"
        ]
        >= 0
    ).all()

    net_recovery_valid = (
        result[
            "expected_net_recovery"
        ]
        >= 0
    ).all()

    print(
        f"[{'PASS' if expected_recovery_valid else 'FAIL'}] "
        "Expected recovery values are non-negative"
    )

    print(
        f"[{'PASS' if net_recovery_valid else 'FAIL'}] "
        "Expected net recovery values are non-negative"
    )

    # --------------------------------------------------------
    # Action validation
    # --------------------------------------------------------

    valid_actions = set(
        ACTION_EFFECTIVENESS.keys()
    )

    actions_valid = (
        set(
            result[
                "recommended_action"
            ].unique()
        )
        <= valid_actions
    )

    print(
        f"[{'PASS' if actions_valid else 'FAIL'}] "
        "Recommended actions are bounded"
    )

    # --------------------------------------------------------
    # Stopping rule
    # --------------------------------------------------------

    no_action_rows = result[
        result[
            "recommended_action"
        ]
        == "NO_ACTION"
    ]

    stopping_rule_valid = (
        no_action_rows[
            "expected_net_recovery"
        ]
        == 0
    ).all()

    print(
        f"[{'PASS' if stopping_rule_valid else 'FAIL'}] "
        "Stopping rule enforced"
    )

    # --------------------------------------------------------
    # Explanation coverage
    # --------------------------------------------------------

    explanation_valid = (
        result[
            "decision_reason"
        ]
        .astype(str)
        .str.len()
        .gt(0)
        .all()
    )

    print(
        f"[{'PASS' if explanation_valid else 'FAIL'}] "
        "Every decision has an explanation"
    )


# ============================================================
# REPORT
# ============================================================

def print_report(
    result: pd.DataFrame,
) -> None:

    print_section(
        "RECLAIM OPPORTUNITY REPORT"
    )

    total_failed = len(result)

    total_failed_value = (
        result["amount"]
        .sum()
    )

    total_expected_recovery = (
        result[
            "expected_recovery"
        ]
        .sum()
    )

    total_expected_net = (
        result[
            "expected_net_recovery"
        ]
        .sum()
    )

    intervention_count = (
        result[
            "recommended_action"
        ]
        .ne("NO_ACTION")
        .sum()
    )

    intervention_rate = (
        intervention_count
        /
        total_failed
        *
        100
    )

    print(
        f"Failed payments:       "
        f"{total_failed:,}"
    )

    print(
        f"Failed payment value:  "
        f"₹{total_failed_value:,.2f}"
    )

    print(
        f"Expected recovery:     "
        f"₹{total_expected_recovery:,.2f}"
    )

    print(
        f"Expected net recovery: "
        f"₹{total_expected_net:,.2f}"
    )

    print(
        f"Interventions:         "
        f"{intervention_count:,}"
    )

    print(
        f"Intervention rate:     "
        f"{intervention_rate:.2f}%"
    )

    # --------------------------------------------------------
    # Action distribution
    # --------------------------------------------------------

    print(
        "\nRecommended action distribution:"
    )

    action_distribution = (
        result[
            "recommended_action"
        ]
        .value_counts()
    )

    for action, count in (
        action_distribution.items()
    ):

        percentage = (
            count
            /
            total_failed
            *
            100
        )

        print(
            f"{action:<18}: "
            f"{count:>5,} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Priority distribution
    # --------------------------------------------------------

    print(
        "\nPriority distribution:"
    )

    priority_distribution = (
        result[
            "priority"
        ]
        .value_counts()
        .sort_index()
    )

    for priority, count in (
        priority_distribution.items()
    ):

        percentage = (
            count
            /
            total_failed
            *
            100
        )

        print(
            f"{str(priority):<10}: "
            f"{count:>5,} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Top opportunities
    # --------------------------------------------------------

    print(
        "\nTop 10 recovery opportunities:"
    )

    top = (
        result
        .sort_values(
            "expected_net_recovery",
            ascending=False,
        )
        .head(10)
    )

    display_columns = [
        "transaction_id",
        "customer_id",
        "amount",
        "failure_reason",
        "recoverability_score",
        "base_recovery_probability",
        "recommended_action",
        "expected_recovery",
        "expected_net_recovery",
        "priority",
    ]

    print(
        top[
            display_columns
        ]
        .to_string(
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
    print("RECLAIM RECOVERY INTELLIGENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Input check
    # --------------------------------------------------------

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Recovery context not found: "
            f"{INPUT_PATH}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    context = pd.read_csv(
        INPUT_PATH,
        parse_dates=["timestamp"],
    )

    print(
        f"Recovery context loaded: "
        f"{len(context):,} failed payments"
    )

    # --------------------------------------------------------
    # Intelligence
    # --------------------------------------------------------

    result = build_intelligence(
        context
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_intelligence(
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
        "\nRECOVERY INTELLIGENCE STATUS: READY"
    )


if __name__ == "__main__":
    main()