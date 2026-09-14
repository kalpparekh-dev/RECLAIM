# simulator/economic_optimizer.py

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "recovery_context.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "economic_opportunities.csv"
)


# ============================================================
# ECONOMIC PARAMETERS
# ============================================================

# Approximate cost of attempting each intervention.
#
# These are simulator economics, NOT Razorpay pricing.
# They exist to make the optimization measurable.
# ============================================================

ACTION_COST = {
    "RETRY_NOW": 1.50,
    "DELAYED_RETRY": 1.00,
    "NO_ACTION": 0.00,
}


# Customer-friction cost represents the estimated economic
# downside of repeatedly bothering a customer.
#
# Again: this is a synthetic modelling parameter.
# ============================================================

FRICTION_COST = {
    "RETRY_NOW": 0.25,
    "DELAYED_RETRY": 0.50,
    "NO_ACTION": 0.00,
}


# ============================================================
# SAFETY / DECISION PARAMETERS
# ============================================================

# Minimum expected net value required before intervention.

MIN_EXPECTED_NET_VALUE = 5.00


# Minimum confidence required for an intervention.

MIN_CONFIDENCE = 0.35


# Penalize uncertain predictions.
#
# confidence = 1.0 → no penalty
# confidence = 0.5 → moderate penalty
# confidence = 0.2 → strong penalty
# ============================================================

UNCERTAINTY_PENALTY = 0.50


# ============================================================
# DISPLAY
# ============================================================

def print_section(title: str) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(df: pd.DataFrame) -> None:

    required_columns = {
        "transaction_id",
        "customer_id",
        "amount",
        "failure_reason",
        "historical_transaction_count",
        "historical_success_rate",
        "transactions_last_7d",
        "failed_payments_last_7d",
        "failure_rate_last_7d",
        "avg_amount_last_7d",
        "failure_streak",
    }

    missing = (
        required_columns
        -
        set(df.columns)
    )

    if missing:

        print(
            "[FAIL] Missing required columns:"
        )

        for column in sorted(missing):

            print(
                f"  - {column}"
            )

        raise ValueError(
            "Recovery context validation failed."
        )

    print(
        "[PASS] Required economic features"
    )

    if df.empty:

        raise ValueError(
            "Recovery context is empty."
        )

    print(
        f"[PASS] Loaded {len(df):,} "
        f"failed payments"
    )


# ============================================================
# BASE RECOVERY PROBABILITY
# ============================================================

def estimate_base_probability(
    row: pd.Series,
) -> float:

    """
    Estimate probability of successful recovery.

    This is intentionally interpretable.

    We use observable information only:

    - historical success
    - recent failure rate
    - transaction history
    - failure streak
    - failure reason
    - amount relative to recent behaviour
    """

    historical_success = float(
        row["historical_success_rate"]
    )

    recent_failure_rate = float(
        row["failure_rate_last_7d"]
    )

    history_count = float(
        row["historical_transaction_count"]
    )

    failure_streak = float(
        row["failure_streak"]
    )

    amount = float(
        row["amount"]
    )

    avg_recent_amount = float(
        row["avg_amount_last_7d"]
    )

    reason = str(
        row["failure_reason"]
    )


    # --------------------------------------------------------
    # Start with historical reliability
    # --------------------------------------------------------

    probability = (
        0.55
        * historical_success
    )


    # --------------------------------------------------------
    # Recent behaviour
    # --------------------------------------------------------

    probability += (
        0.20
        * (1.0 - recent_failure_rate)
    )


    # --------------------------------------------------------
    # Historical evidence
    # --------------------------------------------------------

    history_strength = min(
        history_count / 30.0,
        1.0,
    )

    probability += (
        0.10
        * history_strength
        * historical_success
    )


    # --------------------------------------------------------
    # Failure streak penalty
    # --------------------------------------------------------

    streak_penalty = (
        max(
            failure_streak - 1,
            0,
        )
        * 0.06
    )

    probability -= streak_penalty


    # --------------------------------------------------------
    # Amount behaviour
    # --------------------------------------------------------

    if avg_recent_amount > 0:

        amount_ratio = (
            amount
            /
            avg_recent_amount
        )

        if amount_ratio > 3:

            probability -= 0.08

        elif amount_ratio > 2:

            probability -= 0.04

        elif amount_ratio < 0.50:

            probability += 0.02


    # --------------------------------------------------------
    # Failure reason priors
    # --------------------------------------------------------

    reason_adjustment = {

        "NETWORK_ERROR":
            0.10,

        "PAYMENT_TIMEOUT":
            0.07,

        "BANK_DECLINE":
            -0.03,

        "LIMIT_EXCEEDED":
            -0.05,

        "INSUFFICIENT_FUNDS":
            -0.08,
    }

    probability += (
        reason_adjustment.get(
            reason,
            0.0,
        )
    )


    # --------------------------------------------------------
    # Bound probability
    # --------------------------------------------------------

    return float(
        np.clip(
            probability,
            0.02,
            0.95,
        )
    )


# ============================================================
# CONFIDENCE
# ============================================================

def estimate_confidence(
    row: pd.Series,
) -> float:

    """
    Confidence reflects how much observable evidence
    we have about the customer.

    More history = higher confidence.

    Very recent instability = lower confidence.
    """

    history_count = float(
        row["historical_transaction_count"]
    )

    recent_transactions = float(
        row["transactions_last_7d"]
    )

    history_component = min(
        history_count / 40.0,
        1.0,
    )

    recent_component = min(
        recent_transactions / 15.0,
        1.0,
    )

    confidence = (
        0.65
        * history_component
        +
        0.35
        * recent_component
    )

    return float(
        np.clip(
            confidence,
            0.05,
            1.0,
        )
    )


# ============================================================
# ACTION PROBABILITY
# ============================================================

def action_probability(
    base_probability: float,
    action: str,
    row: pd.Series,
) -> float:

    """
    Estimate recovery probability for each intervention.

    Different failure modes respond differently to timing.
    """

    probability = base_probability

    reason = str(
        row["failure_reason"]
    )

    if action == "RETRY_NOW":

        if reason == "NETWORK_ERROR":

            probability += 0.08

        elif reason == "PAYMENT_TIMEOUT":

            probability += 0.05

        elif reason == "INSUFFICIENT_FUNDS":

            probability -= 0.06

        elif reason == "LIMIT_EXCEEDED":

            probability -= 0.04


    elif action == "DELAYED_RETRY":

        if reason == "INSUFFICIENT_FUNDS":

            probability += 0.08

        elif reason == "BANK_DECLINE":

            probability += 0.05

        elif reason == "NETWORK_ERROR":

            probability -= 0.02

        elif reason == "PAYMENT_TIMEOUT":

            probability += 0.02

        elif reason == "LIMIT_EXCEEDED":

            probability += 0.03


    return float(
        np.clip(
            probability,
            0.01,
            0.95,
        )
    )


# ============================================================
# ECONOMIC SCORE
# ============================================================

def calculate_action_value(
    amount: float,
    probability: float,
    confidence: float,
    action: str,
) -> dict:

    intervention_cost = (
        ACTION_COST[action]
    )

    friction_cost = (
        FRICTION_COST[action]
    )


    # --------------------------------------------------------
    # Expected gross recovery
    # --------------------------------------------------------

    expected_recovery = (
        amount
        *
        probability
    )


    # --------------------------------------------------------
    # Uncertainty penalty
    # --------------------------------------------------------

    uncertainty = (
        1.0
        -
        confidence
    )

    uncertainty_penalty = (
        expected_recovery
        *
        UNCERTAINTY_PENALTY
        *
        uncertainty
    )


    # --------------------------------------------------------
    # Risk-adjusted expected recovery
    # --------------------------------------------------------

    risk_adjusted_recovery = (
        expected_recovery
        -
        uncertainty_penalty
    )


    # --------------------------------------------------------
    # Expected net value
    # --------------------------------------------------------

    expected_net = (
        risk_adjusted_recovery
        -
        intervention_cost
        -
        friction_cost
    )


    return {
        "expected_recovery":
            expected_recovery,

        "uncertainty_penalty":
            uncertainty_penalty,

        "risk_adjusted_recovery":
            risk_adjusted_recovery,

        "intervention_cost":
            intervention_cost,

        "friction_cost":
            friction_cost,

        "expected_net_value":
            max(
                expected_net,
                0.0,
            ),
    }


# ============================================================
# OPTIMIZE ONE PAYMENT
# ============================================================

def optimize_payment(
    row: pd.Series,
) -> dict:

    amount = float(
        row["amount"]
    )

    base_probability = (
        estimate_base_probability(
            row
        )
    )

    confidence = (
        estimate_confidence(
            row
        )
    )


    candidate_actions = [
        "RETRY_NOW",
        "DELAYED_RETRY",
    ]


    candidates = []


    for action in candidate_actions:

        probability = (
            action_probability(
                base_probability,
                action,
                row,
            )
        )

        economics = (
            calculate_action_value(
                amount,
                probability,
                confidence,
                action,
            )
        )

        candidates.append(
            {
                "action":
                    action,

                "probability":
                    probability,

                **economics,
            }
        )


    # --------------------------------------------------------
    # Find best action
    # --------------------------------------------------------

    best = max(
        candidates,
        key=lambda x:
            x["expected_net_value"],
    )


    # --------------------------------------------------------
    # Apply abstention rule
    # --------------------------------------------------------

    if confidence < MIN_CONFIDENCE:

        selected_action = (
            "NO_ACTION"
        )

        reason = (
            "ABSTAIN_LOW_CONFIDENCE"
        )

    elif (
        best[
            "expected_net_value"
        ]
        <
        MIN_EXPECTED_NET_VALUE
    ):

        selected_action = (
            "NO_ACTION"
        )

        reason = (
            "ABSTAIN_LOW_EXPECTED_VALUE"
        )

    else:

        selected_action = (
            best["action"]
        )

        reason = (
            "INTERVENE_POSITIVE_EXPECTED_VALUE"
        )


    # --------------------------------------------------------
    # Return intelligence
    # --------------------------------------------------------

    selected = (
        best
        if selected_action
        != "NO_ACTION"
        else {
            "probability": 0.0,
            "expected_recovery": 0.0,
            "uncertainty_penalty": 0.0,
            "risk_adjusted_recovery": 0.0,
            "intervention_cost": 0.0,
            "friction_cost": 0.0,
            "expected_net_value": 0.0,
        }
    )


    return {
        "base_recovery_probability":
            base_probability,

        "model_confidence":
            confidence,

        "retry_now_probability":
            candidates[0]["probability"],

        "delayed_retry_probability":
            candidates[1]["probability"],

        "retry_now_expected_net":
            candidates[0]["expected_net_value"],

        "delayed_retry_expected_net":
            candidates[1]["expected_net_value"],

        "recommended_action":
            selected_action,

        "selected_recovery_probability":
            selected[
                "probability"
            ],

        "expected_recovery":
            selected[
                "expected_recovery"
            ],

        "uncertainty_penalty":
            selected[
                "uncertainty_penalty"
            ],

        "risk_adjusted_recovery":
            selected[
                "risk_adjusted_recovery"
            ],

        "intervention_cost":
            selected[
                "intervention_cost"
            ],

        "friction_cost":
            selected[
                "friction_cost"
            ],

        "expected_net_recovery":
            selected[
                "expected_net_value"
            ],

        "decision_reason":
            reason,
    }


# ============================================================
# BUILD OPTIMIZATION
# ============================================================

def build_optimizer(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_section(
        "CALCULATING ECONOMIC DECISIONS"
    )

    decisions = []

    total = len(
        df
    )

    for index, row in df.iterrows():

        decision = (
            optimize_payment(
                row
            )
        )

        decisions.append(
            decision
        )

        if (
            len(decisions) % 500
            == 0
        ):

            print(
                f"Processed "
                f"{len(decisions):,}"
                f"/{total:,}"
            )


    intelligence = pd.DataFrame(
        decisions,
        index=df.index,
    )


    output = pd.concat(
        [
            df.reset_index(
                drop=True
            ),
            intelligence.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    return output


# ============================================================
# VALIDATION
# ============================================================

def validate_output(
    df: pd.DataFrame,
) -> None:

    print_section(
        "ECONOMIC OPTIMIZER VALIDATION"
    )

    required = [
        "base_recovery_probability",
        "model_confidence",
        "retry_now_probability",
        "delayed_retry_probability",
        "retry_now_expected_net",
        "delayed_retry_expected_net",
        "recommended_action",
        "expected_recovery",
        "expected_net_recovery",
        "decision_reason",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing output columns: {missing}"
        )

    print(
        "[PASS] Intelligence output columns"
    )


    # --------------------------------------------------------
    # Probability validation
    # --------------------------------------------------------

    probability_columns = [
        "base_recovery_probability",
        "model_confidence",
        "retry_now_probability",
        "delayed_retry_probability",
    ]

    for column in probability_columns:

        values = df[
            column
        ]

        if (
            values.min() < 0
            or
            values.max() > 1
        ):

            raise ValueError(
                f"Invalid probability range: "
                f"{column}"
            )

    print(
        "[PASS] Probability ranges"
    )


    # --------------------------------------------------------
    # Economic validation
    # --------------------------------------------------------

    economic_columns = [
        "expected_recovery",
        "expected_net_recovery",
    ]

    for column in economic_columns:

        if (
            df[column] < 0
        ).any():

            raise ValueError(
                f"Negative economic value: "
                f"{column}"
            )

    print(
        "[PASS] Economic values are non-negative"
    )


    # --------------------------------------------------------
    # Action validation
    # --------------------------------------------------------

    valid_actions = {
        "RETRY_NOW",
        "DELAYED_RETRY",
        "NO_ACTION",
    }

    actual_actions = set(
        df[
            "recommended_action"
        ].unique()
    )

    invalid_actions = (
        actual_actions
        -
        valid_actions
    )

    if invalid_actions:

        raise ValueError(
            f"Invalid actions: "
            f"{invalid_actions}"
        )

    print(
        "[PASS] Actions are bounded"
    )


    # --------------------------------------------------------
    # Abstention validation
    # --------------------------------------------------------

    abstained = (
        df[
            "recommended_action"
        ]
        ==
        "NO_ACTION"
    )

    invalid_abstention = (
        abstained
        &
        (
            df[
                "expected_net_recovery"
            ]
            > 0
        )
    )

    if invalid_abstention.any():

        raise ValueError(
            "Abstention contains positive expected net value."
        )

    print(
        "[PASS] Abstention rule enforced"
    )


    # --------------------------------------------------------
    # Explanation validation
    # --------------------------------------------------------

    if (
        df[
            "decision_reason"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Missing decision explanations."
        )

    print(
        "[PASS] Every decision has an explanation"
    )


# ============================================================
# REPORT
# ============================================================

def print_report(
    df: pd.DataFrame,
) -> None:

    print_section(
        "RECLAIM ECONOMIC OPPORTUNITY REPORT"
    )

    total = len(
        df
    )

    failed_value = (
        df["amount"]
        .sum()
    )

    interventions = (
        df[
            "recommended_action"
        ]
        !=
        "NO_ACTION"
    ).sum()

    expected_recovery = (
        df[
            "expected_recovery"
        ]
        .sum()
    )

    expected_net = (
        df[
            "expected_net_recovery"
        ]
        .sum()
    )


    print(
        f"Failed payments:       "
        f"{total:,}"
    )

    print(
        f"Failed payment value:  "
        f"₹{failed_value:,.2f}"
    )

    print(
        f"Expected recovery:     "
        f"₹{expected_recovery:,.2f}"
    )

    print(
        f"Expected net recovery: "
        f"₹{expected_net:,.2f}"
    )

    print(
        f"Interventions:         "
        f"{interventions:,}"
    )

    print(
        f"Intervention rate:     "
        f"{interventions / total * 100:.2f}%"
    )


    print(
        "\nRecommended action distribution:"
    )

    print(
        df[
            "recommended_action"
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\nDecision reason distribution:"
    )

    print(
        df[
            "decision_reason"
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\nTop 10 economic opportunities:"
    )

    top = (
        df.sort_values(
            "expected_net_recovery",
            ascending=False,
        )
        .head(10)
    )


    columns = [
        "transaction_id",
        "customer_id",
        "amount",
        "failure_reason",
        "base_recovery_probability",
        "model_confidence",
        "recommended_action",
        "expected_recovery",
        "expected_net_recovery",
        "decision_reason",
    ]

    print(
        top[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

def save_output(
    df: pd.DataFrame,
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved to:"
        f"\n{OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("RECLAIM ECONOMIC OPTIMIZER")
    print("=" * 70)


    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Recovery context not found:\n"
            f"{INPUT_PATH}"
        )


    df = pd.read_csv(
        INPUT_PATH
    )


    print(
        f"Recovery context loaded: "
        f"{len(df):,} failed payments"
    )


    validate_input(
        df
    )


    output = build_optimizer(
        df
    )


    validate_output(
        output
    )


    print_report(
        output
    )


    save_output(
        output
    )


    print(
        "\nECONOMIC OPTIMIZER STATUS: READY"
    )


if __name__ == "__main__":
    main()