"""
RECLAIM VALUE-OF-INFORMATION / ECONOMIC ACTION SELECTION OPTIMIZER

Purpose
-------
Select recovery actions using transaction-level economic value rather
than probability alone.

Decision logic:

    Expected Recovery Value
            ↓
    Intervention Cost
            ↓
    Friction Cost
            ↓
    Economic Value
            ↓
    RETRY / ABSTAIN

The optimizer evaluates several policies and determines whether
economic selection can outperform broad retry strategies.

Inputs
------
1. recovery_model_predictions.csv
2. recovery_outcomes.csv

Outputs
-------
1. value_of_information_comparison.csv
2. value_of_information_policy.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\RECLAIM")

PREDICTION_FILE = (
    BASE_DIR / "data" / "generated" / "recovery_model_predictions.csv"
)

OUTCOME_FILE = (
    BASE_DIR / "data" / "generated" / "recovery_outcomes.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "generated"

COMPARISON_FILE = OUTPUT_DIR / "value_of_information_comparison.csv"
POLICY_FILE = OUTPUT_DIR / "value_of_information_policy.csv"


# ============================================================
# ECONOMIC PARAMETERS
# ============================================================

INTERVENTION_COST = 1.50
FRICTION_COST = 0.25

# Minimum expected economic advantage required before intervention.
ECONOMIC_MARGIN = 0.0


# ============================================================
# PRINT HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def pass_msg(message):
    print(f"[PASS] {message}")


def warn_msg(message):
    print(f"[WARN] {message}")


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_probability(series):
    return pd.to_numeric(series, errors="coerce").clip(0.0, 1.0)


def normalize_recovered_flag(df):
    if "recovery_flag" in df.columns:
        df["recovery_flag"] = (
            pd.to_numeric(df["recovery_flag"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        return df

    if "recovered" in df.columns:
        df["recovery_flag"] = (
            pd.to_numeric(df["recovered"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        return df

    if "recovery_status" in df.columns:
        status = (
            df["recovery_status"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        df["recovery_flag"] = status.isin(
            [
                "recovered",
                "success",
                "successful",
                "paid",
                "repaid",
                "true",
                "1",
            ]
        ).astype(int)

        return df

    raise ValueError(
        "Unable to derive recovery_flag. "
        "Expected recovery_flag, recovered, or recovery_status."
    )


def normalize_recovered_amount(df):
    """
    Always produce one canonical recovered_amount column.
    """

    if "recovered_amount" in df.columns:
        df["recovered_amount"] = pd.to_numeric(
            df["recovered_amount"],
            errors="coerce"
        )

    elif "actual_recovered_amount" in df.columns:
        df["recovered_amount"] = pd.to_numeric(
            df["actual_recovered_amount"],
            errors="coerce"
        )

    elif "recovered_value" in df.columns:
        df["recovered_amount"] = pd.to_numeric(
            df["recovered_value"],
            errors="coerce"
        )

    elif "amount_recovered" in df.columns:
        df["recovered_amount"] = pd.to_numeric(
            df["amount_recovered"],
            errors="coerce"
        )

    elif "recovery_flag" in df.columns and "amount" in df.columns:
        df["recovered_amount"] = np.where(
            df["recovery_flag"] == 1,
            pd.to_numeric(df["amount"], errors="coerce"),
            0.0,
        )

    else:
        raise ValueError(
            "Unable to derive recovered_amount."
        )

    df["recovered_amount"] = (
        pd.to_numeric(df["recovered_amount"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    header("RECLAIM VALUE-OF-INFORMATION OPTIMIZER")

    predictions = pd.read_csv(PREDICTION_FILE)
    outcomes = pd.read_csv(OUTCOME_FILE)

    print(f"Prediction rows: {len(predictions):,}")
    print(f"Outcome rows:    {len(outcomes):,}")

    # --------------------------------------------------------
    # Prediction validation
    # --------------------------------------------------------

    required_prediction = [
        "transaction_id",
    ]

    for column in required_prediction:
        if column not in predictions.columns:
            raise ValueError(
                f"Prediction dataset missing column: {column}"
            )

    if "recovery_probability" in predictions.columns:
        probability_column = "recovery_probability"

    elif "base_recovery_probability" in predictions.columns:
        probability_column = "base_recovery_probability"

    elif "action_recovery_probability" in predictions.columns:
        probability_column = "action_recovery_probability"

    else:
        raise ValueError(
            "No recovery probability column found."
        )

    predictions["recovery_probability"] = normalize_probability(
        predictions[probability_column]
    )

    pass_msg(
        f"Using probability column: `{probability_column}`"
    )

    if predictions["transaction_id"].duplicated().any():
        raise ValueError(
            "Prediction transaction IDs are not unique."
        )

    pass_msg("Prediction dataset structure")
    pass_msg("Prediction transaction IDs are unique")
    pass_msg("Recovery probabilities normalized")

    # --------------------------------------------------------
    # Outcome normalization
    # --------------------------------------------------------

    print()
    print("Normalizing recovery outcome schema...")

    print("Outcome columns detected:")
    print(list(outcomes.columns))

    outcomes = normalize_recovered_flag(outcomes)
    outcomes = normalize_recovered_amount(outcomes)

    pass_msg("Recovery outcome schema normalized")

    if outcomes["transaction_id"].duplicated().any():
        raise ValueError(
            "Outcome transaction IDs are not unique."
        )

    pass_msg("Outcome transaction IDs are unique")

    # --------------------------------------------------------
    # Population alignment
    # --------------------------------------------------------

    prediction_ids = set(
        predictions["transaction_id"].astype(str)
    )

    outcome_ids = set(
        outcomes["transaction_id"].astype(str)
    )

    missing_in_outcomes = prediction_ids - outcome_ids
    extra_in_outcomes = outcome_ids - prediction_ids

    print()
    print("=" * 70)
    print("ALIGNING PREDICTION AND OUTCOME POPULATIONS")
    print("=" * 70)

    print(
        f"Prediction population: {len(prediction_ids):,}"
    )

    print(
        f"Outcome population:    {len(outcome_ids):,}"
    )

    print(
        f"Missing in outcomes:   {len(missing_in_outcomes):,}"
    )

    print(
        f"Extra in outcomes:     {len(extra_in_outcomes):,}"
    )

    if missing_in_outcomes:
        raise ValueError(
            "Some prediction transactions are missing from outcomes."
        )

    outcomes = outcomes[
        outcomes["transaction_id"]
        .astype(str)
        .isin(prediction_ids)
    ].copy()

    pass_msg(
        f"Filtered outcomes to prediction population: "
        f"{len(outcomes):,}"
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    pred = predictions.copy()
    out = outcomes.copy()

    pred["transaction_id"] = pred[
        "transaction_id"
    ].astype(str)

    out["transaction_id"] = out[
        "transaction_id"
    ].astype(str)

    # Only bring outcome fields that are needed.
    outcome_columns = [
        "transaction_id",
        "amount",
        "recovery_flag",
        "recovered_amount",
    ]

    outcome_columns = [
        c for c in outcome_columns
        if c in out.columns
    ]

    df = pred.merge(
        out[outcome_columns],
        on="transaction_id",
        how="inner",
        suffixes=("", "_outcome"),
    )

    pass_msg(
        f"Merged dataset: {len(df):,} transactions"
    )

    # --------------------------------------------------------
    # Canonical amount
    # --------------------------------------------------------

    if "amount" not in df.columns:

        if "amount_outcome" in df.columns:
            df["amount"] = df["amount_outcome"]

        else:
            raise ValueError(
                "No canonical amount column available."
            )

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    if df["amount"].isna().any():
        raise ValueError(
            "Invalid transaction amounts detected."
        )

    if (df["amount"] <= 0).any():
        raise ValueError(
            "Transaction amounts must be positive."
        )

    pass_msg("Canonical amount column: `amount`")

    # --------------------------------------------------------
    # Canonical recovered amount
    # --------------------------------------------------------

    if "recovered_amount" not in df.columns:

        if "recovered_amount_outcome" in df.columns:
            df["recovered_amount"] = (
                df["recovered_amount_outcome"]
            )

        elif "recovery_flag" in df.columns:
            df["recovered_amount"] = np.where(
                df["recovery_flag"] == 1,
                df["amount"],
                0.0,
            )

        else:
            raise ValueError(
                "Unable to construct recovered_amount."
            )

    df["recovered_amount"] = (
        pd.to_numeric(
            df["recovered_amount"],
            errors="coerce"
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    pass_msg(
        "Canonical recovered amount column: "
        "`recovered_amount`"
    )

    # --------------------------------------------------------
    # Recovery flag
    # --------------------------------------------------------

    if "recovery_flag" not in df.columns:

        if "recovery_flag_outcome" in df.columns:
            df["recovery_flag"] = (
                df["recovery_flag_outcome"]
            )

        elif "recovered" in df.columns:
            df["recovery_flag"] = (
                pd.to_numeric(
                    df["recovered"],
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )

        else:
            df["recovery_flag"] = (
                df["recovered_amount"] > 0
            ).astype(int)

    df["recovery_flag"] = (
        pd.to_numeric(
            df["recovery_flag"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    pass_msg(
        "Canonical recovery flag column: "
        "`recovery_flag`"
    )

    if len(df) == 0:
        raise ValueError(
            "Merged dataset is empty."
        )

    pass_msg(
        f"Final modeling population: {len(df):,}"
    )

    return df


# ============================================================
# ECONOMIC VALUE
# ============================================================

def calculate_economic_value(df):

    header("CALCULATING VALUE-OF-INFORMATION")

    df = df.copy()

    p = df["recovery_probability"]
    amount = df["amount"]

    # Expected value if we intervene.
    df["expected_recovery"] = (
        p * amount
    )

    # Direct intervention costs.
    df["intervention_cost"] = (
        INTERVENTION_COST
    )

    df["friction_cost"] = (
        FRICTION_COST
    )

    df["expected_action_cost"] = (
        df["intervention_cost"]
        + df["friction_cost"]
    )

    # Expected economic value of acting.
    df["expected_net_recovery"] = (
        df["expected_recovery"]
        - df["expected_action_cost"]
    )

    # Value of information:
    #
    # If we do nothing:
    #       economic value = 0
    #
    # If we intervene:
    #       expected_net_recovery
    #
    # Therefore:
    #
    # incremental_value = max(expected_net_recovery, 0)

    df["incremental_economic_value"] = (
        df["expected_net_recovery"]
        - ECONOMIC_MARGIN
    )

    df["should_intervene"] = (
        df["incremental_economic_value"] > 0
    )

    pass_msg("Expected recovery calculated")
    pass_msg("Expected intervention cost calculated")
    pass_msg("Expected net recovery calculated")
    pass_msg("Incremental economic value calculated")

    return df


# ============================================================
# POLICY EVALUATION
# ============================================================

def evaluate_policy(
    df,
    mask,
    policy_name,
):

    selected = df.loc[mask].copy()

    interventions = len(selected)

    if interventions == 0:

        return {
            "policy": policy_name,
            "interventions": 0,
            "intervention_rate": 0.0,
            "recovered_payments": 0,
            "recovered_value": 0.0,
            "recovery_rate_value": 0.0,
            "intervention_cost": 0.0,
            "friction_cost": 0.0,
            "net_recovered_value": 0.0,
            "net_per_intervention": 0.0,
            "expected_value": 0.0,
        }

    recovered = selected[
        selected["recovery_flag"] == 1
    ]

    recovered_payments = len(recovered)

    recovered_value = (
        recovered["recovered_amount"].sum()
    )

    total_amount = df["amount"].sum()

    recovery_rate_value = (
        recovered_value / total_amount
        if total_amount > 0
        else 0.0
    )

    intervention_cost = (
        interventions * INTERVENTION_COST
    )

    friction_cost = (
        interventions * FRICTION_COST
    )

    net_recovered_value = (
        recovered_value
        - intervention_cost
        - friction_cost
    )

    expected_value = (
        selected["expected_net_recovery"].sum()
    )

    return {
        "policy": policy_name,
        "interventions": interventions,
        "intervention_rate": (
            interventions / len(df)
        ),
        "recovered_payments": recovered_payments,
        "recovered_value": recovered_value,
        "recovery_rate_value": recovery_rate_value,
        "intervention_cost": intervention_cost,
        "friction_cost": friction_cost,
        "net_recovered_value": net_recovered_value,
        "net_per_intervention": (
            net_recovered_value / interventions
        ),
        "expected_value": expected_value,
    }


# ============================================================
# POLICY TESTING
# ============================================================

def test_policies(df):

    header("TESTING ECONOMIC ACTION POLICIES")

    policies = []

    n = len(df)

    # --------------------------------------------------------
    # Retry all
    # --------------------------------------------------------

    policies.append(
        evaluate_policy(
            df,
            pd.Series(True, index=df.index),
            "RETRY_ALL",
        )
    )

    # --------------------------------------------------------
    # Top-k policies by expected net value
    # --------------------------------------------------------

    ranked = df.sort_values(
        "expected_net_recovery",
        ascending=False,
    )

    for percent in [
        5,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
    ]:

        count = max(
            1,
            int(np.ceil(n * percent / 100))
        )

        selected_ids = set(
            ranked.head(count)[
                "transaction_id"
            ]
        )

        mask = df["transaction_id"].isin(
            selected_ids
        )

        policies.append(
            evaluate_policy(
                df,
                mask,
                f"TOP_{percent}",
            )
        )

    # --------------------------------------------------------
    # Probability thresholds
    # --------------------------------------------------------

    for threshold in [
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]:

        mask = (
            df["recovery_probability"]
            >= threshold
        )

        policies.append(
            evaluate_policy(
                df,
                mask,
                f"PROBABILITY_{threshold:.2f}",
            )
        )

    # --------------------------------------------------------
    # Pure economic value policy
    # --------------------------------------------------------

    mask = (
        df["incremental_economic_value"] > 0
    )

    policies.append(
        evaluate_policy(
            df,
            mask,
            "POSITIVE_ECONOMIC_VALUE",
        )
    )

    # --------------------------------------------------------
    # High-value economic policy
    # --------------------------------------------------------

    # Require both:
    # 1. positive economic value
    # 2. recovery probability >= observed rate

    observed_rate = (
        df["recovery_flag"].mean()
    )

    mask = (
        (df["incremental_economic_value"] > 0)
        &
        (
            df["recovery_probability"]
            >= observed_rate
        )
    )

    policies.append(
        evaluate_policy(
            df,
            mask,
            "HIGH_CONFIDENCE_ECONOMIC_VALUE",
        )
    )

    return pd.DataFrame(policies)


# ============================================================
# BUILD FINAL POLICY
# ============================================================

def build_final_policy(df):

    header("BUILDING VALUE-OF-INFORMATION POLICY")

    result = df.copy()

    # Main decision.
    result["action"] = np.where(
        result["incremental_economic_value"] > 0,
        "RETRY",
        "ABSTAIN",
    )

    # Confidence tiers.
    result["economic_priority"] = np.select(
        [
            result["incremental_economic_value"]
            >= 1000,

            result["incremental_economic_value"]
            >= 500,

            result["incremental_economic_value"]
            > 0,
        ],
        [
            "HIGH",
            "MEDIUM",
            "LOW",
        ],
        default="ABSTAIN",
    )

    # Economic rank.
    result["economic_rank"] = (
        result["incremental_economic_value"]
        .rank(
            ascending=False,
            method="first",
        )
        .astype(int)
    )

    interventions = (
        result["action"] == "RETRY"
    ).sum()

    print(
        f"RECLAIM interventions: {interventions:,}"
    )

    print(
        f"RECLAIM intervention rate: "
        f"{interventions / len(result):.2%}"
    )

    return result


# ============================================================
# REPORT
# ============================================================

def print_comparison(results):

    header("POLICY COMPARISON")

    display = results.copy()

    display["intervention_rate"] = (
        display["intervention_rate"]
        .map(lambda x: f"{x:.2%}")
    )

    for column in [
        "recovered_value",
        "intervention_cost",
        "friction_cost",
        "net_recovered_value",
        "expected_value",
        "net_per_intervention",
    ]:
        display[column] = display[column].map(
            lambda x: f"₹{x:,.2f}"
        )

    print(
        display.to_string(index=False)
    )


# ============================================================
# LIFT ANALYSIS
# ============================================================

def calculate_lift(
    results,
    final_policy,
):

    header("VALUE-OF-INFORMATION LIFT")

    retry = results[
        results["policy"] == "RETRY_ALL"
    ].iloc[0]

    reclaim = evaluate_policy(
        final_policy,
        final_policy["action"] == "RETRY",
        "VALUE_OF_INFORMATION_RECLAIM",
    )

    net_difference = (
        reclaim["net_recovered_value"]
        - retry["net_recovered_value"]
    )

    interventions_avoided = (
        retry["interventions"]
        - reclaim["interventions"]
    )

    intervention_reduction = (
        interventions_avoided
        / retry["interventions"]
        if retry["interventions"] > 0
        else 0
    )

    print(
        "VALUE-OF-INFORMATION RECLAIM vs RETRY ALL"
    )

    print("-" * 55)

    print(
        f"Net recovery difference: "
        f"₹{net_difference:,.2f}"
    )

    print(
        f"Interventions avoided: "
        f"{interventions_avoided:,}"
    )

    print(
        f"Intervention reduction: "
        f"{intervention_reduction:.2%}"
    )

    if net_difference > 0:

        print(
            "[PASS] VALUE-OF-INFORMATION RECLAIM "
            "beats RETRY_ALL on net recovered value."
        )

    else:

        warn_msg(
            "VALUE-OF-INFORMATION RECLAIM "
            "does NOT yet beat RETRY_ALL."
        )

    return reclaim


# ============================================================
# VALIDATION
# ============================================================

def validate_policy(
    df,
    final_policy,
):

    header("VALUE-OF-INFORMATION POLICY VALIDATION")

    required = [
        "transaction_id",
        "recovery_probability",
        "expected_recovery",
        "expected_net_recovery",
        "incremental_economic_value",
        "action",
        "economic_priority",
        "economic_rank",
    ]

    missing = [
        column
        for column in required
        if column not in final_policy.columns
    ]

    if missing:
        raise ValueError(
            f"Missing final policy columns: {missing}"
        )

    pass_msg("Required policy columns")

    if not (
        final_policy["recovery_probability"]
        .between(0, 1)
        .all()
    ):
        raise ValueError(
            "Invalid recovery probabilities."
        )

    pass_msg("Probability range")

    if final_policy[
        "expected_recovery"
    ].isna().any():

        raise ValueError(
            "Invalid expected recovery values."
        )

    pass_msg("Expected recovery values")

    if final_policy[
        "expected_net_recovery"
    ].isna().any():

        raise ValueError(
            "Invalid expected net recovery values."
        )

    pass_msg("Expected net recovery values")

    valid_actions = {
        "RETRY",
        "ABSTAIN",
    }

    if not set(
        final_policy["action"]
    ).issubset(valid_actions):

        raise ValueError(
            "Invalid policy actions."
        )

    pass_msg("Actions valid")

    if final_policy[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs in policy."
        )

    pass_msg("Transaction IDs unique")


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = calculate_economic_value(df)

    results = test_policies(df)

    print(
        f"\nPolicies evaluated: "
        f"{len(results)}"
    )

    final_policy = build_final_policy(df)

    print_comparison(results)

    # --------------------------------------------------------
    # Best measured policy
    # --------------------------------------------------------

    best = results.loc[
        results["net_recovered_value"].idxmax()
    ]

    header("OPTIMAL ECONOMIC POLICY")

    print(
        f"Best policy:             "
        f"{best['policy']}"
    )

    print(
        f"Interventions:           "
        f"{int(best['interventions']):,}"
    )

    print(
        f"Intervention rate:       "
        f"{best['intervention_rate']:.2%}"
    )

    print(
        f"Recovered value:         "
        f"₹{best['recovered_value']:,.2f}"
    )

    print(
        f"Net recovered value:     "
        f"₹{best['net_recovered_value']:,.2f}"
    )

    print(
        f"Net ₹ / intervention:    "
        f"₹{best['net_per_intervention']:,.2f}"
    )

    # --------------------------------------------------------
    # Lift
    # --------------------------------------------------------

    reclaim = calculate_lift(
        results,
        final_policy,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_policy(
        df,
        final_policy,
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        COMPARISON_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save final policy
    # --------------------------------------------------------

    save_columns = [
        "transaction_id",
        "amount",
        "recovery_probability",
        "recovery_flag",
        "recovered_amount",
        "expected_recovery",
        "intervention_cost",
        "friction_cost",
        "expected_action_cost",
        "expected_net_recovery",
        "incremental_economic_value",
        "should_intervene",
        "action",
        "economic_priority",
        "economic_rank",
    ]

    save_columns = [
        c
        for c in save_columns
        if c in final_policy.columns
    ]

    final_policy[
        save_columns
    ].sort_values(
        "economic_rank"
    ).to_csv(
        POLICY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    header(
        "VALUE-OF-INFORMATION OPTIMIZATION COMPLETE"
    )

    print(
        "Best measured policy: "
        f"{best['policy']}"
    )

    print(
        "Best net recovered value: "
        f"₹{best['net_recovered_value']:,.2f}"
    )

    print()
    print(
        "Saved comparison to:"
    )

    print(COMPARISON_FILE)

    print()
    print(
        "Saved final policy to:"
    )

    print(POLICY_FILE)

    print()
    print(
        "VALUE-OF-INFORMATION STATUS: READY"
    )


if __name__ == "__main__":
    main()