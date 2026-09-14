"""
RECLAIM COST-CONSTRAINED POLICY OPTIMIZER
=========================================

Purpose:
    Find a recovery intervention policy under a constrained intervention
    budget while maximizing expected / measured recovered value.

Inputs:
    data/generated/recovery_model_predictions.csv
    data/generated/recovery_outcomes.csv

Outputs:
    data/generated/cost_constrained_policy_comparison.csv
    data/generated/cost_constrained_final_policy.csv
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREDICTION_PATH = os.path.join(
    BASE_DIR,
    "data",
    "generated",
    "recovery_model_predictions.csv",
)

OUTCOME_PATH = os.path.join(
    BASE_DIR,
    "data",
    "generated",
    "recovery_outcomes.csv",
)

COMPARISON_OUTPUT = os.path.join(
    BASE_DIR,
    "data",
    "generated",
    "cost_constrained_policy_comparison.csv",
)

FINAL_POLICY_OUTPUT = os.path.join(
    BASE_DIR,
    "data",
    "generated",
    "cost_constrained_final_policy.csv",
)


# Intervention economics
INTERVENTION_COST = 1.50
FRICTION_COST = 0.25

TOTAL_BUDGET = None

# If TOTAL_BUDGET is None, test percentages of the population.
BUDGET_PERCENTAGES = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

# Minimum calibrated/model probability for intervention.
MIN_PROBABILITY = 0.0

RANDOM_SEED = 42


# ============================================================
# DISPLAY HELPERS
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def money(value: float) -> str:
    return f"₹{value:,.2f}"


# ============================================================
# COLUMN HELPERS
# ============================================================

def first_existing(
    df: pd.DataFrame,
    candidates: List[str],
) -> str | None:

    for column in candidates:
        if column in df.columns:
            return column

    return None


def ensure_numeric(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> None:

    if column not in df.columns:
        df[column] = default

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(default)


# ============================================================
# OUTCOME NORMALIZATION
# ============================================================

def normalize_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:

    print()
    print("Normalizing recovery outcome schema...")

    print("Outcome columns detected:")
    print(list(outcomes.columns))

    df = outcomes.copy()

    # --------------------------------------------------------
    # recovery_flag
    # --------------------------------------------------------

    if "recovery_flag" not in df.columns:

        if "recovered" in df.columns:

            df["recovery_flag"] = (
                pd.to_numeric(
                    df["recovered"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                .clip(0, 1)
            )

            print("[PASS] Derived `recovery_flag` from `recovered`")

        elif "recovery_status" in df.columns:

            df["recovery_flag"] = (
                df["recovery_status"]
                .astype(str)
                .str.upper()
                .eq("RECOVERED")
                .astype(int)
            )

            print(
                "[PASS] Derived `recovery_flag` "
                "from `recovery_status`"
            )

        elif "actual_recovery_probability" in df.columns:

            df["recovery_flag"] = (
                pd.to_numeric(
                    df["actual_recovery_probability"],
                    errors="coerce",
                )
                .fillna(0)
                > 0
            ).astype(int)

            print(
                "[PASS] Derived `recovery_flag` "
                "from `actual_recovery_probability`"
            )

        else:
            raise ValueError(
                "Could not derive recovery_flag from outcome dataset."
            )

    # --------------------------------------------------------
    # recovered_amount
    # --------------------------------------------------------

    if "recovered_amount" not in df.columns:

        if "recovered_value" in df.columns:

            df["recovered_amount"] = pd.to_numeric(
                df["recovered_value"],
                errors="coerce",
            ).fillna(0)

            print(
                "[PASS] Derived `recovered_amount` "
                "from `recovered_value`"
            )

        elif "recovery_flag" in df.columns and "amount" in df.columns:

            df["recovered_amount"] = np.where(
                df["recovery_flag"].astype(int) == 1,
                pd.to_numeric(
                    df["amount"],
                    errors="coerce",
                ).fillna(0),
                0.0,
            )

            print(
                "[PASS] Rebuilt `recovered_amount` "
                "from recovery_flag + amount"
            )

        else:
            raise ValueError(
                "Could not construct recovered_amount."
            )

    # --------------------------------------------------------
    # amount
    # --------------------------------------------------------

    if "amount" not in df.columns:

        source = first_existing(
            df,
            [
                "transaction_amount",
                "payment_amount",
                "failed_amount",
            ],
        )

        if source is None:
            raise ValueError(
                "Outcome dataset has no recognizable amount column."
            )

        df["amount"] = pd.to_numeric(
            df[source],
            errors="coerce",
        ).fillna(0)

        print(
            f"[PASS] Derived canonical `amount` from `{source}`"
        )

    # --------------------------------------------------------
    # Normalize numeric columns
    # --------------------------------------------------------

    ensure_numeric(df, "amount")
    ensure_numeric(df, "recovered_amount")
    ensure_numeric(df, "recovery_flag")

    df["recovery_flag"] = (
        df["recovery_flag"]
        .astype(int)
        .clip(0, 1)
    )

    # Safety:
    # recovered payments cannot recover more than transaction amount.
    df["recovered_amount"] = (
        df["recovered_amount"]
        .clip(lower=0)
    )

    # If recovered flag is zero, recovered amount must be zero.
    df.loc[
        df["recovery_flag"] == 0,
        "recovered_amount",
    ] = 0.0

    print("[PASS] Recovery outcome schema normalized")

    return df


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

    if not os.path.exists(PREDICTION_PATH):
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTION_PATH}"
        )

    if not os.path.exists(OUTCOME_PATH):
        raise FileNotFoundError(
            f"Outcome file not found:\n{OUTCOME_PATH}"
        )

    predictions = pd.read_csv(PREDICTION_PATH)
    outcomes_raw = pd.read_csv(OUTCOME_PATH)

    outcomes = normalize_outcomes(outcomes_raw)

    header("RECLAIM COST-CONSTRAINED POLICY OPTIMIZER")

    print(f"Prediction rows: {len(predictions):,}")
    print(f"Outcome rows:    {len(outcomes):,}")

    # --------------------------------------------------------
    # Validate prediction transaction IDs
    # --------------------------------------------------------

    required_prediction = [
        "transaction_id",
    ]

    for column in required_prediction:
        if column not in predictions.columns:
            raise ValueError(
                f"Missing prediction column: {column}"
            )

    if predictions["transaction_id"].duplicated().any():
        raise ValueError(
            "Prediction transaction IDs are not unique."
        )

    print("[PASS] Prediction transaction IDs are unique")

    # --------------------------------------------------------
    # Validate outcome transaction IDs
    # --------------------------------------------------------

    if "transaction_id" not in outcomes.columns:
        raise ValueError(
            "Outcome dataset missing transaction_id."
        )

    if outcomes["transaction_id"].duplicated().any():
        raise ValueError(
            "Outcome transaction IDs are not unique."
        )

    print("[PASS] Outcome transaction IDs are unique")

    # --------------------------------------------------------
    # Probability column
    # --------------------------------------------------------

    probability_column = first_existing(
        predictions,
        [
            "recovery_probability",
            "calibrated_recovery_probability",
            "base_recovery_probability",
            "action_recovery_probability",
        ],
    )

    if probability_column is None:
        raise ValueError(
            "No recovery probability column found in predictions."
        )

    print(
        f"[PASS] Using probability column: `{probability_column}`"
    )

    predictions["recovery_probability"] = pd.to_numeric(
        predictions[probability_column],
        errors="coerce",
    ).fillna(0).clip(0, 1)

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    prediction_ids = set(
        predictions["transaction_id"].astype(str)
    )

    outcome_ids = set(
        outcomes["transaction_id"].astype(str)
    )

    if prediction_ids != outcome_ids.intersection(prediction_ids):
        print(
            "[WARN] Prediction and outcome populations "
            "are not identical."
        )

    merged = predictions.merge(
        outcomes,
        on="transaction_id",
        how="left",
        suffixes=("_prediction", "_outcome"),
    )

    print(
        f"[PASS] Merged dataset: {len(merged):,} transactions"
    )

    # ========================================================
    # CRITICAL FIX
    # ========================================================
    #
    # Pandas can create:
    #
    # amount_prediction / amount_outcome
    # recovered_amount_prediction / recovered_amount_outcome
    # recovery_flag_prediction / recovery_flag_outcome
    #
    # after a merge.
    #
    # Therefore we ALWAYS rebuild canonical columns here.
    # ========================================================

    # --------------------------------------------------------
    # Canonical amount
    # --------------------------------------------------------

    if "amount" not in merged.columns:

        amount_source = first_existing(
            merged,
            [
                "amount_prediction",
                "amount_outcome",
                "transaction_amount",
                "payment_amount",
            ],
        )

        if amount_source is None:
            raise ValueError(
                "Could not construct canonical amount column."
            )

        merged["amount"] = pd.to_numeric(
            merged[amount_source],
            errors="coerce",
        )

    else:
        merged["amount"] = pd.to_numeric(
            merged["amount"],
            errors="coerce",
        )

    merged["amount"] = merged["amount"].fillna(0)

    print("[PASS] Canonical amount column: `amount`")

    # --------------------------------------------------------
    # Canonical recovery flag
    # --------------------------------------------------------

    if "recovery_flag" not in merged.columns:

        flag_source = first_existing(
            merged,
            [
                "recovery_flag_outcome",
                "recovery_flag_prediction",
                "recovered_outcome",
                "recovered_prediction",
            ],
        )

        if flag_source is not None:

            merged["recovery_flag"] = (
                pd.to_numeric(
                    merged[flag_source],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                .clip(0, 1)
            )

        elif "recovery_status" in merged.columns:

            merged["recovery_flag"] = (
                merged["recovery_status"]
                .astype(str)
                .str.upper()
                .eq("RECOVERED")
                .astype(int)
            )

        else:
            raise ValueError(
                "Could not construct canonical recovery_flag."
            )

    else:

        merged["recovery_flag"] = (
            pd.to_numeric(
                merged["recovery_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .clip(0, 1)
        )

    print("[PASS] Canonical recovery flag: `recovery_flag`")

    # --------------------------------------------------------
    # Canonical recovered amount
    # --------------------------------------------------------

    recovered_source = first_existing(
        merged,
        [
            "recovered_amount",
            "recovered_amount_outcome",
            "recovered_amount_prediction",
            "recovered_value",
        ],
    )

    if recovered_source is not None:

        merged["recovered_amount"] = pd.to_numeric(
            merged[recovered_source],
            errors="coerce",
        ).fillna(0)

    else:

        merged["recovered_amount"] = np.where(
            merged["recovery_flag"] == 1,
            merged["amount"],
            0.0,
        )

    merged["recovered_amount"] = (
        merged["recovered_amount"]
        .clip(lower=0)
    )

    merged.loc[
        merged["recovery_flag"] == 0,
        "recovered_amount",
    ] = 0.0

    print(
        "[PASS] Canonical recovered amount: "
        "`recovered_amount`"
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if merged["amount"].isna().any():
        raise ValueError(
            "Canonical amount contains missing values."
        )

    if (merged["amount"] <= 0).any():
        raise ValueError(
            "Transaction amounts must be positive."
        )

    if not merged["recovery_flag"].isin([0, 1]).all():
        raise ValueError(
            "Recovery flag contains invalid values."
        )

    if (merged["recovered_amount"] < 0).any():
        raise ValueError(
            "Recovered amount cannot be negative."
        )

    print("[PASS] Transaction amounts valid")
    print(
        f"[PASS] Observed recovery rate: "
        f"{merged['recovery_flag'].mean():.2%}"
    )

    return merged


# ============================================================
# ECONOMIC CALCULATIONS
# ============================================================

def calculate_expected_values(
    df: pd.DataFrame,
) -> pd.DataFrame:

    header("CALCULATING ECONOMIC VALUE")

    df = df.copy()

    # Expected recovery based on model probability.
    df["expected_recovery"] = (
        df["amount"]
        * df["recovery_probability"]
    )

    df["intervention_cost_per_action"] = (
        INTERVENTION_COST
    )

    df["friction_cost_per_action"] = (
        FRICTION_COST
    )

    df["expected_net_recovery_if_action"] = (
        df["expected_recovery"]
        - INTERVENTION_COST
        - FRICTION_COST
    )

    print("[PASS] Expected recovery calculated")
    print("[PASS] Expected net recovery calculated")

    return df


# ============================================================
# POLICY MASKS
# ============================================================

def select_top_k(
    df: pd.DataFrame,
    fraction: float,
) -> pd.Series:

    count = int(
        np.floor(
            len(df) * fraction
        )
    )

    count = max(
        0,
        min(count, len(df)),
    )

    mask = pd.Series(
        False,
        index=df.index,
    )

    if count > 0:

        ranked = (
            df.sort_values(
                [
                    "expected_net_recovery_if_action",
                    "recovery_probability",
                ],
                ascending=False,
            )
            .head(count)
        )

        mask.loc[ranked.index] = True

    return mask


def select_probability(
    df: pd.DataFrame,
    threshold: float,
) -> pd.Series:

    return (
        df["recovery_probability"]
        >= threshold
    )


def select_expected_value(
    df: pd.DataFrame,
) -> pd.Series:

    return (
        df["expected_net_recovery_if_action"]
        > 0
    )


# ============================================================
# POLICY EVALUATION
# ============================================================

def evaluate_policy(
    df: pd.DataFrame,
    policy_name: str,
    intervention_mask: pd.Series,
) -> Dict:

    mask = intervention_mask.astype(bool)

    interventions = int(mask.sum())

    intervention_rate = (
        interventions / len(df)
        if len(df) > 0
        else 0
    )

    selected = df.loc[mask].copy()

    recovered_payments = int(
        selected["recovery_flag"].sum()
    )

    recovered_value = float(
        selected["recovered_amount"].sum()
    )

    attempted_value = float(
        selected["amount"].sum()
    )

    total_failed_value = float(
        df["amount"].sum()
    )

    recovery_rate_value = (
        recovered_value / total_failed_value
        if total_failed_value > 0
        else 0
    )

    recovery_rate_count = (
        recovered_payments / interventions
        if interventions > 0
        else 0
    )

    intervention_cost = (
        interventions
        * INTERVENTION_COST
    )

    friction_cost = (
        interventions
        * FRICTION_COST
    )

    net_recovered_value = (
        recovered_value
        - intervention_cost
        - friction_cost
    )

    recovery_per_intervention = (
        recovered_value / interventions
        if interventions > 0
        else 0
    )

    net_per_intervention = (
        net_recovered_value / interventions
        if interventions > 0
        else 0
    )

    expected_value = float(
        selected["expected_recovery"].sum()
    )

    expected_net_value = float(
        selected[
            "expected_net_recovery_if_action"
        ].sum()
    )

    return {
        "policy": policy_name,
        "transactions": len(df),
        "interventions": interventions,
        "intervention_rate": intervention_rate,
        "recovered_payments": recovered_payments,
        "recovered_value": recovered_value,
        "attempted_value": attempted_value,
        "recovery_rate_value": recovery_rate_value,
        "recovery_rate_count": recovery_rate_count,
        "intervention_cost": intervention_cost,
        "friction_cost": friction_cost,
        "net_recovered_value": net_recovered_value,
        "recovery_per_intervention": recovery_per_intervention,
        "net_per_intervention": net_per_intervention,
        "expected_recovery": expected_value,
        "expected_net_recovery": expected_net_value,
    }


# ============================================================
# BUDGET OPTIMIZATION
# ============================================================

def optimize_under_budget(
    df: pd.DataFrame,
    budget: float,
) -> Tuple[pd.Series, Dict]:

    if budget <= 0:

        mask = pd.Series(
            False,
            index=df.index,
        )

        result = evaluate_policy(
            df,
            f"BUDGET_₹{budget:.2f}",
            mask,
        )

        return mask, result

    max_actions = int(
        np.floor(
            budget
            / (
                INTERVENTION_COST
                + FRICTION_COST
            )
        )
    )

    max_actions = min(
        max_actions,
        len(df),
    )

    eligible = df[
        (
            df["recovery_probability"]
            >= MIN_PROBABILITY
        )
        &
        (
            df["expected_net_recovery_if_action"]
            > 0
        )
    ].copy()

    eligible = eligible.sort_values(
        [
            "expected_net_recovery_if_action",
            "expected_recovery",
            "recovery_probability",
        ],
        ascending=False,
    )

    selected = eligible.head(
        max_actions
    )

    mask = pd.Series(
        False,
        index=df.index,
    )

    mask.loc[selected.index] = True

    result = evaluate_policy(
        df,
        f"BUDGET_₹{budget:.2f}",
        mask,
    )

    return mask, result


# ============================================================
# POLICY TESTING
# ============================================================

def test_policies(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:

    header("TESTING COST-CONSTRAINED POLICIES")

    policies: Dict[str, pd.Series] = {}

    # --------------------------------------------------------
    # RETRY ALL
    # --------------------------------------------------------

    policies["RETRY_ALL"] = pd.Series(
        True,
        index=df.index,
    )

    # --------------------------------------------------------
    # TOP-K POLICIES
    # --------------------------------------------------------

    for percent in [
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
    ]:

        policies[
            f"TOP_{percent}"
        ] = select_top_k(
            df,
            percent / 100,
        )

    # --------------------------------------------------------
    # PROBABILITY POLICIES
    # --------------------------------------------------------

    for threshold in [
        0.40,
        0.50,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]:

        policies[
            f"PROBABILITY_{threshold:.2f}"
        ] = select_probability(
            df,
            threshold,
        )

    # --------------------------------------------------------
    # POSITIVE EXPECTED VALUE
    # --------------------------------------------------------

    policies[
        "POSITIVE_EXPECTED_VALUE"
    ] = select_expected_value(df)

    # --------------------------------------------------------
    # BUDGET POLICIES
    # --------------------------------------------------------

    if TOTAL_BUDGET is not None:

        budget_values = [
            TOTAL_BUDGET
        ]

    else:

        total_possible_cost = (
            len(df)
            * (
                INTERVENTION_COST
                + FRICTION_COST
            )
        )

        budget_values = [
            total_possible_cost * p
            for p in BUDGET_PERCENTAGES
        ]

    for budget in budget_values:

        mask, _ = optimize_under_budget(
            df,
            budget,
        )

        policies[
            f"BUDGET_₹{budget:.2f}"
        ] = mask

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = []

    for name, mask in policies.items():

        result = evaluate_policy(
            df,
            name,
            mask,
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    print(
        f"Policies evaluated: "
        f"{len(results_df)}"
    )

    return results_df, policies


# ============================================================
# BUILD FINAL POLICY
# ============================================================

def build_final_policy(
    df: pd.DataFrame,
    best_policy_name: str,
    policies: Dict[str, pd.Series],
) -> pd.DataFrame:

    final = df.copy()

    mask = policies[
        best_policy_name
    ].astype(bool)

    final["recommended_action"] = np.where(
        mask,
        "INTERVENE",
        "NO_ACTION",
    )

    final["policy_name"] = (
        best_policy_name
    )

    final["intervention_flag"] = (
        mask.astype(int)
    )

    final["expected_recovery_if_action"] = (
        final["expected_recovery"]
    )

    final["expected_net_recovery_if_action"] = (
        final[
            "expected_net_recovery_if_action"
        ]
    )

    final["actual_recovered_amount"] = (
        final["recovered_amount"]
    )

    final["actual_recovery_flag"] = (
        final["recovery_flag"]
    )

    final["actual_net_recovery"] = np.where(
        mask,
        final["recovered_amount"]
        - INTERVENTION_COST
        - FRICTION_COST,
        0.0,
    )

    final["policy_rank"] = (
        final[
            "expected_net_recovery_if_action"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    return final


# ============================================================
# VALIDATION
# ============================================================

def validate_results(
    df: pd.DataFrame,
    comparison: pd.DataFrame,
    final: pd.DataFrame,
) -> None:

    header("COST-CONSTRAINED POLICY VALIDATION")

    required = [
        "transaction_id",
        "amount",
        "recovery_probability",
        "recovery_flag",
        "recovered_amount",
    ]

    for column in required:

        if column not in df.columns:
            raise ValueError(
                f"Missing required final column: {column}"
            )

    print(
        "[PASS] Required canonical columns present"
    )

    if not df["transaction_id"].is_unique:
        raise ValueError(
            "Transaction IDs are not unique."
        )

    print("[PASS] Transaction IDs unique")

    if not df["recovery_probability"].between(
        0,
        1,
    ).all():

        raise ValueError(
            "Recovery probabilities outside [0,1]."
        )

    print(
        "[PASS] Recovery probabilities valid"
    )

    if not df["recovery_flag"].isin(
        [0, 1]
    ).all():

        raise ValueError(
            "Recovery flags are invalid."
        )

    print("[PASS] Recovery flags valid")

    if (
        df["recovered_amount"] < 0
    ).any():

        raise ValueError(
            "Recovered amounts cannot be negative."
        )

    print(
        "[PASS] Recovered amounts valid"
    )

    if len(comparison) == 0:
        raise ValueError(
            "No policy results generated."
        )

    print(
        "[PASS] Policy comparison generated"
    )

    if len(final) != len(df):
        raise ValueError(
            "Final policy population changed."
        )

    print(
        "[PASS] Final policy population preserved"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # ECONOMIC VALUE
    # --------------------------------------------------------

    df = calculate_expected_values(
        df
    )

    # --------------------------------------------------------
    # POLICIES
    # --------------------------------------------------------

    comparison, policies = test_policies(
        df
    )

    # --------------------------------------------------------
    # BEST POLICY
    # --------------------------------------------------------

    # IMPORTANT:
    # Choose by measured net recovered value, not
    # net-per-intervention. This prevents selecting a tiny
    # high-value subset simply because its ratio is high.

    comparison_sorted = comparison.sort_values(
        [
            "net_recovered_value",
            "recovered_value",
        ],
        ascending=False,
    ).reset_index(drop=True)

    best = comparison_sorted.iloc[0]

    best_policy_name = str(
        best["policy"]
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    header("COST-CONSTRAINED POLICY COMPARISON")

    display_columns = [
        "policy",
        "interventions",
        "intervention_rate",
        "recovered_payments",
        "recovered_value",
        "recovery_rate_value",
        "intervention_cost",
        "friction_cost",
        "net_recovered_value",
        "net_per_intervention",
    ]

    display_df = comparison_sorted[
        display_columns
    ].copy()

    for column in [
        "intervention_rate",
        "recovery_rate_value",
    ]:

        display_df[column] = (
            display_df[column]
            * 100
        ).round(2)

    print(
        display_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BEST POLICY
    # --------------------------------------------------------

    header("OPTIMAL COST-CONSTRAINED POLICY")

    print(
        f"Best policy:             "
        f"{best_policy_name}"
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
        f"{money(best['recovered_value'])}"
    )

    print(
        f"Net recovered value:     "
        f"{money(best['net_recovered_value'])}"
    )

    print(
        f"Net ₹ / intervention:    "
        f"{money(best['net_per_intervention'])}"
    )

    # --------------------------------------------------------
    # RETRY ALL COMPARISON
    # --------------------------------------------------------

    retry_all = comparison[
        comparison["policy"]
        == "RETRY_ALL"
    ]

    if len(retry_all) > 0:

        retry_all = retry_all.iloc[0]

        net_difference = (
            best["net_recovered_value"]
            - retry_all[
                "net_recovered_value"
            ]
        )

        avoided = (
            retry_all["interventions"]
            - best["interventions"]
        )

        reduction = (
            avoided
            / retry_all["interventions"]
            if retry_all["interventions"] > 0
            else 0
        )

        header(
            "COST-CONSTRAINED RECLAIM LIFT"
        )

        print(
            "BEST POLICY vs RETRY ALL"
        )

        print("-" * 55)

        print(
            f"Net recovery difference: "
            f"{money(net_difference)}"
        )

        print(
            f"Interventions avoided:   "
            f"{int(avoided):,}"
        )

        print(
            f"Intervention reduction:   "
            f"{reduction:.2%}"
        )

        if net_difference > 0:

            print(
                "[PASS] Optimized policy beats RETRY_ALL "
                "on net recovered value."
            )

        elif net_difference == 0:

            print(
                "[PASS] Optimized policy matches RETRY_ALL."
            )

        else:

            print(
                "[WARN] Optimized policy does NOT beat "
                "RETRY_ALL on net recovered value."
            )

    # --------------------------------------------------------
    # BUILD FINAL POLICY
    # --------------------------------------------------------

    final_df = build_final_policy(
        df,
        best_policy_name,
        policies,
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validate_results(
        df,
        comparison,
        final_df,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            COMPARISON_OUTPUT
        ),
        exist_ok=True,
    )

    comparison.to_csv(
        COMPARISON_OUTPUT,
        index=False,
    )

    final_columns = [
        "transaction_id",
        "customer_id",
        "amount",
        "failure_reason",
        "recovery_probability",
        "expected_recovery",
        "expected_net_recovery_if_action",
        "recovery_flag",
        "recovered_amount",
        "recommended_action",
        "intervention_flag",
        "policy_name",
        "policy_rank",
        "actual_recovery_flag",
        "actual_recovered_amount",
        "actual_net_recovery",
    ]

    # Only save columns that actually exist.
    final_columns = [
        column
        for column in final_columns
        if column in final_df.columns
    ]

    final_df[
        final_columns
    ].to_csv(
        FINAL_POLICY_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    header(
        "COST-CONSTRAINED POLICY OPTIMIZATION COMPLETE"
    )

    print(
        "Saved comparison to:"
    )

    print(
        COMPARISON_OUTPUT
    )

    print()

    print(
        "Saved final policy to:"
    )

    print(
        FINAL_POLICY_OUTPUT
    )

    print()

    print(
        "COST-CONSTRAINED POLICY STATUS: READY"
    )


if __name__ == "__main__":
    main()