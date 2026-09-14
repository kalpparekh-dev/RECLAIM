"""
RECLAIM ECONOMIC POLICY OPTIMIZER
=================================

Purpose:
    Optimize recovery intervention policies using:
        - Recovery model predictions
        - Observed recovery outcomes
        - Transaction amounts
        - Intervention cost
        - Customer friction cost

Inputs:
    data/generated/recovery_model_predictions.csv
    data/generated/recovery_outcomes.csv

Outputs:
    data/generated/economic_policy_comparison.csv
    data/generated/economic_final_policy.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "generated"

PREDICTION_FILE = DATA_DIR / "recovery_model_predictions.csv"
OUTCOME_FILE = DATA_DIR / "recovery_outcomes.csv"

COMPARISON_FILE = DATA_DIR / "economic_policy_comparison.csv"
FINAL_POLICY_FILE = DATA_DIR / "economic_final_policy.csv"

# Economic assumptions
INTERVENTION_COST = 1.50
FRICTION_COST = 0.25

RANDOM_SEED = 42


# ============================================================
# DISPLAY HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def money(value):
    return f"₹{value:,.2f}"


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTION_FILE}"
        )

    df = pd.read_csv(PREDICTION_FILE)

    required = [
        "transaction_id",
        "amount",
        "recovery_probability",
        "confidence",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Prediction dataset missing columns: {missing}"
        )

    return df


# ============================================================
# NORMALIZE RECOVERY OUTCOMES
# ============================================================

def normalize_outcomes():

    if not OUTCOME_FILE.exists():
        raise FileNotFoundError(
            f"Recovery outcome file not found:\n{OUTCOME_FILE}"
        )

    outcomes = pd.read_csv(OUTCOME_FILE)

    print("Outcome columns detected:")
    print(list(outcomes.columns))

    # --------------------------------------------------------
    # NORMALIZE RECOVERY FLAG
    # --------------------------------------------------------

    if "recovery_flag" not in outcomes.columns:

        if "recovered" in outcomes.columns:

            outcomes["recovery_flag"] = (
                pd.to_numeric(
                    outcomes["recovered"],
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )

            print(
                "[PASS] Derived `recovery_flag` from `recovered`"
            )

        elif "recovery_status" in outcomes.columns:

            outcomes["recovery_flag"] = (
                outcomes["recovery_status"]
                .astype(str)
                .str.upper()
                .eq("RECOVERED")
                .astype(int)
            )

            print(
                "[PASS] Derived `recovery_flag` from "
                "`recovery_status`"
            )

        else:
            raise ValueError(
                "Cannot derive recovery_flag. "
                "Need `recovered` or `recovery_status`."
            )

    # --------------------------------------------------------
    # NORMALIZE RECOVERED AMOUNT
    # --------------------------------------------------------

    if "recovered_amount" not in outcomes.columns:

        if (
            "recovered" in outcomes.columns
            and "amount" in outcomes.columns
        ):

            recovered_flag = (
                pd.to_numeric(
                    outcomes["recovered"],
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )

            amount = pd.to_numeric(
                outcomes["amount"],
                errors="coerce"
            ).fillna(0)

            outcomes["recovered_amount"] = (
                recovered_flag * amount
            )

            print(
                "[PASS] Derived `recovered_amount` "
                "from `recovered` and `amount`"
            )

        elif (
            "recovery_flag" in outcomes.columns
            and "amount" in outcomes.columns
        ):

            recovered_flag = (
                pd.to_numeric(
                    outcomes["recovery_flag"],
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )

            amount = pd.to_numeric(
                outcomes["amount"],
                errors="coerce"
            ).fillna(0)

            outcomes["recovered_amount"] = (
                recovered_flag * amount
            )

            print(
                "[PASS] Derived `recovered_amount` "
                "from `recovery_flag` and `amount`"
            )

        else:
            raise ValueError(
                "Cannot derive recovered_amount. "
                "Need amount + recovered/recovery_flag."
            )

    # --------------------------------------------------------
    # NORMALIZE AMOUNT
    # --------------------------------------------------------

    if "amount" not in outcomes.columns:
        raise ValueError(
            "Recovery outcomes do not contain `amount`."
        )

    # --------------------------------------------------------
    # NUMERIC CLEANUP
    # --------------------------------------------------------

    outcomes["amount"] = pd.to_numeric(
        outcomes["amount"],
        errors="coerce"
    )

    outcomes["recovered_amount"] = pd.to_numeric(
        outcomes["recovered_amount"],
        errors="coerce"
    ).fillna(0)

    outcomes["recovery_flag"] = (
        pd.to_numeric(
            outcomes["recovery_flag"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
        .clip(0, 1)
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    required = [
        "transaction_id",
        "amount",
        "recovery_flag",
        "recovered_amount",
    ]

    missing = [
        column
        for column in required
        if column not in outcomes.columns
    ]

    if missing:
        raise ValueError(
            f"Missing normalized outcome columns: {missing}"
        )

    print("[PASS] Recovery outcome schema normalized")

    return outcomes


# ============================================================
# LOAD + MERGE DATA
# ============================================================

def load_data():

    header("RECLAIM ECONOMIC POLICY OPTIMIZER")

    predictions = load_predictions()
    outcomes = normalize_outcomes()

    print()
    print(
        f"Prediction rows: {len(predictions):,}"
    )

    print(
        f"Outcome rows:    {len(outcomes):,}"
    )

    # --------------------------------------------------------
    # PREDICTION VALIDATION
    # --------------------------------------------------------

    if predictions["transaction_id"].duplicated().any():
        raise ValueError(
            "Prediction transaction IDs are not unique."
        )

    print("[PASS] Prediction dataset structure")

    # --------------------------------------------------------
    # OUTCOME VALIDATION
    # --------------------------------------------------------

    if outcomes["transaction_id"].duplicated().any():
        raise ValueError(
            "Outcome transaction IDs are not unique."
        )

    print("[PASS] Transaction IDs are unique")

    # --------------------------------------------------------
    # KEEP ONLY REQUIRED OUTCOME COLUMNS
    # --------------------------------------------------------

    outcome_subset = outcomes[
        [
            "transaction_id",
            "amount",
            "recovery_flag",
            "recovered_amount",
        ]
    ].copy()

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    df = predictions.merge(
        outcome_subset,
        on="transaction_id",
        how="inner",
        suffixes=("", "_outcome")
    )

    if len(df) == 0:
        raise ValueError(
            "No transactions matched between "
            "prediction and outcome datasets."
        )

    print(
        f"[PASS] Same transaction population"
    )

    print(
        f"[PASS] Merged dataset: {len(df):,} transactions"
    )

    # --------------------------------------------------------
    # CANONICAL AMOUNT
    # --------------------------------------------------------

    if "amount" not in df.columns:

        if "amount_outcome" in df.columns:
            df["amount"] = df["amount_outcome"]

        else:
            raise ValueError(
                "No canonical amount column available."
            )

    # If both exist, use prediction amount.
    if "amount_outcome" in df.columns:

        df["amount"] = pd.to_numeric(
            df["amount"],
            errors="coerce"
        )

        fallback_amount = pd.to_numeric(
            df["amount_outcome"],
            errors="coerce"
        )

        df["amount"] = df["amount"].fillna(
            fallback_amount
        )

    # --------------------------------------------------------
    # NUMERIC NORMALIZATION
    # --------------------------------------------------------

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    ).fillna(0)

    df["recovery_probability"] = pd.to_numeric(
        df["recovery_probability"],
        errors="coerce"
    ).fillna(0).clip(0, 1)

    df["confidence"] = pd.to_numeric(
        df["confidence"],
        errors="coerce"
    ).fillna(0).clip(0, 1)

    df["recovery_flag"] = pd.to_numeric(
        df["recovery_flag"],
        errors="coerce"
    ).fillna(0).astype(int).clip(0, 1)

    df["recovered_amount"] = pd.to_numeric(
        df["recovered_amount"],
        errors="coerce"
    ).fillna(0)

    print(
        "[PASS] Canonical amount column: `amount`"
    )

    print(
        "[PASS] Recovered amount column: "
        "`recovered_amount`"
    )

    return df


# ============================================================
# ECONOMIC VALUE
# ============================================================

def calculate_economic_features(df):

    header("CALCULATING ECONOMIC VALUE")

    df = df.copy()

    # Expected recovery based on model probability
    df["expected_recovery"] = (
        df["amount"]
        * df["recovery_probability"]
    )

    # Confidence-adjusted probability
    df["confidence_adjusted_probability"] = (
        df["recovery_probability"]
        * (
            0.5
            + 0.5 * df["confidence"]
        )
    )

    # Confidence-adjusted expected recovery
    df["confidence_adjusted_recovery"] = (
        df["amount"]
        * df["confidence_adjusted_probability"]
    )

    # Expected cost
    df["expected_intervention_cost"] = (
        INTERVENTION_COST
    )

    df["expected_friction_cost"] = (
        FRICTION_COST
    )

    df["expected_total_cost"] = (
        df["expected_intervention_cost"]
        + df["expected_friction_cost"]
    )

    # Expected net recovery
    df["expected_net_recovery"] = (
        df["confidence_adjusted_recovery"]
        - df["expected_total_cost"]
    )

    # Positive expected value
    df["positive_expected_value"] = (
        df["expected_net_recovery"] > 0
    )

    print(
        "[PASS] Expected recovery calculated"
    )

    print(
        "[PASS] Expected net recovery calculated"
    )

    return df


# ============================================================
# POLICY SELECTION
# ============================================================

def select_top_fraction(df, fraction):

    count = max(
        1,
        int(len(df) * fraction)
    )

    ranked = df.sort_values(
        "confidence_adjusted_recovery",
        ascending=False
    )

    selected_ids = set(
        ranked.head(count)["transaction_id"]
    )

    return df["transaction_id"].isin(
        selected_ids
    )


def build_policies(df):

    policies = {}

    # --------------------------------------------------------
    # RETRY ALL
    # --------------------------------------------------------

    policies["RETRY_ALL"] = np.ones(
        len(df),
        dtype=bool
    )

    # --------------------------------------------------------
    # TOP POLICIES
    # --------------------------------------------------------

    for percentage in [
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

        policies[
            f"TOP_{percentage}"
        ] = select_top_fraction(
            df,
            percentage / 100
        )

    # --------------------------------------------------------
    # PROBABILITY POLICIES
    # --------------------------------------------------------

    for threshold in [
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ]:

        policies[
            f"PROBABILITY_{threshold:.2f}"
        ] = (
            df["recovery_probability"]
            >= threshold
        )

    # --------------------------------------------------------
    # CONFIDENCE POLICIES
    # --------------------------------------------------------

    for threshold in [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
    ]:

        policies[
            f"CONFIDENCE_{threshold:.2f}"
        ] = (
            df["confidence"]
            >= threshold
        )

    # --------------------------------------------------------
    # ECONOMIC RECLAIM
    # --------------------------------------------------------

    # Intervention only if expected net recovery
    # is positive AND model confidence is acceptable.

    reclaim_mask = (
        (df["expected_net_recovery"] > 0)
        &
        (
            df["confidence"]
            >= 0.20
        )
    )

    policies["ECONOMIC_RECLAIM"] = reclaim_mask

    return policies


# ============================================================
# POLICY EVALUATION
# ============================================================

def evaluate_policy(
    df,
    policy_name,
    mask
):

    working = df.copy()

    working["intervene"] = mask.astype(bool)

    interventions = int(
        working["intervene"].sum()
    )

    intervention_rate = (
        interventions / len(working) * 100
        if len(working) > 0
        else 0
    )

    # --------------------------------------------------------
    # Actual recovered payments
    # --------------------------------------------------------

    recovered_mask = (
        working["intervene"]
        &
        (
            working["recovery_flag"] == 1
        )
    )

    recovered_payments = int(
        recovered_mask.sum()
    )

    recovered_value = float(
        working.loc[
            recovered_mask,
            "recovered_amount"
        ].sum()
    )

    total_failed_value = float(
        working["amount"].sum()
    )

    recovery_rate_value = (
        recovered_value
        / total_failed_value
        * 100
        if total_failed_value > 0
        else 0
    )

    recovery_rate_count = (
        recovered_payments
        / interventions
        * 100
        if interventions > 0
        else 0
    )

    # --------------------------------------------------------
    # Costs
    # --------------------------------------------------------

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
        recovered_value
        / interventions
        if interventions > 0
        else 0
    )

    net_per_intervention = (
        net_recovered_value
        / interventions
        if interventions > 0
        else 0
    )

    # --------------------------------------------------------
    # Expected values
    # --------------------------------------------------------

    expected_recovery = float(
        working.loc[
            working["intervene"],
            "expected_recovery"
        ].sum()
    )

    expected_net_recovery = float(
        working.loc[
            working["intervene"],
            "expected_net_recovery"
        ].sum()
    )

    return {
        "policy": policy_name,
        "transactions": len(working),
        "interventions": interventions,
        "intervention_rate": intervention_rate,
        "recovered_payments": recovered_payments,
        "recovered_value": recovered_value,
        "recovery_rate_value": recovery_rate_value,
        "recovery_rate_count": recovery_rate_count,
        "intervention_cost": intervention_cost,
        "friction_cost": friction_cost,
        "net_recovered_value": net_recovered_value,
        "recovery_per_intervention": recovery_per_intervention,
        "net_per_intervention": net_per_intervention,
        "expected_recovery": expected_recovery,
        "expected_net_recovery": expected_net_recovery,
    }


# ============================================================
# PRINT POLICY TABLE
# ============================================================

def print_policy_table(results):

    header("POLICY COMPARISON")

    display = results.copy()

    for column in [
        "intervention_rate",
        "recovery_rate_value",
        "recovery_rate_count",
    ]:

        display[column] = display[column].map(
            lambda x: f"{x:.2f}%"
        )

    for column in [
        "recovered_value",
        "intervention_cost",
        "friction_cost",
        "net_recovered_value",
        "recovery_per_intervention",
        "net_per_intervention",
    ]:

        display[column] = display[column].map(
            money
        )

    columns = [
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

    print(
        display[columns].to_string(
            index=False
        )
    )


# ============================================================
# BEST POLICY
# ============================================================

def determine_best_policy(results):

    # Primary objective:
    # maximize net recovered value.

    best = results.sort_values(
        [
            "net_recovered_value",
            "net_per_intervention",
        ],
        ascending=False
    ).iloc[0]

    return best


# ============================================================
# LIFT ANALYSIS
# ============================================================

def print_lift(results):

    header("ECONOMIC RECLAIM LIFT")

    reclaim = results[
        results["policy"]
        == "ECONOMIC_RECLAIM"
    ].iloc[0]

    retry_all = results[
        results["policy"]
        == "RETRY_ALL"
    ].iloc[0]

    net_difference = (
        reclaim["net_recovered_value"]
        - retry_all["net_recovered_value"]
    )

    interventions_avoided = (
        retry_all["interventions"]
        - reclaim["interventions"]
    )

    intervention_reduction = (
        interventions_avoided
        / retry_all["interventions"]
        * 100
    )

    print(
        "ECONOMIC RECLAIM vs RETRY ALL"
    )

    print("-" * 55)

    print(
        f"Net recovery difference: "
        f"{money(net_difference)}"
    )

    print(
        f"Interventions avoided:    "
        f"{interventions_avoided:,}"
    )

    print(
        f"Intervention reduction:   "
        f"{intervention_reduction:.2f}%"
    )

    if net_difference > 0:

        print(
            "[PASS] ECONOMIC RECLAIM beats "
            "RETRY_ALL on net recovered value."
        )

    else:

        print(
            "[WARN] ECONOMIC RECLAIM does NOT beat "
            "RETRY_ALL on net recovered value."
        )


# ============================================================
# BUILD FINAL POLICY
# ============================================================

def build_final_policy(
    df,
    best_policy_name
):

    if best_policy_name == "ECONOMIC_RECLAIM":

        mask = (
            (df["expected_net_recovery"] > 0)
            &
            (df["confidence"] >= 0.20)
        )

    elif best_policy_name == "RETRY_ALL":

        mask = np.ones(
            len(df),
            dtype=bool
        )

    elif best_policy_name.startswith("TOP_"):

        percentage = float(
            best_policy_name.split("_")[1]
        )

        mask = select_top_fraction(
            df,
            percentage / 100
        )

    elif best_policy_name.startswith(
        "PROBABILITY_"
    ):

        threshold = float(
            best_policy_name.split("_")[1]
        )

        mask = (
            df["recovery_probability"]
            >= threshold
        )

    elif best_policy_name.startswith(
        "CONFIDENCE_"
    ):

        threshold = float(
            best_policy_name.split("_")[1]
        )

        mask = (
            df["confidence"]
            >= threshold
        )

    else:

        raise ValueError(
            f"Unknown policy: {best_policy_name}"
        )

    final = df.copy()

    final["recommended_action"] = np.where(
        mask,
        "INTERVENE",
        "NO_ACTION"
    )

    final["policy_selected"] = best_policy_name

    final["expected_intervention_cost"] = np.where(
        mask,
        INTERVENTION_COST,
        0
    )

    final["expected_friction_cost"] = np.where(
        mask,
        FRICTION_COST,
        0
    )

    final["expected_net_recovery"] = np.where(
        mask,
        final["expected_net_recovery"],
        0
    )

    final["actual_recovered"] = np.where(
        mask,
        final["recovery_flag"],
        0
    )

    final["actual_recovered_amount"] = np.where(
        mask,
        final["recovered_amount"],
        0
    )

    # --------------------------------------------------------
    # Final columns
    # --------------------------------------------------------

    preferred_columns = [
        "transaction_id",
        "amount",
        "recovery_probability",
        "confidence",
        "expected_recovery",
        "expected_net_recovery",
        "recommended_action",
        "policy_selected",
        "recovery_flag",
        "recovered_amount",
        "actual_recovered",
        "actual_recovered_amount",
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in final.columns
    ]

    return final[existing_columns].copy()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # ECONOMIC FEATURES
    # --------------------------------------------------------

    df = calculate_economic_features(df)

    # --------------------------------------------------------
    # BUILD POLICIES
    # --------------------------------------------------------

    header("TESTING ECONOMIC RECOVERY POLICIES")

    policies = build_policies(df)

    print(
        f"Policies evaluated: {len(policies)}"
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    results = []

    for policy_name, mask in policies.items():

        result = evaluate_policy(
            df,
            policy_name,
            mask
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print_policy_table(
        results_df
    )

    # --------------------------------------------------------
    # BEST POLICY
    # --------------------------------------------------------

    header("OPTIMAL ECONOMIC POLICY")

    best = determine_best_policy(
        results_df
    )

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
        f"{best['intervention_rate']:.2f}%"
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
    # LIFT
    # --------------------------------------------------------

    print_lift(
        results_df
    )

    # --------------------------------------------------------
    # BUILD FINAL POLICY
    # --------------------------------------------------------

    final_policy = build_final_policy(
        df,
        best["policy"]
    )

    # --------------------------------------------------------
    # SAVE COMPARISON
    # --------------------------------------------------------

    results_df.to_csv(
        COMPARISON_FILE,
        index=False
    )

    # --------------------------------------------------------
    # SAVE FINAL POLICY
    # --------------------------------------------------------

    final_policy.to_csv(
        FINAL_POLICY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    header("ECONOMIC POLICY OPTIMIZATION COMPLETE")

    print(
        "Saved comparison to:"
    )

    print(
        COMPARISON_FILE
    )

    print()

    print(
        "Saved final policy to:"
    )

    print(
        FINAL_POLICY_FILE
    )

    print()

    print(
        "ECONOMIC POLICY STATUS: READY"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()