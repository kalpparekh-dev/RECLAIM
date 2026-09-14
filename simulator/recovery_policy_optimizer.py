"""
RECLAIM RECOVERY POLICY OPTIMIZER

Purpose:
    Find the intervention policy that maximizes net recovered value
    while controlling intervention volume and customer friction.

Input:
    data/generated/recovery_model_predictions.csv

Output:
    data/generated/recovery_policy_comparison.csv
    data/generated/recovery_final_policy.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "generated"

INPUT_FILE = DATA_DIR / "recovery_model_predictions.csv"

POLICY_OUTPUT = DATA_DIR / "recovery_policy_comparison.csv"
FINAL_OUTPUT = DATA_DIR / "recovery_final_policy.csv"


# ------------------------------------------------------------
# Recovery economics
# ------------------------------------------------------------

RETRY_NOW_COST = 1.50
DELAYED_RETRY_COST = 1.00

RETRY_NOW_FRICTION = 0.25
DELAYED_RETRY_FRICTION = 0.20


# ------------------------------------------------------------
# Policy constraints
# ------------------------------------------------------------

MAX_INTERVENTION_RATE = 0.50
MIN_CONFIDENCE = 0.10
MIN_EXPECTED_NET_RECOVERY = 0.00


# ============================================================
# DISPLAY HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

def load_predictions():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "transaction_id",
        "amount",
        "recovery_probability",
        "confidence",
        "actual_recovered",
        "recovered_amount",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_data(df):

    header("POLICY INPUT VALIDATION")

    if df.empty:
        raise ValueError("Prediction dataset is empty.")

    print(f"[PASS] Dataset loaded: {len(df):,} rows")

    if df["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs detected.")

    print("[PASS] Transaction IDs are unique")

    if not df["amount"].gt(0).all():
        raise ValueError("Invalid transaction amounts.")

    print("[PASS] Transaction amounts are positive")

    if not df["recovery_probability"].between(0, 1).all():
        raise ValueError("Recovery probabilities outside [0,1].")

    print("[PASS] Recovery probabilities valid")

    if not df["confidence"].between(0, 1).all():
        raise ValueError("Confidence values outside [0,1].")

    print("[PASS] Confidence values valid")

    if not df["actual_recovered"].isin([0, 1]).all():
        raise ValueError("actual_recovered must be binary.")

    print("[PASS] Held-out outcomes available")

    print(
        f"Observed recovery rate: "
        f"{df['actual_recovered'].mean() * 100:.2f}%"
    )


# ============================================================
# ECONOMIC SIMULATION
# ============================================================

def simulate_policy(df, mask, policy_name, action_type):

    selected = df.loc[mask].copy()

    if action_type == "RETRY_NOW":

        intervention_cost = RETRY_NOW_COST
        friction_cost = RETRY_NOW_FRICTION

    elif action_type == "DELAYED_RETRY":

        intervention_cost = DELAYED_RETRY_COST
        friction_cost = DELAYED_RETRY_FRICTION

    else:

        intervention_cost = 0.0
        friction_cost = 0.0

    interventions = len(selected)

    recovered_payments = int(
        selected["actual_recovered"].sum()
    )

    recovered_value = float(
        selected["recovered_amount"].sum()
    )

    total_intervention_cost = (
        interventions * intervention_cost
    )

    total_friction_cost = (
        interventions * friction_cost
    )

    net_recovered_value = (
        recovered_value
        - total_intervention_cost
        - total_friction_cost
    )

    total_failed_value = float(df["amount"].sum())

    intervention_rate = (
        interventions / len(df)
        if len(df) > 0
        else 0
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

    return {
        "policy": policy_name,
        "action": action_type,
        "interventions": interventions,
        "intervention_rate": intervention_rate,
        "recovered_payments": recovered_payments,
        "recovered_value": recovered_value,
        "recovery_rate_value": recovery_rate_value,
        "recovery_rate_count": recovery_rate_count,
        "intervention_cost": total_intervention_cost,
        "friction_cost": total_friction_cost,
        "net_recovered_value": net_recovered_value,
        "net_per_intervention": (
            net_recovered_value / interventions
            if interventions > 0
            else 0
        ),
    }


# ============================================================
# EXPECTED ECONOMIC VALUE
# ============================================================

def calculate_expected_values(df):

    df = df.copy()

    # Expected recovery based on model probability
    df["expected_recovery"] = (
        df["amount"]
        * df["recovery_probability"]
    )

    # Select the cheaper action when possible.
    #
    # Retry-now has slightly higher cost/friction,
    # delayed retry is cheaper.
    #
    # The policy will use confidence to decide between them.

    df["retry_now_cost"] = (
        RETRY_NOW_COST
        + RETRY_NOW_FRICTION
    )

    df["delayed_retry_cost"] = (
        DELAYED_RETRY_COST
        + DELAYED_RETRY_FRICTION
    )

    df["expected_net_retry_now"] = (
        df["expected_recovery"]
        - df["retry_now_cost"]
    )

    df["expected_net_delayed"] = (
        df["expected_recovery"]
        - df["delayed_retry_cost"]
    )

    df["best_expected_net"] = np.maximum(
        df["expected_net_retry_now"],
        df["expected_net_delayed"],
    )

    return df


# ============================================================
# POLICY GENERATION
# ============================================================

def build_policies(df):

    policies = []

    # --------------------------------------------------------
    # Baseline: RETRY ALL
    # --------------------------------------------------------

    mask = np.ones(len(df), dtype=bool)

    policies.append(
        simulate_policy(
            df,
            mask,
            "RETRY_ALL",
            "RETRY_NOW",
        )
    )

    # --------------------------------------------------------
    # Top-k policies
    # --------------------------------------------------------

    ranked = df.sort_values(
        "expected_recovery",
        ascending=False,
    )

    percentages = [
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ]

    for percentage in percentages:

        n = max(
            1,
            int(len(df) * percentage)
        )

        selected_ids = set(
            ranked.head(n)["transaction_id"]
        )

        mask = df["transaction_id"].isin(
            selected_ids
        )

        policies.append(
            simulate_policy(
                df,
                mask,
                f"TOP_{int(percentage * 100)}",
                "RETRY_NOW",
            )
        )

    # --------------------------------------------------------
    # Probability thresholds
    # --------------------------------------------------------

    for threshold in [
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ]:

        mask = (
            (df["recovery_probability"] >= threshold)
            &
            (df["confidence"] >= MIN_CONFIDENCE)
        )

        policies.append(
            simulate_policy(
                df,
                mask,
                f"PROBABILITY_{threshold:.2f}",
                "RETRY_NOW",
            )
        )

    # --------------------------------------------------------
    # Confidence thresholds
    # --------------------------------------------------------

    for threshold in [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
    ]:

        mask = (
            (df["confidence"] >= threshold)
            &
            (df["recovery_probability"] >= 0.50)
        )

        policies.append(
            simulate_policy(
                df,
                mask,
                f"CONFIDENCE_{threshold:.2f}",
                "RETRY_NOW",
            )
        )

    return policies


# ============================================================
# RECLAIM POLICY
# ============================================================

def build_reclaim_policy(df):

    df = df.copy()

    # --------------------------------------------------------
    # Economic eligibility
    # --------------------------------------------------------

    eligible = (
        (df["confidence"] >= MIN_CONFIDENCE)
        &
        (df["best_expected_net"] > MIN_EXPECTED_NET_RECOVERY)
    )

    candidates = df.loc[eligible].copy()

    if candidates.empty:

        return (
            np.zeros(len(df), dtype=bool),
            df,
        )

    # --------------------------------------------------------
    # Rank by expected economic value per intervention
    # --------------------------------------------------------

    candidates["economic_priority"] = (
        candidates["best_expected_net"]
        * (
            0.5
            + 0.5 * candidates["confidence"]
        )
    )

    candidates = candidates.sort_values(
        "economic_priority",
        ascending=False,
    )

    # --------------------------------------------------------
    # Enforce maximum intervention rate
    # --------------------------------------------------------

    maximum_actions = int(
        len(df) * MAX_INTERVENTION_RATE
    )

    candidates = candidates.head(
        maximum_actions
    )

    selected_ids = set(
        candidates["transaction_id"]
    )

    mask = df["transaction_id"].isin(
        selected_ids
    )

    # --------------------------------------------------------
    # Action selection
    #
    # High confidence → retry now
    # Lower confidence → delayed retry
    # --------------------------------------------------------

    df["recommended_action"] = "NO_ACTION"

    df.loc[
        mask & (df["confidence"] >= 0.50),
        "recommended_action"
    ] = "RETRY_NOW"

    df.loc[
        mask & (df["confidence"] < 0.50),
        "recommended_action"
    ] = "DELAYED_RETRY"

    return mask, df


# ============================================================
# EVALUATE RECLAIM ACTIONS
# ============================================================

def evaluate_reclaim(df, mask):

    results = []

    retry_mask = (
        mask
        & (df["recommended_action"] == "RETRY_NOW")
    )

    delayed_mask = (
        mask
        & (df["recommended_action"] == "DELAYED_RETRY")
    )

    if retry_mask.any():

        results.append(
            simulate_policy(
                df,
                retry_mask,
                "RECLAIM_RETRY_NOW",
                "RETRY_NOW",
            )
        )

    if delayed_mask.any():

        results.append(
            simulate_policy(
                df,
                delayed_mask,
                "RECLAIM_DELAYED_RETRY",
                "DELAYED_RETRY",
            )
        )

    if mask.any():

        results.append(
            simulate_policy(
                df,
                mask,
                "RECLAIM",
                "MIXED",
            )
        )

    return results


# ============================================================
# RECLAIM MIXED ECONOMICS
# ============================================================

def calculate_mixed_reclaim(df, mask):

    selected = df.loc[mask].copy()

    retry_now = (
        selected["recommended_action"]
        == "RETRY_NOW"
    )

    delayed = (
        selected["recommended_action"]
        == "DELAYED_RETRY"
    )

    interventions = len(selected)

    recovered_payments = int(
        selected["actual_recovered"].sum()
    )

    recovered_value = float(
        selected["recovered_amount"].sum()
    )

    intervention_cost = (
        retry_now.sum() * RETRY_NOW_COST
        + delayed.sum() * DELAYED_RETRY_COST
    )

    friction_cost = (
        retry_now.sum() * RETRY_NOW_FRICTION
        + delayed.sum() * DELAYED_RETRY_FRICTION
    )

    net_value = (
        recovered_value
        - intervention_cost
        - friction_cost
    )

    return {
        "policy": "RECLAIM",
        "action": "MIXED",
        "interventions": interventions,
        "intervention_rate": interventions / len(df),
        "recovered_payments": recovered_payments,
        "recovered_value": recovered_value,
        "recovery_rate_value": (
            recovered_value / df["amount"].sum()
        ),
        "recovery_rate_count": (
            recovered_payments / interventions
            if interventions
            else 0
        ),
        "intervention_cost": intervention_cost,
        "friction_cost": friction_cost,
        "net_recovered_value": net_value,
        "net_per_intervention": (
            net_value / interventions
            if interventions
            else 0
        ),
    }


# ============================================================
# SELECT BEST POLICY
# ============================================================

def select_best_policy(results):

    df = pd.DataFrame(results)

    # Exclude RETRY_ALL when choosing the optimized policy.
    candidates = df[
        df["policy"] != "RETRY_ALL"
    ].copy()

    # Must obey intervention constraint.
    candidates = candidates[
        candidates["intervention_rate"]
        <= MAX_INTERVENTION_RATE
    ]

    if candidates.empty:
        return None, df

    # Primary objective:
    # maximize NET recovered value.
    #
    # Secondary objective:
    # maximize recovery efficiency.

    candidates = candidates.sort_values(
        [
            "net_recovered_value",
            "net_per_intervention",
        ],
        ascending=False,
    )

    best = candidates.iloc[0]

    return best, df


# ============================================================
# PRINT REPORT
# ============================================================

def print_results(results):

    header("POLICY COMPARISON")

    result_df = pd.DataFrame(results)

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

    display_df = result_df[display_columns].copy()

    display_df["intervention_rate"] *= 100
    display_df["recovery_rate_value"] *= 100

    print(
        display_df.to_string(
            index=False,
            formatters={
                "intervention_rate": "{:.2f}%".format,
                "recovery_rate_value": "{:.2f}%".format,
                "recovered_value": "₹{:,.2f}".format,
                "intervention_cost": "₹{:,.2f}".format,
                "friction_cost": "₹{:,.2f}".format,
                "net_recovered_value": "₹{:,.2f}".format,
                "net_per_intervention": "₹{:,.2f}".format,
            },
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    header("RECLAIM RECOVERY POLICY OPTIMIZER")

    df = load_predictions()

    print(f"Prediction rows: {len(df):,}")

    validate_data(df)

    # --------------------------------------------------------
    # Expected economics
    # --------------------------------------------------------

    header("CALCULATING EXPECTED ECONOMIC VALUE")

    df = calculate_expected_values(df)

    print("[PASS] Expected recovery calculated")
    print("[PASS] Expected net recovery calculated")

    # --------------------------------------------------------
    # Generic policies
    # --------------------------------------------------------

    header("TESTING RECOVERY POLICIES")

    results = build_policies(df)

    print(
        f"Policies evaluated: {len(results)}"
    )

    # --------------------------------------------------------
    # RECLAIM policy
    # --------------------------------------------------------

    header("BUILDING RECLAIM POLICY")

    reclaim_mask, df = build_reclaim_policy(df)

    reclaim_result = calculate_mixed_reclaim(
        df,
        reclaim_mask,
    )

    results.append(reclaim_result)

    print(
        f"RECLAIM interventions: "
        f"{reclaim_result['interventions']:,}"
    )

    print(
        f"RECLAIM intervention rate: "
        f"{reclaim_result['intervention_rate'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print_results(results)

    # --------------------------------------------------------
    # Select best policy
    # --------------------------------------------------------

    best, result_df = select_best_policy(results)

    header("OPTIMAL POLICY")

    if best is None:

        print(
            "[WARN] No policy satisfied the intervention constraint."
        )

        return

    print(
        f"Best policy:             {best['policy']}"
    )

    print(
        f"Interventions:           "
        f"{int(best['interventions']):,}"
    )

    print(
        f"Intervention rate:       "
        f"{best['intervention_rate'] * 100:.2f}%"
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
    # Compare against retry all
    # --------------------------------------------------------

    retry_all = result_df[
        result_df["policy"] == "RETRY_ALL"
    ].iloc[0]

    net_lift = (
        best["net_recovered_value"]
        - retry_all["net_recovered_value"]
    )

    interventions_saved = (
        retry_all["interventions"]
        - best["interventions"]
    )

    intervention_reduction = (
        interventions_saved
        / retry_all["interventions"]
    )

    header("RECLAIM LIFT")

    print(
        f"RECLAIM vs RETRY_ALL"
    )

    print(
        f"Net recovery difference: "
        f"₹{net_lift:,.2f}"
    )

    print(
        f"Interventions avoided:   "
        f"{int(interventions_saved):,}"
    )

    print(
        f"Intervention reduction:   "
        f"{intervention_reduction * 100:.2f}%"
    )

    if net_lift > 0:

        print(
            "[PASS] RECLAIM beats RETRY_ALL "
            "on net recovered value."
        )

    else:

        print(
            "[WARN] RECLAIM does not yet beat "
            "RETRY_ALL on net recovered value."
        )

    # --------------------------------------------------------
    # Save policy comparison
    # --------------------------------------------------------

    result_df.to_csv(
        POLICY_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Save final transaction-level policy
    # --------------------------------------------------------

    final_columns = [
    "transaction_id",
    "amount",
    "recovery_probability",
    "confidence",
    "expected_recovery",
    "best_expected_net",
    "recommended_action",
    "actual_recovered",
    "recovered_amount",
]

    final_df = df[
        final_columns
    ].copy()

    final_df["policy"] = "RECLAIM"
    final_df["expected_net_recovery"] = final_df["best_expected_net"]   

    final_df.to_csv(
        FINAL_OUTPUT,
        index=False,
    )

    header("POLICY OPTIMIZATION COMPLETE")

    print(
        f"Saved comparison to:\n{POLICY_OUTPUT}"
    )

    print(
        f"\nSaved final policy to:\n{FINAL_OUTPUT}"
    )

    print()
    print(
        "RECOVERY POLICY STATUS: READY"
    )


if __name__ == "__main__":
    main()