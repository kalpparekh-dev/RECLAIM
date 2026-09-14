"""
======================================================================
RECLAIM CAUSAL UPLIFT DIAGNOSTIC
======================================================================

Purpose:
    Diagnose WHY the causal uplift model is unstable before building
    another optimizer.

This script checks:
    1. Treatment/control balance
    2. Outcome imbalance
    3. Feature variation
    4. Treatment effect variation
    5. Uplift model discrimination
    6. Economic targeting logic
    7. Whether the current experiment is large enough
    8. Whether synthetic treatment effects are too weak/noisy

IMPORTANT:
    This is a diagnostic step.
    It does NOT claim causal deployment readiness.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIG
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "generated"

EXPERIMENT_FILE = DATA_DIR / "reclaim_treatment_control_experiment.csv"

OUTPUT_DIAGNOSTIC = DATA_DIR / "causal_uplift_diagnostic.csv"
OUTPUT_SUMMARY = DATA_DIR / "causal_uplift_diagnostic_summary.csv"

RANDOM_SEED = 42


# ======================================================================
# HELPERS
# ======================================================================

def money(x):
    return f"₹{x:,.2f}"


def pct(x):
    return f"{x * 100:.2f}%"


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_numeric(df, column):
    if column not in df.columns:
        return None

    return pd.to_numeric(df[column], errors="coerce")


# ======================================================================
# LOAD DATA
# ======================================================================

def load_data():

    section("LOADING CAUSAL EXPERIMENT")

    print(f"File: {EXPERIMENT_FILE}")

    if not EXPERIMENT_FILE.exists():
        raise FileNotFoundError(
            f"Experiment file not found:\n{EXPERIMENT_FILE}"
        )

    df = pd.read_csv(EXPERIMENT_FILE)

    print(f"[PASS] Rows loaded: {len(df):,}")

    if "transaction_id" not in df.columns:
        raise ValueError("transaction_id missing")

    if df["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found")

    print("[PASS] Transaction IDs unique")

    required = [
        "treatment",
        "recovered",
        "recovered_amount",
        "amount",
        "intervention_cost",
        "customer_friction_cost",
        "net_recovered_value",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    print("[PASS] Required causal columns present")

    return df


# ======================================================================
# BASIC EXPERIMENT CHECK
# ======================================================================

def validate_experiment(df):

    section("EXPERIMENT STRUCTURE")

    treatment = pd.to_numeric(
        df["treatment"],
        errors="coerce"
    )

    if treatment.isna().any():
        raise ValueError("Invalid treatment values")

    values = sorted(treatment.unique().tolist())

    print(f"Treatment values: {values}")

    if not set(values).issubset({0, 1}):
        raise ValueError(
            "Treatment must contain only 0 and 1"
        )

    treatment_count = int((treatment == 1).sum())
    control_count = int((treatment == 0).sum())

    print(f"Treatment rows: {treatment_count}")
    print(f"Control rows:   {control_count}")

    print("[PASS] Treatment/control groups valid")

    recovery = pd.to_numeric(
        df["recovered"],
        errors="coerce"
    )

    if not recovery.isin([0, 1]).all():
        raise ValueError(
            "`recovered` must contain only 0/1"
        )

    print("[PASS] Recovery outcome valid")

    return treatment_count, control_count


# ======================================================================
# BALANCE CHECK
# ======================================================================

def balance_check(df):

    section("TREATMENT / CONTROL BALANCE")

    numeric_features = [
        "amount",
        "recovery_probability",
        "calibrated_probability",
        "confidence",
        "failure_streak",
        "historical_success_rate",
        "transactions_last_7d",
        "failure_rate_last_7d",
        "recoverability_score",
        "model_confidence",
    ]

    available = [
        c for c in numeric_features
        if c in df.columns
    ]

    rows = []

    for feature in available:

        x = pd.to_numeric(
            df.loc[df["treatment"] == 1, feature],
            errors="coerce"
        )

        y = pd.to_numeric(
            df.loc[df["treatment"] == 0, feature],
            errors="coerce"
        )

        x = x.dropna()
        y = y.dropna()

        if len(x) == 0 or len(y) == 0:
            continue

        mean_t = x.mean()
        mean_c = y.mean()

        pooled_std = np.sqrt(
            (
                x.var(ddof=1)
                + y.var(ddof=1)
            ) / 2
        )

        if pooled_std > 0:
            standardized_difference = (
                mean_t - mean_c
            ) / pooled_std
        else:
            standardized_difference = 0.0

        rows.append({
            "feature": feature,
            "treatment_mean": mean_t,
            "control_mean": mean_c,
            "absolute_difference": abs(
                mean_t - mean_c
            ),
            "standardized_difference":
                standardized_difference,
        })

    balance = pd.DataFrame(rows)

    if len(balance) == 0:
        print("[WARN] No numeric balance features available")
        return balance

    print(balance.to_string(index=False))

    abs_smd = balance[
        "standardized_difference"
    ].abs()

    print()

    print(
        "Features with |standardized difference| > 0.20:",
        int((abs_smd > 0.20).sum())
    )

    print(
        "Features with |standardized difference| > 0.10:",
        int((abs_smd > 0.10).sum())
    )

    return balance


# ======================================================================
# OBSERVED TREATMENT EFFECT
# ======================================================================

def observed_effect(df):

    section("OBSERVED TREATMENT EFFECT")

    treatment = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]

    treatment_recovery = treatment["recovered"].mean()
    control_recovery = control["recovered"].mean()

    recovery_lift = (
        treatment_recovery
        - control_recovery
    )

    treatment_value = treatment[
        "recovered_amount"
    ].sum()

    control_value = control[
        "recovered_amount"
    ].sum()

    treatment_net = treatment[
        "net_recovered_value"
    ].sum()

    control_net = control[
        "net_recovered_value"
    ].sum()

    avg_incremental_net = (
        treatment_net / len(treatment)
        - control_net / len(control)
    )

    print(
        "Treatment recovery rate:",
        pct(treatment_recovery)
    )

    print(
        "Control recovery rate:  ",
        pct(control_recovery)
    )

    print(
        "Observed recovery lift: ",
        f"{recovery_lift * 100:+.2f}%"
    )

    print()

    print(
        "Treatment recovered value:",
        money(treatment_value)
    )

    print(
        "Control recovered value:  ",
        money(control_value)
    )

    print()

    print(
        "Treatment net value:",
        money(treatment_net)
    )

    print(
        "Control net value: ",
        money(control_net)
    )

    print()

    print(
        "Incremental net value / transaction:",
        money(avg_incremental_net)
    )

    return {
        "treatment_recovery_rate":
            treatment_recovery,
        "control_recovery_rate":
            control_recovery,
        "observed_recovery_lift":
            recovery_lift,
        "treatment_recovered_value":
            treatment_value,
        "control_recovered_value":
            control_value,
        "treatment_net_value":
            treatment_net,
        "control_net_value":
            control_net,
        "incremental_net_value_per_transaction":
            avg_incremental_net,
    }


# ======================================================================
# TREATMENT EFFECT DIAGNOSTICS
# ======================================================================

def treatment_effect_check(df):

    section("TREATMENT EFFECT DATA-GENERATING PROCESS")

    columns = [
        "treatment_effect",
        "incremental_recovery_probability",
        "control_recovery_probability",
        "treatment_recovery_probability",
        "expected_incremental_net_value",
    ]

    available = [
        c for c in columns
        if c in df.columns
    ]

    for column in available:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        print(f"\n{column}")

        print(
            "  Mean:",
            values.mean()
        )

        print(
            "  Std:",
            values.std()
        )

        print(
            "  Min:",
            values.min()
        )

        print(
            "  Max:",
            values.max()
        )

        print(
            "  Positive:",
            int((values > 0).sum()),
            f"/ {len(values)}"
        )

    if "treatment_effect" in df.columns:

        effect = pd.to_numeric(
            df["treatment_effect"],
            errors="coerce"
        )

        mean_effect = effect.mean()
        std_effect = effect.std()

        print()

        print(
            "Mean synthetic treatment effect:",
            pct(mean_effect)
        )

        print(
            "Treatment effect std:",
            pct(std_effect)
        )

        if mean_effect < 0.01:

            print(
                "[WARN] Treatment effect is extremely small."
            )

        elif mean_effect < 0.03:

            print(
                "[WARN] Treatment effect may be difficult "
                "to learn reliably with only 581 rows."
            )

        else:

            print(
                "[INFO] Treatment effect has usable variation."
            )


# ======================================================================
# SAMPLE SIZE / POWER WARNING
# ======================================================================

def sample_size_check(df):

    section("EXPERIMENT SAMPLE SIZE CHECK")

    n = len(df)

    treatment_n = int(
        (df["treatment"] == 1).sum()
    )

    control_n = int(
        (df["treatment"] == 0).sum()
    )

    print(f"Total observations: {n}")
    print(f"Treatment:          {treatment_n}")
    print(f"Control:            {control_n}")

    if n < 1000:

        print(
            "[WARN] Only 581 observations are available."
        )

        print(
            "[WARN] Flexible uplift models can be unstable "
            "at this sample size."
        )

    if treatment_n < 300 or control_n < 300:

        print(
            "[WARN] Each treatment arm has fewer than "
            "300 observations."
        )

    print()

    print(
        "Recommendation:"
    )

    print(
        "  Generate a substantially larger randomized "
        "experiment before trusting complex uplift models."
    )


# ======================================================================
# ECONOMIC TARGETING CHECK
# ======================================================================

def economic_check(df):

    section("ECONOMIC TARGETING CHECK")

    if "expected_incremental_net_value" not in df.columns:

        print(
            "[WARN] expected_incremental_net_value missing"
        )

        return

    value = pd.to_numeric(
        df["expected_incremental_net_value"],
        errors="coerce"
    )

    print(
        "Mean expected incremental net value:",
        money(value.mean())
    )

    print(
        "Median expected incremental net value:",
        money(value.median())
    )

    print(
        "Positive economic rows:",
        int((value > 0).sum())
    )

    print(
        "Negative economic rows:",
        int((value < 0).sum())
    )

    print()

    if value.mean() <= 0:

        print(
            "[WARN] Average expected economic uplift "
            "is not positive."
        )

        print(
            "[WARN] Even a statistically useful uplift "
            "model may not create business value."
        )

    else:

        print(
            "[PASS] Average expected economic uplift "
            "is positive."
        )


# ======================================================================
# SIMPLE UPLIFT SIGNAL CHECK
# ======================================================================

def uplift_signal_check(df):

    section("UPLIFT SIGNAL CHECK")

    if "incremental_recovery_probability" not in df.columns:

        print(
            "[WARN] incremental_recovery_probability missing"
        )

        return {}

    uplift = pd.to_numeric(
        df["incremental_recovery_probability"],
        errors="coerce"
    )

    recovered = pd.to_numeric(
        df["recovered"],
        errors="coerce"
    )

    valid = uplift.notna() & recovered.notna()

    uplift = uplift[valid]
    recovered = recovered[valid]

    if len(uplift) < 10:

        print(
            "[WARN] Insufficient observations."
        )

        return {}

    q = pd.qcut(
        uplift,
        q=5,
        labels=False,
        duplicates="drop"
    )

    diagnostic = (
        pd.DataFrame({
            "uplift": uplift,
            "recovered": recovered
        })
        .assign(bucket=q)
        .groupby("bucket")
        .agg(
            mean_uplift=("uplift", "mean"),
            recovery_rate=("recovered", "mean"),
            rows=("recovered", "size")
        )
        .reset_index()
    )

    print(
        diagnostic.to_string(index=False)
    )

    print()

    recovery_rates = diagnostic[
        "recovery_rate"
    ].values

    if len(recovery_rates) >= 2:

        direction = (
            recovery_rates[-1]
            - recovery_rates[0]
        )

        print(
            "Highest-vs-lowest uplift bucket recovery difference:",
            pct(direction)
        )

        if direction <= 0:

            print(
                "[WARN] Uplift ranking is not aligned "
                "with observed recovery."
            )

        else:

            print(
                "[INFO] Some positive ranking signal exists."
            )

    return diagnostic


# ======================================================================
# FINAL VERDICT
# ======================================================================

def final_verdict(
    df,
    observed,
    balance
):

    section("FINAL CAUSAL UPLIFT DIAGNOSTIC VERDICT")

    warnings_found = []

    if len(df) < 1000:

        warnings_found.append(
            "small_experiment"
        )

    if len(balance) > 0:

        bad_balance = (
            balance[
                "standardized_difference"
            ].abs() > 0.20
        ).sum()

        if bad_balance > 0:

            warnings_found.append(
                "treatment_control_imbalance"
            )

    if observed[
        "incremental_net_value_per_transaction"
    ] <= 0:

        warnings_found.append(
            "negative_observed_economic_value"
        )

    if (
        "treatment_effect" in df.columns
    ):

        effect = pd.to_numeric(
            df["treatment_effect"],
            errors="coerce"
        )

        if effect.mean() < 0.03:

            warnings_found.append(
                "weak_treatment_effect"
            )

    print(
        "Diagnostic warnings:",
        len(warnings_found)
    )

    for warning in warnings_found:

        print(
            f"  [WARN] {warning}"
        )

    print()

    if len(warnings_found) >= 2:

        verdict = "EXPERIMENT_NEEDS_IMPROVEMENT"

        print(
            "[WARN] Current experiment should NOT be used "
            "to justify deployment."
        )

        print(
            "The next step should improve the simulator "
            "and generate stronger experimental evidence."
        )

    else:

        verdict = "DIAGNOSTIC_ACCEPTABLE"

        print(
            "[INFO] Experiment passes the basic diagnostic."
        )

    print()

    print(
        "FINAL VERDICT:",
        verdict
    )

    return verdict


# ======================================================================
# SAVE RESULTS
# ======================================================================

def save_results(
    observed,
    verdict,
    df,
    balance
):

    section("SAVING DIAGNOSTIC RESULTS")

    rows = []

    for key, value in observed.items():

        rows.append({
            "metric": key,
            "value": value
        })

    rows.extend([
        {
            "metric": "total_rows",
            "value": len(df)
        },
        {
            "metric": "treatment_rows",
            "value": int(
                (df["treatment"] == 1).sum()
            )
        },
        {
            "metric": "control_rows",
            "value": int(
                (df["treatment"] == 0).sum()
            )
        },
        {
            "metric": "diagnostic_verdict",
            "value": verdict
        }
    ])

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    print(
        "[PASS] Saved diagnostic summary:"
    )

    print(OUTPUT_SUMMARY)

    if len(balance) > 0:

        balance.to_csv(
            OUTPUT_DIAGNOSTIC,
            index=False
        )

        print(
            "[PASS] Saved feature balance diagnostics:"
        )

        print(OUTPUT_DIAGNOSTIC)


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("RECLAIM CAUSAL UPLIFT DIAGNOSTIC")
    print("=" * 70)

    df = load_data()

    treatment_n, control_n = validate_experiment(
        df
    )

    balance = balance_check(df)

    observed = observed_effect(df)

    treatment_effect_check(df)

    sample_size_check(df)

    economic_check(df)

    uplift_signal_check(df)

    verdict = final_verdict(
        df,
        observed,
        balance
    )

    save_results(
        observed,
        verdict,
        df,
        balance
    )

    section("CAUSAL UPLIFT DIAGNOSTIC COMPLETE")

    print(
        f"Transactions analyzed: {len(df):,}"
    )

    print(
        f"Treatment rows:        {treatment_n:,}"
    )

    print(
        f"Control rows:          {control_n:,}"
    )

    print(
        "Observed recovery lift:",
        f"{observed['observed_recovery_lift'] * 100:+.2f}%"
    )

    print(
        "Observed incremental net value:",
        money(
            observed[
                "incremental_net_value_per_transaction"
            ]
        )
    )

    print(
        "FINAL STATUS:",
        verdict
    )


if __name__ == "__main__":
    main()