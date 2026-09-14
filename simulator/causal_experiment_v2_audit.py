"""
======================================================================
RECLAIM CAUSAL EXPERIMENT V2 AUDIT
======================================================================

Purpose:
    Validate the improved 10,000-row randomized treatment/control
    experiment BEFORE any causal uplift model is trained.

Input:
    D:\RECLAIM\data\generated\reclaim_causal_experiment_v2.csv

Output:
    D:\RECLAIM\data\generated\causal_experiment_v2_audit_summary.csv

IMPORTANT:
    This audit does NOT train an uplift model.

    It verifies:
      - experiment structure
      - treatment/control assignment
      - randomization balance
      - feature availability
      - counterfactual probabilities
      - treatment effects
      - observed outcomes
      - economic consistency
      - possible leakage
      - readiness for causal uplift modeling
======================================================================
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(r"D:\RECLAIM")

DATA_DIR = PROJECT_ROOT / "data" / "generated"

EXPERIMENT_FILE = DATA_DIR / "reclaim_causal_experiment_v2.csv"

AUDIT_OUTPUT = DATA_DIR / "causal_experiment_v2_audit_summary.csv"

EXPECTED_ROWS = 10_000

TREATMENT_MIN = 0.40
TREATMENT_MAX = 0.60

BALANCE_WARNING_THRESHOLD = 0.10

EPS = 1e-9


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def subheader(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def passed(message):
    print(f"[PASS] {message}")


def warning(message):
    print(f"[WARN] {message}")


def failed(message):
    print(f"[FAIL] {message}")


def info(message):
    print(f"[INFO] {message}")


# ======================================================================
# SAFE NUMERIC HELPERS
# ======================================================================

def numeric_series(df, column):
    if column not in df.columns:
        return None

    return pd.to_numeric(df[column], errors="coerce")


def mean_value(df, column, mask=None):
    if column not in df.columns:
        return np.nan

    series = pd.to_numeric(df[column], errors="coerce")

    if mask is not None:
        series = series.loc[mask]

    return series.mean()


def pct(value):
    if pd.isna(value):
        return "N/A"

    return f"{value * 100:.2f}%"


def rupees(value):
    if pd.isna(value):
        return "N/A"

    return f"₹{value:,.2f}"


# ======================================================================
# REQUIRED COLUMNS
# ======================================================================

REQUIRED_COLUMNS = [
    "transaction_id",
    "amount",

    "treatment",
    "control",
    "treatment_group",

    "recovery_probability",
    "calibrated_probability",
    "confidence",

    "control_recovery_probability",
    "treatment_recovery_probability",
    "incremental_recovery_probability",

    "observed_recovery_probability",

    "recovered",
    "recovery_flag",
    "recovered_amount",

    "intervention_cost",
    "customer_friction_cost",
    "net_recovered_value",

    "expected_control_recovery",
    "expected_treatment_recovery",
    "expected_incremental_recovery",
    "expected_incremental_net_value",

    "treatment_effect",
    "recovery_status",
]


# ======================================================================
# FEATURE SET
# ======================================================================

BEHAVIORAL_FEATURES = [
    "amount",
    "payment_method",
    "failure_reason",
    "historical_transaction_count",
    "historical_success_count",
    "historical_failure_count",
    "historical_success_rate",
    "hours_since_previous_transaction",
    "previous_transaction_success",
    "transactions_last_7d",
    "successful_payments_last_7d",
    "failed_payments_last_7d",
    "avg_amount_last_7d",
    "failure_rate_last_7d",
    "amount_vs_recent_average",
    "payment_momentum",
    "failure_streak",
    "hours_since_last_success",
    "hours_since_last_failure",
    "method_share",
    "recoverability_score",
    "model_confidence",
    "base_recovery_probability",
]


COUNTERFACTUAL_COLUMNS = [
    "control_recovery_probability",
    "treatment_recovery_probability",
    "incremental_recovery_probability",
    "expected_control_recovery",
    "expected_treatment_recovery",
    "expected_incremental_recovery",
    "expected_incremental_net_value",
    "treatment_effect",
]


# ======================================================================
# MAIN AUDIT
# ======================================================================

def main():

    warnings.filterwarnings("ignore")

    header("RECLAIM CAUSAL EXPERIMENT V2 AUDIT")

    print(
        "This audit validates the improved 10,000-row randomized "
        "treatment/control experiment BEFORE causal uplift modeling."
    )

    # ------------------------------------------------------------------
    # LOCATING DATA
    # ------------------------------------------------------------------

    header("LOCATING EXPERIMENT DATA")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Experiment file: {EXPERIMENT_FILE}")

    if not EXPERIMENT_FILE.exists():
        failed(f"Experiment dataset not found: {EXPERIMENT_FILE}")
        sys.exit(1)

    passed(f"Experiment dataset found: {EXPERIMENT_FILE}")

    # ------------------------------------------------------------------
    # LOADING DATA
    # ------------------------------------------------------------------

    header("LOADING DATA")

    try:
        df = pd.read_csv(EXPERIMENT_FILE)
    except Exception as exc:
        failed(f"Unable to load experiment dataset: {exc}")
        sys.exit(1)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    # ------------------------------------------------------------------
    # BASIC STRUCTURE
    # ------------------------------------------------------------------

    header("EXPERIMENT STRUCTURE")

    print("Columns:")
    print(list(df.columns))

    missing_required = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_required:
        failed(
            "Required columns missing:\n"
            + "\n".join(f"  [NO ] {c}" for c in missing_required)
        )

        print()
        print("UPLIFT READINESS: NOT READY")
        sys.exit(1)

    passed("All required experiment columns present")

    if len(df) == EXPECTED_ROWS:
        passed(f"Experiment size is exactly {EXPECTED_ROWS:,} rows")
    else:
        warning(
            f"Expected {EXPECTED_ROWS:,} rows but found {len(df):,}"
        )

    # ------------------------------------------------------------------
    # TRANSACTION IDS
    # ------------------------------------------------------------------

    header("TRANSACTION ID VALIDATION")

    transaction_ids = df["transaction_id"]

    null_ids = transaction_ids.isna().sum()

    if null_ids == 0:
        passed("Transaction IDs contain no null values")
    else:
        failed(f"Transaction IDs contain {null_ids} null values")

    duplicate_count = transaction_ids.duplicated().sum()

    if duplicate_count == 0:
        passed("Transaction IDs are unique")
    else:
        failed(
            f"Found {duplicate_count} duplicate transaction IDs"
        )

    # ------------------------------------------------------------------
    # TREATMENT / CONTROL
    # ------------------------------------------------------------------

    header("TREATMENT / CONTROL VALIDATION")

    treatment = pd.to_numeric(
        df["treatment"],
        errors="coerce"
    )

    control = pd.to_numeric(
        df["control"],
        errors="coerce"
    )

    invalid_treatment = (
        treatment.isna()
        | ~treatment.isin([0, 1])
    )

    if invalid_treatment.any():
        failed(
            f"Invalid treatment values: "
            f"{invalid_treatment.sum()}"
        )
    else:
        passed("Treatment values are valid")

    invalid_control = (
        control.isna()
        | ~control.isin([0, 1])
    )

    if invalid_control.any():
        failed(
            f"Invalid control values: "
            f"{invalid_control.sum()}"
        )
    else:
        passed("Control values are valid")

    treatment_rows = int((treatment == 1).sum())
    control_rows = int((treatment == 0).sum())

    print()
    print(f"Treatment rows: {treatment_rows:,}")
    print(f"Control rows:   {control_rows:,}")

    if treatment_rows > 0 and control_rows > 0:
        passed("Both treatment and control groups are present")
    else:
        failed("One treatment/control group is missing")

    treatment_rate = treatment.mean()

    print(f"Treatment allocation: {pct(treatment_rate)}")

    if TREATMENT_MIN <= treatment_rate <= TREATMENT_MAX:
        passed("Treatment allocation is usable")
    else:
        warning(
            "Treatment allocation is outside the recommended "
            f"{TREATMENT_MIN:.0%}-{TREATMENT_MAX:.0%} range"
        )

    # ------------------------------------------------------------------
    # TREATMENT / CONTROL MUTUAL EXCLUSIVITY
    # ------------------------------------------------------------------

    header("TREATMENT / CONTROL CONSISTENCY")

    both_groups = (
        (treatment == 1) &
        (control == 1)
    )

    neither_group = (
        (treatment == 0) &
        (control == 0)
    )

    if not both_groups.any():
        passed("No transaction belongs to both groups")
    else:
        failed(
            f"{both_groups.sum()} transactions are both treatment "
            "and control"
        )

    if not neither_group.any():
        passed("Every transaction belongs to a treatment/control group")
    else:
        failed(
            f"{neither_group.sum()} transactions belong to neither group"
        )

    # ------------------------------------------------------------------
    # TREATMENT GROUP LABEL
    # ------------------------------------------------------------------

    if "treatment_group" in df.columns:

        group_values = (
            df["treatment_group"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        print()
        print("Treatment group values:")
        print(group_values.value_counts())

        valid_group_labels = {
            "TREATMENT",
            "CONTROL",
            "1",
            "0",
            "T",
            "C",
        }

        unexpected = set(group_values.unique()) - valid_group_labels

        if not unexpected:
            passed("Treatment group labels are recognizable")
        else:
            warning(
                f"Unexpected treatment_group labels: {unexpected}"
            )

    # ------------------------------------------------------------------
    # AMOUNT VALIDATION
    # ------------------------------------------------------------------

    header("TRANSACTION ECONOMIC INPUT VALIDATION")

    amount = numeric_series(df, "amount")

    if amount.isna().any():
        failed(
            f"Amount contains {amount.isna().sum()} invalid/null values"
        )
    elif (amount < 0).any():
        failed("Negative transaction amounts detected")
    else:
        passed("Transaction amounts are valid")

    print(
        f"Mean transaction amount: {rupees(amount.mean())}"
    )

    print(
        f"Median transaction amount: {rupees(amount.median())}"
    )

    # ------------------------------------------------------------------
    # PROBABILITY VALIDATION
    # ------------------------------------------------------------------

    header("PROBABILITY VALIDATION")

    probability_columns = [
        "recovery_probability",
        "calibrated_probability",
        "control_recovery_probability",
        "treatment_recovery_probability",
        "incremental_recovery_probability",
        "observed_recovery_probability",
    ]

    probability_results = []

    for column in probability_columns:

        values = numeric_series(df, column)

        if values is None:
            failed(f"{column}: missing")
            probability_results.append(False)
            continue

        invalid = (
            values.isna()
            | (values < -EPS)
            | (values > 1 + EPS)
        )

        if invalid.any():
            failed(
                f"{column}: {invalid.sum()} invalid values"
            )
            probability_results.append(False)
        else:
            passed(f"{column}: values within [0, 1]")
            probability_results.append(True)

    # ------------------------------------------------------------------
    # COUNTERFACTUAL CONSISTENCY
    # ------------------------------------------------------------------

    header("COUNTERFACTUAL PROBABILITY CONSISTENCY")

    control_prob = numeric_series(
        df,
        "control_recovery_probability"
    )

    treatment_prob = numeric_series(
        df,
        "treatment_recovery_probability"
    )

    incremental_prob = numeric_series(
        df,
        "incremental_recovery_probability"
    )

    calculated_incremental = (
        treatment_prob - control_prob
    )

    difference = (
        incremental_prob - calculated_incremental
    ).abs()

    max_difference = difference.max()

    print(
        f"Maximum incremental probability discrepancy: "
        f"{max_difference:.10f}"
    )

    if max_difference < 1e-6:
        passed(
            "Incremental probability equals treatment minus control"
        )
    else:
        warning(
            "Incremental probability does not exactly match "
            "treatment minus control"
        )

    # ------------------------------------------------------------------
    # TREATMENT EFFECT
    # ------------------------------------------------------------------

    header("TREATMENT EFFECT VALIDATION")

    treatment_effect = numeric_series(
        df,
        "treatment_effect"
    )

    if treatment_effect.isna().any():
        failed(
            "Treatment effect contains invalid/null values"
        )
    else:
        passed("Treatment effect values are valid")

        print(
            f"Mean treatment effect: "
            f"{pct(treatment_effect.mean())}"
        )

        print(
            f"Median treatment effect: "
            f"{pct(treatment_effect.median())}"
        )

        print(
            f"Minimum treatment effect: "
            f"{pct(treatment_effect.min())}"
        )

        print(
            f"Maximum treatment effect: "
            f"{pct(treatment_effect.max())}"
        )

    # ------------------------------------------------------------------
    # TREATMENT EFFECT CONSISTENCY
    # ------------------------------------------------------------------

    effect_difference = (
        treatment_effect - incremental_prob
    ).abs()

    max_effect_difference = effect_difference.max()

    print()
    print(
        "Maximum treatment-effect discrepancy: "
        f"{max_effect_difference:.10f}"
    )

    if max_effect_difference < 1e-6:
        passed(
            "Treatment effect matches incremental probability"
        )
    else:
        warning(
            "Treatment effect does not exactly match "
            "incremental probability"
        )

    # ------------------------------------------------------------------
    # OUTCOME VALIDATION
    # ------------------------------------------------------------------

    header("OBSERVED RECOVERY OUTCOME VALIDATION")

    recovered = pd.to_numeric(
        df["recovered"],
        errors="coerce"
    )

    recovery_flag = pd.to_numeric(
        df["recovery_flag"],
        errors="coerce"
    )

    recovered_amount = numeric_series(
        df,
        "recovered_amount"
    )

    if recovered.isna().any():
        failed("Recovered contains invalid values")
    elif not recovered.isin([0, 1]).all():
        failed("Recovered must contain only 0/1")
    else:
        passed("Recovered outcome is valid")

    if recovery_flag.isna().any():
        failed("Recovery flag contains invalid values")
    elif not recovery_flag.isin([0, 1]).all():
        failed("Recovery flag must contain only 0/1")
    else:
        passed("Recovery flag is valid")

    if recovered_amount.isna().any():
        failed("Recovered amount contains invalid values")
    elif (recovered_amount < -EPS).any():
        failed("Negative recovered amounts detected")
    else:
        passed("Recovered amounts are valid")

    # ------------------------------------------------------------------
    # RECOVERY FLAG CONSISTENCY
    # ------------------------------------------------------------------

    recovery_difference = (
        recovered - recovery_flag
    ).abs()

    if recovery_difference.max() < EPS:
        passed(
            "`recovered` and `recovery_flag` are consistent"
        )
    else:
        warning(
            "`recovered` and `recovery_flag` contain inconsistencies"
        )

    # ------------------------------------------------------------------
    # RECOVERED AMOUNT CONSISTENCY
    # ------------------------------------------------------------------

    inconsistent_recovered_amount = (
        ((recovered == 0) & (recovered_amount.abs() > EPS))
        |
        ((recovered == 1) & (recovered_amount <= EPS))
    )

    if not inconsistent_recovered_amount.any():
        passed(
            "Recovered amount is consistent with recovery outcome"
        )
    else:
        warning(
            f"{inconsistent_recovered_amount.sum()} rows have "
            "potentially inconsistent recovered amounts"
        )

    # ------------------------------------------------------------------
    # ECONOMIC VARIABLES
    # ------------------------------------------------------------------

    header("ECONOMIC OUTCOME VALIDATION")

    intervention_cost = numeric_series(
        df,
        "intervention_cost"
    )

    friction_cost = numeric_series(
        df,
        "customer_friction_cost"
    )

    net_value = numeric_series(
        df,
        "net_recovered_value"
    )

    economic_columns = {
        "intervention_cost": intervention_cost,
        "customer_friction_cost": friction_cost,
        "net_recovered_value": net_value,
    }

    for name, values in economic_columns.items():

        if values.isna().any():
            failed(
                f"{name}: contains invalid/null values"
            )
        elif (values < -EPS).any():
            warning(
                f"{name}: contains negative values"
            )
        else:
            passed(f"{name}: values valid")

    # ------------------------------------------------------------------
    # TREATMENT COST CONSISTENCY
    # ------------------------------------------------------------------

    header("TREATMENT COST CONSISTENCY")

    control_cost = intervention_cost.loc[treatment == 0]
    treatment_cost = intervention_cost.loc[treatment == 1]

    print(
        f"Mean control intervention cost: "
        f"{rupees(control_cost.mean())}"
    )

    print(
        f"Mean treatment intervention cost: "
        f"{rupees(treatment_cost.mean())}"
    )

    if (control_cost.abs() <= EPS).all():
        passed("Control rows have zero intervention cost")
    else:
        warning(
            "Some control rows have non-zero intervention cost"
        )

    if (treatment_cost > EPS).all():
        passed("Treatment rows have positive intervention cost")
    else:
        warning(
            "Some treatment rows have zero/non-positive intervention cost"
        )

    # ------------------------------------------------------------------
    # NET VALUE CONSISTENCY
    # ------------------------------------------------------------------

    calculated_net = (
        recovered_amount
        - intervention_cost
        - friction_cost
    )

    net_difference = (
        net_value - calculated_net
    ).abs()

    max_net_difference = net_difference.max()

    print(
        f"Maximum net-value discrepancy: "
        f"{max_net_difference:.6f}"
    )

    if max_net_difference < 1e-4:
        passed(
            "Net recovered value is economically consistent"
        )
    else:
        warning(
            "Net recovered value does not exactly match "
            "recovered amount minus costs"
        )

    # ------------------------------------------------------------------
    # GROUP RECOVERY EFFECT
    # ------------------------------------------------------------------

    header("OBSERVED TREATMENT EFFECT")

    treatment_mask = treatment == 1
    control_mask = treatment == 0

    treatment_recovery_rate = recovered.loc[
        treatment_mask
    ].mean()

    control_recovery_rate = recovered.loc[
        control_mask
    ].mean()

    observed_recovery_lift = (
        treatment_recovery_rate
        - control_recovery_rate
    )

    print(
        f"Treatment recovery rate: "
        f"{pct(treatment_recovery_rate)}"
    )

    print(
        f"Control recovery rate:   "
        f"{pct(control_recovery_rate)}"
    )

    print(
        f"Observed recovery lift:  "
        f"{observed_recovery_lift:+.2%}"
    )

    print(
        f"Observed recovery lift:  "
        f"{observed_recovery_lift * 100:+.2f} percentage points"
    )

    # ------------------------------------------------------------------
    # ECONOMIC EFFECT
    # ------------------------------------------------------------------

    header("OBSERVED ECONOMIC EFFECT")

    treatment_net_total = net_value.loc[
        treatment_mask
    ].sum()

    control_net_total = net_value.loc[
        control_mask
    ].sum()

    treatment_net_mean = net_value.loc[
        treatment_mask
    ].mean()

    control_net_mean = net_value.loc[
        control_mask
    ].mean()

    print(
        f"Treatment net value: "
        f"{rupees(treatment_net_total)}"
    )

    print(
        f"Control net value:   "
        f"{rupees(control_net_total)}"
    )

    print(
        f"Treatment mean net value: "
        f"{rupees(treatment_net_mean)}"
    )

    print(
        f"Control mean net value:   "
        f"{rupees(control_net_mean)}"
    )

    observed_incremental_net_per_transaction = (
        treatment_net_mean
        - control_net_mean
    )

    print(
        f"Observed incremental net / transaction: "
        f"{rupees(observed_incremental_net_per_transaction)}"
    )

    if observed_incremental_net_per_transaction > 0:
        passed(
            "Treatment produces positive observed incremental "
            "economic value"
        )
    else:
        warning(
            "Treatment does NOT produce positive observed "
            "incremental economic value"
        )

    # ------------------------------------------------------------------
    # EXPECTED ECONOMIC EFFECT
    # ------------------------------------------------------------------

    header("EXPECTED COUNTERFACTUAL ECONOMIC EFFECT")

    expected_incremental_net = numeric_series(
        df,
        "expected_incremental_net_value"
    )

    expected_incremental_mean = (
        expected_incremental_net.mean()
    )

    print(
        f"Mean expected incremental net value: "
        f"{rupees(expected_incremental_mean)}"
    )

    if expected_incremental_mean > 0:
        passed(
            "Expected incremental economic value is positive"
        )
    else:
        warning(
            "Expected incremental economic value is not positive"
        )

    # ------------------------------------------------------------------
    # EXPECTED VS OBSERVED
    # ------------------------------------------------------------------

    header("EXPECTED VS OBSERVED ECONOMIC EFFECT")

    print(
        f"Expected incremental net / transaction: "
        f"{rupees(expected_incremental_mean)}"
    )

    print(
        f"Observed incremental net / transaction: "
        f"{rupees(observed_incremental_net_per_transaction)}"
    )

    expected_observed_gap = (
        observed_incremental_net_per_transaction
        - expected_incremental_mean
    )

    print(
        f"Observed - expected gap: "
        f"{rupees(expected_observed_gap)}"
    )

    if expected_incremental_mean != 0:

        relative_gap = abs(
            expected_observed_gap
        ) / abs(expected_incremental_mean)

        print(
            f"Relative expected/observed gap: "
            f"{relative_gap:.2%}"
        )

        if relative_gap <= 0.35:
            passed(
                "Observed economic effect is reasonably close "
                "to expected counterfactual effect"
            )
        else:
            warning(
                "Observed economic effect differs substantially "
                "from expected counterfactual effect"
            )

    # ------------------------------------------------------------------
    # RANDOMIZATION BALANCE
    # ------------------------------------------------------------------

    header("RANDOMIZATION BALANCE CHECK")

    balance_features = [
        "amount",
        "recovery_probability",
        "calibrated_probability",
        "confidence",
        "recoverability_score",
        "historical_success_rate",
        "failure_rate_last_7d",
        "payment_momentum",
        "failure_streak",
    ]

    balance_records = []

    available_balance_features = []

    for feature in balance_features:

        if feature not in df.columns:
            continue

        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        if values.notna().sum() == 0:
            continue

        available_balance_features.append(feature)

        t_mean = values.loc[treatment_mask].mean()
        c_mean = values.loc[control_mask].mean()

        pooled_std = values.std()

        if pooled_std > EPS:
            standardized_difference = (
                t_mean - c_mean
            ) / pooled_std
        else:
            standardized_difference = 0.0

        balance_records.append({
            "feature": feature,
            "treatment_mean": t_mean,
            "control_mean": c_mean,
            "standardized_difference":
                standardized_difference,
            "absolute_standardized_difference":
                abs(standardized_difference),
        })

        print(
            f"{feature:<32}"
            f"T={t_mean:>12.4f} "
            f"C={c_mean:>12.4f} "
            f"SMD={standardized_difference:+.4f}"
        )

    if balance_records:

        balance_df = pd.DataFrame(balance_records)

        max_smd = balance_df[
            "absolute_standardized_difference"
        ].max()

        print()
        print(
            f"Maximum absolute standardized difference: "
            f"{max_smd:.4f}"
        )

        if max_smd < BALANCE_WARNING_THRESHOLD:
            passed(
                "Randomization balance is strong across "
                "available numeric features"
            )
        else:
            warning(
                "Some features show noticeable treatment/control imbalance"
            )

    # ------------------------------------------------------------------
    # FEATURE AVAILABILITY
    # ------------------------------------------------------------------

    header("FEATURE AVAILABILITY FOR UPLIFT MODELING")

    available_features = []
    missing_features = []

    for feature in BEHAVIORAL_FEATURES:

        if feature in df.columns:
            non_null = df[feature].notna().sum()

            if non_null > 0:
                available_features.append(feature)
                print(
                    f"  [YES] {feature:<40}"
                    f"{non_null:,} non-null"
                )
            else:
                missing_features.append(feature)
                print(
                    f"  [NO ] {feature:<40}"
                    f"all values null"
                )
        else:
            missing_features.append(feature)
            print(
                f"  [NO ] {feature}"
            )

    print()
    print(
        f"Useful features available: {len(available_features)}"
    )

    print(
        f"Useful features missing:    {len(missing_features)}"
    )

    if len(available_features) >= 10:
        passed(
            "Feature-rich experiment available for uplift modeling"
        )
    elif len(available_features) >= 5:
        warning(
            "Only moderate feature coverage is available"
        )
    else:
        warning(
            "Very few modeling features are available"
        )

    # ------------------------------------------------------------------
    # COUNTERFACTUAL COLUMNS
    # ------------------------------------------------------------------

    header("COUNTERFACTUAL DATA AVAILABILITY")

    counterfactual_missing = []

    for column in COUNTERFACTUAL_COLUMNS:

        if column in df.columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            valid_count = values.notna().sum()

            if valid_count == len(df):
                print(
                    f"  [YES] {column:<40}"
                    f"{valid_count:,}/{len(df):,}"
                )
            else:
                print(
                    f"  [WARN] {column:<40}"
                    f"{valid_count:,}/{len(df):,}"
                )

                counterfactual_missing.append(column)

        else:
            print(
                f"  [NO ] {column}"
            )

            counterfactual_missing.append(column)

    if not counterfactual_missing:
        passed(
            "All counterfactual/effect variables available"
        )
    else:
        warning(
            "Some counterfactual variables are missing or incomplete"
        )

    # ------------------------------------------------------------------
    # POTENTIAL LEAKAGE CHECK
    # ------------------------------------------------------------------

    header("POTENTIAL OUTCOME LEAKAGE CHECK")

    leakage_candidates = [
        "recovered",
        "recovery_flag",
        "recovered_amount",
        "observed_recovery_probability",
        "recovery_status",
        "net_recovered_value",
    ]

    leakage_present = []

    for column in leakage_candidates:

        if column in df.columns:
            leakage_present.append(column)

            print(
                f"  [INFO] Outcome-derived column present: {column}"
            )

    if leakage_present:
        warning(
            "Outcome-derived columns are present in the dataset."
        )

        warning(
            "They MUST NOT be used as input features when training "
            "the uplift model."
        )

    else:
        passed(
            "No obvious outcome-derived columns detected"
        )

    # ------------------------------------------------------------------
    # TREATMENT EFFECT HETEROGENEITY
    # ------------------------------------------------------------------

    header("TREATMENT EFFECT HETEROGENEITY")

    effect_std = treatment_effect.std()

    positive_effect_rows = (
        treatment_effect > 0
    ).sum()

    negative_effect_rows = (
        treatment_effect < 0
    ).sum()

    zero_effect_rows = (
        treatment_effect.abs() <= EPS
    ).sum()

    print(
        f"Treatment effect mean:   {pct(treatment_effect.mean())}"
    )

    print(
        f"Treatment effect std dev: "
        f"{pct(effect_std)}"
    )

    print(
        f"Positive effect rows:     {positive_effect_rows:,}"
    )

    print(
        f"Negative effect rows:     {negative_effect_rows:,}"
    )

    print(
        f"Zero effect rows:         {zero_effect_rows:,}"
    )

    if effect_std > EPS:
        passed(
            "Treatment effect is heterogeneous"
        )
    else:
        warning(
            "Treatment effect is effectively constant"
        )

    # ------------------------------------------------------------------
    # TREATMENT EFFECT / OBSERVED LIFT COMPARISON
    # ------------------------------------------------------------------

    header("TREATMENT EFFECT VS OBSERVED LIFT")

    mean_generated_effect = treatment_effect.mean()

    print(
        f"Mean generated treatment effect: "
        f"{pct(mean_generated_effect)}"
    )

    print(
        f"Observed recovery lift:           "
        f"{observed_recovery_lift:+.2%}"
    )

    effect_gap = (
        observed_recovery_lift
        - mean_generated_effect
    )

    print(
        f"Observed - generated lift:        "
        f"{effect_gap:+.2%}"
    )

    # ------------------------------------------------------------------
    # EXPERIMENT SIZE
    # ------------------------------------------------------------------

    header("STATISTICAL SAMPLE SIZE CHECK")

    print(
        f"Total sample size: {len(df):,}"
    )

    print(
        f"Treatment sample:  {treatment_rows:,}"
    )

    print(
        f"Control sample:    {control_rows:,}"
    )

    if len(df) >= 5000:
        passed(
            "Experiment has a substantially larger sample than "
            "the original 581-row experiment"
        )
    elif len(df) >= 1000:
        passed(
            "Experiment has a usable sample size"
        )
    else:
        warning(
            "Experiment sample size may be insufficient for "
            "stable heterogeneous treatment-effect estimation"
        )

    # ------------------------------------------------------------------
    # READINESS DECISION
    # ------------------------------------------------------------------

    header("FINAL CAUSAL EXPERIMENT V2 READINESS DECISION")

    readiness_checks = []

    # Required columns
    readiness_checks.append(
        len(missing_required) == 0
    )

    # Unique IDs
    readiness_checks.append(
        duplicate_count == 0
    )

    # Treatment validity
    readiness_checks.append(
        not invalid_treatment.any()
    )

    # Both groups
    readiness_checks.append(
        treatment_rows > 0 and control_rows > 0
    )

    # Allocation
    readiness_checks.append(
        TREATMENT_MIN <= treatment_rate <= TREATMENT_MAX
    )

    # Outcomes
    readiness_checks.append(
        not recovered.isna().any()
    )

    # Counterfactual probability
    readiness_checks.append(
        max_difference < 1e-5
    )

    # Economic data
    readiness_checks.append(
        not intervention_cost.isna().any()
        and not friction_cost.isna().any()
        and not net_value.isna().any()
    )

    # Features
    readiness_checks.append(
        len(available_features) >= 5
    )

    # Counterfactuals
    readiness_checks.append(
        len(counterfactual_missing) == 0
    )

    # Sample size
    readiness_checks.append(
        len(df) >= 5000
    )

    passed_checks = sum(readiness_checks)
    total_checks = len(readiness_checks)

    print()
    print(
        f"Readiness checks passed: "
        f"{passed_checks}/{total_checks}"
    )

    print()

    critical_failures = []

    if missing_required:
        critical_failures.append(
            "Required columns missing"
        )

    if duplicate_count > 0:
        critical_failures.append(
            "Duplicate transaction IDs"
        )

    if invalid_treatment.any():
        critical_failures.append(
            "Invalid treatment assignment"
        )

    if treatment_rows == 0 or control_rows == 0:
        critical_failures.append(
            "Missing treatment/control group"
        )

    if max_difference >= 1e-5:
        critical_failures.append(
            "Counterfactual probability inconsistency"
        )

    if len(available_features) < 5:
        critical_failures.append(
            "Too few modeling features"
        )

    if len(counterfactual_missing) > 0:
        critical_failures.append(
            "Incomplete counterfactual variables"
        )

    if critical_failures:

        warning(
            "Critical experiment-quality issues detected:"
        )

        for failure in critical_failures:
            print(f"  - {failure}")

        print()
        print(
            "UPLIFT READINESS: NOT READY"
        )

        final_status = "NOT_READY"

    else:

        passed(
            "Treatment/control experiment structure is valid"
        )

        passed(
            "Experiment is sufficiently large for causal modeling"
        )

        passed(
            "Feature coverage is sufficient"
        )

        passed(
            "Counterfactual variables are available"
        )

        passed(
            "Treatment/control groups are usable"
        )

        print()
        print(
            "The V2 experiment is suitable for the next stage:"
        )

        print(
            "  causal uplift modeling + held-out policy evaluation"
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "The uplift model MUST NOT use outcome-derived columns "
            "as input features."
        )

        print(
            "The final model should be evaluated on held-out data "
            "before any deployment decision."
        )

        print()
        print(
            "UPLIFT READINESS: READY"
        )

        final_status = "READY"

    # ------------------------------------------------------------------
    # BUILD AUDIT SUMMARY
    # ------------------------------------------------------------------

    header("BUILDING AUDIT SUMMARY")

    summary_rows = [

        ("experiment_file", str(EXPERIMENT_FILE)),
        ("rows", len(df)),
        ("columns", len(df.columns)),

        ("treatment_rows", treatment_rows),
        ("control_rows", control_rows),
        ("treatment_rate", treatment_rate),

        (
            "treatment_recovery_rate",
            treatment_recovery_rate
        ),

        (
            "control_recovery_rate",
            control_recovery_rate
        ),

        (
            "observed_recovery_lift",
            observed_recovery_lift
        ),

        (
            "mean_treatment_effect",
            mean_generated_effect
        ),

        (
            "treatment_effect_std",
            effect_std
        ),

        (
            "expected_incremental_net_value_mean",
            expected_incremental_mean
        ),

        (
            "observed_incremental_net_value_per_transaction",
            observed_incremental_net_per_transaction
        ),

        (
            "expected_observed_economic_gap",
            expected_observed_gap
        ),

        (
            "features_available",
            len(available_features)
        ),

        (
            "features_missing",
            len(missing_features)
        ),

        (
            "counterfactual_columns_available",
            len(COUNTERFACTUAL_COLUMNS)
            - len(counterfactual_missing)
        ),

        (
            "max_probability_difference",
            max_difference
        ),

        (
            "max_net_value_difference",
            max_net_difference
        ),

        (
            "duplicate_transaction_ids",
            duplicate_count
        ),

        (
            "positive_effect_rows",
            positive_effect_rows
        ),

        (
            "negative_effect_rows",
            negative_effect_rows
        ),

        (
            "zero_effect_rows",
            zero_effect_rows
        ),

        (
            "readiness_checks_passed",
            passed_checks
        ),

        (
            "readiness_checks_total",
            total_checks
        ),

        (
            "final_status",
            final_status
        ),
    ]

    summary_df = pd.DataFrame(
        summary_rows,
        columns=["metric", "value"]
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_df.to_csv(
        AUDIT_OUTPUT,
        index=False
    )

    passed(
        f"Saved audit summary: {AUDIT_OUTPUT}"
    )

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------

    header("CAUSAL EXPERIMENT V2 AUDIT COMPLETE")

    print(
        f"Experiment rows:             {len(df):,}"
    )

    print(
        f"Treatment rows:               {treatment_rows:,}"
    )

    print(
        f"Control rows:                 {control_rows:,}"
    )

    print(
        f"Treatment allocation:         {pct(treatment_rate)}"
    )

    print(
        f"Observed recovery lift:       "
        f"{observed_recovery_lift:+.2%}"
    )

    print(
        f"Mean treatment effect:        "
        f"{mean_generated_effect:+.2%}"
    )

    print(
        f"Expected incremental net:     "
        f"{rupees(expected_incremental_mean)}"
    )

    print(
        f"Observed incremental net:     "
        f"{rupees(observed_incremental_net_per_transaction)}"
    )

    print(
        f"Features available:           "
        f"{len(available_features)}"
    )

    print(
        f"Counterfactual variables:     "
        f"{len(COUNTERFACTUAL_COLUMNS) - len(counterfactual_missing)}"
    )

    print()
    print(
        f"UPLIFT READINESS: {final_status}"
    )

    if final_status == "READY":

        print()
        print(
            "NEXT STEP:"
        )

        print(
            "Build the V2 causal uplift model using the "
            "behavioral features."
        )

        print(
            "Then evaluate it on held-out transactions using:"
        )

        print(
            "  - uplift/QINI"
        )

        print(
            "  - treatment effect calibration"
        )

        print(
            "  - economic targeting value"
        )

        print(
            "  - comparison against RETRY_ALL"
        )

        print(
            "  - comparison against NO_INTERVENTION"
        )

        print(
            "  - repeated held-out validation"
        )

    else:

        print()
        print(
            "NEXT STEP:"
        )

        print(
            "Fix the experiment-quality issues above "
            "before training the V2 uplift model."
        )

    print()
    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()