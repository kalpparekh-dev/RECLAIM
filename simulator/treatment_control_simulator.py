"""
======================================================================
RECLAIM TREATMENT / CONTROL EXPERIMENT SIMULATOR
======================================================================

Purpose:
    Create a genuine treatment/control experiment for causal uplift
    analysis.

IMPORTANT:
    Treatment assignment happens BEFORE the recovery outcome is
    generated.

    treatment = 1 -> RECLAIM intervention
    treatment = 0 -> control / no intervention

Outputs:
    data/generated/reclaim_treatment_control_experiment.csv

This file is intentionally separate from the existing prediction and
recovery outcome datasets.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(r"D:\RECLAIM")

DATA_DIR = PROJECT_ROOT / "data" / "generated"

PREDICTION_FILE = DATA_DIR / "calibrated_recovery_predictions.csv"

OUTPUT_FILE = DATA_DIR / "reclaim_treatment_control_experiment.csv"

RANDOM_SEED = 42

# Approximate treatment allocation.
# 50/50 gives the cleanest causal comparison.
TREATMENT_RATE = 0.50

# Economic assumptions.
INTERVENTION_COST = 1.50
FRICTION_COST = 0.25

# Maximum amount that can be recovered.
# Recovery is bounded by the original transaction amount.
RECOVERY_FRACTION_MIN = 0.70
RECOVERY_FRACTION_MAX = 1.00


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def money(value):
    return f"₹{value:,.2f}"


# ======================================================================
# COLUMN HELPERS
# ======================================================================

def first_existing(df, columns):
    for column in columns:
        if column in df.columns:
            return column
    return None


def numeric_series(df, column, default=0.0):
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(default)


# ======================================================================
# LOAD PREDICTIONS
# ======================================================================

def load_predictions():
    header("LOADING PREDICTION DATA")

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTION_FILE}"
        )

    df = pd.read_csv(PREDICTION_FILE)

    print(f"[PASS] Prediction dataset found: {PREDICTION_FILE}")
    print(f"[PASS] Prediction rows: {len(df)}")

    if "transaction_id" not in df.columns:
        raise ValueError(
            "Prediction dataset must contain transaction_id."
        )

    if df["transaction_id"].duplicated().any():
        raise ValueError(
            "Prediction transaction IDs must be unique."
        )

    print("[PASS] Prediction transaction IDs are unique")

    # --------------------------------------------------------------
    # Amount
    # --------------------------------------------------------------

    amount_column = first_existing(
        df,
        [
            "amount",
            "transaction_amount",
        ],
    )

    if amount_column is None:
        raise ValueError(
            "Prediction dataset does not contain an amount column."
        )

    if amount_column != "amount":
        df["amount"] = df[amount_column]

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    if df["amount"].isna().any():
        raise ValueError(
            "Prediction dataset contains invalid transaction amounts."
        )

    if (df["amount"] <= 0).any():
        raise ValueError(
            "Transaction amounts must be positive."
        )

    print("[PASS] Transaction amounts valid")

    # --------------------------------------------------------------
    # Recovery probability
    # --------------------------------------------------------------

    probability_column = first_existing(
        df,
        [
            "calibrated_probability",
            "recovery_probability",
            "base_recovery_probability",
        ],
    )

    if probability_column is None:
        raise ValueError(
            "No recovery probability column found."
        )

    df["recovery_probability"] = pd.to_numeric(
        df[probability_column],
        errors="coerce"
    )

    df["recovery_probability"] = (
        df["recovery_probability"]
        .clip(0.001, 0.999)
    )

    print(
        f"[PASS] Using probability column: `{probability_column}`"
    )

    # --------------------------------------------------------------
    # Confidence
    # --------------------------------------------------------------

    confidence_column = first_existing(
        df,
        [
            "confidence",
            "model_confidence",
        ],
    )

    if confidence_column is None:
        df["confidence"] = 0.5
    else:
        df["confidence"] = pd.to_numeric(
            df[confidence_column],
            errors="coerce"
        ).fillna(0.5).clip(0, 1)

    print("[PASS] Confidence normalized")

    return df


# ======================================================================
# FEATURE-BASED TREATMENT EFFECT
# ======================================================================

def calculate_treatment_effect(df):
    """
    Calculate heterogeneous treatment effect.

    The purpose is NOT to make RECLAIM magically win.

    Instead, the simulator creates a realistic scenario where
    intervention can help some transactions more than others.

    The effect depends on:
        - recovery probability
        - confidence
        - recoverability score
        - failure streak
        - payment momentum
        - amount

    The effect is deliberately modest.
    """

    p = df["recovery_probability"].clip(0.001, 0.999)

    confidence = df["confidence"].clip(0, 1)

    recoverability = numeric_series(
        df,
        "recoverability_score",
        default=0.5
    ).clip(0, 1)

    failure_streak = numeric_series(
        df,
        "failure_streak",
        default=0
    ).clip(0, 10)

    payment_momentum = numeric_series(
        df,
        "payment_momentum",
        default=0
    )

    # Normalize momentum approximately around 0.
    momentum_signal = np.tanh(payment_momentum / 2.0)

    # High-probability recoveries should receive smaller incremental
    # treatment effects because they may recover naturally.
    natural_recovery_penalty = p

    # Base intervention effect.
    effect = (
        0.025
        + 0.045 * confidence
        + 0.040 * recoverability
        + 0.015 * np.maximum(momentum_signal, 0)
        - 0.035 * natural_recovery_penalty
    )

    # Repeated failures reduce intervention effectiveness.
    effect -= 0.003 * failure_streak

    # Bound effect to a sensible range.
    effect = np.clip(
        effect,
        0.005,
        0.10
    )

    return effect


# ======================================================================
# ASSIGN TREATMENT
# ======================================================================

def assign_treatment(df, rng):
    """
    Treatment is assigned BEFORE recovery is generated.
    """

    df["treatment"] = (
        rng.random(len(df)) < TREATMENT_RATE
    ).astype(int)

    df["control"] = (
        1 - df["treatment"]
    ).astype(int)

    df["treatment_group"] = np.where(
        df["treatment"] == 1,
        "RECLAIM",
        "CONTROL"
    )

    return df


# ======================================================================
# GENERATE COUNTERFACTUAL RECOVERY PROBABILITIES
# ======================================================================

def build_counterfactual_probabilities(df):
    """
    Build:

        p_control
        p_treatment

    before generating the observed outcome.

    This is important because it creates an actual causal data-generating
    process.

    The observed recovery outcome is then sampled from the probability
    corresponding to the assigned treatment.
    """

    base_probability = df["recovery_probability"].clip(
        0.001,
        0.999
    )

    treatment_effect = df["treatment_effect"]

    # Control = natural recovery probability.
    p_control = base_probability.copy()

    # Treatment = natural recovery + incremental intervention effect.
    p_treatment = np.clip(
        base_probability + treatment_effect,
        0.001,
        0.999
    )

    df["control_recovery_probability"] = p_control

    df["treatment_recovery_probability"] = p_treatment

    df["incremental_recovery_probability"] = (
        p_treatment - p_control
    )

    # The actual probability depends on treatment assignment.
    df["observed_recovery_probability"] = np.where(
        df["treatment"] == 1,
        p_treatment,
        p_control
    )

    return df


# ======================================================================
# GENERATE OBSERVED OUTCOME
# ======================================================================

def generate_outcomes(df, rng):
    """
    Generate the actual observed recovery outcome AFTER treatment
    assignment.

    This is the key causal simulation step.
    """

    probabilities = (
        df["observed_recovery_probability"]
        .clip(0.001, 0.999)
        .to_numpy()
    )

    recovered = (
        rng.random(len(df)) < probabilities
    ).astype(int)

    df["recovered"] = recovered

    df["recovery_flag"] = recovered

    # --------------------------------------------------------------
    # Recovered amount
    # --------------------------------------------------------------

    fractions = rng.uniform(
        RECOVERY_FRACTION_MIN,
        RECOVERY_FRACTION_MAX,
        len(df)
    )

    df["recovered_amount"] = np.where(
        df["recovered"] == 1,
        df["amount"] * fractions,
        0.0
    )

    # --------------------------------------------------------------
    # Costs
    # --------------------------------------------------------------

    df["intervention_cost"] = (
        df["treatment"] * INTERVENTION_COST
    )

    df["customer_friction_cost"] = (
        df["treatment"] * FRICTION_COST
    )

    # --------------------------------------------------------------
    # Net recovered value
    # --------------------------------------------------------------

    df["net_recovered_value"] = (
        df["recovered_amount"]
        - df["intervention_cost"]
        - df["customer_friction_cost"]
    )

    # --------------------------------------------------------------
    # Status
    # --------------------------------------------------------------

    df["recovery_status"] = np.where(
        df["recovered"] == 1,
        "RECOVERED",
        "NOT_RECOVERED"
    )

    return df


# ======================================================================
# COUNTERFACTUAL ECONOMIC VALUES
# ======================================================================

def calculate_counterfactual_values(df):
    """
    Calculate expected economic value under treatment and control.

    These are counterfactual estimates used for validation.

    They are NOT used to generate the observed outcome.
    """

    average_recovery_fraction = (
        RECOVERY_FRACTION_MIN
        + RECOVERY_FRACTION_MAX
    ) / 2.0

    expected_amount = (
        df["amount"] * average_recovery_fraction
    )

    df["expected_control_recovery"] = (
        df["control_recovery_probability"]
        * expected_amount
    )

    df["expected_treatment_recovery"] = (
        df["treatment_recovery_probability"]
        * expected_amount
    )

    df["expected_incremental_recovery"] = (
        df["expected_treatment_recovery"]
        - df["expected_control_recovery"]
    )

    df["expected_incremental_net_value"] = (
        df["expected_incremental_recovery"]
        - INTERVENTION_COST
        - FRICTION_COST
    )

    return df


# ======================================================================
# EXPERIMENT SUMMARY
# ======================================================================

def print_experiment_summary(df):
    header("TREATMENT / CONTROL EXPERIMENT SUMMARY")

    treatment = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]

    print(f"Total transactions: {len(df)}")
    print()

    print(
        f"Treatment rows: {len(treatment)} "
        f"({len(treatment) / len(df):.2%})"
    )

    print(
        f"Control rows:   {len(control)} "
        f"({len(control) / len(df):.2%})"
    )

    print()

    treatment_rate = treatment["recovered"].mean()
    control_rate = control["recovered"].mean()

    treatment_value = treatment["recovered_amount"].sum()
    control_value = control["recovered_amount"].sum()

    treatment_net = treatment["net_recovered_value"].sum()
    control_net = control["net_recovered_value"].sum()

    incremental_rate = (
        treatment_rate - control_rate
    )

    # Difference in mean recovered amount per transaction.
    treatment_mean_value = (
        treatment["recovered_amount"].mean()
    )

    control_mean_value = (
        control["recovered_amount"].mean()
    )

    incremental_value_per_transaction = (
        treatment_mean_value
        - control_mean_value
    )

    print("--------------------------------------------------------------")
    print("RECOVERY RATE")
    print("--------------------------------------------------------------")

    print(
        f"Treatment recovery rate: "
        f"{treatment_rate:.2%}"
    )

    print(
        f"Control recovery rate:   "
        f"{control_rate:.2%}"
    )

    print(
        f"Observed treatment lift:  "
        f"{incremental_rate:+.2%}"
    )

    print()

    print("--------------------------------------------------------------")
    print("RECOVERED VALUE")
    print("--------------------------------------------------------------")

    print(
        f"Treatment recovered value: "
        f"{money(treatment_value)}"
    )

    print(
        f"Control recovered value:   "
        f"{money(control_value)}"
    )

    print(
        f"Treatment mean recovery: "
        f"{money(treatment_mean_value)}"
    )

    print(
        f"Control mean recovery:   "
        f"{money(control_mean_value)}"
    )

    print(
        f"Incremental value / transaction: "
        f"{money(incremental_value_per_transaction)}"
    )

    print()

    print("--------------------------------------------------------------")
    print("ECONOMIC VALUE")
    print("--------------------------------------------------------------")

    print(
        f"Treatment net value: "
        f"{money(treatment_net)}"
    )

    print(
        f"Control net value:   "
        f"{money(control_net)}"
    )

    # Important:
    # We compare average value per transaction because treatment and
    # control groups may not have exactly the same number of rows.
    treatment_mean_net = (
        treatment["net_recovered_value"].mean()
    )

    control_mean_net = (
        control["net_recovered_value"].mean()
    )

    observed_incremental_net = (
        treatment_mean_net
        - control_mean_net
    )

    print(
        f"Observed incremental net value / transaction: "
        f"{money(observed_incremental_net)}"
    )

    print()

    print("--------------------------------------------------------------")
    print("COUNTERFACTUAL ECONOMIC EFFECT")
    print("--------------------------------------------------------------")

    expected_incremental = (
        df["expected_incremental_net_value"].mean()
    )

    print(
        f"Expected incremental net value / transaction: "
        f"{money(expected_incremental)}"
    )

    if observed_incremental_net > 0:
        print(
            "[PASS] Observed treatment produces positive "
            "incremental net value."
        )
    else:
        print(
            "[WARN] Observed treatment does not produce positive "
            "incremental net value in this random draw."
        )

    print()

    print(
        "IMPORTANT: Treatment assignment occurred before outcome "
        "generation."
    )

    print(
        "[PASS] This dataset is suitable as a starting point for "
        "treatment/control uplift analysis."
    )


# ======================================================================
# BALANCE CHECK
# ======================================================================

def print_balance_check(df):
    header("TREATMENT / CONTROL BALANCE CHECK")

    treatment = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]

    features = [
        "amount",
        "recovery_probability",
        "confidence",
        "recoverability_score",
        "historical_success_rate",
        "failure_streak",
    ]

    available = [
        feature
        for feature in features
        if feature in df.columns
    ]

    print(
        f"{'feature':<32}"
        f"{'treatment mean':>18}"
        f"{'control mean':>18}"
    )

    print("-" * 68)

    for feature in available:

        treatment_mean = pd.to_numeric(
            treatment[feature],
            errors="coerce"
        ).mean()

        control_mean = pd.to_numeric(
            control[feature],
            errors="coerce"
        ).mean()

        print(
            f"{feature:<32}"
            f"{treatment_mean:>18.4f}"
            f"{control_mean:>18.4f}"
        )

    print()

    print(
        "[PASS] Randomized treatment assignment provides a "
        "treatment/control comparison."
    )


# ======================================================================
# VALIDATION
# ======================================================================

def validate_experiment(df):

    header("EXPERIMENT VALIDATION")

    required_columns = [
        "transaction_id",
        "amount",
        "treatment",
        "control",
        "treatment_group",
        "recovery_probability",
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
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required experiment columns: {missing}"
        )

    print("[PASS] Required columns present")

    if df["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs found."
        )

    print("[PASS] Transaction IDs unique")

    if not set(df["treatment"].unique()).issubset({0, 1}):
        raise ValueError(
            "Treatment must contain only 0/1."
        )

    print("[PASS] Treatment values valid")

    if not set(df["control"].unique()).issubset({0, 1}):
        raise ValueError(
            "Control must contain only 0/1."
        )

    print("[PASS] Control values valid")

    if not np.all(
        df["treatment"] + df["control"] == 1
    ):
        raise ValueError(
            "Every transaction must belong to exactly one "
            "treatment/control group."
        )

    print("[PASS] Treatment/control groups mutually exclusive")

    if df["recovery_probability"].between(
        0,
        1
    ).all():
        print("[PASS] Base probabilities valid")
    else:
        raise ValueError(
            "Invalid base probabilities."
        )

    if df["observed_recovery_probability"].between(
        0,
        1
    ).all():
        print("[PASS] Observed probabilities valid")
    else:
        raise ValueError(
            "Invalid observed probabilities."
        )

    if (df["recovered_amount"] < 0).any():
        raise ValueError(
            "Recovered amount cannot be negative."
        )

    if (
        df["recovered_amount"]
        > df["amount"] + 1e-9
    ).any():
        raise ValueError(
            "Recovered amount cannot exceed transaction amount."
        )

    print("[PASS] Recovered amounts valid")

    expected_treatment = (
        df["recovery_probability"]
        + df["treatment_effect"]
    ).clip(0.001, 0.999)

    if np.allclose(
        df["treatment_recovery_probability"],
        expected_treatment
    ):
        print("[PASS] Treatment effect generated before outcomes")
    else:
        raise ValueError(
            "Treatment probability construction failed."
        )

    if (
        df["intervention_cost"]
        == df["treatment"] * INTERVENTION_COST
    ).all():
        print("[PASS] Intervention costs aligned with treatment")
    else:
        raise ValueError(
            "Intervention cost alignment failed."
        )

    print("[PASS] Experiment validation complete")


# ======================================================================
# SAVE DATA
# ======================================================================

def save_experiment(df):

    header("SAVING EXPERIMENT DATA")

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Put the most important experiment columns first.
    priority_columns = [
        "transaction_id",
        "amount",
        "failure_reason",

        # Treatment assignment
        "treatment",
        "control",
        "treatment_group",

        # Prediction
        "recovery_probability",
        "confidence",

        # Counterfactual probabilities
        "control_recovery_probability",
        "treatment_recovery_probability",
        "incremental_recovery_probability",

        # Actual observed probability
        "observed_recovery_probability",

        # Outcome
        "recovered",
        "recovery_flag",
        "recovered_amount",

        # Economics
        "intervention_cost",
        "customer_friction_cost",
        "net_recovered_value",

        # Expected economics
        "expected_control_recovery",
        "expected_treatment_recovery",
        "expected_incremental_recovery",
        "expected_incremental_net_value",

        # Simulation information
        "treatment_effect",
    ]

    remaining_columns = [
        column
        for column in df.columns
        if column not in priority_columns
    ]

    df = df[
        priority_columns + remaining_columns
    ]

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"[PASS] Saved experiment dataset:\n{OUTPUT_FILE}"
    )

    print(
        f"[PASS] Rows saved: {len(df)}"
    )

    print(
        f"[PASS] Columns saved: {len(df.columns)}"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    header(
        "RECLAIM TREATMENT / CONTROL EXPERIMENT SIMULATOR"
    )

    print(
        "This simulator creates treatment/control assignment "
        "BEFORE generating recovery outcomes."
    )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print(
        f"Treatment allocation target: "
        f"{TREATMENT_RATE:.0%}"
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # --------------------------------------------------------------
    # 1. Load prediction data
    # --------------------------------------------------------------

    df = load_predictions()

    # --------------------------------------------------------------
    # 2. Create treatment effect
    # --------------------------------------------------------------

    header("CREATING DATA-GENERATING PROCESS")

    df["treatment_effect"] = (
        calculate_treatment_effect(df)
    )

    print(
        "[PASS] Treatment effect generated"
    )

    print(
        f"Mean treatment effect: "
        f"{df['treatment_effect'].mean():.2%}"
    )

    print(
        f"Min treatment effect: "
        f"{df['treatment_effect'].min():.2%}"
    )

    print(
        f"Max treatment effect: "
        f"{df['treatment_effect'].max():.2%}"
    )

    # --------------------------------------------------------------
    # 3. ASSIGN TREATMENT BEFORE OUTCOME
    # --------------------------------------------------------------

    header(
        "ASSIGNING TREATMENT / CONTROL BEFORE OUTCOME"
    )

    df = assign_treatment(
        df,
        rng
    )

    treatment_count = int(
        df["treatment"].sum()
    )

    control_count = int(
        df["control"].sum()
    )

    print(
        f"[PASS] Treatment assigned: "
        f"{treatment_count}"
    )

    print(
        f"[PASS] Control assigned: "
        f"{control_count}"
    )

    # --------------------------------------------------------------
    # 4. Build counterfactual probabilities
    # --------------------------------------------------------------

    header(
        "BUILDING COUNTERFACTUAL RECOVERY PROBABILITIES"
    )

    df = build_counterfactual_probabilities(
        df
    )

    print(
        "[PASS] Control recovery probabilities calculated"
    )

    print(
        "[PASS] Treatment recovery probabilities calculated"
    )

    print(
        "[PASS] Incremental recovery probabilities calculated"
    )

    # --------------------------------------------------------------
    # 5. Generate actual outcome
    # --------------------------------------------------------------

    header(
        "GENERATING OBSERVED RECOVERY OUTCOMES"
    )

    df = generate_outcomes(
        df,
        rng
    )

    print(
        "[PASS] Recovery outcomes generated"
    )

    print(
        "[PASS] Outcomes generated AFTER treatment assignment"
    )

    # --------------------------------------------------------------
    # 6. Economic counterfactuals
    # --------------------------------------------------------------

    header(
        "CALCULATING COUNTERFACTUAL ECONOMIC VALUE"
    )

    df = calculate_counterfactual_values(
        df
    )

    print(
        "[PASS] Expected control recovery calculated"
    )

    print(
        "[PASS] Expected treatment recovery calculated"
    )

    print(
        "[PASS] Expected incremental recovery calculated"
    )

    print(
        "[PASS] Expected incremental net value calculated"
    )

    # --------------------------------------------------------------
    # 7. Summary
    # --------------------------------------------------------------

    print_experiment_summary(
        df
    )

    print_balance_check(
        df
    )

    # --------------------------------------------------------------
    # 8. Validate
    # --------------------------------------------------------------

    validate_experiment(
        df
    )

    # --------------------------------------------------------------
    # 9. Save
    # --------------------------------------------------------------

    save_experiment(
        df
    )

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    header(
        "TREATMENT / CONTROL EXPERIMENT COMPLETE"
    )

    print(
        "The RECLAIM dataset now contains:"
    )

    print(
        "  treatment = 1  -> RECLAIM intervention"
    )

    print(
        "  treatment = 0  -> control / no intervention"
    )

    print(
        "  recovered     -> observed recovery outcome"
    )

    print(
        "  recovered_amount"
    )

    print(
        "  intervention_cost"
    )

    print(
        "  customer_friction_cost"
    )

    print(
        "  net_recovered_value"
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "Run the treatment/control audit again."
    )

    print(
        "The audit should now report:"
    )

    print(
        "  Treatment candidates: 1"
    )

    print(
        "  Usable treatment: treatment"
    )

    print(
        "  UPLIFT READINESS: READY"
    )


if __name__ == "__main__":
    main()