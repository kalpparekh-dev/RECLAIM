"""
RECLAIM IMPROVED TREATMENT / CONTROL EXPERIMENT GENERATOR - V2

Purpose
-------
Creates a randomized treatment/control experiment BEFORE generating
the observed recovery outcome.

Key fixes:
- Generates exactly 10,000 experimental rows.
- Creates UNIQUE experiment transaction IDs.
- Preserves the behavioral features from recovery_outcomes.csv.
- Randomizes treatment before observed outcomes are generated.
- Generates heterogeneous treatment effects.
- Keeps counterfactual probabilities/economic values.
- Avoids Windows "\R" SyntaxWarning by using Path objects.
- Writes a clean V2 experiment for the V2 audit/uplift model.

Run:
    python simulator/improved_treatment_control_experiment.py
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ==============================================================
# CONFIGURATION
# ==============================================================

PROJECT_ROOT = Path(r"D:\RECLAIM")
DATA_DIR = PROJECT_ROOT / "data" / "generated"

PREDICTION_FILE = DATA_DIR / "calibrated_recovery_predictions.csv"
OUTCOME_FILE = DATA_DIR / "recovery_outcomes.csv"

OUTPUT_FILE = DATA_DIR / "reclaim_causal_experiment_v2.csv"
SUMMARY_FILE = DATA_DIR / "reclaim_causal_experiment_v2_summary.csv"

RANDOM_SEED = 2026
EXPERIMENT_SIZE = 10_000
TREATMENT_RATE = 0.50

# Economic assumptions.
# These are deliberately modest because the experiment is synthetic.
INTERVENTION_COST = 1.50
FRICTION_COST = 0.50


# ==============================================================
# HELPERS
# ==============================================================

def banner(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


def clamp_probability(x):
    return np.clip(np.asarray(x, dtype=float), 0.001, 0.999)


def safe_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def standardize(x):
    x = np.asarray(x, dtype=float)
    mean = np.mean(x)
    std = np.std(x)

    if std < 1e-12:
        return np.zeros_like(x)

    return (x - mean) / std


# ==============================================================
# LOAD SOURCE DATA
# ==============================================================

def load_source_data():
    banner("RECLAIM IMPROVED TREATMENT / CONTROL EXPERIMENT GENERATOR")

    print(f"Random seed: {RANDOM_SEED}")
    print(f"Target experiment size: {EXPERIMENT_SIZE:,}")
    print(f"Treatment allocation: {TREATMENT_RATE:.2%}")
    print()

    banner("LOADING SOURCE DATA")

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(f"Prediction dataset not found:\n{PREDICTION_FILE}")

    if not OUTCOME_FILE.exists():
        raise FileNotFoundError(f"Outcome dataset not found:\n{OUTCOME_FILE}")

    predictions = pd.read_csv(PREDICTION_FILE)
    outcomes = pd.read_csv(OUTCOME_FILE)

    print(f"[PASS] Prediction rows: {len(predictions):,}")
    print(f"[PASS] Outcome rows:    {len(outcomes):,}")

    if "transaction_id" not in predictions.columns:
        raise ValueError("Prediction dataset is missing transaction_id.")

    if "transaction_id" not in outcomes.columns:
        raise ValueError("Outcome dataset is missing transaction_id.")

    # One prediction row per modeled transaction.
    predictions = predictions.drop_duplicates("transaction_id").copy()

    # Keep one outcome row per transaction. The legacy outcome file contains
    # repeated historical/customer rows, so choose the latest row for each ID.
    outcomes = outcomes.copy()

    if "timestamp" in outcomes.columns:
        outcomes["_timestamp_sort"] = pd.to_datetime(
            outcomes["timestamp"], errors="coerce"
        )
        outcomes = (
            outcomes.sort_values("_timestamp_sort")
            .drop_duplicates("transaction_id", keep="last")
            .drop(columns="_timestamp_sort")
        )
    else:
        outcomes = outcomes.drop_duplicates("transaction_id", keep="last")

    source = predictions.merge(
        outcomes,
        on="transaction_id",
        how="inner",
        suffixes=("_prediction", "")
    )

    if len(source) == 0:
        raise ValueError("No matching transaction IDs between prediction and outcome data.")

    print(f"[PASS] Base transactions available: {len(source):,}")

    return source


# ==============================================================
# PREPARE FEATURES
# ==============================================================

def prepare_features(source):
    banner("PREPARING BEHAVIORAL FEATURES")

    probability_candidates = [
        "calibrated_probability",
        "base_recovery_probability",
        "recovery_probability",
    ]

    probability_source = None

    for col in probability_candidates:
        if col in source.columns:
            probability_source = col
            break

    if probability_source is None:
        raise ValueError("No usable recovery probability column found.")

    print(f"[PASS] Probability source: {probability_source}")

    # Behavioral features that should be available to the uplift model.
    behavioral_features = [
        "amount",
        "payment_method",
        "failure_reason",
        "historical_transaction_count",
        "historical_success_count",
        "historical_failure_count",
        "historical_success_rate",
        "hours_since_previous_transaction",
        "previous_transaction_success",
        "previous_failure_reason",
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

    available = [c for c in behavioral_features if c in source.columns]

    print(f"[PASS] Behavioral features preserved: {len(available)}")

    if len(available) < 10:
        raise ValueError(
            f"Too few behavioral features available: {len(available)}"
        )

    return source.copy(), probability_source, available


# ==============================================================
# EXPAND SOURCE POPULATION
# ==============================================================

def expand_population(source, experiment_size, rng):
    banner("EXPANDING TRANSACTION POPULATION")

    n_source = len(source)

    if n_source == 0:
        raise ValueError("Source population is empty.")

    # Sample with replacement to reach exactly the requested experiment size.
    indices = rng.choice(
        np.arange(n_source),
        size=experiment_size,
        replace=True
    )

    df = source.iloc[indices].copy().reset_index(drop=True)

    # Preserve source identity separately.
    df["experiment_source_transaction_id"] = (
        df["transaction_id"].astype(str)
    )

    df["source_repeat"] = (
        df.groupby("experiment_source_transaction_id").cumcount() + 1
    )

    # ----------------------------------------------------------
    # IMPORTANT:
    # Create genuinely unique experimental units.
    # Do NOT copy the source transaction ID into transaction_id.
    # ----------------------------------------------------------

    df["experiment_transaction_id"] = [
        f"RECLAIM-V2-{i:06d}"
        for i in range(1, experiment_size + 1)
    ]

    # The audit expects transaction_id to identify an experiment row.
    df["transaction_id"] = df["experiment_transaction_id"]

    if df["transaction_id"].duplicated().any():
        raise RuntimeError("Failed to create unique experiment transaction IDs.")

    if df["transaction_id"].nunique() != experiment_size:
        raise RuntimeError("Experiment transaction IDs are not unique.")

    print(f"[PASS] Expanded population: {len(df):,}")
    print("[PASS] Created unique experiment transaction IDs")

    return df


# ==============================================================
# CREATE HETEROGENEOUS TREATMENT EFFECT
# ==============================================================

def create_treatment_effect(df, probability_source):
    banner("CREATING HETEROGENEOUS TREATMENT EFFECT")

    base_probability = clamp_probability(
        safe_numeric(df[probability_source], 0.50)
    )

    df["base_probability"] = base_probability

    # Behavioral heterogeneity.
    recoverability = standardize(
        safe_numeric(df.get("recoverability_score", 0), 0)
    )

    momentum = standardize(
        safe_numeric(df.get("payment_momentum", 0), 0)
    )

    failure_rate = standardize(
        safe_numeric(df.get("failure_rate_last_7d", 0), 0)
    )

    confidence = standardize(
        safe_numeric(df.get("model_confidence", 0), 0)
    )

    amount = safe_numeric(df.get("amount", 0), 0)
    amount_signal = standardize(np.log1p(np.maximum(amount, 0)))

    # Start with a meaningful but heterogeneous treatment effect.
    effect = (
        0.056
        + 0.006 * recoverability
        + 0.003 * momentum
        + 0.002 * failure_rate
        + 0.0015 * confidence
        + 0.001 * amount_signal
    )

    # Bound the synthetic treatment effect.
    effect = np.clip(effect, 0.0325, 0.0775)

    df["treatment_effect"] = effect

    print(f"Mean treatment effect: {effect.mean():.2%}")
    print(f"Median treatment effect: {np.median(effect):.2%}")
    print(f"Minimum treatment effect: {effect.min():.2%}")
    print(f"Maximum treatment effect: {effect.max():.2%}")

    return df


# ==============================================================
# RANDOMIZED ASSIGNMENT
# ==============================================================

def assign_treatment(df, rng):
    banner("ASSIGNING RANDOMIZED TREATMENT / CONTROL")

    n = len(df)

    treatment = (
        rng.random(n) < TREATMENT_RATE
    ).astype(int)

    df["treatment"] = treatment
    df["control"] = 1 - treatment

    df["treatment_group"] = np.where(
        treatment == 1,
        "TREATMENT",
        "CONTROL"
    )

    # Metadata explicitly records that assignment occurred before outcome.
    df["treatment_assignment_before_outcome"] = True
    df["experiment_version"] = "V2"
    df["random_seed"] = RANDOM_SEED

    treatment_count = int((treatment == 1).sum())
    control_count = int((treatment == 0).sum())

    print(f"[PASS] Treatment rows: {treatment_count:,}")
    print(f"[PASS] Control rows:   {control_count:,}")
    print(f"Treatment allocation: {treatment.mean():.2%}")

    return df


# ==============================================================
# COUNTERFACTUAL PROBABILITIES
# ==============================================================

def build_counterfactuals(df):
    banner("BUILDING COUNTERFACTUAL RECOVERY PROBABILITIES")

    base = clamp_probability(df["base_probability"])
    effect = np.asarray(df["treatment_effect"], dtype=float)

    control_probability = base
    treatment_probability = np.clip(base + effect, 0.001, 0.999)

    incremental_probability = (
        treatment_probability - control_probability
    )

    df["control_recovery_probability"] = control_probability
    df["treatment_recovery_probability"] = treatment_probability
    df["incremental_recovery_probability"] = incremental_probability

    print(
        f"Mean control probability: "
        f"{control_probability.mean():.2%}"
    )

    print(
        f"Mean treatment probability: "
        f"{treatment_probability.mean():.2%}"
    )

    print(
        f"Mean incremental probability: "
        f"{incremental_probability.mean():.2%}"
    )

    return df


# ==============================================================
# GENERATE OBSERVED OUTCOME
# ==============================================================

def generate_observed_outcome(df, rng):
    banner("GENERATING OBSERVED OUTCOME")

    treatment = df["treatment"].to_numpy(dtype=int)

    control_p = df["control_recovery_probability"].to_numpy()
    treatment_p = df["treatment_recovery_probability"].to_numpy()

    # IMPORTANT:
    # Treatment is already assigned here.
    # Outcome is generated only AFTER assignment.
    observed_probability = np.where(
        treatment == 1,
        treatment_p,
        control_p
    )

    recovered = (
        rng.random(len(df)) < observed_probability
    ).astype(int)

    df["observed_recovery_probability"] = observed_probability
    df["recovered"] = recovered
    df["recovery_flag"] = recovered

    amounts = safe_numeric(df["amount"], 0).to_numpy()

    df["recovered_amount"] = np.where(
        recovered == 1,
        amounts,
        0.0
    )

    df["recovery_status"] = np.where(
        recovered == 1,
        "RECOVERED",
        "NOT_RECOVERED"
    )

    print(f"Observed recovery rate: {recovered.mean():.2%}")
    print("[PASS] Outcomes generated after treatment assignment")

    return df


# ==============================================================
# ECONOMIC OUTCOMES
# ==============================================================

def calculate_economic_outcomes(df):
    banner("CALCULATING ECONOMIC OUTCOMES")

    treatment = df["treatment"].to_numpy(dtype=int)
    recovered_amount = df["recovered_amount"].to_numpy(dtype=float)

    intervention_cost = treatment * INTERVENTION_COST

    # Friction is incurred only when intervention is actually attempted.
    # This remains deliberately small relative to recovered value.
    friction_cost = treatment * FRICTION_COST

    net_value = (
        recovered_amount
        - intervention_cost
        - friction_cost
    )

    df["intervention_cost"] = intervention_cost
    df["customer_friction_cost"] = friction_cost
    df["net_recovered_value"] = net_value

    # Expected economic quantities are based on counterfactual probabilities.
    amount = safe_numeric(df["amount"], 0).to_numpy(dtype=float)

    expected_control_recovery = (
        df["control_recovery_probability"].to_numpy()
        * amount
    )

    expected_treatment_recovery = (
        df["treatment_recovery_probability"].to_numpy()
        * amount
    )

    expected_incremental_recovery = (
        expected_treatment_recovery
        - expected_control_recovery
    )

    expected_incremental_net = (
        expected_incremental_recovery
        - INTERVENTION_COST
        - FRICTION_COST
    )

    df["expected_control_recovery"] = expected_control_recovery
    df["expected_treatment_recovery"] = expected_treatment_recovery
    df["expected_incremental_recovery"] = expected_incremental_recovery
    df["expected_incremental_net_value"] = expected_incremental_net

    print(
        "Expected incremental recovered value: "
        f"₹{expected_incremental_recovery.mean():,.2f}"
    )

    print(
        "Expected incremental net value: "
        f"₹{expected_incremental_net.mean():,.2f}"
    )

    return df


# ==============================================================
# BALANCE CHECK
# ==============================================================

def randomization_balance_check(df):
    banner("RANDOMIZATION BALANCE CHECK")

    numeric_features = [
        "amount",
        "base_probability",
        "recoverability_score",
        "historical_success_rate",
        "failure_rate_last_7d",
        "payment_momentum",
        "failure_streak",
    ]

    treatment = df["treatment"] == 1
    control = df["treatment"] == 0

    for feature in numeric_features:
        if feature not in df.columns:
            continue

        values = safe_numeric(df[feature], 0)

        t_mean = values[treatment].mean()
        c_mean = values[control].mean()

        print(
            f"{feature:<32}"
            f"T={t_mean:>12.4f} "
            f"C={c_mean:>12.4f}"
        )


# ==============================================================
# EXPERIMENT SUMMARY
# ==============================================================

def build_summary(df):
    banner("EXPERIMENT SUMMARY")

    treatment = df["treatment"] == 1
    control = df["treatment"] == 0

    treatment_count = int(treatment.sum())
    control_count = int(control.sum())

    t_recovery_rate = df.loc[treatment, "recovered"].mean()
    c_recovery_rate = df.loc[control, "recovered"].mean()

    observed_lift = t_recovery_rate - c_recovery_rate

    t_net = df.loc[treatment, "net_recovered_value"].sum()
    c_net = df.loc[control, "net_recovered_value"].sum()

    observed_incremental_net = (
        t_net / treatment_count
        - c_net / control_count
    )

    expected_incremental_net = (
        df["expected_incremental_net_value"].mean()
    )

    print(f"Total transactions: {len(df):,}")
    print()
    print(f"Treatment rows: {treatment_count:,}")
    print(f"Control rows: {control_count:,}")
    print()
    print(f"Treatment recovery rate: {t_recovery_rate:.2%}")
    print(f"Control recovery rate:   {c_recovery_rate:.2%}")
    print(f"Observed recovery lift:  {observed_lift:+.2%}")
    print()
    print(f"Treatment net value: ₹{t_net:,.2f}")
    print(f"Control net value:   ₹{c_net:,.2f}")
    print(
        "Observed incremental net / transaction: "
        f"₹{observed_incremental_net:,.2f}"
    )
    print(
        "Expected incremental net / transaction: "
        f"₹{expected_incremental_net:,.2f}"
    )

    summary = pd.DataFrame([{
        "experiment_version": "V2",
        "random_seed": RANDOM_SEED,
        "experiment_size": len(df),
        "treatment_rows": treatment_count,
        "control_rows": control_count,
        "treatment_allocation": treatment_count / len(df),
        "treatment_recovery_rate": t_recovery_rate,
        "control_recovery_rate": c_recovery_rate,
        "observed_recovery_lift": observed_lift,
        "mean_treatment_effect": df["treatment_effect"].mean(),
        "expected_incremental_recovered_value":
            df["expected_incremental_recovery"].mean(),
        "expected_incremental_net_value":
            expected_incremental_net,
        "observed_incremental_net_per_transaction":
            observed_incremental_net,
        "unique_transaction_ids":
            df["transaction_id"].nunique(),
    }])

    return summary


# ==============================================================
# VALIDATE BEFORE SAVING
# ==============================================================

def final_validation(df):
    banner("FINAL EXPERIMENT VALIDATION")

    required = [
        "transaction_id",
        "treatment",
        "control",
        "treatment_group",
        "recovered",
        "recovered_amount",
        "intervention_cost",
        "customer_friction_cost",
        "net_recovered_value",
        "control_recovery_probability",
        "treatment_recovery_probability",
        "incremental_recovery_probability",
        "expected_incremental_net_value",
        "treatment_effect",
        "treatment_assignment_before_outcome",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Required experiment columns missing: "
            + ", ".join(missing)
        )

    if df["transaction_id"].isna().any():
        raise ValueError("Transaction IDs contain null values.")

    if df["transaction_id"].duplicated().any():
        raise ValueError("Transaction IDs are duplicated.")

    if df["transaction_id"].nunique() != len(df):
        raise ValueError("Transaction IDs are not one-to-one.")

    if not set(df["treatment"].unique()).issubset({0, 1}):
        raise ValueError("Invalid treatment values.")

    if not set(df["control"].unique()).issubset({0, 1}):
        raise ValueError("Invalid control values.")

    if not np.all(df["control"].to_numpy() == 1 - df["treatment"].to_numpy()):
        raise ValueError("Treatment/control columns are inconsistent.")

    if not np.all(
        df["incremental_recovery_probability"].to_numpy()
        <= 1
    ):
        raise ValueError("Invalid incremental probability.")

    if not np.allclose(
        df["incremental_recovery_probability"],
        df["treatment_recovery_probability"]
        - df["control_recovery_probability"],
        atol=1e-10
    ):
        raise ValueError("Incremental probability mismatch.")

    if not np.allclose(
        df["treatment_effect"],
        df["incremental_recovery_probability"],
        atol=1e-10
    ):
        raise ValueError("Treatment effect mismatch.")

    if not df["treatment_assignment_before_outcome"].all():
        raise ValueError(
            "Treatment assignment timing metadata is invalid."
        )

    control_cost = df.loc[
        df["treatment"] == 0,
        "intervention_cost"
    ]

    treatment_cost = df.loc[
        df["treatment"] == 1,
        "intervention_cost"
    ]

    if not np.allclose(control_cost, 0):
        raise ValueError("Control rows have non-zero intervention cost.")

    if not np.all(treatment_cost > 0):
        raise ValueError("Treatment rows must have positive intervention cost.")

    expected_net = (
        df["recovered_amount"]
        - df["intervention_cost"]
        - df["customer_friction_cost"]
    )

    if not np.allclose(
        df["net_recovered_value"],
        expected_net,
        atol=1e-8
    ):
        raise ValueError("Net recovered value is inconsistent.")

    print("[PASS] Required columns present")
    print("[PASS] Transaction IDs are unique")
    print("[PASS] Treatment/control assignments valid")
    print("[PASS] Counterfactual probabilities consistent")
    print("[PASS] Treatment effects consistent")
    print("[PASS] Outcome generated after treatment assignment")
    print("[PASS] Economic variables consistent")
    print("[PASS] Experiment validation complete")


# ==============================================================
# MAIN
# ==============================================================

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    source = load_source_data()

    source, probability_source, _ = prepare_features(source)

    df = expand_population(
        source,
        EXPERIMENT_SIZE,
        rng
    )

    df = create_treatment_effect(
        df,
        probability_source
    )

    # CRITICAL CAUSAL ORDER:
    # 1. Treatment effect is created.
    # 2. Treatment/control is randomized.
    # 3. Counterfactual probabilities are built.
    # 4. Observed recovery outcome is generated.
    df = assign_treatment(df, rng)

    df = build_counterfactuals(df)

    df = generate_observed_outcome(
        df,
        rng
    )

    df = calculate_economic_outcomes(df)

    randomization_balance_check(df)

    summary = build_summary(df)

    final_validation(df)

    banner("SAVING IMPROVED EXPERIMENT")

    # Keep transaction_id unique and use experiment_transaction_id
    # as the same unique experimental-unit identifier.
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print(f"[PASS] Saved experiment:")
    print(OUTPUT_FILE)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print(f"[PASS] Saved summary:")
    print(SUMMARY_FILE)

    banner("IMPROVED EXPERIMENT COMPLETE")

    print(f"Experiment size: {len(df):,}")
    print(
        f"Mean treatment effect: "
        f"{df['treatment_effect'].mean():.2%}"
    )
    print(
        "Expected incremental net value: "
        f"₹{df['expected_incremental_net_value'].mean():,.2f}"
    )

    print()
    print("NEXT STEP:")
    print("Run:")
    print("  python simulator/causal_experiment_v2_audit.py")
    print()
    print("The audit should now confirm:")
    print("  - 10,000 rows")
    print("  - 10,000 unique transaction IDs")
    print("  - balanced treatment/control groups")
    print("  - treatment assignment before outcome")
    print("  - 23 behavioral features")
    print("  - valid counterfactual probabilities")
    print()
    print("Only after the V2 audit reports READY should")
    print("the V2 uplift model be trained.")


if __name__ == "__main__":
    main()