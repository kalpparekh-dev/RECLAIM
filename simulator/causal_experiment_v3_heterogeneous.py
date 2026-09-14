"""
RECLAIM CAUSAL EXPERIMENT V3
Heterogeneous treatment-effect experiment redesign.

Purpose:
- Preserve randomized treatment/control assignment.
- Create realistic heterogeneous treatment effects.
- Include positive, near-zero, and negative treatment effects.
- Generate observed outcomes AFTER treatment assignment.
- Preserve the existing V2 experiment structure where possible.

Input:
    D:\RECLAIM\data\generated\reclaim_causal_experiment_v2.csv

Output:
    D:\RECLAIM\data\generated\reclaim_causal_experiment_v3.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "generated" / "reclaim_causal_experiment_v2.csv"
OUTPUT_FILE = ROOT / "data" / "generated" / "reclaim_causal_experiment_v3.csv"

SEED = 4242
rng = np.random.default_rng(SEED)

N_EXPECTED = 10_000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def zscore(series):
    values = pd.to_numeric(series, errors="coerce").fillna(series.median()).to_numpy(float)
    std = np.std(values)

    if std < 1e-12:
        return np.zeros(len(values))

    return (values - np.mean(values)) / std


def minmax(series):
    values = pd.to_numeric(series, errors="coerce").fillna(series.median()).to_numpy(float)

    lo = np.min(values)
    hi = np.max(values)

    if hi - lo < 1e-12:
        return np.full(len(values), 0.5)

    return (values - lo) / (hi - lo)


print("=" * 70)
print("RECLAIM CAUSAL EXPERIMENT V3")
print("HETEROGENEOUS TREATMENT-EFFECT REDESIGN")
print("=" * 70)

print("\nLoading V2 experiment...")
print(f"File: {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print(f"[PASS] Rows loaded: {len(df):,}")
print(f"[PASS] Columns loaded: {len(df.columns)}")

if len(df) != N_EXPECTED:
    raise ValueError(f"Expected {N_EXPECTED} rows, found {len(df)}")


# ---------------------------------------------------------------------
# RANDOMIZED ASSIGNMENT
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("CREATING RANDOMIZED TREATMENT / CONTROL ASSIGNMENT")
print("=" * 70)

# Important:
# Treatment is assigned independently of the generated outcome.

assignment_rng = np.random.default_rng(SEED)

treatment = assignment_rng.binomial(
    1,
    0.50,
    size=len(df)
)

df["treatment"] = treatment
df["control"] = 1 - treatment

df["treatment_group"] = np.where(
    treatment == 1,
    "TREATMENT",
    "CONTROL"
)

df["treatment_assignment_before_outcome"] = True

print(f"Treatment rows: {int(treatment.sum()):,}")
print(f"Control rows:   {int((1 - treatment).sum()):,}")


# ---------------------------------------------------------------------
# BASE CONTROL PROBABILITY
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("BUILDING BASE CONTROL PROBABILITY")
print("=" * 70)

if "control_recovery_probability" in df.columns:
    control_probability = pd.to_numeric(
        df["control_recovery_probability"],
        errors="coerce"
    ).fillna(0.5).to_numpy(float)
else:
    control_probability = pd.to_numeric(
        df["base_recovery_probability"],
        errors="coerce"
    ).fillna(0.5).to_numpy(float)

control_probability = np.clip(
    control_probability,
    0.05,
    0.95
)


# ---------------------------------------------------------------------
# HETEROGENEOUS TREATMENT EFFECT
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("GENERATING HETEROGENEOUS TREATMENT EFFECT")
print("=" * 70)

# Behavioral signals.
#
# These are all PRE-TREATMENT features and therefore safe for causal
# modeling.

amount_z = zscore(df["amount"])

failure_rate_z = zscore(
    df["failure_rate_last_7d"]
)

failure_streak_z = zscore(
    df["failure_streak"]
)

momentum_z = zscore(
    df["payment_momentum"]
)

success_rate_z = zscore(
    df["historical_success_rate"]
)

recoverability_z = zscore(
    df["recoverability_score"]
)

method_share_z = zscore(
    df["method_share"]
)

confidence_z = zscore(
    df["model_confidence"]
)


# ---------------------------------------------------------------------
# SEGMENT SCORE
# ---------------------------------------------------------------------

# The goal is NOT to make every customer benefit.
#
# Some behavioral profiles receive a strong treatment benefit.
# Some receive a weak benefit.
# Some receive approximately zero benefit.
# Some receive negative economic impact.

segment_score = (
    0.28 * recoverability_z
    - 0.22 * failure_rate_z
    + 0.20 * momentum_z
    + 0.14 * success_rate_z
    - 0.10 * failure_streak_z
    + 0.08 * confidence_z
    + 0.05 * method_share_z
    + 0.04 * amount_z
)


# Non-linear component creates genuine subgroup structure.
heterogeneity_signal = np.tanh(segment_score)


# ---------------------------------------------------------------------
# RAW EFFECT
# ---------------------------------------------------------------------

# Base effect is approximately 5-6 percentage points.
#
# Heterogeneity can push the effect below zero or substantially above
# the average.

raw_effect = (
    0.056
    + 0.060 * heterogeneity_signal
)


# Add a small amount of structured noise.
# This avoids making treatment effect perfectly deterministic.

noise = rng.normal(
    loc=0.0,
    scale=0.006,
    size=len(df)
)

raw_effect += noise


# ---------------------------------------------------------------------
# CENTER EFFECT AROUND ~5.6%
# ---------------------------------------------------------------------

# Centering preserves the overall experiment-level treatment lift while
# introducing subgroup heterogeneity.

raw_effect -= np.mean(raw_effect)
raw_effect += 0.056


# Keep treatment effects economically realistic.

treatment_effect = np.clip(
    raw_effect,
    -0.08,
    0.18
)


# Make sure treatment probability remains valid.

treatment_probability = np.clip(
    control_probability + treatment_effect,
    0.01,
    0.99
)

# Actual effective treatment effect after probability clipping.
treatment_effect = treatment_probability - control_probability


df["control_recovery_probability"] = control_probability
df["treatment_recovery_probability"] = treatment_probability
df["incremental_recovery_probability"] = treatment_effect
df["treatment_effect"] = treatment_effect


print(
    f"Mean treatment effect: "
    f"{np.mean(treatment_effect) * 100:.2f}%"
)

print(
    f"Std treatment effect: "
    f"{np.std(treatment_effect) * 100:.2f}%"
)

print(
    f"Minimum treatment effect: "
    f"{np.min(treatment_effect) * 100:.2f}%"
)

print(
    f"Maximum treatment effect: "
    f"{np.max(treatment_effect) * 100:.2f}%"
)

positive = np.sum(treatment_effect > 0)
negative = np.sum(treatment_effect < 0)
zero = np.sum(np.isclose(treatment_effect, 0, atol=1e-6))

print(f"Positive-effect rows: {positive:,}")
print(f"Negative-effect rows: {negative:,}")
print(f"Near-zero rows:       {zero:,}")


# ---------------------------------------------------------------------
# OBSERVED OUTCOME
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("GENERATING OBSERVED RECOVERY OUTCOMES")
print("=" * 70)

# IMPORTANT:
# Outcome is generated AFTER treatment assignment.

observed_probability = np.where(
    treatment == 1,
    treatment_probability,
    control_probability
)

observed_probability = np.clip(
    observed_probability,
    0.01,
    0.99
)

recovered = rng.binomial(
    1,
    observed_probability
)

df["observed_recovery_probability"] = observed_probability
df["recovered"] = recovered
df["recovery_flag"] = recovered


# ---------------------------------------------------------------------
# RECOVERED AMOUNT
# ---------------------------------------------------------------------

amount = pd.to_numeric(
    df["amount"],
    errors="coerce"
).fillna(0.0).to_numpy(float)

df["recovered_amount"] = np.where(
    recovered == 1,
    amount,
    0.0
)


# ---------------------------------------------------------------------
# COST MODEL
# ---------------------------------------------------------------------

# Preserve V2 treatment cost if available.

if "intervention_cost" in df.columns:
    base_cost = pd.to_numeric(
        df["intervention_cost"],
        errors="coerce"
    ).fillna(1.5).to_numpy(float)
else:
    base_cost = np.full(
        len(df),
        1.5
    )

# Treatment incurs intervention cost.
intervention_cost = np.where(
    treatment == 1,
    np.maximum(base_cost, 1.0),
    0.0
)

# Small friction cost for intervention.
customer_friction_cost = np.where(
    treatment == 1,
    np.minimum(
        0.005 * amount,
        25.0
    ),
    0.0
)

df["intervention_cost"] = intervention_cost
df["customer_friction_cost"] = customer_friction_cost


# ---------------------------------------------------------------------
# OBSERVED NET VALUE
# ---------------------------------------------------------------------

df["net_recovered_value"] = (
    df["recovered_amount"]
    - df["intervention_cost"]
    - df["customer_friction_cost"]
)

df["recovery_status"] = np.where(
    recovered == 1,
    "RECOVERED",
    "NOT_RECOVERED"
)


# ---------------------------------------------------------------------
# EXPECTED COUNTERFACTUAL ECONOMICS
# ---------------------------------------------------------------------

expected_control_recovery = (
    control_probability * amount
)

expected_treatment_recovery = (
    treatment_probability * amount
)

expected_control_net = (
    expected_control_recovery
)

expected_treatment_net = (
    expected_treatment_recovery
    - np.maximum(base_cost, 1.0)
    - np.minimum(
        0.005 * amount,
        25.0
    )
)

expected_incremental_net = (
    expected_treatment_net
    - expected_control_net
)

df["expected_control_recovery"] = expected_control_recovery
df["expected_treatment_recovery"] = expected_treatment_recovery
df["expected_incremental_recovery"] = (
    expected_treatment_recovery
    - expected_control_recovery
)

df["expected_incremental_net_value"] = expected_incremental_net


# ---------------------------------------------------------------------
# OBSERVED ECONOMIC CHECK
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("OBSERVED EXPERIMENT RESULTS")
print("=" * 70)

treatment_mask = df["treatment"] == 1
control_mask = df["treatment"] == 0

treatment_recovery_rate = (
    df.loc[treatment_mask, "recovered"].mean()
)

control_recovery_rate = (
    df.loc[control_mask, "recovered"].mean()
)

print(
    f"Treatment recovery rate: "
    f"{treatment_recovery_rate * 100:.2f}%"
)

print(
    f"Control recovery rate:   "
    f"{control_recovery_rate * 100:.2f}%"
)

print(
    f"Observed recovery lift:  "
    f"{(treatment_recovery_rate - control_recovery_rate) * 100:+.2f}%"
)

print(
    f"Treatment mean net:      "
    f"₹{df.loc[treatment_mask, 'net_recovered_value'].mean():,.2f}"
)

print(
    f"Control mean net:        "
    f"₹{df.loc[control_mask, 'net_recovered_value'].mean():,.2f}"
)


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("V3 HETEROGENEITY VALIDATION")
print("=" * 70)

if positive == 0:
    raise RuntimeError(
        "V3 failed: no positive treatment-effect rows."
    )

if negative == 0:
    raise RuntimeError(
        "V3 failed: no negative treatment-effect rows."
    )

if np.std(treatment_effect) < 0.01:
    raise RuntimeError(
        "V3 failed: treatment-effect heterogeneity is too weak."
    )

if not np.all((df["control_recovery_probability"] >= 0) &
              (df["control_recovery_probability"] <= 1)):
    raise RuntimeError(
        "Invalid control probabilities."
    )

if not np.all((df["treatment_recovery_probability"] >= 0) &
              (df["treatment_recovery_probability"] <= 1)):
    raise RuntimeError(
        "Invalid treatment probabilities."
    )

if not np.allclose(
    df["incremental_recovery_probability"],
    df["treatment_recovery_probability"]
    - df["control_recovery_probability"],
    atol=1e-10
):
    raise RuntimeError(
        "Treatment effect consistency failed."
    )

print("[PASS] Positive treatment-effect rows exist")
print("[PASS] Negative treatment-effect rows exist")
print("[PASS] Treatment effect is heterogeneous")
print("[PASS] Probabilities valid")
print("[PASS] Counterfactual consistency valid")


# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("SAVING V3 EXPERIMENT")
print("=" * 70)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"[PASS] Saved: {OUTPUT_FILE}")

print("\n" + "=" * 70)
print("CAUSAL EXPERIMENT V3 COMPLETE")
print("=" * 70)

print(f"Rows:                 {len(df):,}")
print(f"Treatment:             {treatment.sum():,}")
print(f"Control:               {(1 - treatment).sum():,}")
print(
    f"Mean treatment effect: "
    f"{np.mean(treatment_effect) * 100:.2f}%"
)
print(
    f"Effect std dev:        "
    f"{np.std(treatment_effect) * 100:.2f}%"
)
print(
    f"Positive effects:      "
    f"{positive:,}"
)
print(
    f"Negative effects:      "
    f"{negative:,}"
)

print("\nNEXT STEP:")
print("Run:")
print("python simulator/causal_uplift_optimizer_v7.py")