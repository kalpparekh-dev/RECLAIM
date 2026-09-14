
"""
======================================================================
RECLAIM CAUSAL UPLIFT OPTIMIZER V8
======================================================================

FAST ORACLE-GATED ECONOMIC POLICY TEST

Designed specifically for:
    reclaim_causal_experiment_v3.csv

Goals:
    1. Robust heterogeneous treatment-effect validation
    2. Held-out uplift modeling
    3. Economic policy evaluation
    4. Oracle benchmark
    5. RETRY_ALL comparison
    6. Repeated validation
    7. NumPy 2.x compatibility

IMPORTANT:
This script does NOT modify the experiment data.

It evaluates whether selective intervention targeting can outperform
the RETRY_ALL policy on held-out data.

======================================================================
"""

from pathlib import Path

import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# =====================================================================
# CONFIGURATION
# =====================================================================

DATA_FILE = Path(
    r"D:\RECLAIM\data\generated\reclaim_causal_experiment_v3.csv"
)

OUTPUT_DIR = Path(
    r"D:\RECLAIM\data\generated"
)

VALIDATION_RUNS = 20
TEST_SIZE = 0.30

RF_ESTIMATORS = 300
RF_MIN_LEAF = 20

BASE_SEED = 800

# Policy thresholds to test.
# 0 means intervene when predicted incremental recovery > 0.
# Additional thresholds test whether more conservative targeting
# can outperform RETRY_ALL.
UPLIFT_THRESHOLDS = [
    -0.020,
    -0.010,
    0.000,
    0.005,
    0.010,
    0.020,
    0.030,
    0.040,
    0.050,
]

# Optional economic thresholds in rupees.
ECONOMIC_THRESHOLDS = [
    -100.0,
    -50.0,
    0.0,
    25.0,
    50.0,
    100.0,
]


# =====================================================================
# COLUMN DEFINITIONS
# =====================================================================

TREATMENT_COL = "treatment"
EFFECT_COL = "treatment_effect"
RECOVERY_COL = "recovered"

CONTROL_VALUE_COL = "expected_net_recovery"
INCREMENTAL_VALUE_COL = "expected_incremental_net_value"

OBSERVED_NET_COL = "net_recovered_value"


# =====================================================================
# PRINTING
# =====================================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def section(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# =====================================================================
# DATA LOADING
# =====================================================================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"File: {DATA_FILE}")
    print(f"[PASS] Rows loaded: {len(df):,}")
    print(f"[PASS] Columns loaded: {len(df.columns)}")

    return df


# =====================================================================
# VALIDATION
# =====================================================================

def validate_dataset(df):

    header("VALIDATING V3 EXPERIMENT")

    required = [
        TREATMENT_COL,
        EFFECT_COL,
        RECOVERY_COL,
        CONTROL_VALUE_COL,
        INCREMENTAL_VALUE_COL,
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required V3 columns:\n"
            + "\n".join(missing)
        )

    # Treatment
    treatment_values = set(
        pd.to_numeric(
            df[TREATMENT_COL],
            errors="coerce"
        ).dropna().unique()
    )

    if not treatment_values.issubset({0, 1}):
        raise ValueError(
            "Treatment column must contain only 0/1."
        )

    # Effect
    effect = pd.to_numeric(
        df[EFFECT_COL],
        errors="coerce"
    )

    if effect.isna().any():
        raise ValueError(
            "Treatment effect contains NaN values."
        )

    # Recovery
    recovery = pd.to_numeric(
        df[RECOVERY_COL],
        errors="coerce"
    )

    if recovery.isna().any():
        raise ValueError(
            "Recovery outcome contains NaN values."
        )

    # Economic values
    control_value = pd.to_numeric(
        df[CONTROL_VALUE_COL],
        errors="coerce"
    )

    incremental_value = pd.to_numeric(
        df[INCREMENTAL_VALUE_COL],
        errors="coerce"
    )

    if control_value.isna().any():
        raise ValueError(
            f"{CONTROL_VALUE_COL} contains NaN values."
        )

    if incremental_value.isna().any():
        raise ValueError(
            f"{INCREMENTAL_VALUE_COL} contains NaN values."
        )

    print("[PASS] Required columns present")
    print("[PASS] Treatment assignment valid")
    print("[PASS] Recovery outcome valid")
    print("[PASS] Treatment-effect values valid")
    print("[PASS] Economic values valid")

    if OBSERVED_NET_COL in df.columns:
        print(
            f"[PASS] Observed economic column present: "
            f"{OBSERVED_NET_COL}"
        )

    return True


# =====================================================================
# FEATURE SELECTION
# =====================================================================

def get_features(df):

    """
    Explicitly exclude:
        treatment
        treatment effect
        observed outcomes
        counterfactual/economic outcome variables
        experiment metadata

    This prevents leakage.
    """

    excluded = {
        # Treatment
        TREATMENT_COL,
        "treatment_group",
        "treatment_assignment_before_outcome",

        # True treatment effect
        EFFECT_COL,

        # Outcomes
        RECOVERY_COL,
        "recovery",
        "recovery_flag",
        "observed_recovery_probability",
        "actual_recovery_probability",

        # Economic outcomes
        CONTROL_VALUE_COL,
        INCREMENTAL_VALUE_COL,
        OBSERVED_NET_COL,

        "expected_treatment_net_value",
        "expected_control_net_value",
        "expected_incremental_net_recovery",
        "expected_treatment_recovery",
        "expected_control_recovery",
        "expected_incremental_recovery",

        # Experiment metadata
        "experiment_source_transaction_id",
        "experiment_transaction_id",
        "experiment_version",
        "random_seed",
        "control_recovery_probability",
        "treatment_recovery_probability",

        # Explicit counterfactuals
        "control",
        "treatment",
        "base_probability",
        "base_recovery_probability",
    }

    candidates = []

    for col in df.columns:

        if col in excluded:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        candidates.append(col)

    if not candidates:
        raise ValueError(
            "No numeric modeling features remain after leakage removal."
        )

    return candidates


# =====================================================================
# ECONOMIC VALUES
# =====================================================================

def economic_values(df):

    """
    V3 economic structure:

        control value
            =
        expected_net_recovery

        incremental treatment value
            =
        expected_incremental_net_value

        treatment value
            =
        control value + incremental value
    """

    control_value = pd.to_numeric(
        df[CONTROL_VALUE_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    incremental_value = pd.to_numeric(
        df[INCREMENTAL_VALUE_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    treatment_value = (
        control_value
        + incremental_value
    )

    return (
        control_value,
        incremental_value,
        treatment_value,
    )


# =====================================================================
# BASELINE
# =====================================================================

def baseline_analysis(df):

    header("EXPERIMENT BASELINE")

    treatment_mask = (
        df[TREATMENT_COL].to_numpy() == 1
    )

    control_mask = ~treatment_mask

    treatment_recovery = df.loc[
        treatment_mask,
        RECOVERY_COL
    ].mean()

    control_recovery = df.loc[
        control_mask,
        RECOVERY_COL
    ].mean()

    recovery_lift = (
        treatment_recovery
        - control_recovery
    )

    control_value, incremental_value, treatment_value = (
        economic_values(df)
    )

    treatment_mean_net = treatment_value[
        treatment_mask
    ].mean()

    control_mean_net = control_value[
        treatment_mask
    ].mean()

    print(
        f"Treatment recovery rate: "
        f"{treatment_recovery * 100:.2f}%"
    )

    print(
        f"Control recovery rate:   "
        f"{control_recovery * 100:.2f}%"
    )

    print(
        f"Observed recovery lift:   "
        f"{recovery_lift * 100:+.2f}%"
    )

    print(
        f"Treatment mean net:      "
        f"₹{treatment_mean_net:,.2f}"
    )

    print(
        f"Control mean net:        "
        f"₹{control_mean_net:,.2f}"
    )

    return {
        "treatment_recovery": treatment_recovery,
        "control_recovery": control_recovery,
        "recovery_lift": recovery_lift,
        "treatment_mean_net": treatment_mean_net,
        "control_mean_net": control_mean_net,
    }


# =====================================================================
# HETEROGENEITY
# =====================================================================

def heterogeneity_analysis(df):

    header("TREATMENT EFFECT HETEROGENEITY")

    effect = pd.to_numeric(
        df[EFFECT_COL],
        errors="coerce"
    ).to_numpy(dtype=float)

    positive = effect > 0
    negative = effect < 0
    zero = effect == 0

    print(
        f"Mean effect: "
        f"{effect.mean() * 100:.2f}%"
    )

    print(
        f"Std effect: "
        f"{effect.std() * 100:.2f}%"
    )

    print(
        f"Minimum effect: "
        f"{effect.min() * 100:.2f}%"
    )

    print(
        f"Maximum effect: "
        f"{effect.max() * 100:.2f}%"
    )

    print(
        f"Positive effects: "
        f"{positive.sum():,}"
    )

    print(
        f"Negative effects: "
        f"{negative.sum():,}"
    )

    print(
        f"Zero effects: "
        f"{zero.sum():,}"
    )

    if negative.sum() == 0:

        print()
        print(
            "[WARN] No negative treatment-effect rows."
        )

        print(
            "[WARN] Selective targeting cannot gain much "
            "from avoiding treatment on causal grounds."
        )

    else:

        print(
            "[PASS] Negative treatment-effect subgroup exists."
        )

    return {
        "mean_effect": effect.mean(),
        "std_effect": effect.std(),
        "positive_effects": int(positive.sum()),
        "negative_effects": int(negative.sum()),
        "zero_effects": int(zero.sum()),
    }


# =====================================================================
# QINI
# =====================================================================

def qini_score(
    treatment,
    outcome,
    uplift
):
    """
    Simple QINI-style area score.

    NumPy 2.x removed np.trapz.
    Use np.trapezoid when available, otherwise
    fall back to np.trapz for older NumPy versions.
    """

    treatment = np.asarray(
        treatment,
        dtype=float
    )

    outcome = np.asarray(
        outcome,
        dtype=float
    )

    uplift = np.asarray(
        uplift,
        dtype=float
    )

    order = np.argsort(
        -uplift
    )

    treatment = treatment[order]
    outcome = outcome[order]

    treated = treatment == 1
    control = treatment == 0

    n_treated = treated.sum()
    n_control = control.sum()

    if n_treated == 0 or n_control == 0:
        return np.nan

    cumulative_treated = np.cumsum(
        outcome * treated
    )

    cumulative_control = np.cumsum(
        outcome * control
    )

    treated_count = np.cumsum(
        treated
    )

    control_count = np.cumsum(
        control
    )

    expected_control = (
        cumulative_control
        / np.maximum(
            control_count,
            1
        )
    )

    qini_curve = (
        cumulative_treated
        - treated_count * expected_control
    )

    x = np.arange(
        1,
        len(qini_curve) + 1,
        dtype=float
    )

    if hasattr(np, "trapezoid"):
        return float(
            np.trapezoid(
                qini_curve,
                x
            )
        )

    return float(
        np.trapz(
            qini_curve,
            x
        )
    )


# =====================================================================
# MODEL TRAINING
# =====================================================================

def train_uplift_models(
    train,
    features,
    seed
):

    treatment_mask = (
        train[TREATMENT_COL].to_numpy()
        == 1
    )

    control_mask = ~treatment_mask

    model_treatment = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        min_samples_leaf=RF_MIN_LEAF,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    model_control = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        min_samples_leaf=RF_MIN_LEAF,
        random_state=seed + 1000,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    model_treatment.fit(
        train.loc[
            treatment_mask,
            features
        ],
        train.loc[
            treatment_mask,
            RECOVERY_COL
        ]
    )

    model_control.fit(
        train.loc[
            control_mask,
            features
        ],
        train.loc[
            control_mask,
            RECOVERY_COL
        ]
    )

    return (
        model_treatment,
        model_control,
    )


# =====================================================================
# PREDICT UPLIFT
# =====================================================================

def predict_uplift(
    model_treatment,
    model_control,
    test,
    features
):

    p_treatment = (
        model_treatment
        .predict_proba(
            test[features]
        )[:, 1]
    )

    p_control = (
        model_control
        .predict_proba(
            test[features]
        )[:, 1]
    )

    uplift = (
        p_treatment
        - p_control
    )

    return (
        p_treatment,
        p_control,
        uplift,
    )


# =====================================================================
# POLICY EVALUATION
# =====================================================================

def evaluate_policy(
    policy,
    control_value,
    treatment_value
):

    policy = np.asarray(
        policy,
        dtype=bool
    )

    value = np.where(
        policy,
        treatment_value,
        control_value
    )

    return {
        "value": float(
            value.mean()
        ),
        "intervention_rate": float(
            policy.mean()
        ),
    }


# =====================================================================
# ORACLE POLICY
# =====================================================================

def oracle_policy(
    incremental_value,
    threshold=0.0
):

    """
    True oracle policy.

    Treatment is selected only when the true incremental
    economic value exceeds the threshold.
    """

    return (
        incremental_value
        > threshold
    )


# =====================================================================
# RETRY ALL
# =====================================================================

def retry_all_policy(
    n
):

    return np.ones(
        n,
        dtype=bool
    )


# =====================================================================
# LEARNED POLICY SEARCH
# =====================================================================

def search_best_policy(
    uplift,
    incremental_value,
    control_value,
    treatment_value
):

    rows = []

    for threshold in UPLIFT_THRESHOLDS:

        policy = (
            uplift
            > threshold
        )

        result = evaluate_policy(
            policy,
            control_value,
            treatment_value
        )

        retry_value = (
            treatment_value.mean()
        )

        advantage = (
            result["value"]
            - retry_value
        )

        rows.append(
            {
                "policy_type":
                    "UPLIFT_THRESHOLD",

                "threshold":
                    threshold,

                "value":
                    result["value"],

                "retry_all_value":
                    retry_value,

                "advantage_vs_retry_all":
                    advantage,

                "intervention_rate":
                    result[
                        "intervention_rate"
                    ],
            }
        )

    # Economic policy based directly on predicted uplift
    # converted into expected monetary benefit.
    #
    # For V8 the probability uplift is multiplied by the
    # control-side economic opportunity.
    #
    # This creates a more decision-theoretic score without
    # requiring unavailable economic counterfactual columns.

    estimated_incremental_value = (
        uplift
        * np.maximum(
            control_value,
            0.0
        )
    )

    for threshold in ECONOMIC_THRESHOLDS:

        policy = (
            estimated_incremental_value
            > threshold
        )

        result = evaluate_policy(
            policy,
            control_value,
            treatment_value
        )

        retry_value = (
            treatment_value.mean()
        )

        advantage = (
            result["value"]
            - retry_value
        )

        rows.append(
            {
                "policy_type":
                    "ECONOMIC_THRESHOLD",

                "threshold":
                    threshold,

                "value":
                    result["value"],

                "retry_all_value":
                    retry_value,

                "advantage_vs_retry_all":
                    advantage,

                "intervention_rate":
                    result[
                        "intervention_rate"
                    ],
            }
        )

    policy_df = pd.DataFrame(
        rows
    )

    best_idx = (
        policy_df[
            "value"
        ].idxmax()
    )

    best = policy_df.loc[
        best_idx
    ]

    return (
        best,
        policy_df,
    )


# =====================================================================
# SINGLE VALIDATION RUN
# =====================================================================

def run_validation(
    df,
    features,
    seed
):

    train_idx, test_idx = train_test_split(
        np.arange(
            len(df)
        ),
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=df[
            TREATMENT_COL
        ],
    )

    train = df.iloc[
        train_idx
    ].copy()

    test = df.iloc[
        test_idx
    ].copy()

    model_treatment, model_control = (
        train_uplift_models(
            train,
            features,
            seed
        )
    )

    (
        p_treatment,
        p_control,
        uplift,
    ) = predict_uplift(
        model_treatment,
        model_control,
        test,
        features
    )

    (
        control_value,
        incremental_value,
        treatment_value,
    ) = economic_values(
        test
    )

    # ---------------------------------------------------------------
    # Basic uplift
    # ---------------------------------------------------------------

    mean_probability_uplift = (
        uplift.mean()
    )

    positive_probability_uplift = (
        uplift > 0
    )

    # ---------------------------------------------------------------
    # Economic uplift estimate
    # ---------------------------------------------------------------

    estimated_incremental_value = (
        uplift
        * np.maximum(
            control_value,
            0.0
        )
    )

    mean_economic_uplift = (
        estimated_incremental_value.mean()
    )

    positive_economic = (
        estimated_incremental_value
        > 0
    )

    # ---------------------------------------------------------------
    # Learned policy
    # ---------------------------------------------------------------

    (
        best_policy,
        policy_table,
    ) = search_best_policy(
        uplift,
        incremental_value,
        control_value,
        treatment_value,
    )

    # ---------------------------------------------------------------
    # Learned targeting
    # ---------------------------------------------------------------

    if (
        best_policy[
            "policy_type"
        ]
        == "UPLIFT_THRESHOLD"
    ):

        learned_policy = (
            uplift
            > best_policy[
                "threshold"
            ]
        )

    else:

        learned_policy = (
            estimated_incremental_value
            > best_policy[
                "threshold"
            ]
        )

    learned_result = evaluate_policy(
        learned_policy,
        control_value,
        treatment_value
    )

    # ---------------------------------------------------------------
    # RETRY ALL
    # ---------------------------------------------------------------

    retry_policy = retry_all_policy(
        len(test)
    )

    retry_result = evaluate_policy(
        retry_policy,
        control_value,
        treatment_value
    )

    advantage_vs_retry = (
        learned_result["value"]
        - retry_result["value"]
    )

    # ---------------------------------------------------------------
    # NO INTERVENTION
    # ---------------------------------------------------------------

    no_intervention_policy = np.zeros(
        len(test),
        dtype=bool
    )

    no_intervention_result = evaluate_policy(
        no_intervention_policy,
        control_value,
        treatment_value
    )

    advantage_vs_no_intervention = (
        learned_result["value"]
        - no_intervention_result["value"]
    )

    # ---------------------------------------------------------------
    # ORACLE
    # ---------------------------------------------------------------

    oracle = oracle_policy(
        incremental_value,
        threshold=0.0
    )

    oracle_result = evaluate_policy(
        oracle,
        control_value,
        treatment_value
    )

    oracle_advantage = (
        oracle_result["value"]
        - retry_result["value"]
    )

    # ---------------------------------------------------------------
    # QINI
    # ---------------------------------------------------------------

    qini = qini_score(
        test[
            TREATMENT_COL
        ].to_numpy(),
        test[
            RECOVERY_COL
        ].to_numpy(),
        uplift,
    )

    # ---------------------------------------------------------------
    # Return
    # ---------------------------------------------------------------

    return {
        "seed":
            seed,

        "training_rows":
            len(train),

        "test_rows":
            len(test),

        "mean_probability_uplift":
            mean_probability_uplift,

        "mean_economic_uplift":
            mean_economic_uplift,

        "positive_economic_uplift_rate":
            positive_economic.mean(),

        "positive_probability_uplift_rate":
            positive_probability_uplift.mean(),

        "selected_policy_type":
            best_policy[
                "policy_type"
            ],

        "selected_threshold":
            best_policy[
                "threshold"
            ],

        "intervention_rate":
            learned_result[
                "intervention_rate"
            ],

        "targeting_value":
            learned_result[
                "value"
            ],

        "retry_all_value":
            retry_result[
                "value"
            ],

        "no_intervention_value":
            no_intervention_result[
                "value"
            ],

        "oracle_value":
            oracle_result[
                "value"
            ],

        "advantage_vs_retry_all":
            advantage_vs_retry,

        "advantage_vs_no_intervention":
            advantage_vs_no_intervention,

        "oracle_advantage_vs_retry_all":
            oracle_advantage,

        "qini":
            qini,
    }


# =====================================================================
# CONFIDENCE INTERVAL
# =====================================================================

def mean_ci(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    mean = values.mean()

    std = values.std(
        ddof=1
    )

    se = (
        std
        / np.sqrt(
            len(values)
        )
    )

    # Approximate 95% normal CI.
    margin = (
        1.96
        * se
    )

    return (
        mean,
        mean - margin,
        mean + margin,
    )


# =====================================================================
# POLICY STABILITY
# =====================================================================

def policy_stability(
    results_df
):

    section(
        "POLICY STABILITY"
    )

    counts = (
        results_df[
            "selected_policy_type"
        ]
        .value_counts()
    )

    print(
        counts.to_string()
    )

    threshold_counts = (
        results_df[
            "selected_threshold"
        ]
        .value_counts()
    )

    print()
    print(
        "Selected thresholds:"
    )

    print(
        threshold_counts.to_string()
    )


# =====================================================================
# FINAL VERDICT
# =====================================================================

def final_verdict(
    results_df,
    heterogeneity
):

    header(
        "FINAL V8 ECONOMIC POLICY VERDICT"
    )

    advantage = results_df[
        "advantage_vs_retry_all"
    ].to_numpy()

    oracle_advantage = results_df[
        "oracle_advantage_vs_retry_all"
    ].to_numpy()

    qini = results_df[
        "qini"
    ].to_numpy()

    positive_qini_rate = (
        np.mean(
            qini > 0
        )
    )

    beat_retry_rate = (
        np.mean(
            advantage > 0
        )
    )

    mean_advantage = (
        advantage.mean()
    )

    mean_oracle_advantage = (
        oracle_advantage.mean()
    )

    (
        mean_adv,
        ci_lower,
        ci_upper,
    ) = mean_ci(
        advantage
    )

    print(
        f"Positive QINI rate: "
        f"{positive_qini_rate * 100:.2f}%"
    )

    print(
        f"Beat RETRY_ALL rate: "
        f"{beat_retry_rate * 100:.2f}%"
    )

    print(
        f"Mean learned advantage vs RETRY_ALL: "
        f"₹{mean_advantage:,.2f}"
    )

    print(
        f"95% CI: "
        f"[₹{ci_lower:,.2f}, ₹{ci_upper:,.2f}]"
    )

    print(
        f"Mean oracle advantage vs RETRY_ALL: "
        f"₹{mean_oracle_advantage:,.2f}"
    )

    print()

    # ---------------------------------------------------------------
    # Scientific interpretation
    # ---------------------------------------------------------------

    if mean_oracle_advantage > 0:

        print(
            "[PASS] The generated experiment contains "
            "economic value that can theoretically be "
            "improved over RETRY_ALL."
        )

        if (
            mean_advantage > 0
            and ci_lower > 0
        ):

            verdict = (
                "TARGETING_BEATS_RETRY_ALL"
            )

        else:

            verdict = (
                "ORACLE_SUPPORTS_TARGETING_BUT_LEARNER_UNCERTAIN"
            )

    else:

        print(
            "[INFO] Even the oracle does not consistently "
            "outperform RETRY_ALL."
        )

        print(
            "[INFO] In the current economic experiment, "
            "RETRY_ALL remains the stronger policy."
        )

        verdict = (
            "RETRY_ALL_SUPPORTED_BY_CURRENT_EXPERIMENT"
        )

    # ---------------------------------------------------------------
    # Heterogeneity warning
    # ---------------------------------------------------------------

    if (
        heterogeneity[
            "negative_effects"
        ] == 0
    ):

        print()
        print(
            "[INFO] The experiment contains no negative "
            "causal-effect subgroup."
        )

        print(
            "[INFO] Therefore selective targeting has "
            "limited causal headroom."
        )

    print()
    print(
        f"FINAL VERDICT: {verdict}"
    )

    return verdict


# =====================================================================
# SAVE RESULTS
# =====================================================================

def save_results(
    results_df,
    predictions_df,
    policy_df,
    summary
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    validation_path = (
        OUTPUT_DIR
        / "causal_uplift_v8_validation_results.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / "causal_uplift_v8_predictions.csv"
    )

    policies_path = (
        OUTPUT_DIR
        / "causal_uplift_v8_policy_comparison.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "causal_uplift_v8_summary.csv"
    )

    results_df.to_csv(
        validation_path,
        index=False
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    policy_df.to_csv(
        policies_path,
        index=False
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_path,
        index=False
    )

    print()
    print(
        f"[PASS] Validation:  {validation_path}"
    )

    print(
        f"[PASS] Predictions: {predictions_path}"
    )

    print(
        f"[PASS] Policies:    {policies_path}"
    )

    print(
        f"[PASS] Summary:     {summary_path}"
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning
    )

    header(
        "RECLAIM CAUSAL UPLIFT OPTIMIZER V8"
    )

    print(
        "FAST ORACLE-GATED ECONOMIC POLICY TEST"
    )

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------

    df = load_data()

    # ---------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------

    validate_dataset(
        df
    )

    # ---------------------------------------------------------------
    # Leakage protection
    # ---------------------------------------------------------------

    header(
        "LEAKAGE PROTECTION"
    )

    features = get_features(
        df
    )

    print(
        "[PASS] Outcome/treatment/counterfactual "
        "columns excluded"
    )

    print(
        f"Modeling features: {len(features)}"
    )

    # ---------------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------------

    baseline = baseline_analysis(
        df
    )

    # ---------------------------------------------------------------
    # Heterogeneity
    # ---------------------------------------------------------------

    heterogeneity = heterogeneity_analysis(
        df
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    header(
        "RUNNING V8 REPEATED HELD-OUT VALIDATION"
    )

    print(
        f"Validation runs: {VALIDATION_RUNS}"
    )

    print(
        f"Test size: {TEST_SIZE * 100:.0f}%"
    )

    validation_results = []

    for run in range(
        VALIDATION_RUNS
    ):

        seed = (
            BASE_SEED
            + run
        )

        print()
        print(
            "-" * 70
        )

        print(
            f"[{run + 1:02d}/{VALIDATION_RUNS}] "
            f"SEED {seed}"
        )

        result = run_validation(
            df,
            features,
            seed
        )

        validation_results.append(
            result
        )

        print(
            f"Training rows: "
            f"{result['training_rows']:,} | "
            f"Test rows: "
            f"{result['test_rows']:,}"
        )

        print(
            f"Mean probability uplift: "
            f"{result['mean_probability_uplift'] * 100:.2f}%"
        )

        print(
            f"Mean economic uplift: "
            f"₹{result['mean_economic_uplift']:,.2f}"
        )

        print(
            f"Positive economic uplift: "
            f"{result['positive_economic_uplift_rate'] * 100:.2f}%"
        )

        print(
            f"Selected policy: "
            f"{result['selected_policy_type']}"
        )

        print(
            f"Selected threshold: "
            f"{result['selected_threshold']:.4f}"
        )

        print(
            f"Intervention rate: "
            f"{result['intervention_rate'] * 100:.2f}%"
        )

        print(
            f"QINI: "
            f"{result['qini']:,.2f}"
        )

        print(
            f"Targeting value/transaction: "
            f"₹{result['targeting_value']:,.2f}"
        )

        print(
            f"RETRY_ALL value/transaction: "
            f"₹{result['retry_all_value']:,.2f}"
        )

        print(
            f"Advantage vs RETRY_ALL: "
            f"₹{result['advantage_vs_retry_all']:,.2f}"
        )

        print(
            f"Oracle advantage vs RETRY_ALL: "
            f"₹{result['oracle_advantage_vs_retry_all']:,.2f}"
        )

    results_df = pd.DataFrame(
        validation_results
    )

    # ---------------------------------------------------------------
    # Stability summary
    # ---------------------------------------------------------------

    header(
        "V8 STABILITY SUMMARY"
    )

    qini_values = results_df[
        "qini"
    ].to_numpy()

    economic_uplift = results_df[
        "mean_economic_uplift"
    ].to_numpy()

    probability_uplift = results_df[
        "mean_probability_uplift"
    ].to_numpy()

    intervention_rate = results_df[
        "intervention_rate"
    ].to_numpy()

    advantage = results_df[
        "advantage_vs_retry_all"
    ].to_numpy()

    advantage_no_intervention = results_df[
        "advantage_vs_no_intervention"
    ].to_numpy()

    oracle_advantage = results_df[
        "oracle_advantage_vs_retry_all"
    ].to_numpy()

    print(
        f"Validation runs: "
        f"{len(results_df)}"
    )

    print(
        f"Mean QINI: "
        f"{np.nanmean(qini_values):,.2f}"
    )

    print(
        f"QINI std dev: "
        f"{np.nanstd(qini_values):,.2f}"
    )

    print(
        f"Positive-QINI rate: "
        f"{np.nanmean(qini_values > 0) * 100:.2f}%"
    )

    print(
        f"Mean probability uplift: "
        f"{probability_uplift.mean() * 100:.2f}%"
    )

    print(
        f"Mean economic uplift: "
        f"₹{economic_uplift.mean():,.2f}"
    )

    print(
        f"Mean intervention rate: "
        f"{intervention_rate.mean() * 100:.2f}%"
    )

    print()
    print(
        "TARGETING VS NO_INTERVENTION"
    )

    print(
        f"Mean advantage: "
        f"₹{advantage_no_intervention.mean():,.2f}"
    )

    print(
        f"Runs beating NO_INTERVENTION: "
        f"{np.mean(advantage_no_intervention > 0) * 100:.2f}%"
    )

    print()
    print(
        "TARGETING VS RETRY_ALL"
    )

    print(
        f"Mean advantage: "
        f"₹{advantage.mean():,.2f}"
    )

    (
        mean_advantage,
        ci_lower,
        ci_upper,
    ) = mean_ci(
        advantage
    )

    print(
        f"95% CI: "
        f"[₹{ci_lower:,.2f}, ₹{ci_upper:,.2f}]"
    )

    print(
        f"Runs beating RETRY_ALL: "
        f"{np.mean(advantage > 0) * 100:.2f}%"
    )

    print()
    print(
        "ORACLE BENCHMARK"
    )

    print(
        f"Mean oracle advantage vs RETRY_ALL: "
        f"₹{oracle_advantage.mean():,.2f}"
    )

    # ---------------------------------------------------------------
    # Policy stability
    # ---------------------------------------------------------------

    policy_stability(
        results_df
    )

    # ---------------------------------------------------------------
    # Final verdict
    # ---------------------------------------------------------------

    verdict = final_verdict(
        results_df,
        heterogeneity
    )

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------

    # Create one final full-data model for inspection.
    #
    # This is NOT used for the held-out evaluation.
    # It is only used to generate the final prediction file.

    print()
    print(
        "=" * 70
    )

    print(
        "GENERATING FINAL FULL-DATA PREDICTIONS"
    )

    print(
        "=" * 70
    )

    model_treatment, model_control = (
        train_uplift_models(
            df,
            features,
            BASE_SEED
        )
    )

    (
        p_treatment,
        p_control,
        uplift,
    ) = predict_uplift(
        model_treatment,
        model_control,
        df,
        features
    )

    (
        control_value,
        incremental_value,
        treatment_value,
    ) = economic_values(
        df
    )

    estimated_incremental_value = (
        uplift
        * np.maximum(
            control_value,
            0.0
        )
    )

    predictions_df = pd.DataFrame(
        {
            "transaction_id":
                df[
                    "transaction_id"
                ].to_numpy()
                if "transaction_id"
                in df.columns
                else np.arange(
                    len(df)
                ),

            "predicted_treatment_probability":
                p_treatment,

            "predicted_control_probability":
                p_control,

            "predicted_probability_uplift":
                uplift,

            "predicted_economic_uplift":
                estimated_incremental_value,

            "true_treatment_effect":
                df[
                    EFFECT_COL
                ].to_numpy(),

            "true_incremental_economic_value":
                incremental_value,

            "oracle_treatment":
                incremental_value > 0,

            "learned_treatment":
                uplift > 0,

            "control_value":
                control_value,

            "treatment_value":
                treatment_value,
        }
    )

    # ---------------------------------------------------------------
    # Full policy comparison
    # ---------------------------------------------------------------

    policy_rows = []

    retry_value = (
        treatment_value.mean()
    )

    no_intervention_value = (
        control_value.mean()
    )

    oracle_policy_mask = (
        incremental_value > 0
    )

    oracle_value = np.where(
        oracle_policy_mask,
        treatment_value,
        control_value
    ).mean()

    for threshold in UPLIFT_THRESHOLDS:

        policy = (
            uplift
            > threshold
        )

        value = np.where(
            policy,
            treatment_value,
            control_value
        ).mean()

        policy_rows.append(
            {
                "policy":
                    "UPLIFT_THRESHOLD",

                "threshold":
                    threshold,

                "intervention_rate":
                    policy.mean(),

                "value_per_transaction":
                    value,

                "advantage_vs_retry_all":
                    value
                    - retry_value,

                "advantage_vs_no_intervention":
                    value
                    - no_intervention_value,
            }
        )

    for threshold in ECONOMIC_THRESHOLDS:

        policy = (
            estimated_incremental_value
            > threshold
        )

        value = np.where(
            policy,
            treatment_value,
            control_value
        ).mean()

        policy_rows.append(
            {
                "policy":
                    "ECONOMIC_THRESHOLD",

                "threshold":
                    threshold,

                "intervention_rate":
                    policy.mean(),

                "value_per_transaction":
                    value,

                "advantage_vs_retry_all":
                    value
                    - retry_value,

                "advantage_vs_no_intervention":
                    value
                    - no_intervention_value,
            }
        )

    policy_rows.append(
        {
            "policy":
                "RETRY_ALL",

            "threshold":
                np.nan,

            "intervention_rate":
                1.0,

            "value_per_transaction":
                retry_value,

            "advantage_vs_retry_all":
                0.0,

            "advantage_vs_no_intervention":
                retry_value
                - no_intervention_value,
        }
    )

    policy_rows.append(
        {
            "policy":
                "NO_INTERVENTION",

            "threshold":
                np.nan,

            "intervention_rate":
                0.0,

            "value_per_transaction":
                no_intervention_value,

            "advantage_vs_retry_all":
                no_intervention_value
                - retry_value,

            "advantage_vs_no_intervention":
                0.0,
        }
    )

    policy_rows.append(
        {
            "policy":
                "ORACLE",

            "threshold":
                0.0,

            "intervention_rate":
                oracle_policy_mask.mean(),

            "value_per_transaction":
                oracle_value,

            "advantage_vs_retry_all":
                oracle_value
                - retry_value,

            "advantage_vs_no_intervention":
                oracle_value
                - no_intervention_value,
        }
    )

    policy_df = pd.DataFrame(
        policy_rows
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary = {
        "version":
            "V8",

        "dataset":
            str(DATA_FILE),

        "rows":
            len(df),

        "features":
            len(features),

        "validation_runs":
            VALIDATION_RUNS,

        "mean_qini":
            np.nanmean(qini_values),

        "qini_std":
            np.nanstd(qini_values),

        "positive_qini_rate":
            np.nanmean(
                qini_values > 0
            ),

        "mean_probability_uplift":
            probability_uplift.mean(),

        "mean_economic_uplift":
            economic_uplift.mean(),

        "mean_intervention_rate":
            intervention_rate.mean(),

        "mean_advantage_vs_retry_all":
            advantage.mean(),

        "ci_lower_advantage_vs_retry_all":
            ci_lower,

        "ci_upper_advantage_vs_retry_all":
            ci_upper,

        "beat_retry_all_rate":
            np.mean(
                advantage > 0
            ),

        "mean_advantage_vs_no_intervention":
            advantage_no_intervention.mean(),

        "mean_oracle_advantage_vs_retry_all":
            oracle_advantage.mean(),

        "negative_effect_rows":
            heterogeneity[
                "negative_effects"
            ],

        "positive_effect_rows":
            heterogeneity[
                "positive_effects"
            ],

        "final_verdict":
            verdict,
    }

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    header(
        "SAVING V8 RESULTS"
    )

    save_results(
        results_df,
        predictions_df,
        policy_df,
        summary
    )

    # ---------------------------------------------------------------
    # Complete
    # ---------------------------------------------------------------

    header(
        "V8 COMPLETE"
    )

    print(
        f"Mean QINI: "
        f"{np.nanmean(qini_values):,.2f}"
    )

    print(
        f"Positive QINI rate: "
        f"{np.nanmean(qini_values > 0) * 100:.2f}%"
    )

    print(
        f"Mean learned targeting advantage: "
        f"₹{advantage.mean():,.2f}"
    )

    print(
        f"Mean oracle advantage: "
        f"₹{oracle_advantage.mean():,.2f}"
    )

    print(
        f"95% CI: "
        f"[₹{ci_lower:,.2f}, ₹{ci_upper:,.2f}]"
    )

    print(
        f"FINAL VERDICT: {verdict}"
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()

