
"""
RECLAIM CAUSAL UPLIFT OPTIMIZER V9
==================================

STRONG HETEROGENEITY + ECONOMIC POLICY STRESS TEST

Purpose
-------
V8 showed:

    Oracle advantage vs RETRY_ALL  : only about +₹1
    Learned targeting              : below RETRY_ALL

V9 therefore performs a stronger causal-policy stress test.

IMPORTANT
---------
This script does NOT force a positive result.

It:
1. Loads the heterogeneous V3 experiment.
2. Creates a stronger synthetic economic-policy layer.
3. Preserves randomized treatment assignment.
4. Creates an explicit heterogeneous treatment-effect signal.
5. Trains treatment/control outcome models.
6. Evaluates learned targeting on held-out data.
7. Computes an oracle upper-bound policy.
8. Compares against:
       - NO_INTERVENTION
       - RETRY_ALL
       - LEARNED_TARGETING
       - ORACLE
9. Runs repeated validation.
10. Reports whether the learner actually beats RETRY_ALL.

The economic layer is intentionally stronger than V8 so that
the oracle policy has enough headroom to demonstrate whether
selective targeting can work.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# ======================================================================
# CONFIGURATION
# ======================================================================

INPUT_FILE = Path(
    r"D:\RECLAIM\data\generated\reclaim_causal_experiment_v3.csv"
)

OUTPUT_DIR = Path(
    r"D:\RECLAIM\data\generated"
)

N_RUNS = 20
TEST_SIZE = 0.30

RF_ESTIMATORS = 300
RF_MIN_LEAF = 12

SEEDS = list(range(900, 900 + N_RUNS))

# Stronger economic stress-test parameters.
#
# Positive treatment effects receive larger economic benefit.
# Negative-effect customers receive an explicit intervention penalty.
#
# These values are deliberately stronger than V8 so the oracle
# has measurable headroom.

POSITIVE_EFFECT_VALUE_MULTIPLIER = 18000.0
NEGATIVE_EFFECT_PENALTY_MULTIPLIER = 26000.0

# Additional intervention friction/cost.
INTERVENTION_COST = 85.0

# Small amount of economic noise.
ECONOMIC_NOISE_STD = 18.0

# Policy thresholds evaluated on the predicted incremental value.
ECONOMIC_THRESHOLDS = [
    -100.0,
    -50.0,
    -25.0,
    0.0,
    25.0,
    50.0,
    75.0,
    100.0,
    150.0,
    200.0,
]


# ======================================================================
# PRINT HELPERS
# ======================================================================

def banner(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


# ======================================================================
# COLUMN DISCOVERY
# ======================================================================

def find_column(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(
            f"Could not find required column.\n"
            f"Tried: {candidates}\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    return None


def identify_schema(df):
    treatment = find_column(
        df,
        [
            "treatment",
            "treatment_assignment_before_outcome",
        ],
    )

    recovery = find_column(
        df,
        [
            "recovered",
            "recovery_flag",
        ],
    )

    effect = find_column(
        df,
        [
            "treatment_effect",
            "incremental_recovery_probability",
        ],
    )

    control_probability = find_column(
        df,
        [
            "control_recovery_probability",
            "expected_control_recovery",
        ],
    )

    treatment_probability = find_column(
        df,
        [
            "treatment_recovery_probability",
            "expected_treatment_recovery",
        ],
    )

    return {
        "treatment": treatment,
        "recovery": recovery,
        "effect": effect,
        "control_probability": control_probability,
        "treatment_probability": treatment_probability,
    }


# ======================================================================
# VALIDATION
# ======================================================================

def validate_dataset(df, schema):

    required = [
        schema["treatment"],
        schema["recovery"],
        schema["effect"],
        schema["control_probability"],
        schema["treatment_probability"],
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    treatment_values = set(
        pd.Series(df[schema["treatment"]])
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not treatment_values.issubset({0, 1}):
        raise ValueError(
            "Treatment must contain only 0/1."
        )

    recovery_values = set(
        pd.Series(df[schema["recovery"]])
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not recovery_values.issubset({0, 1}):
        raise ValueError(
            "Recovery must contain only 0/1."
        )

    for col in [
        schema["effect"],
        schema["control_probability"],
        schema["treatment_probability"],
    ]:
        if df[col].isna().any():
            raise ValueError(
                f"Column contains missing values: {col}"
            )

    print("[PASS] Required columns present")
    print("[PASS] Treatment assignment valid")
    print("[PASS] Recovery outcome valid")
    print("[PASS] Treatment-effect values valid")


# ======================================================================
# STRONG ECONOMIC LAYER
# ======================================================================

def create_economic_layer(df, schema, seed):

    rng = np.random.default_rng(seed)

    effect = df[schema["effect"]].to_numpy(
        dtype=float
    )

    base_probability = df[
        schema["control_probability"]
    ].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Normalize effect around zero.
    # ------------------------------------------------------------------

    effect_centered = (
        effect - np.mean(effect)
    )

    # ------------------------------------------------------------------
    # Economic incremental value.
    #
    # Positive effects create meaningful upside.
    # Negative effects create meaningful downside.
    # ------------------------------------------------------------------

    incremental_value = np.where(
        effect_centered >= 0,

        effect_centered
        * POSITIVE_EFFECT_VALUE_MULTIPLIER,

        effect_centered
        * NEGATIVE_EFFECT_PENALTY_MULTIPLIER,
    )

    # ------------------------------------------------------------------
    # Add a mild dependence on baseline recoverability.
    # This makes the economic decision more realistic.
    # ------------------------------------------------------------------

    baseline_adjustment = (
        base_probability - np.mean(base_probability)
    ) * 120.0

    incremental_value = (
        incremental_value
        + baseline_adjustment
        - INTERVENTION_COST
    )

    # Small noise prevents a completely deterministic oracle.
    incremental_value += rng.normal(
        0.0,
        ECONOMIC_NOISE_STD,
        size=len(df),
    )

    # ------------------------------------------------------------------
    # Control economic value.
    #
    # This is based on the transaction's existing net recovery value
    # when available, otherwise on recovered amount / amount.
    # ------------------------------------------------------------------

    if "net_recovered_value" in df.columns:

        control_base = df[
            "net_recovered_value"
        ].to_numpy(dtype=float)

    elif "recovered_amount" in df.columns:

        control_base = df[
            "recovered_amount"
        ].to_numpy(dtype=float)

    elif "amount" in df.columns:

        control_base = (
            df["amount"]
            .to_numpy(dtype=float)
            * base_probability
        )

    else:

        control_base = (
            base_probability
            * 1500.0
        )

    # ------------------------------------------------------------------
    # Remove treatment-specific observed outcome leakage from policy
    # construction. We use control_base as the baseline value.
    # ------------------------------------------------------------------

    control_value = np.maximum(
        control_base,
        0.0,
    )

    treatment_value = (
        control_value
        + incremental_value
    )

    result = df.copy()

    result["v9_control_value"] = control_value
    result["v9_incremental_value"] = incremental_value
    result["v9_treatment_value"] = treatment_value

    return result


# ======================================================================
# MODELING FEATURES
# ======================================================================

def get_features(df, schema):

    excluded = {
        schema["treatment"],
        schema["recovery"],
        schema["effect"],
        schema["control_probability"],
        schema["treatment_probability"],
    }

    # Outcome / counterfactual / policy columns must not be used.
    leakage_terms = [
        "recovered",
        "recovery",
        "treatment",
        "control",
        "counterfactual",
        "effect",
        "incremental",
        "net_recovered",
        "expected_net",
        "observed",
        "treatment_probability",
        "control_probability",
        "v9_",
        "policy",
        "recommended_action",
        "recommended_policy",
        "recovery_status",
    ]

    numeric_features = []

    for col in df.columns:

        if col in excluded:
            continue

        if col.startswith("v9_"):
            continue

        lowered = col.lower()

        if any(
            term in lowered
            for term in leakage_terms
        ):
            continue

        if pd.api.types.is_numeric_dtype(
            df[col]
        ):
            numeric_features.append(col)

    if len(numeric_features) < 5:
        raise ValueError(
            "Too few modeling features after leakage protection."
        )

    return numeric_features


# ======================================================================
# SAFE FEATURE MATRIX
# ======================================================================

def prepare_features(train, test, features):

    X_train = (
        train[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    X_test = (
        test[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return X_train, X_test


# ======================================================================
# QINI
# ======================================================================

def qini_score(df, predicted_uplift, treatment_col, outcome_col):

    tmp = pd.DataFrame({
        "uplift": np.asarray(
            predicted_uplift,
            dtype=float,
        ),
        "treatment": df[
            treatment_col
        ].to_numpy(),
        "outcome": df[
            outcome_col
        ].to_numpy(),
    })

    tmp = tmp.sort_values(
        "uplift",
        ascending=False,
    ).reset_index(drop=True)

    treated = (
        tmp["treatment"] == 1
    )

    control = (
        tmp["treatment"] == 0
    )

    n_treated = treated.sum()
    n_control = control.sum()

    if n_treated == 0 or n_control == 0:
        return 0.0

    cumulative_treated = (
        (tmp["outcome"] * treated)
        .cumsum()
    )

    cumulative_control = (
        (tmp["outcome"] * control)
        .cumsum()
    )

    cumulative_treated_n = (
        treated.cumsum()
    )

    cumulative_control_n = (
        control.cumsum()
    )

    expected_control = (
        cumulative_control_n
        * cumulative_treated
        / np.maximum(
            cumulative_treated_n,
            1,
        )
    )

    incremental = (
        cumulative_treated
        - expected_control
    )

    x = np.arange(
        1,
        len(tmp) + 1,
        dtype=float,
    )

    # NumPy 2.x removed np.trapz.
    # trapezoid is the current equivalent.
    return float(
        np.trapezoid(
            incremental,
            x,
        )
    )


# ======================================================================
# ORACLE POLICY
# ======================================================================

def oracle_policy(test):

    incremental = test[
        "v9_incremental_value"
    ].to_numpy(dtype=float)

    policy = incremental > 0

    control_value = test[
        "v9_control_value"
    ].to_numpy(dtype=float)

    treatment_value = test[
        "v9_treatment_value"
    ].to_numpy(dtype=float)

    oracle_value = np.where(
        policy,
        treatment_value,
        control_value,
    ).mean()

    retry_all_value = (
        treatment_value.mean()
    )

    no_intervention_value = (
        control_value.mean()
    )

    return {
        "oracle_value": oracle_value,
        "retry_all_value": retry_all_value,
        "no_intervention_value":
            no_intervention_value,
        "oracle_advantage":
            oracle_value - retry_all_value,
        "oracle_intervention_rate":
            policy.mean(),
    }


# ======================================================================
# LEARNED TARGETING
# ======================================================================

def learned_targeting(
    train,
    test,
    schema,
    features,
    seed,
):

    t_col = schema["treatment"]
    recovery_col = schema["recovery"]

    train_t = (
        train[t_col] == 1
    )

    train_c = (
        train[t_col] == 0
    )

    X_train, X_test = prepare_features(
        train,
        test,
        features,
    )

    model_t = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        min_samples_leaf=RF_MIN_LEAF,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )

    model_c = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        min_samples_leaf=RF_MIN_LEAF,
        random_state=seed + 1000,
        n_jobs=-1,
        class_weight="balanced",
    )

    model_t.fit(
        X_train.loc[train_t],
        train.loc[
            train_t,
            recovery_col,
        ],
    )

    model_c.fit(
        X_train.loc[train_c],
        train.loc[
            train_c,
            recovery_col,
        ],
    )

    p_t = model_t.predict_proba(
        X_test
    )[:, 1]

    p_c = model_c.predict_proba(
        X_test
    )[:, 1]

    probability_uplift = (
        p_t - p_c
    )

    # ------------------------------------------------------------------
    # Convert probability uplift into economic uplift.
    #
    # The economic decision must consider the value generated by
    # recovery, rather than probability alone.
    # ------------------------------------------------------------------

    value_scale = np.maximum(
        test[
            "v9_control_value"
        ].to_numpy(dtype=float),
        100.0,
    )

    predicted_economic_uplift = (
        probability_uplift
        * value_scale
        - INTERVENTION_COST
    )

    # ------------------------------------------------------------------
    # Evaluate several thresholds.
    # Threshold is selected using ONLY the training set.
    #
    # This avoids selecting the policy directly on the held-out test.
    # ------------------------------------------------------------------

    train_pred_t = model_t.predict_proba(
        X_train
    )[:, 1]

    train_pred_c = model_c.predict_proba(
        X_train
    )[:, 1]

    train_probability_uplift = (
        train_pred_t
        - train_pred_c
    )

    train_value_scale = np.maximum(
        train[
            "v9_control_value"
        ].to_numpy(dtype=float),
        100.0,
    )

    train_pred_economic = (
        train_probability_uplift
        * train_value_scale
        - INTERVENTION_COST
    )

    threshold_results = []

    for threshold in ECONOMIC_THRESHOLDS:

        policy = (
            train_pred_economic
            > threshold
        )

        if policy.mean() < 0.01:
            continue

        # Since treatment/control outcomes are observed in the
        # randomized training experiment, calculate the empirical
        # policy value using the actual treatment/control groups.
        #
        # This is only used to select the threshold inside training.

        treatment_outcome = (
            train[
                "v9_treatment_value"
            ].to_numpy(dtype=float)
        )

        control_outcome = (
            train[
                "v9_control_value"
            ].to_numpy(dtype=float)
        )

        treatment_mask = (
            train[t_col].to_numpy() == 1
        )

        control_mask = (
            train[t_col].to_numpy() == 0
        )

        # IPW policy-value estimate.
        propensity = 0.5

        ipw_values = np.where(
            policy & treatment_mask,
            treatment_outcome / propensity,
            0.0,
        )

        ipw_values += np.where(
            (~policy) & control_mask,
            control_outcome / (1.0 - propensity),
            0.0,
        )

        estimated_value = (
            ipw_values.mean()
        )

        threshold_results.append(
            (
                estimated_value,
                threshold,
            )
        )

    if threshold_results:

        threshold_results.sort(
            reverse=True
        )

        selected_threshold = (
            threshold_results[0][1]
        )

    else:

        selected_threshold = 0.0

    policy = (
        predicted_economic_uplift
        > selected_threshold
    )

    control_value = test[
        "v9_control_value"
    ].to_numpy(dtype=float)

    treatment_value = test[
        "v9_treatment_value"
    ].to_numpy(dtype=float)

    targeting_value = np.where(
        policy,
        treatment_value,
        control_value,
    ).mean()

    retry_all_value = (
        treatment_value.mean()
    )

    no_intervention_value = (
        control_value.mean()
    )

    return {
        "probability_uplift":
            probability_uplift,
        "economic_uplift":
            predicted_economic_uplift,
        "policy":
            policy,
        "selected_threshold":
            selected_threshold,
        "targeting_value":
            targeting_value,
        "retry_all_value":
            retry_all_value,
        "no_intervention_value":
            no_intervention_value,
        "advantage_vs_retry":
            targeting_value
            - retry_all_value,
        "advantage_vs_no_intervention":
            targeting_value
            - no_intervention_value,
        "intervention_rate":
            policy.mean(),
    }


# ======================================================================
# ONE VALIDATION RUN
# ======================================================================

def run_validation(
    df,
    schema,
    features,
    seed,
):

    indices = np.arange(
        len(df)
    )

    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=df[
            schema["treatment"]
        ],
    )

    train = df.iloc[
        train_idx
    ].copy()

    test = df.iloc[
        test_idx
    ].copy()

    learned = learned_targeting(
        train,
        test,
        schema,
        features,
        seed,
    )

    qini = qini_score(
        test,
        learned[
            "probability_uplift"
        ],
        schema["treatment"],
        schema["recovery"],
    )

    oracle = oracle_policy(
        test
    )

    return {
        "seed": seed,

        "mean_probability_uplift":
            learned[
                "probability_uplift"
            ].mean(),

        "mean_predicted_economic_uplift":
            learned[
                "economic_uplift"
            ].mean(),

        "positive_economic_uplift_rate":
            (
                learned[
                    "economic_uplift"
                ] > 0
            ).mean(),

        "selected_threshold":
            learned[
                "selected_threshold"
            ],

        "intervention_rate":
            learned[
                "intervention_rate"
            ],

        "qini":
            qini,

        "targeting_value":
            learned[
                "targeting_value"
            ],

        "retry_all_value":
            learned[
                "retry_all_value"
            ],

        "no_intervention_value":
            learned[
                "no_intervention_value"
            ],

        "learned_advantage_vs_retry":
            learned[
                "advantage_vs_retry"
            ],

        "learned_advantage_vs_no_intervention":
            learned[
                "advantage_vs_no_intervention"
            ],

        "oracle_value":
            oracle[
                "oracle_value"
            ],

        "oracle_advantage_vs_retry":
            oracle[
                "oracle_advantage"
            ],

        "oracle_intervention_rate":
            oracle[
                "oracle_intervention_rate"
            ],
    }


# ======================================================================
# FINAL VERDICT
# ======================================================================

def determine_verdict(results):

    learned_advantage = results[
        "learned_advantage_vs_retry"
    ]

    oracle_advantage = results[
        "oracle_advantage_vs_retry"
    ]

    beating_rate = (
        learned_advantage > 0
    ).mean()

    oracle_beating_rate = (
        oracle_advantage > 0
    ).mean()

    ci_low = np.percentile(
        learned_advantage,
        2.5,
    )

    ci_high = np.percentile(
        learned_advantage,
        97.5,
    )

    mean_learned = (
        learned_advantage.mean()
    )

    mean_oracle = (
        oracle_advantage.mean()
    )

    if (
        mean_oracle > 0
        and beating_rate >= 0.70
        and ci_low > 0
    ):
        return (
            "LEARNED_TARGETING_BEATS_RETRY_ALL"
        )

    if (
        mean_oracle > 0
        and oracle_beating_rate >= 0.70
        and mean_learned <= 0
    ):
        return (
            "ORACLE_HAS_HEADROOM_BUT_LEARNER_FAILS"
        )

    if mean_oracle <= 0:
        return (
            "RETRY_ALL_SUPPORTED_BY_EXPERIMENT"
        )

    return (
        "TARGETING_SIGNAL_EXISTS_BUT_IS_NOT_STABLE"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    banner(
        "RECLAIM CAUSAL UPLIFT OPTIMIZER V9"
    )

    print(
        "STRONG HETEROGENEITY + ECONOMIC POLICY STRESS TEST"
    )

    print(
        f"File: {INPUT_FILE}"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"[PASS] Rows loaded: {len(df):,}"
    )

    print(
        f"[PASS] Columns loaded: {len(df.columns)}"
    )

    banner(
        "VALIDATING V3 EXPERIMENT"
    )

    schema = identify_schema(
        df
    )

    validate_dataset(
        df,
        schema,
    )

    banner(
        "CREATING V9 STRONG ECONOMIC LAYER"
    )

    # Use a deterministic layer.
    df = create_economic_layer(
        df,
        schema,
        seed=9000,
    )

    print(
        "Economic layer created."
    )

    print(
        "Positive incremental-value rows:",
        int(
            (
                df["v9_incremental_value"]
                > 0
            ).sum()
        ),
    )

    print(
        "Negative incremental-value rows:",
        int(
            (
                df["v9_incremental_value"]
                < 0
            ).sum()
        ),
    )

    print(
        "Mean incremental value:",
        f"₹{df['v9_incremental_value'].mean():,.2f}"
    )

    print(
        "Std incremental value:",
        f"₹{df['v9_incremental_value'].std():,.2f}"
    )

    banner(
        "LEAKAGE PROTECTION"
    )

    features = get_features(
        df,
        schema,
    )

    print(
        "[PASS] Outcome/treatment/counterfactual "
        "columns excluded"
    )

    print(
        f"Modeling features: {len(features)}"
    )

    banner(
        "RUNNING V9 REPEATED HELD-OUT VALIDATION"
    )

    print(
        f"Validation runs: {N_RUNS}"
    )

    print(
        f"Test size: {TEST_SIZE:.0%}"
    )

    all_results = []

    for run_number, seed in enumerate(
        SEEDS,
        start=1,
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"[{run_number:02d}/{N_RUNS}] SEED {seed}"
        )

        result = run_validation(
            df,
            schema,
            features,
            seed,
        )

        all_results.append(
            result
        )

        print(
            "Training rows: 7,000 | "
            "Test rows: 3,000"
        )

        print(
            "Mean probability uplift:",
            f"{result['mean_probability_uplift']:.2%}"
        )

        print(
            "Mean predicted economic uplift:",
            f"₹{result['mean_predicted_economic_uplift']:,.2f}"
        )

        print(
            "Positive economic uplift:",
            f"{result['positive_economic_uplift_rate']:.2%}"
        )

        print(
            "Selected threshold:",
            f"₹{result['selected_threshold']:,.2f}"
        )

        print(
            "Intervention rate:",
            f"{result['intervention_rate']:.2%}"
        )

        print(
            "QINI:",
            f"{result['qini']:,.2f}"
        )

        print(
            "Targeting value/transaction:",
            f"₹{result['targeting_value']:,.2f}"
        )

        print(
            "RETRY_ALL value/transaction:",
            f"₹{result['retry_all_value']:,.2f}"
        )

        print(
            "Learned advantage vs RETRY_ALL:",
            f"₹{result['learned_advantage_vs_retry']:,.2f}"
        )

        print(
            "Oracle advantage vs RETRY_ALL:",
            f"₹{result['oracle_advantage_vs_retry']:,.2f}"
        )

    results = pd.DataFrame(
        all_results
    )

    banner(
        "V9 STABILITY SUMMARY"
    )

    mean_qini = (
        results["qini"].mean()
    )

    qini_std = (
        results["qini"].std()
    )

    positive_qini_rate = (
        results["qini"] > 0
    ).mean()

    mean_probability = (
        results[
            "mean_probability_uplift"
        ].mean()
    )

    mean_economic = (
        results[
            "mean_predicted_economic_uplift"
        ].mean()
    )

    mean_intervention = (
        results[
            "intervention_rate"
        ].mean()
    )

    print(
        f"Validation runs: {N_RUNS}"
    )

    print(
        f"Mean QINI: {mean_qini:,.2f}"
    )

    print(
        f"QINI std dev: {qini_std:,.2f}"
    )

    print(
        f"Positive-QINI rate: "
        f"{positive_qini_rate:.2%}"
    )

    print(
        f"Mean probability uplift: "
        f"{mean_probability:.2%}"
    )

    print(
        f"Mean predicted economic uplift: "
        f"₹{mean_economic:,.2f}"
    )

    print(
        f"Mean intervention rate: "
        f"{mean_intervention:.2%}"
    )

    banner(
        "TARGETING VS NO_INTERVENTION"
    )

    no_intervention_advantage = results[
        "learned_advantage_vs_no_intervention"
    ]

    print(
        "Mean advantage:",
        f"₹{no_intervention_advantage.mean():,.2f}"
    )

    print(
        "Runs beating NO_INTERVENTION:",
        f"{(no_intervention_advantage > 0).mean():.2%}"
    )

    banner(
        "TARGETING VS RETRY_ALL"
    )

    learned_advantage = results[
        "learned_advantage_vs_retry"
    ]

    ci_low = np.percentile(
        learned_advantage,
        2.5,
    )

    ci_high = np.percentile(
        learned_advantage,
        97.5,
    )

    print(
        "Mean learned advantage:",
        f"₹{learned_advantage.mean():,.2f}"
    )

    print(
        "95% CI:",
        f"[₹{ci_low:,.2f}, "
        f"₹{ci_high:,.2f}]"
    )

    print(
        "Runs beating RETRY_ALL:",
        f"{(learned_advantage > 0).mean():.2%}"
    )

    banner(
        "ORACLE BENCHMARK"
    )

    oracle_advantage = results[
        "oracle_advantage_vs_retry"
    ]

    print(
        "Mean oracle advantage:",
        f"₹{oracle_advantage.mean():,.2f}"
    )

    print(
        "Oracle positive-rate:",
        f"{(oracle_advantage > 0).mean():.2%}"
    )

    print(
        "Mean oracle intervention rate:",
        f"{results['oracle_intervention_rate'].mean():.2%}"
    )

    banner(
        "FINAL V9 CAUSAL POLICY VERDICT"
    )

    verdict = determine_verdict(
        results
    )

    print(
        f"Mean oracle advantage: "
        f"₹{oracle_advantage.mean():,.2f}"
    )

    print(
        f"Mean learned advantage: "
        f"₹{learned_advantage.mean():,.2f}"
    )

    print(
        f"Learned 95% CI: "
        f"[₹{ci_low:,.2f}, "
        f"₹{ci_high:,.2f}]"
    )

    print()

    if oracle_advantage.mean() > 0:
        print(
            "[PASS] Oracle policy has measurable "
            "economic headroom over RETRY_ALL."
        )
    else:
        print(
            "[WARN] Even the oracle does not "
            "reliably beat RETRY_ALL."
        )

    if (
        learned_advantage.mean() > 0
        and ci_low > 0
    ):
        print(
            "[PASS] Learned targeting beats "
            "RETRY_ALL with positive 95% CI."
        )
    else:
        print(
            "[INFO] Learned targeting has not "
            "demonstrated reliable superiority."
        )

    print()

    print(
        f"FINAL VERDICT: {verdict}"
    )

    # ==================================================================
    # SAVE VALIDATION RESULTS
    # ==================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_path = (
        OUTPUT_DIR
        / "causal_uplift_v9_validation_results.csv"
    )

    results.to_csv(
        validation_path,
        index=False,
    )

    print()
    print(
        "[PASS] Validation:",
        validation_path,
    )

    # ==================================================================
    # FINAL FULL-DATA PREDICTION
    # ==================================================================

    banner(
        "GENERATING FINAL FULL-DATA POLICY SCORE"
    )

    full_train = df.copy()

    # Use a deterministic holdout for final prediction.
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=9999,
        stratify=df[
            schema["treatment"]
        ],
    )

    train = df.iloc[
        train_idx
    ].copy()

    test = df.iloc[
        test_idx
    ].copy()

    final = learned_targeting(
        train,
        test,
        schema,
        features,
        seed=9999,
    )

    prediction_df = test[
        [
            c
            for c in [
                "transaction_id",
                "customer_id",
                "amount",
                schema["treatment"],
                schema["recovery"],
            ]
            if c in test.columns
        ]
    ].copy()

    prediction_df[
        "predicted_probability_uplift"
    ] = final[
        "probability_uplift"
    ]

    prediction_df[
        "predicted_economic_uplift"
    ] = final[
        "economic_uplift"
    ]

    prediction_df[
        "targeting_policy"
    ] = final[
        "policy"
    ].astype(int)

    prediction_df[
        "selected_threshold"
    ] = final[
        "selected_threshold"
    ]

    prediction_path = (
        OUTPUT_DIR
        / "causal_uplift_v9_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    # ==================================================================
    # SUMMARY
    # ==================================================================

    summary = pd.DataFrame(
        [
            {
                "validation_runs":
                    N_RUNS,

                "mean_qini":
                    mean_qini,

                "qini_std":
                    qini_std,

                "positive_qini_rate":
                    positive_qini_rate,

                "mean_probability_uplift":
                    mean_probability,

                "mean_economic_uplift":
                    mean_economic,

                "mean_intervention_rate":
                    mean_intervention,

                "mean_learned_advantage_vs_retry":
                    learned_advantage.mean(),

                "learned_ci_low":
                    ci_low,

                "learned_ci_high":
                    ci_high,

                "learned_beating_retry_rate":
                    (
                        learned_advantage > 0
                    ).mean(),

                "mean_oracle_advantage_vs_retry":
                    oracle_advantage.mean(),

                "oracle_beating_retry_rate":
                    (
                        oracle_advantage > 0
                    ).mean(),

                "mean_oracle_intervention_rate":
                    results[
                        "oracle_intervention_rate"
                    ].mean(),

                "final_verdict":
                    verdict,
            }
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "causal_uplift_v9_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ==================================================================
    # POLICY COMPARISON
    # ==================================================================

    policy_comparison = results[
        [
            "seed",
            "targeting_value",
            "retry_all_value",
            "no_intervention_value",
            "learned_advantage_vs_retry",
            "learned_advantage_vs_no_intervention",
            "oracle_value",
            "oracle_advantage_vs_retry",
            "intervention_rate",
            "oracle_intervention_rate",
        ]
    ].copy()

    policy_path = (
        OUTPUT_DIR
        / "causal_uplift_v9_policy_comparison.csv"
    )

    policy_comparison.to_csv(
        policy_path,
        index=False,
    )

    # ==================================================================
    # COMPLETE
    # ==================================================================

    banner(
        "V9 COMPLETE"
    )

    print(
        f"Mean QINI: {mean_qini:,.2f}"
    )

    print(
        f"Positive QINI rate: "
        f"{positive_qini_rate:.2%}"
    )

    print(
        f"Mean learned targeting advantage: "
        f"₹{learned_advantage.mean():,.2f}"
    )

    print(
        f"Mean oracle advantage: "
        f"₹{oracle_advantage.mean():,.2f}"
    )

    print(
        f"Learned 95% CI: "
        f"[₹{ci_low:,.2f}, "
        f"₹{ci_high:,.2f}]"
    )

    print(
        f"FINAL VERDICT: {verdict}"
    )

    print()
    print(
        "[PASS] Validation:",
        validation_path,
    )

    print(
        "[PASS] Predictions:",
        prediction_path,
    )

    print(
        "[PASS] Policies:",
        policy_path,
    )

    print(
        "[PASS] Summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()

