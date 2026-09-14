"""
======================================================================
RECLAIM CAUSAL UPLIFT OPTIMIZER V10.1
======================================================================

V10 PRESERVED + POSITIVE INTERVENTION POLICY SELECTION

Main goals:
    1. Preserve V10's successful economic/QINI framework.
    2. Preserve repeated outer held-out validation.
    3. Preserve leakage protection.
    4. Preserve validation-only policy selection.
    5. Preserve untouched outer test evaluation.
    6. Prevent degenerate 0% intervention policies.
    7. Select the strongest economically justified POSITIVE
       intervention policy from the inner validation set.

IMPORTANT:
    This version does NOT force a fixed 10%, 20%, etc. intervention rate.

    Instead:
        - rank test/validation rows by learned economic uplift
        - evaluate several positive intervention candidates
        - select the candidate with the strongest validation economics
        - require intervention_rate > 0
        - apply that policy to the untouched outer test set

======================================================================
"""

from pathlib import Path
import warnings

import numpy as np
# NumPy compatibility: np.trapz was removed in NumPy 2.x
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

VERSION = "V10.1"

DATA_FILE = Path(
    r"D:\RECLAIM\data\generated\reclaim_causal_experiment_v3.csv"
)

OUTPUT_DIR = Path(
    r"D:\RECLAIM\data\generated"
)

N_RUNS = 20
OUTER_TEST_SIZE = 0.30
INNER_VALIDATION_SIZE = 0.25

RF_ESTIMATORS = 250
RF_MIN_LEAF = 8

BASE_SEED = 1000

# Positive intervention candidates.
#
# These are NOT forced intervention rates.
# They are candidate policies evaluated on the validation set.
INTERVENTION_RATES = np.array([
    0.01,
    0.02,
    0.03,
    0.05,
    0.07,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
])

# Economic layer parameters.
# Kept deliberately simple and deterministic.
#
# The policy decision uses incremental economic value:
#
#       treatment economic value
#       -
#       control economic value
#
# Positive incremental value means intervention is economically
# preferable for that transaction.
#
# V10's strong result came from preserving the economic policy layer.
# This version therefore does not modify the causal outcome columns.
ECONOMIC_COST_MULTIPLIER = 0.05
FRICTION_MULTIPLIER = 1.0


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def money(x):
    return f"₹{x:,.2f}"


# ======================================================================
# COLUMN DISCOVERY
# ======================================================================

def first_existing(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def identify_schema(df):

    treatment = first_existing(
        df,
        [
            "treatment",
            "treatment_group",
            "treatment_assignment_before_outcome",
        ],
    )

    recovery = first_existing(
        df,
        [
            "recovered",
            "recovery",
            "recovery_flag",
        ],
    )

    effect = first_existing(
        df,
        [
            "treatment_effect",
            "effect",
            "incremental_recovery_probability",
        ],
    )

    amount = first_existing(
        df,
        [
            "amount",
            "amount_prediction",
        ],
    )

    control_probability = first_existing(
        df,
        [
            "control_recovery_probability",
            "expected_control_recovery",
            "base_probability",
        ],
    )

    treatment_probability = first_existing(
        df,
        [
            "treatment_recovery_probability",
            "expected_treatment_recovery",
        ],
    )

    incremental_probability = first_existing(
        df,
        [
            "incremental_recovery_probability",
            "treatment_effect",
        ],
    )

    observed_net = first_existing(
        df,
        [
            "net_recovered_value",
            "expected_net_recovery",
            "expected_net_recovery_prediction",
        ],
    )

    control_net = first_existing(
        df,
        [
            "expected_control_net",
            "control_net",
            "net_control",
        ],
    )

    treatment_net = first_existing(
        df,
        [
            "expected_treatment_net",
            "treatment_net",
            "net_treatment",
        ],
    )

    incremental_net = first_existing(
        df,
        [
            "expected_incremental_net_value",
            "incremental_net",
        ],
    )

    return {
        "treatment": treatment,
        "recovery": recovery,
        "effect": effect,
        "amount": amount,
        "control_probability": control_probability,
        "treatment_probability": treatment_probability,
        "incremental_probability": incremental_probability,
        "observed_net": observed_net,
        "control_net": control_net,
        "treatment_net": treatment_net,
        "incremental_net": incremental_net,
    }


# ======================================================================
# VALIDATION
# ======================================================================

def validate_dataset(df, schema):

    required = [
        schema["treatment"],
        schema["recovery"],
        schema["effect"],
    ]

    missing = [
        x for x in required
        if x is None
    ]

    if missing:
        raise ValueError(
            f"Missing core causal columns: {missing}"
        )

    print("[PASS] Required columns present")

    treatment_values = set(
        pd.to_numeric(
            df[schema["treatment"]],
            errors="coerce"
        ).dropna().unique()
    )

    if not treatment_values.issubset({0, 1}):
        raise ValueError(
            f"Treatment must be binary 0/1. Found: "
            f"{sorted(treatment_values)}"
        )

    print("[PASS] Treatment assignment valid")

    recovery_values = set(
        pd.to_numeric(
            df[schema["recovery"]],
            errors="coerce"
        ).dropna().unique()
    )

    if not recovery_values.issubset({0, 1}):
        raise ValueError(
            f"Recovery outcome must be binary 0/1. "
            f"Found: {sorted(recovery_values)}"
        )

    print("[PASS] Recovery outcome valid")

    effect_values = pd.to_numeric(
        df[schema["effect"]],
        errors="coerce"
    )

    if effect_values.isna().any():
        raise ValueError(
            "Treatment-effect column contains NaN values."
        )

    print("[PASS] Treatment-effect values valid")


# ======================================================================
# ECONOMIC LAYER
# ======================================================================

def create_economic_layer(df, schema):

    banner("CREATING V10.1 ECONOMIC POLICY LAYER")

    out = df.copy()

    amount_col = schema["amount"]

    if amount_col is None:
        amount = np.ones(len(out)) * 1000.0
    else:
        amount = pd.to_numeric(
            out[amount_col],
            errors="coerce"
        ).fillna(
            out[amount_col].median()
        ).to_numpy(dtype=float)

    # --------------------------------------------------------------
    # Existing economic columns are preferred.
    # --------------------------------------------------------------

    if schema["control_net"] is not None:
        control_value = pd.to_numeric(
            out[schema["control_net"]],
            errors="coerce"
        ).to_numpy(dtype=float)

    elif "expected_control_recovery" in out.columns:

        control_probability = pd.to_numeric(
            out["expected_control_recovery"],
            errors="coerce"
        ).fillna(0).to_numpy(dtype=float)

        control_value = (
            amount * control_probability
        )

    else:

        control_probability = (
            pd.to_numeric(
                out[schema["control_probability"]],
                errors="coerce"
            )
            .fillna(0)
            .to_numpy(dtype=float)
        )

        control_value = (
            amount * control_probability
        )

    # --------------------------------------------------------------
    # Existing incremental economic value.
    # --------------------------------------------------------------

    if schema["incremental_net"] is not None:

        incremental_value = pd.to_numeric(
            out[schema["incremental_net"]],
            errors="coerce"
        ).fillna(0).to_numpy(dtype=float)

    else:

        # V10-style economic construction.
        #
        # Incremental recovery probability is converted into
        # incremental monetary value and friction/cost is removed.
        if schema["incremental_probability"] is not None:

            probability_uplift = pd.to_numeric(
                out[schema["incremental_probability"]],
                errors="coerce"
            ).fillna(0).to_numpy(dtype=float)

        else:

            probability_uplift = np.zeros(
                len(out),
                dtype=float
            )

        gross_incremental_value = (
            amount *
            probability_uplift
        )

        # Intervention cost.
        if "intervention_cost" in out.columns:

            intervention_cost = pd.to_numeric(
                out["intervention_cost"],
                errors="coerce"
            ).fillna(0).to_numpy(dtype=float)

        elif "expected_intervention_cost" in out.columns:

            intervention_cost = pd.to_numeric(
                out["expected_intervention_cost"],
                errors="coerce"
            ).fillna(0).to_numpy(dtype=float)

        else:

            intervention_cost = (
                amount *
                ECONOMIC_COST_MULTIPLIER
            )

        # Customer friction.
        if "customer_friction_cost" in out.columns:

            friction = pd.to_numeric(
                out["customer_friction_cost"],
                errors="coerce"
            ).fillna(0).to_numpy(dtype=float)

        elif "expected_friction_cost" in out.columns:

            friction = pd.to_numeric(
                out["expected_friction_cost"],
                errors="coerce"
            ).fillna(0).to_numpy(dtype=float)

        else:

            friction = np.zeros(
                len(out),
                dtype=float
            )

        incremental_value = (
            gross_incremental_value
            - intervention_cost
            - (
                friction *
                FRICTION_MULTIPLIER
            )
        )

    treatment_value = (
        control_value +
        incremental_value
    )

    out["_v101_control_value"] = control_value
    out["_v101_incremental_value"] = incremental_value
    out["_v101_treatment_value"] = treatment_value

    positive = incremental_value > 0

    print("Economic layer created.")
    print(
        "Positive incremental-value rows:",
        int(positive.sum())
    )
    print(
        "Negative incremental-value rows:",
        int((~positive).sum())
    )
    print(
        "Mean incremental value:",
        money(np.mean(incremental_value))
    )
    print(
        "Std incremental value:",
        money(np.std(incremental_value))
    )

    return out


# ======================================================================
# FEATURE SELECTION / LEAKAGE PROTECTION
# ======================================================================

def build_features(df, schema):

    forbidden = {
        schema["treatment"],
        schema["recovery"],
        schema["effect"],
        schema["control_probability"],
        schema["treatment_probability"],
        schema["incremental_probability"],
        schema["observed_net"],
        schema["control_net"],
        schema["treatment_net"],
        schema["incremental_net"],
    }

    forbidden.update(
        {
            "_v101_control_value",
            "_v101_incremental_value",
            "_v101_treatment_value",

            "recovered_amount",
            "net_recovered_value",
            "recovery_status",

            "treatment_group",
            "treatment_assignment_before_outcome",
            "experiment_version",
            "experiment_transaction_id",
            "experiment_source_transaction_id",

            "treatment_effect",
            "incremental_recovery_probability",
            "observed_recovery_probability",

            "expected_control_recovery",
            "expected_treatment_recovery",
            "expected_incremental_recovery",
            "expected_incremental_net_value",

            "control_recovery_probability",
            "treatment_recovery_probability",

            "customer_friction_cost",
            "intervention_cost",
        }
    )

    features = []

    for col in df.columns:

        if col in forbidden:
            continue

        if col.startswith("_v101_"):
            continue

        if col in [
            "transaction_id",
            "customer_id",
            "timestamp",
            "status",
            "failure_reason",
            "previous_failure_reason",
            "recommended_policy",
            "recommended_action",
            "decision_reason",
            "priority",
        ]:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):

            if df[col].nunique(dropna=True) > 1:
                features.append(col)

    if not features:
        raise ValueError(
            "No usable numeric modeling features found."
        )

    X = df[features].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    X = X.fillna(0)

    print()
    print("=" * 70)
    print("LEAKAGE PROTECTION")
    print("=" * 70)
    print(
        "[PASS] Outcome/treatment/counterfactual "
        "columns excluded"
    )
    print(
        "Modeling features:",
        len(features)
    )

    return features, X


# ======================================================================
# MODEL
# ======================================================================

def fit_uplift_models(
    train,
    X_train,
    features,
    schema,
    seed
):

    t_col = schema["treatment"]
    y_col = schema["recovery"]

    treatment_mask = (
        train[t_col].to_numpy() == 1
    )

    control_mask = (
        train[t_col].to_numpy() == 0
    )

    if treatment_mask.sum() < 20:
        raise ValueError(
            "Too few treatment observations."
        )

    if control_mask.sum() < 20:
        raise ValueError(
            "Too few control observations."
        )

    model_t = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        min_samples_leaf=RF_MIN_LEAF,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    model_c = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        min_samples_leaf=RF_MIN_LEAF,
        random_state=seed + 10000,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    model_t.fit(
        X_train.loc[
            treatment_mask,
            features
        ],
        train.loc[
            treatment_mask,
            y_col
        ]
    )

    model_c.fit(
        X_train.loc[
            control_mask,
            features
        ],
        train.loc[
            control_mask,
            y_col
        ]
    )

    return model_t, model_c


def predict_uplift(
    model_t,
    model_c,
    X
):

    p_t = model_t.predict_proba(X)[:, 1]
    p_c = model_c.predict_proba(X)[:, 1]

    uplift = p_t - p_c

    return p_t, p_c, uplift


# ======================================================================
# QINI
# ======================================================================

def calculate_qini(
    treatment,
    outcome,
    uplift_score
):

    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome)
    uplift_score = np.asarray(uplift_score)

    order = np.argsort(
        -uplift_score
    )

    t = treatment[order]
    y = outcome[order]

    treated = np.cumsum(t)
    control = np.cumsum(1 - t)

    treated_outcomes = np.cumsum(
        y * t
    )

    control_outcomes = np.cumsum(
        y * (1 - t)
    )

    treated_rate = np.divide(
        treated_outcomes,
        treated,
        out=np.zeros_like(
            treated_outcomes,
            dtype=float
        ),
        where=treated > 0
    )

    control_rate = np.divide(
        control_outcomes,
        control,
        out=np.zeros_like(
            control_outcomes,
            dtype=float
        ),
        where=control > 0
    )

    incremental = (
        treated_rate -
        control_rate
    )

    qini_curve = (
        incremental *
        np.arange(1, len(y) + 1)
    )

    return float(
        np.trapezoid(
            qini_curve
        )
    )


# ======================================================================
# POLICY EVALUATION
# ======================================================================

def evaluate_policy(
    policy,
    control_value,
    treatment_value
):

    policy = np.asarray(policy).astype(bool)

    selected_value = np.where(
        policy,
        treatment_value,
        control_value
    )

    return {
        "value": float(
            selected_value.mean()
        ),
        "intervention_rate": float(
            policy.mean()
        ),
    }


# ======================================================================
# POSITIVE INTERVENTION POLICY SELECTOR
# ======================================================================

def select_positive_intervention_policy(
    validation,
    economic_score,
    seed
):

    control_value = validation[
        "_v101_control_value"
    ].to_numpy(dtype=float)

    treatment_value = validation[
        "_v101_treatment_value"
    ].to_numpy(dtype=float)

    economic_score = np.asarray(
        economic_score,
        dtype=float
    )

    n = len(validation)

    ranking = np.argsort(
        -economic_score
    )

    candidates = []

    for requested_rate in INTERVENTION_RATES:

        k = max(
            1,
            int(
                np.ceil(
                    n * requested_rate
                )
            )
        )

        k = min(k, n)

        policy = np.zeros(
            n,
            dtype=bool
        )

        policy[
            ranking[:k]
        ] = True

        result = evaluate_policy(
            policy,
            control_value,
            treatment_value
        )

        candidates.append(
            {
                "requested_rate":
                    requested_rate,
                "actual_rate":
                    result[
                        "intervention_rate"
                    ],
                "value":
                    result["value"],
                "policy":
                    policy.copy(),
            }
        )

    # --------------------------------------------------------------
    # IMPORTANT V10.1 RULE
    #
    # Do NOT permit 0%.
    #
    # Every candidate above has a strictly positive intervention
    # rate. Therefore the selected policy is guaranteed to have
    # positive intervention on the validation set.
    # --------------------------------------------------------------

    candidates = sorted(
        candidates,
        key=lambda x: x["value"],
        reverse=True
    )

    selected = candidates[0]

    return selected, candidates


# ======================================================================
# ORACLE
# ======================================================================

def oracle_policy(
    df
):

    incremental = df[
        "_v101_incremental_value"
    ].to_numpy(dtype=float)

    control = df[
        "_v101_control_value"
    ].to_numpy(dtype=float)

    treatment = df[
        "_v101_treatment_value"
    ].to_numpy(dtype=float)

    policy = incremental > 0

    value = np.where(
        policy,
        treatment,
        control
    )

    retry_all = treatment.mean()

    return {
        "value":
            float(value.mean()),

        "retry_all":
            float(retry_all),

        "advantage":
            float(
                value.mean() -
                retry_all
            ),

        "intervention_rate":
            float(policy.mean()),
    }


# ======================================================================
# ONE VALIDATION RUN
# ======================================================================

def run_single_validation(
    df,
    X,
    features,
    schema,
    seed
):

    indices = np.arange(
        len(df)
    )

    train_idx, test_idx = train_test_split(
        indices,
        test_size=OUTER_TEST_SIZE,
        random_state=seed,
        stratify=df[
            schema["treatment"]
        ]
    )

    train_outer = df.iloc[
        train_idx
    ].copy()

    test_outer = df.iloc[
        test_idx
    ].copy()

    X_train_outer = X.iloc[
        train_idx
    ].copy()

    X_test_outer = X.iloc[
        test_idx
    ].copy()

    # --------------------------------------------------------------
    # INNER SPLIT
    # --------------------------------------------------------------

    inner_train_idx, inner_val_idx = train_test_split(
        np.arange(
            len(train_outer)
        ),
        test_size=INNER_VALIDATION_SIZE,
        random_state=seed + 50000,
        stratify=train_outer[
            schema["treatment"]
        ]
    )

    inner_train = train_outer.iloc[
        inner_train_idx
    ].copy()

    inner_val = train_outer.iloc[
        inner_val_idx
    ].copy()

    X_inner_train = X_train_outer.iloc[
        inner_train_idx
    ].copy()

    X_inner_val = X_train_outer.iloc[
        inner_val_idx
    ].copy()

    # --------------------------------------------------------------
    # FIT ONLY ON INNER TRAIN
    # --------------------------------------------------------------

    model_t, model_c = fit_uplift_models(
        inner_train,
        X_inner_train,
        features,
        schema,
        seed
    )

    # --------------------------------------------------------------
    # INNER VALIDATION PREDICTIONS
    # --------------------------------------------------------------

    p_t_val, p_c_val, uplift_val = predict_uplift(
        model_t,
        model_c,
        X_inner_val[features]
    )

    # --------------------------------------------------------------
    # ECONOMIC SCORE
    #
    # Probability uplift is retained as an important causal signal.
    # Economic ranking is used as the policy score.
    #
    # This is the crucial V10.1 change:
    # the ranking is preserved, but a positive intervention
    # candidate MUST be selected.
    # --------------------------------------------------------------

    amount_col = schema["amount"]

    if amount_col is not None:

        amount_val = pd.to_numeric(
            inner_val[amount_col],
            errors="coerce"
        ).fillna(
            inner_val[amount_col].median()
        ).to_numpy(dtype=float)

    else:

        amount_val = np.ones(
            len(inner_val)
        )

    economic_score_val = (
        uplift_val *
        amount_val
    )

    # Blend probability and economics.
    #
    # This prevents a noisy economic magnitude from completely
    # replacing the causal uplift signal.
    economic_score_val = (
        0.50 * uplift_val +
        0.50 * (
            economic_score_val /
            (
                np.median(
                    np.abs(
                        economic_score_val
                    )
                ) + 1e-9
            )
        )
    )

    selected, candidates = (
        select_positive_intervention_policy(
            inner_val,
            economic_score_val,
            seed
        )
    )

    selected_rate = selected[
        "actual_rate"
    ]

    selected_value = selected[
        "value"
    ]

    # --------------------------------------------------------------
    # REFIT ON COMPLETE OUTER TRAIN
    # --------------------------------------------------------------

    final_model_t, final_model_c = fit_uplift_models(
        train_outer,
        X_train_outer,
        features,
        schema,
        seed + 70000
    )

    # --------------------------------------------------------------
    # OUTER TEST
    # --------------------------------------------------------------

    p_t_test, p_c_test, uplift_test = predict_uplift(
        final_model_t,
        final_model_c,
        X_test_outer[features]
    )

    amount_test = pd.to_numeric(
        test_outer[amount_col],
        errors="coerce"
    ).fillna(
        df[amount_col].median()
    ).to_numpy(dtype=float)

    economic_score_test = (
        uplift_test *
        amount_test
    )

    economic_score_test = (
        0.50 * uplift_test +
        0.50 * (
            economic_score_test /
            (
                np.median(
                    np.abs(
                        economic_score_test
                    )
                ) + 1e-9
            )
        )
    )

    # --------------------------------------------------------------
    # APPLY THE VALIDATION-SELECTED RATE.
    #
    # RATE is selected ONLY on inner validation.
    # Ranking is generated on the outer test.
    # --------------------------------------------------------------

    ranking_test = np.argsort(
        -economic_score_test
    )

    k_test = max(
        1,
        int(
            np.ceil(
                len(test_outer) *
                selected_rate
            )
        )
    )

    k_test = min(
        k_test,
        len(test_outer)
    )

    policy_test = np.zeros(
        len(test_outer),
        dtype=bool
    )

    policy_test[
        ranking_test[:k_test]
    ] = True

    control_test = test_outer[
        "_v101_control_value"
    ].to_numpy(dtype=float)

    treatment_test = test_outer[
        "_v101_treatment_value"
    ].to_numpy(dtype=float)

    targeting_value = np.where(
        policy_test,
        treatment_test,
        control_test
    ).mean()

    retry_all_value = (
        treatment_test.mean()
    )

    no_intervention_value = (
        control_test.mean()
    )

    learned_advantage = (
        targeting_value -
        retry_all_value
    )

    no_intervention_advantage = (
        targeting_value -
        no_intervention_value
    )

    oracle = oracle_policy(
        test_outer
    )

    qini = calculate_qini(
        test_outer[
            schema["treatment"]
        ].to_numpy(),
        test_outer[
            schema["recovery"]
        ].to_numpy(),
        uplift_test
    )

    return {
        "seed": seed,

        "train_rows":
            len(train_outer),

        "test_rows":
            len(test_outer),

        "mean_probability_uplift":
            float(uplift_test.mean()),

        "selected_rate":
            float(selected_rate),

        "intervention_rate":
            float(policy_test.mean()),

        "selected_validation_value":
            float(selected_value),

        "qini":
            float(qini),

        "targeting_value":
            float(targeting_value),

        "retry_all_value":
            float(retry_all_value),

        "no_intervention_value":
            float(no_intervention_value),

        "learned_advantage":
            float(learned_advantage),

        "no_intervention_advantage":
            float(no_intervention_advantage),

        "oracle_advantage":
            float(oracle["advantage"]),

        "oracle_intervention_rate":
            float(oracle["intervention_rate"]),

        "oracle_opportunity_captured":
            float(
                learned_advantage /
                oracle["advantage"]
            )
            if oracle["advantage"] > 0
            else np.nan,
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    banner(
        "RECLAIM CAUSAL UPLIFT OPTIMIZER V10.1"
    )

    print(
        "V10 PRESERVED + POSITIVE INTERVENTION POLICY SELECTION"
    )

    print(
        f"File: {DATA_FILE}"
    )

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"[PASS] Rows loaded: {len(df)}"
    )

    print(
        f"[PASS] Columns loaded: {len(df.columns)}"
    )

    # --------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------

    banner(
        "VALIDATING V3 EXPERIMENT"
    )

    schema = identify_schema(
        df
    )

    validate_dataset(
        df,
        schema
    )

    # --------------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------------

    treatment_mask = (
        pd.to_numeric(
            df[schema["treatment"]],
            errors="coerce"
        ) == 1
    )

    control_mask = ~treatment_mask

    recovery = pd.to_numeric(
        df[schema["recovery"]],
        errors="coerce"
    )

    treatment_recovery = recovery[
        treatment_mask
    ].mean()

    control_recovery = recovery[
        control_mask
    ].mean()

    effect = pd.to_numeric(
        df[schema["effect"]],
        errors="coerce"
    )

    banner(
        "EXPERIMENT BASELINE"
    )

    print(
        "Treatment recovery rate:",
        f"{treatment_recovery:.2%}"
    )

    print(
        "Control recovery rate:  ",
        f"{control_recovery:.2%}"
    )

    print(
        "Observed recovery lift: ",
        f"{treatment_recovery - control_recovery:+.2%}"
    )

    # --------------------------------------------------------------
    # HETEROGENEITY
    # --------------------------------------------------------------

    banner(
        "TREATMENT EFFECT HETEROGENEITY"
    )

    print(
        "Mean effect:",
        f"{effect.mean():.2%}"
    )

    print(
        "Std effect:",
        f"{effect.std():.2%}"
    )

    print(
        "Minimum effect:",
        f"{effect.min():.2%}"
    )

    print(
        "Maximum effect:",
        f"{effect.max():.2%}"
    )

    print(
        "Positive effects:",
        int((effect > 0).sum())
    )

    print(
        "Negative effects:",
        int((effect < 0).sum())
    )

    print(
        "Zero effects:",
        int((effect == 0).sum())
    )

    # --------------------------------------------------------------
    # ECONOMIC LAYER
    # --------------------------------------------------------------

    df = create_economic_layer(
        df,
        schema
    )

    # --------------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------------

    features, X = build_features(
        df,
        schema
    )

    # --------------------------------------------------------------
    # REPEATED VALIDATION
    # --------------------------------------------------------------

    banner(
        "RUNNING V10.1 REPEATED HELD-OUT VALIDATION"
    )

    print(
        f"Validation runs: {N_RUNS}"
    )

    print(
        f"Outer test size: "
        f"{OUTER_TEST_SIZE:.0%}"
    )

    print(
        f"Inner validation size: "
        f"{INNER_VALIDATION_SIZE:.0%}"
    )

    results = []

    for i in range(N_RUNS):

        seed = BASE_SEED + i

        print()
        print("-" * 70)

        print(
            f"[{i + 1:02d}/{N_RUNS}] "
            f"SEED {seed}"
        )

        result = run_single_validation(
            df,
            X,
            features,
            schema,
            seed
        )

        results.append(
            result
        )

        print(
            f"Training rows: "
            f"{result['train_rows']} | "
            f"Test rows: "
            f"{result['test_rows']}"
        )

        print(
            "Mean probability uplift:",
            f"{result['mean_probability_uplift']:.2%}"
        )

        print(
            "Selected intervention rate:",
            f"{result['selected_rate']:.2%}"
        )

        print(
            "Actual intervention rate:",
            f"{result['intervention_rate']:.2%}"
        )

        print(
            "QINI:",
            f"{result['qini']:,.2f}"
        )

        print(
            "Targeting value/transaction:",
            money(
                result[
                    "targeting_value"
                ]
            )
        )

        print(
            "RETRY_ALL value/transaction:",
            money(
                result[
                    "retry_all_value"
                ]
            )
        )

        print(
            "Learned advantage vs RETRY_ALL:",
            money(
                result[
                    "learned_advantage"
                ]
            )
        )

        print(
            "Oracle advantage vs RETRY_ALL:",
            money(
                result[
                    "oracle_advantage"
                ]
            )
        )

        print(
            "Oracle opportunity captured:",
            (
                f"{result['oracle_opportunity_captured']:.1%}"
                if np.isfinite(
                    result[
                        "oracle_opportunity_captured"
                    ]
                )
                else "N/A"
            )
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------

    banner(
        "V10.1 STABILITY SUMMARY"
    )

    qini_mean = results_df[
        "qini"
    ].mean()

    qini_std = results_df[
        "qini"
    ].std(ddof=1)

    qini_positive_rate = (
        results_df[
            "qini"
        ] > 0
    ).mean()

    qini_se = (
        qini_std /
        np.sqrt(
            len(results_df)
        )
    )

    qini_ci_low = (
        qini_mean -
        1.96 * qini_se
    )

    qini_ci_high = (
        qini_mean +
        1.96 * qini_se
    )

    mean_prob_uplift = results_df[
        "mean_probability_uplift"
    ].mean()

    mean_intervention = results_df[
        "intervention_rate"
    ].mean()

    print(
        "Validation runs:",
        N_RUNS
    )

    print(
        "Mean QINI:",
        f"{qini_mean:,.2f}"
    )

    print(
        "QINI 95% CI:",
        f"[{qini_ci_low:,.2f}, "
        f"{qini_ci_high:,.2f}]"
    )

    print(
        "Positive-QINI rate:",
        f"{qini_positive_rate:.2%}"
    )

    print(
        "Mean probability uplift:",
        f"{mean_prob_uplift:.2%}"
    )

    print(
        "Mean intervention rate:",
        f"{mean_intervention:.2%}"
    )

    # --------------------------------------------------------------
    # NO INTERVENTION
    # --------------------------------------------------------------

    banner(
        "TARGETING VS NO_INTERVENTION"
    )

    no_int_advantage = results_df[
        "no_intervention_advantage"
    ]

    print(
        "Mean advantage:",
        money(
            no_int_advantage.mean()
        )
    )

    print(
        "Runs beating NO_INTERVENTION:",
        f"{(no_int_advantage > 0).mean():.2%}"
    )

    # --------------------------------------------------------------
    # RETRY ALL
    # --------------------------------------------------------------

    banner(
        "TARGETING VS RETRY_ALL"
    )

    advantages = results_df[
        "learned_advantage"
    ]

    mean_advantage = (
        advantages.mean()
    )

    advantage_std = (
        advantages.std(ddof=1)
    )

    advantage_se = (
        advantage_std /
        np.sqrt(
            len(advantages)
        )
    )

    ci_low = (
        mean_advantage -
        1.96 * advantage_se
    )

    ci_high = (
        mean_advantage +
        1.96 * advantage_se
    )

    print(
        "Mean learned advantage:",
        money(
            mean_advantage
        )
    )

    print(
        "95% CI:",
        f"[{money(ci_low)}, "
        f"{money(ci_high)}]"
    )

    print(
        "Runs beating RETRY_ALL:",
        f"{(advantages > 0).mean():.2%}"
    )

    # --------------------------------------------------------------
    # ORACLE
    # --------------------------------------------------------------

    banner(
        "ORACLE BENCHMARK"
    )

    oracle_advantage = results_df[
        "oracle_advantage"
    ]

    oracle_intervention = results_df[
        "oracle_intervention_rate"
    ]

    opportunity = results_df[
        "oracle_opportunity_captured"
    ]

    print(
        "Mean oracle advantage:",
        money(
            oracle_advantage.mean()
        )
    )

    print(
        "Oracle positive-rate:",
        f"{(oracle_advantage > 0).mean():.2%}"
    )

    print(
        "Mean oracle intervention rate:",
        f"{oracle_intervention.mean():.2%}"
    )

    print(
        "Mean oracle opportunity captured:",
        (
            f"{opportunity.mean():.2%}"
            if opportunity.notna().any()
            else "N/A"
        )
    )

    # --------------------------------------------------------------
    # POLICY STABILITY
    # --------------------------------------------------------------

    banner(
        "POLICY STABILITY"
    )

    threshold_summary = (
        results_df[
            "selected_rate"
        ]
        .round(4)
        .value_counts()
        .sort_index()
    )

    print(
        threshold_summary
        .to_string()
    )

    # --------------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------------

    banner(
        "FINAL V10.1 CAUSAL POLICY VERDICT"
    )

    print(
        "Mean QINI:",
        f"{qini_mean:,.2f}"
    )

    print(
        "Positive QINI rate:",
        f"{qini_positive_rate:.2%}"
    )

    print(
        "Mean learned advantage:",
        money(
            mean_advantage
        )
    )

    print(
        "Learned 95% CI:",
        f"[{money(ci_low)}, "
        f"{money(ci_high)}]"
    )

    print(
        "Mean oracle advantage:",
        money(
            oracle_advantage.mean()
        )
    )

    print(
        "Mean intervention rate:",
        f"{mean_intervention:.2%}"
    )

    # --------------------------------------------------------------
    # VERDICT LOGIC
    # --------------------------------------------------------------

    if (
        mean_intervention > 0
        and
        mean_advantage > 0
        and
        ci_low > 0
        and
        qini_positive_rate >= 0.50
    ):

        verdict = (
            "V10.1_TARGETING_SUPPORTED_WITH_POSITIVE_INTERVENTION"
        )

    elif (
        mean_intervention > 0
        and
        mean_advantage > 0
    ):

        verdict = (
            "V10.1_LEARNED_TARGETING_BEATS_RETRY_ALL"
        )

    elif (
        mean_intervention > 0
        and
        qini_positive_rate >= 0.50
    ):

        verdict = (
            "V10.1_POSITIVE_INTERVENTION_QINI_SUPPORTED"
        )

    else:

        verdict = (
            "V10.1_INTERVENTION_NEEDS_REVIEW"
        )

    print()
    print(
        "FINAL VERDICT:",
        verdict
    )

    # --------------------------------------------------------------
    # FINAL FULL-DATA POLICY
    # --------------------------------------------------------------

    banner(
        "GENERATING FINAL FULL-DATA POLICY SCORE"
    )

    # Train final models on all available data.
    final_model_t, final_model_c = fit_uplift_models(
        df,
        X,
        features,
        schema,
        BASE_SEED + 99999
    )

    _, _, full_uplift = predict_uplift(
        final_model_t,
        final_model_c,
        X[features]
    )

    amount_full = pd.to_numeric(
        df[
            schema["amount"]
        ],
        errors="coerce"
    ).fillna(
        df[
            schema["amount"]
        ].median()
    ).to_numpy(dtype=float)

    full_economic_score = (
        0.50 * full_uplift
        +
        0.50 * (
            (
                full_uplift *
                amount_full
            )
            /
            (
                np.median(
                    np.abs(
                        full_uplift *
                        amount_full
                    )
                )
                + 1e-9
            )
        )
    )

    median_selected_rate = float(
        results_df[
            "selected_rate"
        ].median()
    )

    ranking_full = np.argsort(
        -full_economic_score
    )

    k_full = max(
        1,
        int(
            np.ceil(
                len(df) *
                median_selected_rate
            )
        )
    )

    policy_full = np.zeros(
        len(df),
        dtype=bool
    )

    policy_full[
        ranking_full[:k_full]
    ] = True

    full_control = df[
        "_v101_control_value"
    ].to_numpy(dtype=float)

    full_treatment = df[
        "_v101_treatment_value"
    ].to_numpy(dtype=float)

    full_targeting_value = np.where(
        policy_full,
        full_treatment,
        full_control
    ).mean()

    full_retry_all = (
        full_treatment.mean()
    )

    full_advantage = (
        full_targeting_value -
        full_retry_all
    )

    print(
        "Median selected intervention rate:",
        f"{median_selected_rate:.2%}"
    )

    print(
        "Full-data intervention rate:",
        f"{policy_full.mean():.2%}"
    )

    print(
        "Full-data targeting value:",
        money(
            full_targeting_value
        )
    )

    print(
        "Full-data RETRY_ALL value:",
        money(
            full_retry_all
        )
    )

    print(
        "Full-data advantage:",
        money(
            full_advantage
        )
    )

    # --------------------------------------------------------------
    # SAVE VALIDATION RESULTS
    # --------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    validation_file = (
        OUTPUT_DIR /
        "causal_uplift_v10_1_validation_results.csv"
    )

    predictions_file = (
        OUTPUT_DIR /
        "causal_uplift_v10_1_predictions.csv"
    )

    policy_file = (
        OUTPUT_DIR /
        "causal_uplift_v10_1_policy_comparison.csv"
    )

    summary_file = (
        OUTPUT_DIR /
        "causal_uplift_v10_1_summary.csv"
    )

    results_df.to_csv(
        validation_file,
        index=False
    )

    # --------------------------------------------------------------
    # FULL PREDICTIONS
    # --------------------------------------------------------------

    predictions = pd.DataFrame(
        {
            "uplift_probability":
                full_uplift,

            "economic_score":
                full_economic_score,

            "intervention":
                policy_full.astype(int),

            "control_value":
                full_control,

            "treatment_value":
                full_treatment,

            "incremental_value":
                full_treatment -
                full_control,
        }
    )

    if "transaction_id" in df.columns:

        predictions.insert(
            0,
            "transaction_id",
            df[
                "transaction_id"
            ].values
        )

    predictions.to_csv(
        predictions_file,
        index=False
    )

    # --------------------------------------------------------------
    # POLICY COMPARISON
    # --------------------------------------------------------------

    policy_comparison = pd.DataFrame(
        {
            "control_value":
                full_control,

            "treatment_value":
                full_treatment,

            "incremental_value":
                full_treatment -
                full_control,

            "targeting_policy":
                policy_full.astype(int),

            "retry_all_policy":
                np.ones(
                    len(df),
                    dtype=int
                ),

            "no_intervention_policy":
                np.zeros(
                    len(df),
                    dtype=int
                ),

            "targeting_selected_value":
                np.where(
                    policy_full,
                    full_treatment,
                    full_control
                ),
        }
    )

    policy_comparison.to_csv(
        policy_file,
        index=False
    )

    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------

    summary = pd.DataFrame(
        {
            "metric": [
                "version",
                "validation_runs",
                "mean_qini",
                "qini_positive_rate",
                "mean_probability_uplift",
                "mean_intervention_rate",
                "mean_learned_advantage_vs_retry_all",
                "learned_advantage_ci_low",
                "learned_advantage_ci_high",
                "mean_oracle_advantage",
                "mean_oracle_intervention_rate",
                "full_data_intervention_rate",
                "full_data_targeting_value",
                "full_data_retry_all_value",
                "full_data_advantage",
                "final_verdict",
            ],

            "value": [
                VERSION,
                N_RUNS,
                qini_mean,
                qini_positive_rate,
                mean_prob_uplift,
                mean_intervention,
                mean_advantage,
                ci_low,
                ci_high,
                oracle_advantage.mean(),
                oracle_intervention.mean(),
                policy_full.mean(),
                full_targeting_value,
                full_retry_all,
                full_advantage,
                verdict,
            ],
        }
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    # --------------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------------

    banner(
        "V10.1 COMPLETE"
    )

    print(
        "Mean QINI:",
        f"{qini_mean:,.2f}"
    )

    print(
        "Positive QINI rate:",
        f"{qini_positive_rate:.2%}"
    )

    print(
        "Mean learned targeting advantage:",
        money(
            mean_advantage
        )
    )

    print(
        "Learned 95% CI:",
        f"[{money(ci_low)}, "
        f"{money(ci_high)}]"
    )

    print(
        "Mean oracle advantage:",
        money(
            oracle_advantage.mean()
        )
    )

    print(
        "Mean intervention rate:",
        f"{mean_intervention:.2%}"
    )

    print(
        "Full-data intervention rate:",
        f"{policy_full.mean():.2%}"
    )

    print(
        "FINAL VERDICT:",
        verdict
    )

    print()
    print(
        "[PASS] Validation:",
        validation_file
    )

    print(
        "[PASS] Predictions:",
        predictions_file
    )

    print(
        "[PASS] Policies:",
        policy_file
    )

    print(
        "[PASS] Summary:",
        summary_file
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()