"""
RECLAIM CAUSAL UPLIFT OPTIMIZER V2

Purpose
-------
Train and evaluate a causal T-learner on the improved randomized
treatment/control experiment.

IMPORTANT
---------
Outcome-derived columns are NEVER used as model features.

The model estimates:

    uplift(x) = P(Y=1 | X=x, T=1) - P(Y=1 | X=x, T=0)

where:
    T=1 -> RECLAIM intervention
    T=0 -> control
    Y   -> recovered

Evaluation is performed on held-out data.

Outputs
-------
data/generated/causal_uplift_v2_predictions.csv
data/generated/causal_uplift_v2_policy_comparison.csv
data/generated/causal_uplift_v2_summary.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 2026

TEST_SIZE = 0.30

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "reclaim_causal_experiment_v2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "generated"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "causal_uplift_v2_predictions.csv"
)

POLICY_FILE = (
    OUTPUT_DIR
    / "causal_uplift_v2_policy_comparison.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "causal_uplift_v2_summary.csv"
)


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "amount",
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
    "recovery_probability",
    "calibrated_probability",
    "confidence",
]

CATEGORICAL_FEATURES = [
    "payment_method",
    "failure_reason",
]

# Outcome-derived / post-treatment columns.
# NONE of these may enter the model.
LEAKAGE_COLUMNS = {
    "recovered",
    "recovery_flag",
    "recovered_amount",
    "observed_recovery_probability",
    "recovery_status",
    "net_recovered_value",
    "intervention_cost",
    "customer_friction_cost",
    "treatment",
    "control",
    "treatment_group",
    "control_recovery_probability",
    "treatment_recovery_probability",
    "incremental_recovery_probability",
    "expected_control_recovery",
    "expected_treatment_recovery",
    "expected_incremental_recovery",
    "expected_incremental_net_value",
    "treatment_effect",
    "treatment_assignment_before_outcome",
}


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


def pct(value):
    return f"{value * 100:.2f}%"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    header("RECLAIM CAUSAL UPLIFT OPTIMIZER V2")

    print("Loading V2 randomized treatment/control experiment...")
    print(f"File: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Experiment file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"[PASS] Rows loaded: {len(df):,}")
    print(f"[PASS] Columns loaded: {len(df.columns)}")

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_data(df):

    header("VALIDATING V2 CAUSAL EXPERIMENT")

    required = [
        "transaction_id",
        "amount",
        "treatment",
        "recovered",
        "recovered_amount",
        "intervention_cost",
        "customer_friction_cost",
        "net_recovered_value",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df["transaction_id"].is_unique:
        print("[PASS] Transaction IDs are unique")
    else:
        raise ValueError(
            "Transaction IDs are not unique."
        )

    treatments = sorted(
        df["treatment"]
        .dropna()
        .unique()
        .tolist()
    )

    print(f"Treatment values: {treatments}")

    if set(treatments) != {0, 1}:
        raise ValueError(
            "Treatment must contain exactly 0 and 1."
        )

    print("[PASS] Treatment values valid")

    treatment_rows = int(
        (df["treatment"] == 1).sum()
    )

    control_rows = int(
        (df["treatment"] == 0).sum()
    )

    print(f"Treatment rows: {treatment_rows:,}")
    print(f"Control rows:   {control_rows:,}")

    if treatment_rows == 0 or control_rows == 0:
        raise ValueError(
            "Both treatment and control groups are required."
        )

    print("[PASS] Treatment/control groups usable")

    if not df["recovered"].isin([0, 1]).all():
        raise ValueError(
            "`recovered` must contain only 0/1."
        )

    print("[PASS] Recovery outcome valid")

    for col in [
        "amount",
        "recovered_amount",
        "intervention_cost",
        "customer_friction_cost",
        "net_recovered_value",
    ]:

        values = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if values.isna().any():
            raise ValueError(
                f"Invalid numeric values in {col}"
            )

    print("[PASS] Economic variables valid")


# ============================================================
# FEATURE CHECK
# ============================================================

def select_features(df):

    header("SELECTING V2 UPLIFT FEATURES")

    numeric = [
        col
        for col in NUMERIC_FEATURES
        if col in df.columns
    ]

    categorical = [
        col
        for col in CATEGORICAL_FEATURES
        if col in df.columns
    ]

    print(
        f"Numeric features available: "
        f"{len(numeric)}"
    )

    for col in numeric:
        print(f"  [YES] {col}")

    print(
        f"Categorical features available: "
        f"{len(categorical)}"
    )

    for col in categorical:
        print(f"  [YES] {col}")

    if len(numeric) + len(categorical) < 5:
        raise ValueError(
            "Too few usable features for V2 uplift modeling."
        )

    # Safety check.
    selected = set(numeric + categorical)

    leaked = selected.intersection(
        LEAKAGE_COLUMNS
    )

    if leaked:
        raise RuntimeError(
            f"LEAKAGE DETECTED: {sorted(leaked)}"
        )

    print(
        f"[PASS] Total modeling features: "
        f"{len(selected)}"
    )

    return numeric, categorical


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(
    numeric_features,
    categorical_features
):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# MODEL
# ============================================================

def build_model(
    numeric_features,
    categorical_features,
    seed
):

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    classifier = RandomForestClassifier(
        n_estimators=350,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                classifier,
            ),
        ]
    )


# ============================================================
# OBSERVED EXPERIMENT EFFECT
# ============================================================

def print_observed_effect(df):

    header("OBSERVED EXPERIMENT EFFECT")

    treatment = df[
        df["treatment"] == 1
    ]

    control = df[
        df["treatment"] == 0
    ]

    t_rate = treatment["recovered"].mean()
    c_rate = control["recovered"].mean()

    lift = t_rate - c_rate

    print(
        f"Treatment recovery rate: "
        f"{pct(t_rate)}"
    )

    print(
        f"Control recovery rate:   "
        f"{pct(c_rate)}"
    )

    print(
        f"Observed recovery lift:  "
        f"{lift:+.2%}"
    )

    t_value = treatment[
        "net_recovered_value"
    ].sum()

    c_value = control[
        "net_recovered_value"
    ].sum()

    t_mean = treatment[
        "net_recovered_value"
    ].mean()

    c_mean = control[
        "net_recovered_value"
    ].mean()

    economic_lift = t_mean - c_mean

    print()
    print(
        f"Treatment net value: "
        f"{money(t_value)}"
    )

    print(
        f"Control net value:   "
        f"{money(c_value)}"
    )

    print(
        f"Treatment mean net:  "
        f"{money(t_mean)}"
    )

    print(
        f"Control mean net:    "
        f"{money(c_mean)}"
    )

    print(
        f"Observed incremental "
        f"net / transaction: "
        f"{money(economic_lift)}"
    )

    return {
        "treatment_rate": t_rate,
        "control_rate": c_rate,
        "observed_lift": lift,
        "treatment_value": t_value,
        "control_value": c_value,
        "economic_lift": economic_lift,
    }


# ============================================================
# TRAIN T-LEARNER
# ============================================================

def train_t_learner(
    train_df,
    numeric_features,
    categorical_features
):

    header("BUILDING V2 CAUSAL T-LEARNER")

    treatment_train = train_df[
        train_df["treatment"] == 1
    ].copy()

    control_train = train_df[
        train_df["treatment"] == 0
    ].copy()

    print(
        f"Treatment training rows: "
        f"{len(treatment_train):,}"
    )

    print(
        f"Control training rows:   "
        f"{len(control_train):,}"
    )

    print(
        f"Treatment recovery rate: "
        f"{pct(treatment_train['recovered'].mean())}"
    )

    print(
        f"Control recovery rate:   "
        f"{pct(control_train['recovered'].mean())}"
    )

    feature_columns = (
        numeric_features +
        categorical_features
    )

    X_treatment = treatment_train[
        feature_columns
    ]

    y_treatment = treatment_train[
        "recovered"
    ].astype(int)

    X_control = control_train[
        feature_columns
    ]

    y_control = control_train[
        "recovered"
    ].astype(int)

    treatment_model = build_model(
        numeric_features,
        categorical_features,
        RANDOM_SEED + 1,
    )

    control_model = build_model(
        numeric_features,
        categorical_features,
        RANDOM_SEED + 2,
    )

    treatment_model.fit(
        X_treatment,
        y_treatment,
    )

    print(
        "[PASS] Treatment outcome model trained"
    )

    control_model.fit(
        X_control,
        y_control,
    )

    print(
        "[PASS] Control outcome model trained"
    )

    return treatment_model, control_model


# ============================================================
# PREDICT UPLIFT
# ============================================================

def estimate_uplift(
    model_treatment,
    model_control,
    test_df,
    numeric_features,
    categorical_features
):

    header("ESTIMATING INDIVIDUAL TREATMENT EFFECT")

    feature_columns = (
        numeric_features +
        categorical_features
    )

    X_test = test_df[
        feature_columns
    ]

    p_treatment = (
        model_treatment
        .predict_proba(X_test)[:, 1]
    )

    p_control = (
        model_control
        .predict_proba(X_test)[:, 1]
    )

    uplift = (
        p_treatment -
        p_control
    )

    result = test_df.copy()

    result[
        "predicted_treatment_probability"
    ] = p_treatment

    result[
        "predicted_control_probability"
    ] = p_control

    result[
        "predicted_uplift"
    ] = uplift

    result[
        "predicted_incremental_recovery"
    ] = uplift

    print(
        f"Mean predicted treatment probability: "
        f"{pct(p_treatment.mean())}"
    )

    print(
        f"Mean predicted control probability:   "
        f"{pct(p_control.mean())}"
    )

    print(
        f"Mean predicted uplift: "
        f"{uplift.mean():+.2%}"
    )

    print(
        f"Positive uplift rows: "
        f"{(uplift > 0).sum():,}"
    )

    print(
        f"Negative uplift rows: "
        f"{(uplift < 0).sum():,}"
    )

    print("[PASS] Individual uplift estimated")

    return result


# ============================================================
# MODEL QUALITY
# ============================================================

def evaluate_models(
    result,
    model_treatment,
    model_control,
    numeric_features,
    categorical_features
):

    header("HELD-OUT MODEL QUALITY")

    feature_columns = (
        numeric_features +
        categorical_features
    )

    # Actual outcomes.
    y = result["recovered"].astype(int)

    p_t = result[
        "predicted_treatment_probability"
    ]

    p_c = result[
        "predicted_control_probability"
    ]

    # Brier scores.
    brier_t = brier_score_loss(
        y,
        p_t,
    )

    brier_c = brier_score_loss(
        y,
        p_c,
    )

    print(
        f"Treatment Brier score: "
        f"{brier_t:.6f}"
    )

    print(
        f"Control Brier score:   "
        f"{brier_c:.6f}"
    )

    # AUC is descriptive because the test set contains
    # both treatment/control observations.
    try:
        auc_t = roc_auc_score(
            y,
            p_t,
        )
    except Exception:
        auc_t = np.nan

    try:
        auc_c = roc_auc_score(
            y,
            p_c,
        )
    except Exception:
        auc_c = np.nan

    print(
        f"Treatment probability AUC: "
        f"{auc_t:.4f}"
    )

    print(
        f"Control probability AUC:   "
        f"{auc_c:.4f}"
    )

    return {
        "brier_treatment": brier_t,
        "brier_control": brier_c,
        "auc_treatment": auc_t,
        "auc_control": auc_c,
    }


# ============================================================
# QINI
# ============================================================

def calculate_qini(df):

    """
    QINI-style cumulative incremental gain.

    Important:
    This implementation avoids np.trapz because NumPy 2.x
    removed that alias.

    The curve is constructed by ranking transactions by
    predicted uplift and estimating cumulative treatment
    effect using randomized treatment/control observations.

    QINI is reported as the area under the incremental gain
    curve relative to the straight baseline.
    """

    if len(df) == 0:
        return 0.0

    work = df.copy()

    work = work.sort_values(
        "predicted_uplift",
        ascending=False,
    ).reset_index(drop=True)

    t = (
        work["treatment"]
        .astype(int)
        .to_numpy()
    )

    y = (
        work["recovered"]
        .astype(float)
        .to_numpy()
    )

    cumulative_t = np.cumsum(t)
    cumulative_c = np.cumsum(1 - t)

    cumulative_y_t = np.cumsum(
        y * t
    )

    cumulative_y_c = np.cumsum(
        y * (1 - t)
    )

    # Avoid division by zero.
    treatment_rate = np.divide(
        cumulative_y_t,
        cumulative_t,
        out=np.zeros_like(
            cumulative_y_t,
            dtype=float,
        ),
        where=cumulative_t > 0,
    )

    control_rate = np.divide(
        cumulative_y_c,
        cumulative_c,
        out=np.zeros_like(
            cumulative_y_c,
            dtype=float,
        ),
        where=cumulative_c > 0,
    )

    cumulative_effect = (
        cumulative_t
        * (
            treatment_rate -
            control_rate
        )
    )

    x = np.arange(
        1,
        len(work) + 1,
        dtype=float,
    )

    # np.trapezoid is the current NumPy
    # replacement for np.trapz.
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(
            cumulative_effect,
            x,
        )
    else:
        area = np.sum(
            (
                cumulative_effect[:-1]
                + cumulative_effect[1:]
            )
            / 2.0
            * np.diff(x)
        )

    return float(area)


# ============================================================
# ECONOMIC TARGETING
# ============================================================

def evaluate_targeting_policy(
    test_df,
    name,
    mask,
):

    selected = test_df[
        mask
    ].copy()

    interventions = len(selected)

    if interventions == 0:

        return {
            "policy": name,
            "interventions": 0,
            "intervention_rate": 0.0,
            "observed_recovered_payments": 0,
            "observed_recovered_value": 0.0,
            "observed_net_value": 0.0,
            "observed_net_per_intervention": 0.0,
            "predicted_incremental_net_value": 0.0,
        }

    # Because the held-out experiment contains randomized
    # treatment/control assignments, evaluate the policy
    # using the observed outcomes of selected rows.

    observed_recovered = int(
        selected["recovered"].sum()
    )

    observed_value = float(
        selected["recovered_amount"].sum()
    )

    observed_net = float(
        selected["net_recovered_value"].sum()
    )

    # Economic value predicted from uplift.
    #
    # Use amount as the recoverable value proxy and subtract
    # expected intervention/friction costs.
    predicted_incremental = (
        selected[
            "predicted_uplift"
        ]
        * selected["amount"]
    )

    predicted_incremental_net = (
        predicted_incremental
        - selected["intervention_cost"]
        - selected["customer_friction_cost"]
    )

    return {
        "policy": name,
        "interventions": interventions,
        "intervention_rate": (
            interventions / len(test_df)
        ),
        "observed_recovered_payments": (
            observed_recovered
        ),
        "observed_recovered_value": (
            observed_value
        ),
        "observed_net_value": (
            observed_net
        ),
        "observed_net_per_intervention": (
            observed_net / interventions
        ),
        "predicted_incremental_net_value": (
            float(
                predicted_incremental_net.sum()
            )
        ),
    }


# ============================================================
# POLICY EVALUATION
# ============================================================

def evaluate_policies(result):

    header("TESTING HELD-OUT CAUSAL TARGETING POLICIES")

    policies = []

    n = len(result)

    # --------------------------------------------------------
    # NO INTERVENTION
    # --------------------------------------------------------

    policies.append(
        evaluate_targeting_policy(
            result,
            "NO_INTERVENTION",
            np.zeros(
                n,
                dtype=bool,
            ),
        )
    )

    # --------------------------------------------------------
    # RETRY ALL
    # --------------------------------------------------------

    policies.append(
        evaluate_targeting_policy(
            result,
            "RETRY_ALL",
            np.ones(
                n,
                dtype=bool,
            ),
        )
    )

    # --------------------------------------------------------
    # TOP K UPLIFT POLICIES
    # --------------------------------------------------------

    ranking = result.sort_values(
        "predicted_uplift",
        ascending=False,
    )

    top_percentages = [
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
        100,
    ]

    for percent in top_percentages:

        k = max(
            1,
            int(
                np.ceil(
                    len(ranking)
                    * percent
                    / 100
                )
            ),
        )

        selected_ids = set(
            ranking.head(k)[
                "transaction_id"
            ]
        )

        mask = result[
            "transaction_id"
        ].isin(selected_ids)

        policies.append(
            evaluate_targeting_policy(
                result,
                f"TOP_{percent}",
                mask,
            )
        )

    # --------------------------------------------------------
    # UPLIFT THRESHOLDS
    # --------------------------------------------------------

    thresholds = [
        0.00,
        0.01,
        0.02,
        0.03,
        0.05,
        0.10,
    ]

    for threshold in thresholds:

        mask = (
            result["predicted_uplift"]
            >= threshold
        )

        policies.append(
            evaluate_targeting_policy(
                result,
                f"UPLIFT_{threshold:.2f}",
                mask,
            )
        )

    policy_df = pd.DataFrame(
        policies
    )

    print(
        f"Policies evaluated: "
        f"{len(policy_df)}"
    )

    return policy_df


# ============================================================
# SELECT POLICY
# ============================================================

def select_policy(policy_df):

    header("SELECTING BEST V2 CAUSAL POLICY")

    # IMPORTANT:
    #
    # Do NOT select a policy using observed test-set
    # economic outcomes. That would leak the test outcome
    # into policy selection.
    #
    # Select based on predicted incremental net value.

    candidates = policy_df[
        ~policy_df["policy"].isin(
            [
                "NO_INTERVENTION",
                "RETRY_ALL",
            ]
        )
    ].copy()

    if candidates.empty:

        best = policy_df.iloc[0]

    else:

        best = candidates.loc[
            candidates[
                "predicted_incremental_net_value"
            ].idxmax()
        ]

    print(
        f"Best predicted policy: "
        f"{best['policy']}"
    )

    print(
        f"Interventions: "
        f"{int(best['interventions']):,}"
    )

    print(
        f"Intervention rate: "
        f"{best['intervention_rate']:.2%}"
    )

    print(
        f"Predicted incremental net value: "
        f"{money(best['predicted_incremental_net_value'])}"
    )

    return best


# ============================================================
# CALIBRATION / UPLIFT BINS
# ============================================================

def uplift_calibration(result):

    header("UPLIFT CALIBRATION CHECK")

    work = result.copy()

    work["uplift_decile"] = pd.qcut(
        work["predicted_uplift"],
        q=10,
        labels=False,
        duplicates="drop",
    )

    rows = []

    for decile, group in work.groupby(
        "uplift_decile",
        observed=True,
    ):

        treatment = group[
            group["treatment"] == 1
        ]

        control = group[
            group["treatment"] == 0
        ]

        if (
            len(treatment) == 0
            or len(control) == 0
        ):
            continue

        observed_lift = (
            treatment["recovered"].mean()
            -
            control["recovered"].mean()
        )

        predicted_lift = (
            group["predicted_uplift"].mean()
        )

        rows.append(
            {
                "uplift_decile": int(decile) + 1,
                "rows": len(group),
                "predicted_uplift": predicted_lift,
                "observed_lift": observed_lift,
            }
        )

    calibration = pd.DataFrame(
        rows
    )

    if not calibration.empty:

        print(
            calibration.to_string(
                index=False
            )
        )

        mae = np.mean(
            np.abs(
                calibration[
                    "predicted_uplift"
                ]
                -
                calibration[
                    "observed_lift"
                ]
            )
        )

        print()
        print(
            f"Uplift calibration MAE: "
            f"{mae:.4%}"
        )

    return calibration


# ============================================================
# SAVE RESULTS
# ============================================================

def save_outputs(
    result,
    policy_df,
    model_metrics,
    calibration,
    observed_effect,
    best_policy,
    qini,
):

    header("SAVING V2 CAUSAL UPLIFT RESULTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    result.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        "[PASS] Saved causal uplift predictions:"
    )

    print(
        PREDICTIONS_FILE
    )

    # --------------------------------------------------------
    # Policies
    # --------------------------------------------------------

    policy_df.to_csv(
        POLICY_FILE,
        index=False,
    )

    print(
        "[PASS] Saved policy comparison:"
    )

    print(
        POLICY_FILE
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "rows_total": len(result),
        "treatment_rows": int(
            (result["treatment"] == 1).sum()
        ),
        "control_rows": int(
            (result["treatment"] == 0).sum()
        ),
        "features_used": (
            len(NUMERIC_FEATURES)
            + len(CATEGORICAL_FEATURES)
        ),
        "observed_treatment_recovery_rate": (
            observed_effect[
                "treatment_rate"
            ]
        ),
        "observed_control_recovery_rate": (
            observed_effect[
                "control_rate"
            ]
        ),
        "observed_recovery_lift": (
            observed_effect[
                "observed_lift"
            ]
        ),
        "observed_incremental_net_per_transaction": (
            observed_effect[
                "economic_lift"
            ]
        ),
        "mean_predicted_treatment_probability": (
            result[
                "predicted_treatment_probability"
            ].mean()
        ),
        "mean_predicted_control_probability": (
            result[
                "predicted_control_probability"
            ].mean()
        ),
        "mean_predicted_uplift": (
            result[
                "predicted_uplift"
            ].mean()
        ),
        "positive_predicted_uplift_rows": int(
            (
                result[
                    "predicted_uplift"
                ] > 0
            ).sum()
        ),
        "qini": qini,
        "treatment_brier_score": (
            model_metrics[
                "brier_treatment"
            ]
        ),
        "control_brier_score": (
            model_metrics[
                "brier_control"
            ]
        ),
        "treatment_auc": (
            model_metrics[
                "auc_treatment"
            ]
        ),
        "control_auc": (
            model_metrics[
                "auc_control"
            ]
        ),
        "best_policy": (
            best_policy["policy"]
        ),
        "best_policy_interventions": int(
            best_policy[
                "interventions"
            ]
        ),
        "best_policy_intervention_rate": (
            best_policy[
                "intervention_rate"
            ]
        ),
        "best_policy_predicted_incremental_net": (
            best_policy[
                "predicted_incremental_net_value"
            ]
        ),
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        "[PASS] Saved causal uplift summary:"
    )

    print(
        SUMMARY_FILE
    )

    return summary_df


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    df = load_data()

    validate_data(
        df
    )

    numeric_features, categorical_features = (
        select_features(df)
    )

    # --------------------------------------------------------
    # Observed experiment
    # --------------------------------------------------------

    observed_effect = (
        print_observed_effect(
            df
        )
    )

    # --------------------------------------------------------
    # Train/test split
    #
    # Stratify by treatment so both groups remain balanced.
    # --------------------------------------------------------

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["treatment"],
    )

    print()
    print(
        f"Held-out validation split:"
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Test rows:     "
        f"{len(test_df):,}"
    )

    print(
        f"Training treatment rows: "
        f"{(train_df['treatment'] == 1).sum():,}"
    )

    print(
        f"Training control rows:   "
        f"{(train_df['treatment'] == 0).sum():,}"
    )

    print(
        f"Test treatment rows: "
        f"{(test_df['treatment'] == 1).sum():,}"
    )

    print(
        f"Test control rows:   "
        f"{(test_df['treatment'] == 0).sum():,}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    treatment_model, control_model = (
        train_t_learner(
            train_df,
            numeric_features,
            categorical_features,
        )
    )

    # --------------------------------------------------------
    # Predict held-out uplift
    # --------------------------------------------------------

    result = estimate_uplift(
        treatment_model,
        control_model,
        test_df,
        numeric_features,
        categorical_features,
    )

    # --------------------------------------------------------
    # Model quality
    # --------------------------------------------------------

    model_metrics = evaluate_models(
        result,
        treatment_model,
        control_model,
        numeric_features,
        categorical_features,
    )

    # --------------------------------------------------------
    # QINI
    # --------------------------------------------------------

    header("HELD-OUT QINI EVALUATION")

    qini = calculate_qini(
        result
    )

    print(
        f"QINI score: "
        f"{qini:,.4f}"
    )

    print(
        "[PASS] QINI calculated on held-out data"
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    calibration = (
        uplift_calibration(
            result
        )
    )

    # --------------------------------------------------------
    # Economic targeting
    # --------------------------------------------------------

    policy_df = evaluate_policies(
        result
    )

    header("POLICY COMPARISON")

    display_columns = [
        "policy",
        "interventions",
        "intervention_rate",
        "observed_recovered_payments",
        "observed_net_value",
        "observed_net_per_intervention",
        "predicted_incremental_net_value",
    ]

    display_df = policy_df[
        display_columns
    ].copy()

    display_df[
        "intervention_rate"
    ] = display_df[
        "intervention_rate"
    ].map(
        lambda x: f"{x:.2%}"
    )

    display_df[
        "observed_net_value"
    ] = display_df[
        "observed_net_value"
    ].map(
        lambda x: money(x)
    )

    display_df[
        "observed_net_per_intervention"
    ] = display_df[
        "observed_net_per_intervention"
    ].map(
        lambda x: money(x)
    )

    display_df[
        "predicted_incremental_net_value"
    ] = display_df[
        "predicted_incremental_net_value"
    ].map(
        lambda x: money(x)
    )

    print(
        display_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best policy
    # --------------------------------------------------------

    best_policy = select_policy(
        policy_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        result=result,
        policy_df=policy_df,
        model_metrics=model_metrics,
        calibration=calibration,
        observed_effect=observed_effect,
        best_policy=best_policy,
        qini=qini,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    header(
        "CAUSAL UPLIFT V2 COMPLETE"
    )

    print(
        f"Transactions evaluated: "
        f"{len(result):,}"
    )

    print(
        f"Training transactions: "
        f"{len(train_df):,}"
    )

    print(
        f"Held-out transactions: "
        f"{len(test_df):,}"
    )

    print(
        f"Features used: "
        f"{len(numeric_features) + len(categorical_features)}"
    )

    print(
        f"Mean predicted uplift: "
        f"{result['predicted_uplift'].mean():+.2%}"
    )

    print(
        f"Positive predicted uplift rows: "
        f"{(result['predicted_uplift'] > 0).sum():,}"
    )

    print(
        f"Held-out QINI: "
        f"{qini:,.4f}"
    )

    print(
        f"Best predicted policy: "
        f"{best_policy['policy']}"
    )

    print(
        f"Predicted incremental net value: "
        f"{money(best_policy['predicted_incremental_net_value'])}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The V2 model uses held-out predictions."
    )

    print(
        "Outcome-derived columns are excluded from modeling."
    )

    print(
        "The best policy is selected using predicted "
        "incremental economic value, not held-out outcomes."
    )

    print(
        "Do NOT deploy the policy until repeated "
        "held-out validation confirms stability."
    )

    print()
    print(
        "STATUS: V2 CAUSAL UPLIFT BASELINE COMPLETE"
    )


if __name__ == "__main__":
    main()