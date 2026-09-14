"""
======================================================================
RECLAIM CAUSAL UPLIFT OPTIMIZER
======================================================================

Purpose
-------
Estimate transaction-level causal uplift:

    uplift =
        P(recovery | RECLAIM intervention, X)
        -
        P(recovery | control, X)

Then convert uplift into economic value and determine whether
targeted intervention beats:

    1. NO_INTERVENTION
    2. RETRY_ALL

IMPORTANT
---------
Treatment assignment must already have happened BEFORE the outcome.
This script does NOT randomly assign treatment.

Input
-----
D:/RECLAIM/data/generated/reclaim_treatment_control_experiment.csv

Outputs
-------
causal_uplift_predictions.csv
causal_uplift_policy_comparison.csv
causal_uplift_summary.csv

======================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(r"D:\RECLAIM")

DATA_DIR = PROJECT_ROOT / "data" / "generated"

EXPERIMENT_FILE = DATA_DIR / "reclaim_treatment_control_experiment.csv"

OUTPUT_PREDICTIONS = DATA_DIR / "causal_uplift_predictions.csv"
OUTPUT_POLICY = DATA_DIR / "causal_uplift_policy_comparison.csv"
OUTPUT_SUMMARY = DATA_DIR / "causal_uplift_summary.csv"

RANDOM_STATE = 42

MIN_GROUP_SIZE = 30

# Policy percentages
TOP_PERCENTAGES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Positive uplift threshold
UPLIFT_THRESHOLDS = [
    0.00,
    0.01,
    0.02,
    0.03,
    0.05,
    0.10,
]


# =====================================================================
# PRINT HELPERS
# =====================================================================

def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def money(value):
    return f"₹{value:,.2f}"


# =====================================================================
# DATA LOADING
# =====================================================================

def load_data():

    banner("RECLAIM CAUSAL UPLIFT OPTIMIZER")

    print("Loading treatment/control experiment...")
    print(f"File: {EXPERIMENT_FILE}")

    if not EXPERIMENT_FILE.exists():
        raise FileNotFoundError(
            f"\nExperiment file not found:\n{EXPERIMENT_FILE}\n"
            "\nRun the treatment/control experiment simulator first."
        )

    df = pd.read_csv(EXPERIMENT_FILE)

    print(f"[PASS] Rows loaded: {len(df):,}")

    if "transaction_id" not in df.columns:
        raise ValueError("Missing transaction_id")

    if df["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs detected.")

    print("[PASS] Transaction IDs are unique")

    required = [
        "transaction_id",
        "treatment",
        "recovered",
        "recovered_amount",
        "intervention_cost",
        "customer_friction_cost",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("[PASS] Required causal columns present")

    return df


# =====================================================================
# VALIDATE EXPERIMENT
# =====================================================================

def validate_experiment(df):

    section("VALIDATING CAUSAL EXPERIMENT")

    treatment_values = sorted(df["treatment"].dropna().unique())

    print(f"Treatment values: {treatment_values}")

    if not set(treatment_values).issubset({0, 1}):
        raise ValueError(
            "Treatment must contain only 0 and 1."
        )

    print("[PASS] Treatment values valid")

    treatment_count = int((df["treatment"] == 1).sum())
    control_count = int((df["treatment"] == 0).sum())

    print(f"Treatment rows: {treatment_count}")
    print(f"Control rows:   {control_count}")

    if treatment_count < MIN_GROUP_SIZE:
        raise ValueError(
            f"Too few treatment rows: {treatment_count}"
        )

    if control_count < MIN_GROUP_SIZE:
        raise ValueError(
            f"Too few control rows: {control_count}"
        )

    print("[PASS] Treatment/control groups usable")

    if df["recovered"].isna().any():
        raise ValueError("Missing recovered values")

    df["recovered"] = (
        pd.to_numeric(df["recovered"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if not df["recovered"].isin([0, 1]).all():
        raise ValueError(
            "Recovered must contain only 0/1."
        )

    print("[PASS] Recovery outcome valid")

    df["recovered_amount"] = pd.to_numeric(
        df["recovered_amount"],
        errors="coerce"
    ).fillna(0)

    df["intervention_cost"] = pd.to_numeric(
        df["intervention_cost"],
        errors="coerce"
    ).fillna(0)

    df["customer_friction_cost"] = pd.to_numeric(
        df["customer_friction_cost"],
        errors="coerce"
    ).fillna(0)

    print("[PASS] Economic variables valid")

    return df


# =====================================================================
# FIND FEATURES
# =====================================================================

def find_features(df):

    section("SELECTING UPLIFT MODEL FEATURES")

    preferred_features = [
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
        "recovery_probability",
        "calibrated_probability",
        "confidence",
    ]

    excluded = {
        "transaction_id",
        "treatment",
        "control",
        "treatment_group",
        "recovered",
        "recovery_flag",
        "recovered_amount",
        "intervention_cost",
        "customer_friction_cost",
        "net_recovered_value",
        "control_recovery_probability",
        "treatment_recovery_probability",
        "incremental_recovery_probability",
        "observed_recovery_probability",
        "expected_control_recovery",
        "expected_treatment_recovery",
        "expected_incremental_recovery",
        "expected_incremental_net_value",
        "treatment_effect",
        "recovery_status",
        "recommended_policy",
        "policy_threshold",
        "expected_recovery",
        "expected_intervention_cost",
        "expected_friction_cost",
        "expected_net_recovery",
    }

    features = []

    for column in preferred_features:

        if column not in df.columns:
            continue

        if column in excluded:
            continue

        if df[column].nunique(dropna=True) <= 1:
            continue

        features.append(column)

    # Also automatically include useful numeric/categorical columns
    # that may have been carried into the experiment.
    for column in df.columns:

        if column in excluded:
            continue

        if column in features:
            continue

        if column not in preferred_features:

            if (
                pd.api.types.is_numeric_dtype(df[column])
                or pd.api.types.is_object_dtype(df[column])
                or pd.api.types.is_string_dtype(df[column])
            ):
                if df[column].nunique(dropna=True) > 1:
                    features.append(column)

    print(f"Features selected: {len(features)}")

    for feature in features:
        print(f"  [YES] {feature}")

    if len(features) == 0:
        raise ValueError(
            "No usable features available for uplift modeling."
        )

    if len(features) < 3:
        print()
        print(
            "[WARN] Very few features are available."
        )
        print(
            "The model can still run, but this is a weak baseline."
        )

    return features


# =====================================================================
# PREPROCESSOR
# =====================================================================

def build_preprocessor(df, features):

    numeric_features = []
    categorical_features = []

    for feature in features:

        if pd.api.types.is_numeric_dtype(df[feature]):
            numeric_features.append(feature)

        else:
            categorical_features.append(feature)

    transformers = []

    if numeric_features:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# =====================================================================
# BUILD T-LEARNER
# =====================================================================

def build_model(df, features):

    section("BUILDING CAUSAL T-LEARNER")

    treatment_df = df[df["treatment"] == 1].copy()
    control_df = df[df["treatment"] == 0].copy()

    X_treatment = treatment_df[features]
    y_treatment = treatment_df["recovered"]

    X_control = control_df[features]
    y_control = control_df["recovered"]

    print(
        f"Treatment training rows: {len(treatment_df)}"
    )

    print(
        f"Control training rows:   {len(control_df)}"
    )

    print(
        f"Treatment recovery rate: {y_treatment.mean():.2%}"
    )

    print(
        f"Control recovery rate:   {y_control.mean():.2%}"
    )

    preprocessor_treatment = build_preprocessor(
        df,
        features
    )

    preprocessor_control = build_preprocessor(
        df,
        features
    )

    # Logistic regression provides a stable baseline for a relatively
    # small randomized experiment.
    treatment_model = Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor_treatment
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            ),
        ]
    )

    control_model = Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor_control
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            ),
        ]
    )

    treatment_model.fit(
        X_treatment,
        y_treatment
    )

    control_model.fit(
        X_control,
        y_control
    )

    print("[PASS] Treatment outcome model trained")
    print("[PASS] Control outcome model trained")

    return treatment_model, control_model


# =====================================================================
# PREDICT INDIVIDUAL UPLIFT
# =====================================================================

def calculate_uplift(
    df,
    features,
    treatment_model,
    control_model
):

    section("ESTIMATING INDIVIDUAL TREATMENT EFFECT")

    X = df[features]

    treatment_probability = treatment_model.predict_proba(
        X
    )[:, 1]

    control_probability = control_model.predict_proba(
        X
    )[:, 1]

    uplift = (
        treatment_probability
        - control_probability
    )

    result = df.copy()

    result[
        "predicted_treatment_probability"
    ] = np.clip(
        treatment_probability,
        0,
        1
    )

    result[
        "predicted_control_probability"
    ] = np.clip(
        control_probability,
        0,
        1
    )

    result["predicted_uplift"] = uplift

    result["predicted_uplift"] = np.clip(
        result["predicted_uplift"],
        -1,
        1
    )

    print(
        f"Mean predicted treatment probability: "
        f"{result['predicted_treatment_probability'].mean():.2%}"
    )

    print(
        f"Mean predicted control probability: "
        f"{result['predicted_control_probability'].mean():.2%}"
    )

    print(
        f"Mean predicted uplift: "
        f"{result['predicted_uplift'].mean():+.2%}"
    )

    print(
        f"Positive uplift rows: "
        f"{(result['predicted_uplift'] > 0).sum()}"
    )

    print(
        f"Negative uplift rows: "
        f"{(result['predicted_uplift'] < 0).sum()}"
    )

    print("[PASS] Individual uplift estimated")

    return result


# =====================================================================
# ECONOMIC UPLIFT
# =====================================================================

def calculate_economic_uplift(df):

    section("CALCULATING ECONOMIC UPLIFT")

    result = df.copy()

    # Expected incremental recovered value.
    result["expected_incremental_recovered_value"] = (
        result["predicted_uplift"]
        * result["amount"]
    )

    # Intervention costs only apply if we treat.
    result["expected_intervention_cost"] = (
        pd.to_numeric(
            result["intervention_cost"],
            errors="coerce"
        ).fillna(0)
    )

    result["expected_friction_cost"] = (
        pd.to_numeric(
            result["customer_friction_cost"],
            errors="coerce"
        ).fillna(0)
    )

    result["predicted_incremental_net_value"] = (
        result["expected_incremental_recovered_value"]
        - result["expected_intervention_cost"]
        - result["expected_friction_cost"]
    )

    print(
        "Mean expected incremental recovered value: "
        + money(
            result[
                "expected_incremental_recovered_value"
            ].mean()
        )
    )

    print(
        "Mean predicted incremental net value: "
        + money(
            result[
                "predicted_incremental_net_value"
            ].mean()
        )
    )

    print(
        "Positive predicted economic value rows: "
        f"{(
            result[
                "predicted_incremental_net_value"
            ] > 0
        ).sum()}"
    )

    print("[PASS] Economic uplift calculated")

    return result


# =====================================================================
# POLICY EVALUATION
# =====================================================================

def evaluate_policy(
    df,
    mask,
    policy_name
):

    selected = df.loc[mask].copy()

    all_rows = len(df)

    interventions = len(selected)

    intervention_rate = (
        interventions / all_rows
        if all_rows
        else 0
    )

    if interventions == 0:

        return {
            "policy": policy_name,
            "interventions": 0,
            "intervention_rate": 0,
            "observed_recovered_payments": 0,
            "observed_recovered_value": 0,
            "intervention_cost": 0,
            "friction_cost": 0,
            "observed_net_value": 0,
            "observed_net_per_intervention": 0,
            "predicted_incremental_value": 0,
            "predicted_incremental_net_value": 0,
        }

    recovered_payments = int(
        selected["recovered"].sum()
    )

    recovered_value = float(
        selected["recovered_amount"].sum()
    )

    intervention_cost = float(
        selected["intervention_cost"].sum()
    )

    friction_cost = float(
        selected["customer_friction_cost"].sum()
    )

    net_value = (
        recovered_value
        - intervention_cost
        - friction_cost
    )

    predicted_incremental_value = float(
        selected[
            "expected_incremental_recovered_value"
        ].sum()
    )

    predicted_incremental_net = float(
        selected[
            "predicted_incremental_net_value"
        ].sum()
    )

    return {
        "policy": policy_name,
        "interventions": interventions,
        "intervention_rate": intervention_rate,
        "observed_recovered_payments": recovered_payments,
        "observed_recovered_value": recovered_value,
        "intervention_cost": intervention_cost,
        "friction_cost": friction_cost,
        "observed_net_value": net_value,
        "observed_net_per_intervention": (
            net_value / interventions
            if interventions
            else 0
        ),
        "predicted_incremental_value": predicted_incremental_value,
        "predicted_incremental_net_value": predicted_incremental_net,
    }


# =====================================================================
# POLICY COMPARISON
# =====================================================================

def build_policies(df):

    section("TESTING CAUSAL TARGETING POLICIES")

    policies = []

    # ---------------------------------------------------------------
    # NO INTERVENTION
    # ---------------------------------------------------------------

    no_intervention = np.zeros(
        len(df),
        dtype=bool
    )

    policies.append(
        evaluate_policy(
            df,
            no_intervention,
            "NO_INTERVENTION"
        )
    )

    # ---------------------------------------------------------------
    # RETRY ALL
    # ---------------------------------------------------------------

    retry_all = np.ones(
        len(df),
        dtype=bool
    )

    policies.append(
        evaluate_policy(
            df,
            retry_all,
            "RETRY_ALL"
        )
    )

    # ---------------------------------------------------------------
    # TOP K UPLIFT
    # ---------------------------------------------------------------

    ranked = df.sort_values(
        "predicted_incremental_net_value",
        ascending=False
    )

    for percentage in TOP_PERCENTAGES:

        k = max(
            1,
            int(
                np.ceil(
                    len(df) * percentage / 100
                )
            )
        )

        selected_ids = set(
            ranked.head(k)[
                "transaction_id"
            ]
        )

        mask = df[
            "transaction_id"
        ].isin(selected_ids)

        policies.append(
            evaluate_policy(
                df,
                mask,
                f"TOP_{percentage}"
            )
        )

    # ---------------------------------------------------------------
    # POSITIVE UPLIFT
    # ---------------------------------------------------------------

    for threshold in UPLIFT_THRESHOLDS:

        mask = (
            df["predicted_uplift"]
            >= threshold
        )

        policies.append(
            evaluate_policy(
                df,
                mask,
                f"UPLIFT_{threshold:.2f}"
            )
        )

    # ---------------------------------------------------------------
    # POSITIVE ECONOMIC VALUE
    # ---------------------------------------------------------------

    mask = (
        df[
            "predicted_incremental_net_value"
        ] > 0
    )

    policies.append(
        evaluate_policy(
            df,
            mask,
            "POSITIVE_ECONOMIC_UPLIFT"
        )
    )

    # ---------------------------------------------------------------
    # NEGATIVE ECONOMIC VALUE
    # ---------------------------------------------------------------

    mask = (
        df[
            "predicted_incremental_net_value"
        ] >= 0
    )

    policies.append(
        evaluate_policy(
            df,
            mask,
            "NON_NEGATIVE_ECONOMIC_UPLIFT"
        )
    )

    policy_df = pd.DataFrame(policies)

    print(
        f"Policies evaluated: {len(policy_df)}"
    )

    return policy_df


# =====================================================================
# PRINT POLICY TABLE
# =====================================================================

def print_policy_comparison(policy_df):

    section("POLICY COMPARISON")

    display_columns = [
        "policy",
        "interventions",
        "intervention_rate",
        "observed_recovered_payments",
        "observed_recovered_value",
        "observed_net_value",
        "observed_net_per_intervention",
        "predicted_incremental_net_value",
    ]

    table = policy_df[
        display_columns
    ].copy()

    table[
        "intervention_rate"
    ] = table[
        "intervention_rate"
    ].map(
        lambda x: f"{x:.2%}"
    )

    for column in [
        "observed_recovered_value",
        "observed_net_value",
        "observed_net_per_intervention",
        "predicted_incremental_net_value",
    ]:

        table[column] = table[column].map(
            money
        )

    print(
        table.to_string(
            index=False
        )
    )


# =====================================================================
# SELECT BEST POLICY
# =====================================================================

def select_best_policy(policy_df):

    section("SELECTING OPTIMAL CAUSAL POLICY")

    # We primarily optimize predicted incremental economic value,
    # because this is the quantity the causal model is designed to
    # estimate for future targeting.

    candidate = policy_df[
        policy_df["policy"]
        != "NO_INTERVENTION"
    ].copy()

    best = candidate.loc[
        candidate[
            "predicted_incremental_net_value"
        ].idxmax()
    ]

    print(
        f"Best predicted policy: "
        f"{best['policy']}"
    )

    print(
        f"Interventions: "
        f"{int(best['interventions'])}"
    )

    print(
        f"Intervention rate: "
        f"{best['intervention_rate']:.2%}"
    )

    print(
        "Predicted incremental net value: "
        + money(
            best[
                "predicted_incremental_net_value"
            ]
        )
    )

    return best


# =====================================================================
# OBSERVED EXPERIMENT EFFECT
# =====================================================================

def observed_experiment_effect(df):

    section("OBSERVED EXPERIMENT EFFECT")

    treatment = df[
        df["treatment"] == 1
    ]

    control = df[
        df["treatment"] == 0
    ]

    treatment_rate = treatment[
        "recovered"
    ].mean()

    control_rate = control[
        "recovered"
    ].mean()

    lift = (
        treatment_rate
        - control_rate
    )

    treatment_value = treatment[
        "recovered_amount"
    ].sum()

    control_value = control[
        "recovered_amount"
    ].sum()

    treatment_cost = (
        treatment["intervention_cost"].sum()
        + treatment[
            "customer_friction_cost"
        ].sum()
    )

    control_cost = (
        control["intervention_cost"].sum()
        + control[
            "customer_friction_cost"
        ].sum()
    )

    treatment_net = (
        treatment_value
        - treatment_cost
    )

    control_net = (
        control_value
        - control_cost
    )

    incremental_net = (
        treatment_net / len(treatment)
        -
        control_net / len(control)
    )

    print(
        f"Treatment recovery rate: {treatment_rate:.2%}"
    )

    print(
        f"Control recovery rate:   {control_rate:.2%}"
    )

    print(
        f"Observed recovery lift:  {lift:+.2%}"
    )

    print()

    print(
        "Treatment recovered value: "
        + money(treatment_value)
    )

    print(
        "Control recovered value:   "
        + money(control_value)
    )

    print()

    print(
        "Treatment net value: "
        + money(treatment_net)
    )

    print(
        "Control net value:   "
        + money(control_net)
    )

    print()

    print(
        "Observed incremental net value / transaction: "
        + money(incremental_net)
    )

    return {
        "treatment_recovery_rate": treatment_rate,
        "control_recovery_rate": control_rate,
        "observed_recovery_lift": lift,
        "treatment_recovered_value": treatment_value,
        "control_recovered_value": control_value,
        "treatment_net_value": treatment_net,
        "control_net_value": control_net,
        "observed_incremental_net_value_per_transaction": incremental_net,
    }


# =====================================================================
# MODEL QUALITY CHECK
# =====================================================================

def model_quality_check(
    df,
    features,
    treatment_model,
    control_model
):

    section("CAUSAL MODEL QUALITY CHECK")

    treatment = df[
        df["treatment"] == 1
    ]

    control = df[
        df["treatment"] == 0
    ]

    # This is an in-sample diagnostic only.
    # It is NOT a causal validation score.
    try:

        treatment_pred = treatment_model.predict_proba(
            treatment[features]
        )[:, 1]

        control_pred = control_model.predict_proba(
            control[features]
        )[:, 1]

        treatment_brier = brier_score_loss(
            treatment["recovered"],
            treatment_pred
        )

        control_brier = brier_score_loss(
            control["recovered"],
            control_pred
        )

        print(
            f"Treatment Brier score: {treatment_brier:.6f}"
        )

        print(
            f"Control Brier score:   {control_brier:.6f}"
        )

        print(
            "[INFO] These are in-sample diagnostics, "
            "not final causal validation."
        )

    except Exception as exc:

        print(
            f"[WARN] Model quality diagnostic failed: {exc}"
        )


# =====================================================================
# SAVE PREDICTIONS
# =====================================================================

def save_predictions(df):

    columns = [
        "transaction_id",
        "amount",
        "failure_reason",
        "treatment",
        "recovered",
        "recovered_amount",
        "intervention_cost",
        "customer_friction_cost",
        "predicted_control_probability",
        "predicted_treatment_probability",
        "predicted_uplift",
        "expected_incremental_recovered_value",
        "predicted_incremental_net_value",
    ]

    columns = [
        c for c in columns
        if c in df.columns
    ]

    output = df[columns].copy()

    output = output.sort_values(
        "predicted_incremental_net_value",
        ascending=False
    )

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False
    )

    print()
    print(
        "[PASS] Saved causal uplift predictions:"
    )

    print(
        OUTPUT_PREDICTIONS
    )


# =====================================================================
# SAVE POLICY
# =====================================================================

def save_policy(policy_df):

    policy_df.to_csv(
        OUTPUT_POLICY,
        index=False
    )

    print(
        "[PASS] Saved policy comparison:"
    )

    print(
        OUTPUT_POLICY
    )


# =====================================================================
# SAVE SUMMARY
# =====================================================================

def save_summary(
    df,
    features,
    observed,
    best_policy
):

    positive_uplift = int(
        (
            df["predicted_uplift"] > 0
        ).sum()
    )

    positive_economic = int(
        (
            df[
                "predicted_incremental_net_value"
            ] > 0
        ).sum()
    )

    summary = pd.DataFrame([
        {
            "transactions": len(df),
            "treatment_rows": int(
                (df["treatment"] == 1).sum()
            ),
            "control_rows": int(
                (df["treatment"] == 0).sum()
            ),
            "features_used": len(features),
            "mean_predicted_treatment_probability":
                df[
                    "predicted_treatment_probability"
                ].mean(),
            "mean_predicted_control_probability":
                df[
                    "predicted_control_probability"
                ].mean(),
            "mean_predicted_uplift":
                df[
                    "predicted_uplift"
                ].mean(),
            "positive_uplift_rows":
                positive_uplift,
            "positive_economic_uplift_rows":
                positive_economic,
            "observed_recovery_lift":
                observed[
                    "observed_recovery_lift"
                ],
            "observed_incremental_net_value":
                observed[
                    "observed_incremental_net_value_per_transaction"
                ],
            "best_policy":
                best_policy["policy"],
            "best_policy_interventions":
                int(
                    best_policy["interventions"]
                ),
            "best_policy_intervention_rate":
                best_policy[
                    "intervention_rate"
                ],
            "best_predicted_incremental_net_value":
                best_policy[
                    "predicted_incremental_net_value"
                ],
        }
    ])

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    print(
        "[PASS] Saved causal uplift summary:"
    )

    print(
        OUTPUT_SUMMARY
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    # ---------------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------------

    df = load_data()

    # ---------------------------------------------------------------
    # VALIDATE
    # ---------------------------------------------------------------

    df = validate_experiment(df)

    # ---------------------------------------------------------------
    # FEATURES
    # ---------------------------------------------------------------

    features = find_features(df)

    # ---------------------------------------------------------------
    # OBSERVED EXPERIMENT EFFECT
    # ---------------------------------------------------------------

    observed = observed_experiment_effect(df)

    # ---------------------------------------------------------------
    # T-LEARNER
    # ---------------------------------------------------------------

    treatment_model, control_model = build_model(
        df,
        features
    )

    # ---------------------------------------------------------------
    # MODEL DIAGNOSTICS
    # ---------------------------------------------------------------

    model_quality_check(
        df,
        features,
        treatment_model,
        control_model
    )

    # ---------------------------------------------------------------
    # INDIVIDUAL UPLIFT
    # ---------------------------------------------------------------

    df = calculate_uplift(
        df,
        features,
        treatment_model,
        control_model
    )

    # ---------------------------------------------------------------
    # ECONOMIC UPLIFT
    # ---------------------------------------------------------------

    df = calculate_economic_uplift(df)

    # ---------------------------------------------------------------
    # POLICY TESTING
    # ---------------------------------------------------------------

    policy_df = build_policies(df)

    print_policy_comparison(
        policy_df
    )

    # ---------------------------------------------------------------
    # BEST POLICY
    # ---------------------------------------------------------------

    best_policy = select_best_policy(
        policy_df
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    section("SAVING CAUSAL UPLIFT RESULTS")

    save_predictions(df)

    save_policy(policy_df)

    save_summary(
        df,
        features,
        observed,
        best_policy
    )

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    banner("CAUSAL UPLIFT OPTIMIZATION COMPLETE")

    print(
        f"Transactions analyzed: {len(df)}"
    )

    print(
        f"Features used: {len(features)}"
    )

    print(
        f"Mean predicted uplift: "
        f"{df['predicted_uplift'].mean():+.2%}"
    )

    print(
        "Positive predicted uplift rows: "
        f"{(
            df['predicted_uplift'] > 0
        ).sum()}"
    )

    print(
        "Positive predicted economic uplift rows: "
        f"{(
            df[
                'predicted_incremental_net_value'
            ] > 0
        ).sum()}"
    )

    print()

    print(
        f"Best policy: {best_policy['policy']}"
    )

    print(
        "Predicted incremental net value: "
        + money(
            best_policy[
                "predicted_incremental_net_value"
            ]
        )
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The observed experiment effect and the "
        "model-predicted targeting effect are separate."
    )

    print(
        "Do not claim the uplift model is proven "
        "until it is evaluated on held-out data "
        "or a future randomized experiment."
    )

    print()
    print("STATUS: CAUSAL UPLIFT BASELINE READY")


if __name__ == "__main__":
    main()