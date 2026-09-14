"""
RECLAIM CAUSAL UPLIFT STABILITY VALIDATOR V2

Purpose
-------
Repeated held-out validation of the V2 causal uplift model.

This validator:
1. Loads the 10,000-row randomized V2 experiment.
2. Uses only pre-treatment features.
3. Repeats randomized train/test splits.
4. Trains separate treatment/control outcome models.
5. Estimates individual treatment uplift.
6. Calculates held-out QINI.
7. Evaluates economic targeting.
8. Compares targeting against NO_INTERVENTION and RETRY_ALL.
9. Calculates uplift calibration.
10. Produces a final stability verdict.

IMPORTANT
---------
Outcome-derived columns are NEVER used as model features.

Compatible with Python 3.14 / modern NumPy.
Does not use np.trapz().
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ======================================================================
# CONFIGURATION
# ======================================================================

RANDOM_SEED = 2026

N_SPLITS = 20
TEST_SIZE = 0.30

MIN_ROWS = 5000

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "reclaim_causal_experiment_v2.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"

RESULTS_FILE = (
    OUTPUT_DIR
    / "causal_uplift_v2_stability_results.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "causal_uplift_v2_stability_summary.csv"
)


# ======================================================================
# FEATURE DEFINITIONS
# ======================================================================

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

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ======================================================================
# OUTCOME / LEAKAGE COLUMNS
# ======================================================================

OUTCOME_COLUMNS = [
    "recovered",
    "recovery_flag",
    "recovered_amount",
    "observed_recovery_probability",
    "recovery_status",
    "net_recovered_value",
    "customer_friction_cost",
    "intervention_cost",
]

TREATMENT_COLUMNS = [
    "treatment",
    "control",
    "treatment_group",
    "treatment_assignment_before_outcome",
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
# UTILITY FUNCTIONS
# ======================================================================

def header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def money(value: float) -> str:
    return f"₹{value:,.2f}"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def safe_mean(values) -> float:
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return 0.0

    return float(np.mean(values))


def safe_std(values) -> float:
    values = np.asarray(values, dtype=float)

    if len(values) <= 1:
        return 0.0

    return float(np.std(values, ddof=1))


def ci95(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan, np.nan

    mean = np.mean(values)

    if len(values) == 1:
        return float(mean), float(mean)

    se = np.std(values, ddof=1) / np.sqrt(len(values))

    margin = 1.96 * se

    return float(mean - margin), float(mean + margin)


# ======================================================================
# LOAD DATA
# ======================================================================

def load_experiment() -> pd.DataFrame:

    header("RECLAIM CAUSAL UPLIFT STABILITY VALIDATOR V2")

    print("Loading V2 randomized treatment/control experiment...")
    print(f"File: {EXPERIMENT_FILE}")

    if not EXPERIMENT_FILE.exists():
        raise FileNotFoundError(
            f"Experiment file not found:\n{EXPERIMENT_FILE}"
        )

    df = pd.read_csv(EXPERIMENT_FILE)

    print(f"[PASS] Rows loaded: {len(df):,}")
    print(f"[PASS] Columns loaded: {len(df.columns):,}")

    return df


# ======================================================================
# VALIDATE DATA
# ======================================================================

def validate_experiment(df: pd.DataFrame) -> None:

    header("VALIDATING V2 RANDOMIZED EXPERIMENT")

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Experiment has only {len(df)} rows. "
            f"Expected at least {MIN_ROWS}."
        )

    if "experiment_transaction_id" in df.columns:

        if df["experiment_transaction_id"].is_unique:
            print("[PASS] Experiment transaction IDs are unique")
        else:
            raise ValueError(
                "experiment_transaction_id contains duplicates."
            )

    elif "transaction_id" in df.columns:

        if df["transaction_id"].is_unique:
            print("[PASS] Transaction IDs are unique")
        else:
            raise ValueError(
                "transaction_id contains duplicates."
            )

    required = [
        "treatment",
        "recovered",
        "amount",
        "net_recovered_value",
        "intervention_cost",
        "customer_friction_cost",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    treatments = sorted(
        df["treatment"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    print(f"Treatment values: {treatments}")

    if set(treatments) != {0, 1}:
        raise ValueError(
            "Treatment must contain exactly 0 and 1."
        )

    print("[PASS] Treatment values valid")

    treatment_count = int(
        (df["treatment"] == 1).sum()
    )

    control_count = int(
        (df["treatment"] == 0).sum()
    )

    print(f"Treatment rows: {treatment_count:,}")
    print(f"Control rows:   {control_count:,}")

    if treatment_count == 0 or control_count == 0:
        raise ValueError(
            "Both treatment and control groups are required."
        )

    print("[PASS] Treatment/control groups usable")

    recovered_values = set(
        pd.to_numeric(
            df["recovered"],
            errors="coerce"
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not recovered_values.issubset({0, 1}):
        raise ValueError(
            "Recovered must contain only 0/1 values."
        )

    print("[PASS] Recovery outcome valid")

    for column in [
        "amount",
        "net_recovered_value",
        "intervention_cost",
        "customer_friction_cost",
    ]:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if values.isna().any():
            raise ValueError(
                f"{column} contains invalid numeric values."
            )

    print("[PASS] Economic variables valid")


# ======================================================================
# FEATURE VALIDATION
# ======================================================================

def select_features(df: pd.DataFrame):

    header("SELECTING AVAILABLE V2 UPLIFT FEATURES")

    numeric = [
        column
        for column in NUMERIC_FEATURES
        if column in df.columns
    ]

    categorical = [
        column
        for column in CATEGORICAL_FEATURES
        if column in df.columns
    ]

    print(f"Numeric features available: {len(numeric)}")

    for column in numeric:
        print(f"  [YES] {column}")

    print(
        f"Categorical features available: "
        f"{len(categorical)}"
    )

    for column in categorical:
        print(f"  [YES] {column}")

    features = numeric + categorical

    if len(features) < 10:
        warnings.warn(
            "Very few modeling features are available."
        )

    print(
        f"[PASS] Total modeling features: {len(features)}"
    )

    print()
    print("[LEAKAGE PROTECTION]")

    for column in OUTCOME_COLUMNS:
        if column in features:
            raise RuntimeError(
                f"LEAKAGE ERROR: outcome column "
                f"{column} selected as a feature."
            )

    for column in TREATMENT_COLUMNS:
        if column in features:
            raise RuntimeError(
                f"LEAKAGE ERROR: treatment column "
                f"{column} selected as a feature."
            )

    for column in COUNTERFACTUAL_COLUMNS:
        if column in features:
            raise RuntimeError(
                f"LEAKAGE ERROR: counterfactual column "
                f"{column} selected as a feature."
            )

    print("[PASS] No outcome/treatment/counterfactual leakage")

    return numeric, categorical, features


# ======================================================================
# PREPROCESSOR
# ======================================================================

def build_preprocessor(
    numeric_features,
    categorical_features,
):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
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


# ======================================================================
# BUILD MODEL
# ======================================================================

def build_model(
    numeric_features,
    categorical_features,
    seed,
):

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=25,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ======================================================================
# QINI
# ======================================================================

def calculate_qini(
    treatment,
    outcome,
    uplift,
):

    treatment = np.asarray(treatment).astype(int)
    outcome = np.asarray(outcome).astype(float)
    uplift = np.asarray(uplift).astype(float)

    order = np.argsort(
        -uplift,
        kind="mergesort"
    )

    treatment = treatment[order]
    outcome = outcome[order]

    cumulative_treatment = np.cumsum(
        treatment
    )

    cumulative_control = np.cumsum(
        1 - treatment
    )

    cumulative_treated_outcome = np.cumsum(
        treatment * outcome
    )

    cumulative_control_outcome = np.cumsum(
        (1 - treatment) * outcome
    )

    treated_rate = np.divide(
        cumulative_treated_outcome,
        cumulative_treatment,
        out=np.zeros_like(
            cumulative_treated_outcome,
            dtype=float,
        ),
        where=cumulative_treatment > 0,
    )

    control_rate = np.divide(
        cumulative_control_outcome,
        cumulative_control,
        out=np.zeros_like(
            cumulative_control_outcome,
            dtype=float,
        ),
        where=cumulative_control > 0,
    )

    uplift_curve = (
        treated_rate - control_rate
    )

    population = np.arange(
        1,
        len(outcome) + 1,
    )

    gain = uplift_curve * population

    # NumPy 2.x removed np.trapz.
    # Use trapezoidal integration manually.
    qini = np.sum(
        (
            gain[:-1] + gain[1:]
        )
        * 0.5
    )

    return float(qini)


# ======================================================================
# UPLIFT CALIBRATION
# ======================================================================

def calculate_uplift_calibration(
    test_df,
    uplift,
    n_bins=10,
):

    calibration = pd.DataFrame(
        {
            "uplift": uplift,
            "treatment": test_df[
                "treatment"
            ].to_numpy(),
            "outcome": test_df[
                "recovered"
            ].to_numpy(),
        }
    )

    calibration[
        "uplift_decile"
    ] = pd.qcut(
        calibration["uplift"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    ) + 1

    rows = []

    for decile, group in calibration.groupby(
        "uplift_decile",
        observed=True,
    ):

        treatment_group = group[
            group["treatment"] == 1
        ]

        control_group = group[
            group["treatment"] == 0
        ]

        treatment_rate = (
            treatment_group["outcome"].mean()
            if len(treatment_group)
            else np.nan
        )

        control_rate = (
            control_group["outcome"].mean()
            if len(control_group)
            else np.nan
        )

        observed_lift = (
            treatment_rate - control_rate
            if (
                not np.isnan(treatment_rate)
                and not np.isnan(control_rate)
            )
            else np.nan
        )

        rows.append(
            {
                "uplift_decile": int(decile),
                "rows": len(group),
                "predicted_uplift": group[
                    "uplift"
                ].mean(),
                "observed_lift": observed_lift,
            }
        )

    result = pd.DataFrame(rows)

    valid = result.dropna(
        subset=[
            "predicted_uplift",
            "observed_lift",
        ]
    )

    if len(valid):
        mae = mean_absolute_error(
            valid["observed_lift"],
            valid["predicted_uplift"],
        )
    else:
        mae = np.nan

    return result, float(mae)


# ======================================================================
# ECONOMIC TARGETING
# ======================================================================

def evaluate_targeting(
    test_df,
    predicted_uplift,
    policy,
):

    df = test_df.copy()

    df["_uplift"] = predicted_uplift

    intervention_cost = pd.to_numeric(
        df["intervention_cost"],
        errors="coerce",
    ).fillna(0)

    friction_cost = pd.to_numeric(
        df["customer_friction_cost"],
        errors="coerce",
    ).fillna(0)

    recovered_amount = pd.to_numeric(
        df["recovered_amount"],
        errors="coerce",
    ).fillna(0)

    recovered = pd.to_numeric(
        df["recovered"],
        errors="coerce",
    ).fillna(0)

    observed_net = (
        recovered_amount
        - intervention_cost
        - friction_cost
    )

    if policy == "NO_INTERVENTION":

        mask = np.zeros(
            len(df),
            dtype=bool,
        )

    elif policy == "RETRY_ALL":

        mask = np.ones(
            len(df),
            dtype=bool,
        )

    elif policy.startswith("TOP_"):

        percentage = float(
            policy.split("_")[1]
        )

        count = int(
            np.ceil(
                len(df)
                * percentage
                / 100
            )
        )

        order = np.argsort(
            -df["_uplift"].to_numpy()
        )

        mask = np.zeros(
            len(df),
            dtype=bool,
        )

        mask[
            order[:count]
        ] = True

    elif policy.startswith("UPLIFT_"):

        threshold = float(
            policy.split("_")[1]
        )

        mask = (
            df["_uplift"]
            >= threshold
        )

    else:
        raise ValueError(
            f"Unknown policy: {policy}"
        )

    interventions = int(mask.sum())

    if interventions == 0:

        observed_recovered = 0.0
        observed_value = 0.0
        observed_net = 0.0
        net_per_intervention = 0.0

    else:

        selected_recovered = recovered[
            mask
        ]

        selected_net = (
            observed_net[mask]
        )

        observed_recovered = float(
            selected_recovered.sum()
        )

        observed_value = float(
            recovered_amount[mask].sum()
        )

        observed_net = float(
            selected_net.sum()
        )

        net_per_intervention = (
            observed_net
            / interventions
        )

    # Economic uplift is estimated from predicted
    # incremental recovery probability.
    #
    # We estimate incremental value as:
    #
    # predicted uplift * amount
    #
    # Then subtract expected intervention/friction
    # costs for selected transactions.

    predicted_incremental_value = float(
        (
            df.loc[
                mask,
                "_uplift"
            ]
            * df.loc[
                mask,
                "amount"
            ]
        ).sum()
    )

    predicted_cost = float(
        (
            intervention_cost[mask]
            + friction_cost[mask]
        ).sum()
    )

    predicted_incremental_net = (
        predicted_incremental_value
        - predicted_cost
    )

    return {
        "policy": policy,
        "interventions": interventions,
        "intervention_rate": (
            interventions / len(df)
            if len(df)
            else 0
        ),
        "observed_recovered_payments": int(
            recovered[mask].sum()
        ),
        "observed_recovered_value": (
            observed_value
        ),
        "observed_net_value": (
            observed_net
        ),
        "observed_net_per_intervention": (
            net_per_intervention
        ),
        "predicted_incremental_net_value": (
            predicted_incremental_net
        ),
    }


# ======================================================================
# EVALUATE ONE SPLIT
# ======================================================================

def evaluate_split(
    df,
    numeric_features,
    categorical_features,
    seed,
):

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=df["treatment"],
    )

    train_treatment = train_df[
        train_df["treatment"] == 1
    ].copy()

    train_control = train_df[
        train_df["treatment"] == 0
    ].copy()

    test_treatment = test_df[
        test_df["treatment"] == 1
    ].copy()

    test_control = test_df[
        test_df["treatment"] == 0
    ].copy()

    X_train_treatment = train_treatment[
        numeric_features
        + categorical_features
    ]

    X_train_control = train_control[
        numeric_features
        + categorical_features
    ]

    y_train_treatment = train_treatment[
        "recovered"
    ].astype(float)

    y_train_control = train_control[
        "recovered"
    ].astype(float)

    X_test = test_df[
        numeric_features
        + categorical_features
    ]

    y_test = test_df[
        "recovered"
    ].astype(float)

    # --------------------------------------------------------------
    # Treatment model
    # --------------------------------------------------------------

    treatment_model = build_model(
        numeric_features,
        categorical_features,
        seed,
    )

    treatment_model.fit(
        X_train_treatment,
        y_train_treatment,
    )

    # --------------------------------------------------------------
    # Control model
    # --------------------------------------------------------------

    control_model = build_model(
        numeric_features,
        categorical_features,
        seed + 10000,
    )

    control_model.fit(
        X_train_control,
        y_train_control,
    )

    # --------------------------------------------------------------
    # Predict counterfactual probabilities
    # --------------------------------------------------------------

    predicted_treatment = np.clip(
        treatment_model.predict(X_test),
        0,
        1,
    )

    predicted_control = np.clip(
        control_model.predict(X_test),
        0,
        1,
    )

    uplift = (
        predicted_treatment
        - predicted_control
    )

    # --------------------------------------------------------------
    # Probability quality
    # --------------------------------------------------------------

    treatment_brier = brier_score_loss(
        test_treatment["recovered"],
        np.clip(
            treatment_model.predict(
                test_treatment[
                    numeric_features
                    + categorical_features
                ]
            ),
            0,
            1,
        ),
    )

    control_brier = brier_score_loss(
        test_control["recovered"],
        np.clip(
            control_model.predict(
                test_control[
                    numeric_features
                    + categorical_features
                ]
            ),
            0,
            1,
        ),
    )

    try:

        treatment_auc = roc_auc_score(
            test_treatment["recovered"],
            treatment_model.predict(
                test_treatment[
                    numeric_features
                    + categorical_features
                ]
            ),
        )

    except ValueError:

        treatment_auc = np.nan

    try:

        control_auc = roc_auc_score(
            test_control["recovered"],
            control_model.predict(
                test_control[
                    numeric_features
                    + categorical_features
                ]
            ),
        )

    except ValueError:

        control_auc = np.nan

    # --------------------------------------------------------------
    # QINI
    # --------------------------------------------------------------

    qini = calculate_qini(
        test_df["treatment"].to_numpy(),
        test_df["recovered"].to_numpy(),
        uplift,
    )

    # --------------------------------------------------------------
    # Calibration
    # --------------------------------------------------------------

    calibration_df, calibration_mae = (
        calculate_uplift_calibration(
            test_df,
            uplift,
        )
    )

    # --------------------------------------------------------------
    # Policies
    # --------------------------------------------------------------

    policies = [
        "NO_INTERVENTION",
        "RETRY_ALL",
        "TOP_5",
        "TOP_10",
        "TOP_20",
        "TOP_30",
        "TOP_40",
        "TOP_50",
        "TOP_60",
        "TOP_70",
        "TOP_80",
        "TOP_90",
        "UPLIFT_0.00",
        "UPLIFT_0.01",
        "UPLIFT_0.02",
        "UPLIFT_0.03",
        "UPLIFT_0.05",
        "UPLIFT_0.10",
    ]

    policy_results = []

    for policy in policies:

        result = evaluate_targeting(
            test_df,
            uplift,
            policy,
        )

        policy_results.append(result)

    policy_df = pd.DataFrame(
        policy_results
    )

    best_row = policy_df.loc[
        policy_df[
            "predicted_incremental_net_value"
        ].idxmax()
    ]

    retry_all_row = policy_df[
        policy_df["policy"] == "RETRY_ALL"
    ].iloc[0]

    no_intervention_row = policy_df[
        policy_df["policy"]
        == "NO_INTERVENTION"
    ].iloc[0]

    targeting_advantage = (
        best_row[
            "predicted_incremental_net_value"
        ]
    )

    return {
        "seed": seed,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_treatment_rows": len(
            train_treatment
        ),
        "train_control_rows": len(
            train_control
        ),
        "test_treatment_rows": len(
            test_treatment
        ),
        "test_control_rows": len(
            test_control
        ),
        "qini": qini,
        "mean_predicted_uplift": float(
            np.mean(uplift)
        ),
        "std_predicted_uplift": float(
            np.std(uplift)
        ),
        "positive_uplift_rate": float(
            np.mean(uplift > 0)
        ),
        "treatment_brier": treatment_brier,
        "control_brier": control_brier,
        "treatment_auc": treatment_auc,
        "control_auc": control_auc,
        "calibration_mae": calibration_mae,
        "best_policy": best_row["policy"],
        "best_policy_interventions": int(
            best_row["interventions"]
        ),
        "best_policy_rate": float(
            best_row["intervention_rate"]
        ),
        "predicted_targeting_value": float(
            best_row[
                "predicted_incremental_net_value"
            ]
        ),
        "observed_targeting_net": float(
            best_row[
                "observed_net_value"
            ]
        ),
        "retry_all_observed_net": float(
            retry_all_row[
                "observed_net_value"
            ]
        ),
        "retry_all_predicted_incremental": float(
            retry_all_row[
                "predicted_incremental_net_value"
            ]
        ),
        "no_intervention_observed_net": float(
            no_intervention_row[
                "observed_net_value"
            ]
        ),
        "targeting_vs_retry_all_observed": float(
            best_row[
                "observed_net_value"
            ]
            - retry_all_row[
                "observed_net_value"
            ]
        ),
        "targeting_vs_no_intervention_observed": float(
            best_row[
                "observed_net_value"
            ]
            - no_intervention_row[
                "observed_net_value"
            ]
        ),
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    df = load_experiment()

    validate_experiment(df)

    numeric_features, categorical_features, features = (
        select_features(df)
    )

    header("EXPERIMENT BASELINE")

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

    observed_lift = (
        treatment_rate
        - control_rate
    )

    treatment_net = treatment[
        "net_recovered_value"
    ].sum()

    control_net = control[
        "net_recovered_value"
    ].sum()

    treatment_mean_net = treatment[
        "net_recovered_value"
    ].mean()

    control_mean_net = control[
        "net_recovered_value"
    ].mean()

    observed_incremental_net = (
        treatment_mean_net
        - control_mean_net
    )

    print(
        f"Treatment recovery rate: "
        f"{pct(treatment_rate)}"
    )

    print(
        f"Control recovery rate:   "
        f"{pct(control_rate)}"
    )

    print(
        f"Observed recovery lift:  "
        f"{pct(observed_lift)}"
    )

    print()

    print(
        f"Treatment net value: "
        f"{money(treatment_net)}"
    )

    print(
        f"Control net value:   "
        f"{money(control_net)}"
    )

    print(
        f"Treatment mean net:  "
        f"{money(treatment_mean_net)}"
    )

    print(
        f"Control mean net:    "
        f"{money(control_mean_net)}"
    )

    print(
        f"Observed incremental net / transaction: "
        f"{money(observed_incremental_net)}"
    )

    header("RUNNING REPEATED HELD-OUT VALIDATION")

    print(
        f"Randomized splits: {N_SPLITS}"
    )

    print(
        f"Test size: {TEST_SIZE:.0%}"
    )

    results = []

    for i in range(N_SPLITS):

        seed = 100 + i

        print(
            f"\n[{i + 1:02d}/{N_SPLITS}] "
            f"Evaluating seed {seed}..."
        )

        try:

            result = evaluate_split(
                df,
                numeric_features,
                categorical_features,
                seed,
            )

            results.append(result)

            print(
                f"  QINI: "
                f"{result['qini']:,.4f}"
            )

            print(
                f"  Mean uplift: "
                f"{pct(result['mean_predicted_uplift'])}"
            )

            print(
                f"  Calibration MAE: "
                f"{pct(result['calibration_mae'])}"
            )

            print(
                "  Best policy: "
                f"{result['best_policy']}"
            )

            print(
                "  Predicted targeting value: "
                f"{money(result['predicted_targeting_value'])}"
            )

            print(
                "  Targeting vs RETRY_ALL "
                "observed: "
                f"{money(result['targeting_vs_retry_all_observed'])}"
            )

        except Exception as exc:

            print(
                f"  [WARN] Split failed: {exc}"
            )

    if not results:
        raise RuntimeError(
            "All validation splits failed."
        )

    results_df = pd.DataFrame(
        results
    )

    # ==================================================================
    # STABILITY SUMMARY
    # ==================================================================

    header("V2 STABILITY SUMMARY")

    qini_mean = safe_mean(
        results_df["qini"]
    )

    qini_std = safe_std(
        results_df["qini"]
    )

    positive_qini_runs = int(
        (
            results_df["qini"] > 0
        ).sum()
    )

    uplift_mean = safe_mean(
        results_df[
            "mean_predicted_uplift"
        ]
    )

    uplift_std = safe_std(
        results_df[
            "mean_predicted_uplift"
        ]
    )

    calibration_mean = safe_mean(
        results_df[
            "calibration_mae"
        ]
    )

    targeting_vs_retry_mean = safe_mean(
        results_df[
            "targeting_vs_retry_all_observed"
        ]
    )

    targeting_vs_retry_std = safe_std(
        results_df[
            "targeting_vs_retry_all_observed"
        ]
    )

    targeting_retry_ci_low, targeting_retry_ci_high = (
        ci95(
            results_df[
                "targeting_vs_retry_all_observed"
            ]
        )
    )

    targeting_vs_no_intervention_mean = safe_mean(
        results_df[
            "targeting_vs_no_intervention_observed"
        ]
    )

    beat_retry_rate = safe_mean(
        (
            results_df[
                "targeting_vs_retry_all_observed"
            ] > 0
        ).astype(float)
    )

    beat_no_intervention_rate = safe_mean(
        (
            results_df[
                "targeting_vs_no_intervention_observed"
            ] > 0
        ).astype(float)
    )

    print(
        f"Validation runs: {len(results_df)}"
    )

    print()

    print(
        f"Mean QINI: {qini_mean:,.4f}"
    )

    print(
        f"QINI std dev: {qini_std:,.4f}"
    )

    print(
        f"Positive-QINI runs: "
        f"{positive_qini_runs}/"
        f"{len(results_df)} "
        f"({pct(positive_qini_runs / len(results_df))})"
    )

    print()

    print(
        f"Mean predicted uplift: "
        f"{pct(uplift_mean)}"
    )

    print(
        f"Uplift std dev: "
        f"{pct(uplift_std)}"
    )

    print()

    print(
        f"Mean uplift calibration MAE: "
        f"{pct(calibration_mean)}"
    )

    print()

    print(
        "TARGETING VS RETRY_ALL"
    )

    print(
        f"Mean economic difference: "
        f"{money(targeting_vs_retry_mean)}"
    )

    print(
        f"Std dev: "
        f"{money(targeting_vs_retry_std)}"
    )

    print(
        f"95% CI: "
        f"[{money(targeting_retry_ci_low)}, "
        f"{money(targeting_retry_ci_high)}]"
    )

    print(
        f"Runs where targeting beats RETRY_ALL: "
        f"{int(beat_retry_rate * len(results_df))}/"
        f"{len(results_df)} "
        f"({pct(beat_retry_rate)})"
    )

    print()

    print(
        "TARGETING VS NO_INTERVENTION"
    )

    print(
        f"Mean economic difference: "
        f"{money(targeting_vs_no_intervention_mean)}"
    )

    print(
        f"Runs where targeting beats NO_INTERVENTION: "
        f"{int(beat_no_intervention_rate * len(results_df))}/"
        f"{len(results_df)} "
        f"({pct(beat_no_intervention_rate)})"
    )

    # ==================================================================
    # POLICY DISTRIBUTION
    # ==================================================================

    header("BEST POLICY STABILITY")

    policy_counts = (
        results_df[
            "best_policy"
        ]
        .value_counts()
    )

    for policy, count in policy_counts.items():

        print(
            f"{policy:<22} "
            f"{count:>3}/{len(results_df)} "
            f"({pct(count / len(results_df))})"
        )

    # ==================================================================
    # VERDICT
    # ==================================================================

    header("FINAL V2 CAUSAL UPLIFT STABILITY VERDICT")

    # Conservative deployment criteria.
    #
    # 1. Most QINI runs must be positive.
    # 2. Targeting must beat RETRY_ALL in a majority
    #    of validation runs.
    # 3. Mean targeting advantage must be positive.
    # 4. 95% CI should not be entirely negative.
    # 5. Targeting must beat NO_INTERVENTION in most runs.
    #
    # Calibration is reported separately because good
    # ranking and perfect probability calibration are
    # different properties.

    qini_stable = (
        positive_qini_runs
        / len(results_df)
        >= 0.70
    )

    targeting_stable = (
        beat_retry_rate
        >= 0.70
    )

    positive_mean_economic_value = (
        targeting_vs_retry_mean > 0
    )

    retry_ci_supports_value = (
        targeting_retry_ci_low > 0
    )

    no_intervention_stable = (
        beat_no_intervention_rate
        >= 0.70
    )

    print()

    print(
        f"Positive QINI rate: "
        f"{pct(positive_qini_runs / len(results_df))}"
    )

    print(
        f"Beat RETRY_ALL rate: "
        f"{pct(beat_retry_rate)}"
    )

    print(
        f"Beat NO_INTERVENTION rate: "
        f"{pct(beat_no_intervention_rate)}"
    )

    print(
        f"Mean targeting advantage vs RETRY_ALL: "
        f"{money(targeting_vs_retry_mean)}"
    )

    print(
        f"95% CI lower bound: "
        f"{money(targeting_retry_ci_low)}"
    )

    print()

    if (
        qini_stable
        and targeting_stable
        and positive_mean_economic_value
        and retry_ci_supports_value
        and no_intervention_stable
    ):

        verdict = "TARGETING_STABLE"

        print(
            "[PASS] Causal targeting is stable "
            "across repeated held-out validation."
        )

    else:

        verdict = "TARGETING_NOT_STABLE"

        print(
            "[WARN] Causal targeting is NOT "
            "stable enough to deploy."
        )

    print()

    print(
        f"FINAL VERDICT: {verdict}"
    )

    # ==================================================================
    # SAVE RESULTS
    # ==================================================================

    header("SAVING V2 STABILITY RESULTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print(
        "[PASS] Saved stability results:"
    )

    print(
        RESULTS_FILE
    )

    summary = pd.DataFrame(
        [
            {
                "validation_runs": len(results_df),
                "test_size": TEST_SIZE,
                "mean_qini": qini_mean,
                "qini_std": qini_std,
                "positive_qini_runs": positive_qini_runs,
                "positive_qini_rate": (
                    positive_qini_runs
                    / len(results_df)
                ),
                "mean_predicted_uplift": uplift_mean,
                "uplift_std": uplift_std,
                "mean_calibration_mae": calibration_mean,
                "mean_targeting_vs_retry_all": (
                    targeting_vs_retry_mean
                ),
                "std_targeting_vs_retry_all": (
                    targeting_vs_retry_std
                ),
                "targeting_vs_retry_all_ci_low": (
                    targeting_retry_ci_low
                ),
                "targeting_vs_retry_all_ci_high": (
                    targeting_retry_ci_high
                ),
                "beat_retry_all_rate": (
                    beat_retry_rate
                ),
                "mean_targeting_vs_no_intervention": (
                    targeting_vs_no_intervention_mean
                ),
                "beat_no_intervention_rate": (
                    beat_no_intervention_rate
                ),
                "verdict": verdict,
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        "[PASS] Saved stability summary:"
    )

    print(
        SUMMARY_FILE
    )

    # ==================================================================
    # FINAL OUTPUT
    # ==================================================================

    header("CAUSAL UPLIFT V2 STABILITY VALIDATION COMPLETE")

    print(
        f"Validation runs:              "
        f"{len(results_df)}"
    )

    print(
        f"Mean QINI:                    "
        f"{qini_mean:,.4f}"
    )

    print(
        f"Positive QINI runs:           "
        f"{positive_qini_runs}/"
        f"{len(results_df)}"
    )

    print(
        f"Mean predicted uplift:        "
        f"{pct(uplift_mean)}"
    )

    print(
        f"Mean calibration MAE:         "
        f"{pct(calibration_mean)}"
    )

    print(
        f"Mean targeting advantage:     "
        f"{money(targeting_vs_retry_mean)}"
    )

    print(
        f"Beat RETRY_ALL:               "
        f"{pct(beat_retry_rate)}"
    )

    print(
        f"Beat NO_INTERVENTION:         "
        f"{pct(beat_no_intervention_rate)}"
    )

    print(
        f"95% CI vs RETRY_ALL:          "
        f"[{money(targeting_retry_ci_low)}, "
        f"{money(targeting_retry_ci_high)}]"
    )

    print()

    print(
        f"FINAL VERDICT: {verdict}"
    )

    if verdict == "TARGETING_STABLE":

        print()
        print(
            "NEXT STEP:"
        )

        print(
            "Proceed to the V2 economic policy gate."
        )

        print(
            "The uplift policy has passed repeated "
            "held-out stability checks."
        )

    else:

        print()
        print(
            "NEXT STEP:"
        )

        print(
            "Do NOT deploy the uplift policy."
        )

        print(
            "Improve the uplift model/features or "
            "experimental design and rerun validation."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "A stable uplift model is not automatically "
        "a production deployment approval."
    )

    print(
        "Production deployment should require a "
        "future randomized validation/rollout."
    )

    print()
    print(
        "STATUS: "
        f"{verdict}"
    )


if __name__ == "__main__":
    main()