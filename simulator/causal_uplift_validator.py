"""
RECLAIM CAUSAL UPLIFT VALIDATOR

Purpose:
    Validate whether the causal uplift model can identify transactions
    that benefit from RECLAIM intervention on HELD-OUT randomized data.

Important:
    - Treatment assignment already happened before outcomes were generated.
    - The model is trained only on the training split.
    - Test outcomes are never used to train the model.
    - Policy evaluation is performed on the held-out test set.

Input:
    data/generated/reclaim_treatment_control_experiment.csv

Outputs:
    data/generated/causal_uplift_validation_predictions.csv
    data/generated/causal_uplift_validation_policies.csv
    data/generated/causal_uplift_validation_summary.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ==============================================================
# CONFIGURATION
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "generated"

INPUT_FILE = DATA_DIR / "reclaim_treatment_control_experiment.csv"

PREDICTIONS_FILE = DATA_DIR / "causal_uplift_validation_predictions.csv"
POLICY_FILE = DATA_DIR / "causal_uplift_validation_policies.csv"
SUMMARY_FILE = DATA_DIR / "causal_uplift_validation_summary.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.30

MIN_GROUP_SIZE = 10


# ==============================================================
# DISPLAY
# ==============================================================

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


# ==============================================================
# LOAD DATA
# ==============================================================

def load_data():

    header("RECLAIM CAUSAL UPLIFT VALIDATOR")

    print("Loading treatment/control experiment...")
    print(f"File: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Experiment file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"[PASS] Rows loaded: {len(df)}")

    required = [
        "transaction_id",
        "treatment",
        "recovered",
        "recovered_amount",
        "intervention_cost",
        "customer_friction_cost",
        "net_recovered_value",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required causal columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    if df["transaction_id"].duplicated().any():
        raise ValueError("Transaction IDs are not unique.")

    print("[PASS] Transaction IDs are unique")
    print("[PASS] Required causal columns present")

    return df


# ==============================================================
# VALIDATE EXPERIMENT
# ==============================================================

def validate_experiment(df):

    header("VALIDATING RANDOMIZED EXPERIMENT")

    treatment_values = sorted(df["treatment"].dropna().unique().tolist())

    print(f"Treatment values: {treatment_values}")

    if not set(treatment_values).issubset({0, 1}):
        raise ValueError("Treatment must contain only 0 and 1.")

    print("[PASS] Treatment values valid")

    treatment_count = int((df["treatment"] == 1).sum())
    control_count = int((df["treatment"] == 0).sum())

    print(f"Treatment rows: {treatment_count}")
    print(f"Control rows:   {control_count}")

    if treatment_count < MIN_GROUP_SIZE:
        raise ValueError("Treatment group is too small.")

    if control_count < MIN_GROUP_SIZE:
        raise ValueError("Control group is too small.")

    print("[PASS] Treatment/control groups usable")

    df["recovered"] = pd.to_numeric(
        df["recovered"], errors="coerce"
    ).fillna(0).astype(int)

    df["recovered_amount"] = pd.to_numeric(
        df["recovered_amount"], errors="coerce"
    ).fillna(0)

    df["intervention_cost"] = pd.to_numeric(
        df["intervention_cost"], errors="coerce"
    ).fillna(0)

    df["customer_friction_cost"] = pd.to_numeric(
        df["customer_friction_cost"], errors="coerce"
    ).fillna(0)

    df["net_recovered_value"] = pd.to_numeric(
        df["net_recovered_value"], errors="coerce"
    ).fillna(0)

    if not df["recovered"].isin([0, 1]).all():
        raise ValueError("Recovered must contain only 0/1.")

    print("[PASS] Recovery outcome valid")

    if (df["recovered_amount"] < 0).any():
        raise ValueError("Negative recovered amounts detected.")

    print("[PASS] Economic variables valid")

    return df


# ==============================================================
# FEATURE SELECTION
# ==============================================================

def select_features(df):

    header("SELECTING UPLIFT MODEL FEATURES")

    preferred_numeric = [
        "amount",
        "recovery_probability",
        "calibrated_probability",
        "confidence",
        "historical_transaction_count",
        "historical_success_rate",
        "historical_failure_count",
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
        "base_recovery_probability",
    ]

    categorical = [
        "failure_reason",
        "payment_method",
    ]

    numeric_features = [
        c for c in preferred_numeric
        if c in df.columns
    ]

    categorical_features = [
        c for c in categorical
        if c in df.columns
    ]

    features = numeric_features + categorical_features

    if len(features) < 2:
        raise ValueError(
            "Not enough features available for causal modeling."
        )

    print(f"Features selected: {len(features)}")

    for c in features:
        print(f"  [YES] {c}")

    return features, numeric_features, categorical_features


# ==============================================================
# PREPROCESSOR
# ==============================================================

def build_preprocessor(numeric_features, categorical_features):

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
                numeric_features,
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
                    ),
                ),
            ]
        )

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


# ==============================================================
# OBSERVED EFFECT
# ==============================================================

def observed_effect(df):

    header("FULL EXPERIMENT EFFECT")

    treatment = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]

    treatment_rate = treatment["recovered"].mean()
    control_rate = control["recovered"].mean()

    lift = treatment_rate - control_rate

    print(
        f"Treatment recovery rate: {treatment_rate:.2%}"
    )

    print(
        f"Control recovery rate:   {control_rate:.2%}"
    )

    print(
        f"Observed recovery lift:  {lift:+.2%}"
    )

    treatment_value = treatment["recovered_amount"].sum()
    control_value = control["recovered_amount"].sum()

    treatment_net = treatment["net_recovered_value"].sum()
    control_net = control["net_recovered_value"].sum()

    print()
    print(
        f"Treatment recovered value: ₹{treatment_value:,.2f}"
    )

    print(
        f"Control recovered value:   ₹{control_value:,.2f}"
    )

    print()
    print(
        f"Treatment net value: ₹{treatment_net:,.2f}"
    )

    print(
        f"Control net value:   ₹{control_net:,.2f}"
    )

    print(
        f"Observed net difference: "
        f"₹{treatment_net - control_net:,.2f}"
    )


# ==============================================================
# TRAIN T-LEARNER
# ==============================================================

def train_t_learner(train_df, features, numeric_features, categorical_features):

    header("TRAINING CAUSAL T-LEARNER")

    treatment_train = train_df[
        train_df["treatment"] == 1
    ].copy()

    control_train = train_df[
        train_df["treatment"] == 0
    ].copy()

    print(
        f"Treatment training rows: {len(treatment_train)}"
    )

    print(
        f"Control training rows:   {len(control_train)}"
    )

    if len(treatment_train) < MIN_GROUP_SIZE:
        raise ValueError("Treatment training set too small.")

    if len(control_train) < MIN_GROUP_SIZE:
        raise ValueError("Control training set too small.")

    treatment_preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    control_preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    treatment_model = Pipeline(
        steps=[
            (
                "preprocess",
                treatment_preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    control_model = Pipeline(
        steps=[
            (
                "preprocess",
                control_preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    X_treatment = treatment_train[features]
    y_treatment = treatment_train["recovered"]

    X_control = control_train[features]
    y_control = control_train["recovered"]

    treatment_model.fit(
        X_treatment,
        y_treatment,
    )

    control_model.fit(
        X_control,
        y_control,
    )

    print("[PASS] Treatment model trained")
    print("[PASS] Control model trained")

    return treatment_model, control_model


# ==============================================================
# PREDICT HELD-OUT UPLIFT
# ==============================================================

def predict_uplift(
    model_treatment,
    model_control,
    test_df,
    features,
):

    header("ESTIMATING HELD-OUT INDIVIDUAL TREATMENT EFFECT")

    X_test = test_df[features]

    p_treatment = model_treatment.predict_proba(
        X_test
    )[:, 1]

    p_control = model_control.predict_proba(
        X_test
    )[:, 1]

    uplift = p_treatment - p_control

    result = test_df.copy()

    result["predicted_treatment_probability"] = p_treatment
    result["predicted_control_probability"] = p_control
    result["predicted_uplift"] = uplift

    print(
        f"Mean predicted treatment probability: "
        f"{p_treatment.mean():.2%}"
    )

    print(
        f"Mean predicted control probability:   "
        f"{p_control.mean():.2%}"
    )

    print(
        f"Mean predicted uplift: "
        f"{uplift.mean():+.2%}"
    )

    print(
        f"Positive uplift rows: "
        f"{(uplift > 0).sum()}"
    )

    print(
        f"Negative uplift rows: "
        f"{(uplift < 0).sum()}"
    )

    print("[PASS] Held-out uplift estimated")

    return result


# ==============================================================
# ECONOMIC UPLIFT
# ==============================================================

def calculate_economic_uplift(df):

    header("CALCULATING HELD-OUT ECONOMIC UPLIFT")

    df = df.copy()

    df["expected_incremental_recovery"] = (
        df["predicted_uplift"]
        * df["amount"]
    )

    # Treatment costs are known from the experiment design.
    # For a prospective intervention, use the treatment cost.
    df["prospective_intervention_cost"] = (
        df["intervention_cost"]
        .where(
            df["intervention_cost"] > 0,
            1.5,
        )
    )

    # Friction cost is also estimated from the treatment observations.
    treatment_friction = df.loc[
        df["treatment"] == 1,
        "customer_friction_cost"
    ]

    if len(treatment_friction) > 0:
        friction_estimate = treatment_friction.mean()
    else:
        friction_estimate = 0.0

    df["prospective_friction_cost"] = friction_estimate

    df["predicted_incremental_net_value"] = (
        df["expected_incremental_recovery"]
        - df["prospective_intervention_cost"]
        - df["prospective_friction_cost"]
    )

    print(
        "Mean expected incremental recovery: "
        f"₹{df['expected_incremental_recovery'].mean():,.2f}"
    )

    print(
        "Mean predicted incremental net value: "
        f"₹{df['predicted_incremental_net_value'].mean():,.2f}"
    )

    print(
        "Positive predicted economic uplift rows: "
        f"{(df['predicted_incremental_net_value'] > 0).sum()}"
    )

    print("[PASS] Economic uplift calculated")

    return df


# ==============================================================
# POLICY EVALUATION
# ==============================================================

def evaluate_policy(df, policy_name, selected):

    selected = selected.copy()

    n = len(df)
    interventions = len(selected)

    if interventions == 0:

        return {
            "policy": policy_name,
            "interventions": 0,
            "intervention_rate": 0.0,
            "observed_recovered_payments": 0,
            "observed_recovered_value": 0.0,
            "observed_net_value": 0.0,
            "observed_net_per_intervention": 0.0,
            "predicted_incremental_net_value": 0.0,
        }

    recovered_payments = selected["recovered"].sum()

    recovered_value = selected[
        "recovered_amount"
    ].sum()

    intervention_cost = selected[
        "intervention_cost"
    ].sum()

    friction_cost = selected[
        "customer_friction_cost"
    ].sum()

    observed_net = (
        recovered_value
        - intervention_cost
        - friction_cost
    )

    predicted_incremental = selected[
        "predicted_incremental_net_value"
    ].sum()

    return {
        "policy": policy_name,
        "interventions": interventions,
        "intervention_rate": interventions / n,
        "observed_recovered_payments": recovered_payments,
        "observed_recovered_value": recovered_value,
        "observed_net_value": observed_net,
        "observed_net_per_intervention": (
            observed_net / interventions
        ),
        "predicted_incremental_net_value": predicted_incremental,
    }


def build_policies(df):

    header("TESTING HELD-OUT TARGETING POLICIES")

    results = []

    # ----------------------------------------------------------
    # NO INTERVENTION
    # ----------------------------------------------------------

    results.append(
        evaluate_policy(
            df,
            "NO_INTERVENTION",
            df.iloc[0:0],
        )
    )

    # ----------------------------------------------------------
    # RETRY ALL
    # ----------------------------------------------------------

    results.append(
        evaluate_policy(
            df,
            "RETRY_ALL",
            df,
        )
    )

    # ----------------------------------------------------------
    # TOP K UPLIFT
    # ----------------------------------------------------------

    sorted_df = df.sort_values(
        "predicted_uplift",
        ascending=False,
    )

    for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:

        k = max(
            1,
            int(np.ceil(len(df) * pct / 100))
        )

        selected = sorted_df.head(k)

        results.append(
            evaluate_policy(
                df,
                f"TOP_{pct}_UPLIFT",
                selected,
            )
        )

    # ----------------------------------------------------------
    # UPLIFT THRESHOLDS
    # ----------------------------------------------------------

    for threshold in [
        0.00,
        0.01,
        0.02,
        0.03,
        0.05,
        0.10,
    ]:

        selected = df[
            df["predicted_uplift"] >= threshold
        ]

        results.append(
            evaluate_policy(
                df,
                f"UPLIFT_{threshold:.2f}",
                selected,
            )
        )

    # ----------------------------------------------------------
    # POSITIVE ECONOMIC VALUE
    # ----------------------------------------------------------

    selected = df[
        df["predicted_incremental_net_value"] > 0
    ]

    results.append(
        evaluate_policy(
            df,
            "POSITIVE_ECONOMIC_UPLIFT",
            selected,
        )
    )

    print(
        f"Policies evaluated: {len(results)}"
    )

    return pd.DataFrame(results)


# ==============================================================
# POLICY TABLE
# ==============================================================

def print_policy_table(policy_df):

    header("HELD-OUT POLICY COMPARISON")

    display_df = policy_df.copy()

    display_df["intervention_rate"] = (
        display_df["intervention_rate"]
        .map(lambda x: f"{x:.2%}")
    )

    display_df["observed_recovered_value"] = (
        display_df["observed_recovered_value"]
        .map(lambda x: f"₹{x:,.2f}")
    )

    display_df["observed_net_value"] = (
        display_df["observed_net_value"]
        .map(lambda x: f"₹{x:,.2f}")
    )

    display_df["observed_net_per_intervention"] = (
        display_df["observed_net_per_intervention"]
        .map(lambda x: f"₹{x:,.2f}")
    )

    display_df["predicted_incremental_net_value"] = (
        display_df["predicted_incremental_net_value"]
        .map(lambda x: f"₹{x:,.2f}")
    )

    print(display_df.to_string(index=False))


# ==============================================================
# QINI / UPLIFT CURVE
# ==============================================================

def calculate_qini(df):

    section("QINI / UPLIFT CURVE")

    ranked = df.sort_values(
        "predicted_uplift",
        ascending=False,
    ).reset_index(drop=True)

    rows = []

    cumulative_treatment = 0
    cumulative_control = 0

    treatment_recovered = 0
    control_recovered = 0

    for i, row in ranked.iterrows():

        if row["treatment"] == 1:

            cumulative_treatment += 1
            treatment_recovered += row["recovered"]

        else:

            cumulative_control += 1
            control_recovered += row["recovered"]

        if (
            cumulative_treatment > 0
            and cumulative_control > 0
        ):

            treatment_rate = (
                treatment_recovered
                / cumulative_treatment
            )

            control_rate = (
                control_recovered
                / cumulative_control
            )

            incremental = (
                treatment_rate
                - control_rate
            ) * (i + 1)

        else:

            incremental = 0.0

        rows.append(
            {
                "rank": i + 1,
                "target_fraction": (i + 1) / len(ranked),
                "incremental_recovery": incremental,
            }
        )

    qini_df = pd.DataFrame(rows)

    if len(qini_df) > 1:

        qini_score = np.trapezoid(
            qini_df["incremental_recovery"],
            qini_df["target_fraction"],
        )

    else:

        qini_score = 0.0

    print(
        f"QINI score: {qini_score:,.4f}"
    )

    print("[PASS] QINI curve calculated")

    return qini_df, qini_score


# ==============================================================
# TEST MODEL DIAGNOSTICS
# ==============================================================

def model_diagnostics(
    model_treatment,
    model_control,
    test_df,
    features,
):

    header("HELD-OUT MODEL DIAGNOSTICS")

    treatment_test = test_df[
        test_df["treatment"] == 1
    ]

    control_test = test_df[
        test_df["treatment"] == 0
    ]

    treatment_probability = model_treatment.predict_proba(
        treatment_test[features]
    )[:, 1]

    control_probability = model_control.predict_proba(
        control_test[features]
    )[:, 1]

    treatment_brier = brier_score_loss(
        treatment_test["recovered"],
        treatment_probability,
    )

    control_brier = brier_score_loss(
        control_test["recovered"],
        control_probability,
    )

    print(
        f"Treatment test Brier score: "
        f"{treatment_brier:.6f}"
    )

    print(
        f"Control test Brier score:   "
        f"{control_brier:.6f}"
    )

    print(
        "[PASS] Diagnostics calculated on held-out data"
    )

    return treatment_brier, control_brier


# ==============================================================
# FINAL VERDICT
# ==============================================================

def determine_verdict(policy_df):

    header("FINAL CAUSAL TARGETING VERDICT")

    retry = policy_df[
        policy_df["policy"] == "RETRY_ALL"
    ].iloc[0]

    positive = policy_df[
        policy_df["policy"] == "POSITIVE_ECONOMIC_UPLIFT"
    ].iloc[0]

    top20 = policy_df[
        policy_df["policy"] == "TOP_20_UPLIFT"
    ].iloc[0]

    top30 = policy_df[
        policy_df["policy"] == "TOP_30_UPLIFT"
    ].iloc[0]

    best_targeting = max(
        [positive, top20, top30],
        key=lambda x: x["observed_net_value"],
    )

    print(
        f"RETRY_ALL held-out net value: "
        f"₹{retry['observed_net_value']:,.2f}"
    )

    print(
        f"Best targeting policy held-out net value: "
        f"₹{best_targeting['observed_net_value']:,.2f}"
    )

    difference = (
        best_targeting["observed_net_value"]
        - retry["observed_net_value"]
    )

    print(
        f"Held-out targeting difference: "
        f"₹{difference:,.2f}"
    )

    print()

    if difference > 0:

        print(
            "[PASS] Targeting beats RETRY_ALL "
            "on held-out observed net value."
        )

        verdict = "TARGETING_BEATS_RETRY_ALL"

    else:

        print(
            "[WARN] Targeting does NOT beat RETRY_ALL "
            "on held-out observed net value."
        )

        verdict = "TARGETING_NOT_PROVEN"

    return verdict, difference, best_targeting["policy"]


# ==============================================================
# SAVE RESULTS
# ==============================================================

def save_results(
    uplift_df,
    policy_df,
    qini_df,
    qini_score,
    treatment_brier,
    control_brier,
    verdict,
    difference,
    best_policy,
    train_size,
    test_size,
):

    header("SAVING VALIDATION RESULTS")

    output = uplift_df.copy()

    output.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        "[PASS] Saved held-out uplift predictions:"
    )

    print(PREDICTIONS_FILE)

    policy_df.to_csv(
        POLICY_FILE,
        index=False,
    )

    print(
        "[PASS] Saved held-out policy comparison:"
    )

    print(POLICY_FILE)

    summary = pd.DataFrame(
        [
            {
                "total_rows": train_size + test_size,
                "training_rows": train_size,
                "test_rows": test_size,
                "test_treatment_rows": int(
                    (uplift_df["treatment"] == 1).sum()
                ),
                "test_control_rows": int(
                    (uplift_df["treatment"] == 0).sum()
                ),
                "mean_predicted_uplift": uplift_df[
                    "predicted_uplift"
                ].mean(),
                "positive_predicted_uplift_rows": int(
                    (
                        uplift_df["predicted_uplift"] > 0
                    ).sum()
                ),
                "positive_economic_uplift_rows": int(
                    (
                        uplift_df[
                            "predicted_incremental_net_value"
                        ] > 0
                    ).sum()
                ),
                "treatment_brier": treatment_brier,
                "control_brier": control_brier,
                "qini_score": qini_score,
                "best_policy": best_policy,
                "targeting_vs_retry_all_net_difference": difference,
                "verdict": verdict,
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        "[PASS] Saved validation summary:"
    )

    print(SUMMARY_FILE)

    return summary


# ==============================================================
# MAIN
# ==============================================================

def main():

    df = load_data()

    df = validate_experiment(df)

    observed_effect(df)

    features, numeric_features, categorical_features = (
        select_features(df)
    )

    header("CREATING HELD-OUT RANDOMIZED SPLIT")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["treatment"],
    )

    print(f"Training rows: {len(train_df)}")
    print(f"Test rows:     {len(test_df)}")

    print(
        f"Training treatment: "
        f"{(train_df['treatment'] == 1).sum()}"
    )

    print(
        f"Training control:   "
        f"{(train_df['treatment'] == 0).sum()}"
    )

    print(
        f"Test treatment:     "
        f"{(test_df['treatment'] == 1).sum()}"
    )

    print(
        f"Test control:       "
        f"{(test_df['treatment'] == 0).sum()}"
    )

    print(
        "[PASS] Treatment/control stratification complete"
    )

    model_treatment, model_control = train_t_learner(
        train_df,
        features,
        numeric_features,
        categorical_features,
    )

    treatment_brier, control_brier = model_diagnostics(
        model_treatment,
        model_control,
        test_df,
        features,
    )

    uplift_df = predict_uplift(
        model_treatment,
        model_control,
        test_df,
        features,
    )

    uplift_df = calculate_economic_uplift(
        uplift_df
    )

    policy_df = build_policies(
        uplift_df
    )

    print_policy_table(
        policy_df
    )

    qini_df, qini_score = calculate_qini(
        uplift_df
    )

    verdict, difference, best_policy = determine_verdict(
        policy_df
    )

    save_results(
        uplift_df,
        policy_df,
        qini_df,
        qini_score,
        treatment_brier,
        control_brier,
        verdict,
        difference,
        best_policy,
        len(train_df),
        len(test_df),
    )

    header("CAUSAL UPLIFT VALIDATION COMPLETE")

    print(
        f"Training rows:              {len(train_df)}"
    )

    print(
        f"Held-out test rows:         {len(test_df)}"
    )

    print(
        f"Features used:              {len(features)}"
    )

    print(
        f"Mean predicted uplift:      "
        f"{uplift_df['predicted_uplift'].mean():+.2%}"
    )

    print(
        f"QINI score:                 "
        f"{qini_score:,.4f}"
    )

    print(
        f"Best held-out policy:       {best_policy}"
    )

    print(
        f"Targeting vs RETRY_ALL:     "
        f"₹{difference:,.2f}"
    )

    print()
    print(
        f"FINAL VERDICT: {verdict}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This validation uses held-out randomized data."
    )

    print(
        "A positive result is substantially stronger evidence "
        "than the previous in-sample optimizer result."
    )

    print()
    print("STATUS: VALIDATION COMPLETE")


if __name__ == "__main__":
    main()