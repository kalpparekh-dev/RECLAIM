
"""
RECLAIM CAUSAL UPLIFT OPTIMIZER V4

Economic uplift targeting with:
- leakage protection
- cross-fitted T-learner
- economic treatment-effect scoring
- train-only threshold selection
- held-out policy evaluation
- comparison against NO_INTERVENTION and RETRY_ALL
- policy efficiency analysis
- repeated validation
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 2026

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "reclaim_causal_experiment_v2.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"

PREDICTIONS_FILE = (
    OUTPUT_DIR / "causal_uplift_v4_predictions.csv"
)

POLICY_FILE = (
    OUTPUT_DIR / "causal_uplift_v4_policy_comparison.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "causal_uplift_v4_summary.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR / "causal_uplift_v4_validation_results.csv"
)


TEST_SIZE = 0.30
N_VALIDATION_RUNS = 10
N_FOLDS = 5


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

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# LEAKAGE COLUMNS
# ============================================================

LEAKAGE_COLUMNS = {
    "treatment",
    "control",
    "treatment_group",
    "recovered",
    "recovery_flag",
    "recovered_amount",
    "customer_friction_cost",
    "net_recovered_value",
    "recovery_status",
    "observed_recovery_probability",
    "control_recovery_probability",
    "treatment_recovery_probability",
    "incremental_recovery_probability",
    "expected_control_recovery",
    "expected_treatment_recovery",
    "expected_incremental_recovery",
    "expected_incremental_net_value",
    "treatment_effect",
    "actual_recovery_probability",
}


# ============================================================
# MODEL
# ============================================================

def build_model(seed):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=7,
        min_samples_leaf=25,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(df):

    required = (
        FEATURES
        + [
            "transaction_id",
            "treatment",
            "recovered",
            "amount",
            "intervention_cost",
            "net_recovered_value",
        ]
    )

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df["transaction_id"].duplicated().any():
        raise ValueError(
            "Transaction IDs are not unique."
        )

    if not set(df["treatment"].unique()).issubset({0, 1}):
        raise ValueError(
            "Treatment must contain only 0 and 1."
        )

    if not set(df["recovered"].unique()).issubset({0, 1}):
        raise ValueError(
            "Recovered must contain only 0 and 1."
        )

    print("[PASS] Dataset validation complete")


# ============================================================
# QINI
# ============================================================

def calculate_qini(test_df):

    df = test_df.copy()

    df = df.sort_values(
        "predicted_uplift",
        ascending=False
    ).reset_index(drop=True)

    treated = df["treatment"] == 1
    control = df["treatment"] == 0

    cumulative_treated = treated.cumsum()
    cumulative_control = control.cumsum()

    treated_outcomes = (
        (df["recovered"] * treated).cumsum()
    )

    control_outcomes = (
        (df["recovered"] * control).cumsum()
    )

    incremental = (
        treated_outcomes
        - (
            control_outcomes
            * cumulative_treated
            / cumulative_control.replace(0, np.nan)
        )
    )

    incremental = incremental.fillna(0)

    return float(incremental.max())


# ============================================================
# ECONOMIC UPLIFT
# ============================================================

def calculate_economic_uplift(df):

    """

    Economic value of treating one transaction:

        probability uplift
        × transaction amount
        − intervention cost

    """

    return (
        df["predicted_uplift"]
        * df["amount"]
        - df["intervention_cost"]
    )


# ============================================================
# POLICY EVALUATION
# ============================================================

def evaluate_policy(
    df,
    treatment_mask,
    policy_name,
):

    treated = df[
        treatment_mask
    ].copy()

    intervention_count = len(treated)

    total_rows = len(df)

    if intervention_count == 0:

        return {
            "policy": policy_name,
            "interventions": 0,
            "intervention_rate": 0.0,
            "recovered_payments": 0,
            "total_net_value": 0.0,
            "net_per_intervention": 0.0,
            "incremental_net_value": 0.0,
            "recovery_rate": 0.0,
        }

    recovered_payments = int(
        treated["recovered"].sum()
    )

    total_net_value = float(
        treated["net_recovered_value"].sum()
    )

    net_per_intervention = (
        total_net_value
        / intervention_count
    )

    recovery_rate = (
        recovered_payments
        / intervention_count
    )

    return {
        "policy": policy_name,
        "interventions": intervention_count,
        "intervention_rate": (
            intervention_count / total_rows
        ),
        "recovered_payments": recovered_payments,
        "total_net_value": total_net_value,
        "net_per_intervention": net_per_intervention,
        "incremental_net_value": total_net_value,
        "recovery_rate": recovery_rate,
    }


# ============================================================
# TRAIN-ONLY THRESHOLD SELECTION
# ============================================================

def select_threshold(train_df):

    candidate_thresholds = np.linspace(
        -500,
        500,
        101
    )

    best_threshold = 0.0
    best_value = -np.inf

    for threshold in candidate_thresholds:

        selected = (
            train_df["economic_uplift"]
            >= threshold
        )

        if selected.sum() == 0:
            continue

        value = train_df.loc[
            selected,
            "economic_uplift"
        ].sum()

        if value > best_value:
            best_value = value
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_value),
    )


# ============================================================
# ONE VALIDATION RUN
# ============================================================

def run_validation(df, seed, save_predictions=False):

    print("-" * 70)
    print(f"VALIDATION SEED: {seed}")

    rng = np.random.RandomState(seed)

    shuffled = df.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    split_index = int(
        len(shuffled)
        * (1 - TEST_SIZE)
    )

    train_df = shuffled.iloc[
        :split_index
    ].copy()

    test_df = shuffled.iloc[
        split_index:
    ].copy()

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    train_treatment = (
        train_df["treatment"] == 1
    )

    train_control = (
        train_df["treatment"] == 0
    )

    test_treatment = (
        test_df["treatment"] == 1
    )

    test_control = (
        test_df["treatment"] == 0
    )

    treatment_model = build_model(
        seed
    )

    control_model = build_model(
        seed + 10000
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("Training treatment model...")

    treatment_model.fit(
        train_df.loc[
            train_treatment,
            FEATURES
        ],
        train_df.loc[
            train_treatment,
            "recovered"
        ],
    )

    print("Training control model...")

    control_model.fit(
        train_df.loc[
            train_control,
            FEATURES
        ],
        train_df.loc[
            train_control,
            "recovered"
        ],
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    treatment_probability = (
        treatment_model.predict_proba(
            test_df[FEATURES]
        )[:, 1]
    )

    control_probability = (
        control_model.predict_proba(
            test_df[FEATURES]
        )[:, 1]
    )

    test_df[
        "predicted_treatment_probability"
    ] = treatment_probability

    test_df[
        "predicted_control_probability"
    ] = control_probability

    test_df[
        "predicted_uplift"
    ] = (
        treatment_probability
        - control_probability
    )

    # --------------------------------------------------------
    # ECONOMIC UPLIFT
    # --------------------------------------------------------

    test_df[
        "economic_uplift"
    ] = calculate_economic_uplift(
        test_df
    )

    positive_economic = (
        test_df["economic_uplift"] > 0
    )

    print(
        "Mean probability uplift: "
        f"{test_df['predicted_uplift'].mean():.2%}"
    )

    print(
        "Mean economic uplift: ₹"
        f"{test_df['economic_uplift'].mean():,.2f}"
    )

    print(
        "Positive economic uplift: "
        f"{positive_economic.mean():.2%}"
    )

    # --------------------------------------------------------
    # QINI
    # --------------------------------------------------------

    qini = calculate_qini(
        test_df
    )

    print(
        f"QINI: {qini:,.2f}"
    )

    # --------------------------------------------------------
    # TRAIN-ONLY THRESHOLD
    # --------------------------------------------------------

    # Build an approximate economic uplift
    # on the training data using cross-fitted
    # predictions.

    train_treat_prob = (
        treatment_model.predict_proba(
            train_df[FEATURES]
        )[:, 1]
    )

    train_control_prob = (
        control_model.predict_proba(
            train_df[FEATURES]
        )[:, 1]
    )

    train_df[
        "predicted_uplift"
    ] = (
        train_treat_prob
        - train_control_prob
    )

    train_df[
        "economic_uplift"
    ] = (
        train_df["predicted_uplift"]
        * train_df["amount"]
        - train_df["intervention_cost"]
    )

    threshold, train_value = (
        select_threshold(
            train_df
        )
    )

    print(
        "Selected economic threshold: "
        f"₹{threshold:,.2f}"
    )

    # --------------------------------------------------------
    # TARGETING POLICY
    # --------------------------------------------------------

    targeting_mask = (
        test_df["economic_uplift"]
        >= threshold
    )

    no_intervention_mask = (
        np.zeros(
            len(test_df),
            dtype=bool
        )
    )

    retry_all_mask = (
        np.ones(
            len(test_df),
            dtype=bool
        )
    )

    targeting = evaluate_policy(
        test_df,
        targeting_mask,
        "ECONOMIC_TARGETING",
    )

    no_intervention = evaluate_policy(
        test_df,
        no_intervention_mask,
        "NO_INTERVENTION",
    )

    retry_all = evaluate_policy(
        test_df,
        retry_all_mask,
        "RETRY_ALL",
    )

    policy_df = pd.DataFrame(
        [
            no_intervention,
            retry_all,
            targeting,
        ]
    )

    # --------------------------------------------------------
    # EFFICIENCY
    # --------------------------------------------------------

    targeting_efficiency = (
        targeting["net_per_intervention"]
    )

    retry_efficiency = (
        retry_all["net_per_intervention"]
    )

    efficiency_gain = (
        targeting_efficiency
        - retry_efficiency
    )

    targeting_vs_no_intervention = (
        targeting["total_net_value"]
        - no_intervention["total_net_value"]
    )

    targeting_vs_retry_all = (
        targeting["total_net_value"]
        - retry_all["total_net_value"]
    )

    print(
        f"Intervention rate: "
        f"{targeting['intervention_rate']:.2%}"
    )

    print(
        f"Targeting net/intervention: ₹"
        f"{targeting_efficiency:,.2f}"
    )

    print(
        f"Retry-all net/intervention: ₹"
        f"{retry_efficiency:,.2f}"
    )

    print(
        f"Efficiency gain: ₹"
        f"{efficiency_gain:,.2f}"
    )

    print(
        f"Targeting vs NO_INTERVENTION: ₹"
        f"{targeting_vs_no_intervention:,.2f}"
    )

    print(
        f"Targeting vs RETRY_ALL: ₹"
        f"{targeting_vs_retry_all:,.2f}"
    )

    if save_predictions:

        test_df.to_csv(
            PREDICTIONS_FILE,
            index=False,
        )

        policy_df.to_csv(
            POLICY_FILE,
            index=False,
        )

    return {
        "seed": seed,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "qini": qini,
        "mean_probability_uplift": float(
            test_df["predicted_uplift"].mean()
        ),
        "mean_economic_uplift": float(
            test_df["economic_uplift"].mean()
        ),
        "positive_economic_rate": float(
            positive_economic.mean()
        ),
        "threshold": threshold,
        "intervention_rate": targeting[
            "intervention_rate"
        ],
        "targeting_net_per_intervention": (
            targeting_efficiency
        ),
        "retry_all_net_per_intervention": (
            retry_efficiency
        ),
        "efficiency_gain": efficiency_gain,
        "targeting_vs_no_intervention": (
            targeting_vs_no_intervention
        ),
        "targeting_vs_retry_all": (
            targeting_vs_retry_all
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "RECLAIM CAUSAL UPLIFT OPTIMIZER V4"
    )
    print("=" * 70)

    print(
        "Economic uplift targeting + "
        "held-out policy evaluation"
    )

    print(
        f"File: {DATA_FILE}"
    )

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Experiment file not found:\n"
            f"{DATA_FILE}"
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"[PASS] Rows loaded: {len(df):,}"
    )

    print(
        f"[PASS] Columns loaded: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATING DATA")
    print("=" * 70)

    validate_dataset(df)

    # --------------------------------------------------------
    # LEAKAGE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LEAKAGE PROTECTION")
    print("=" * 70)

    forbidden_features = (
        set(FEATURES)
        & LEAKAGE_COLUMNS
    )

    if forbidden_features:

        raise ValueError(
            "Leakage features detected: "
            f"{forbidden_features}"
        )

    print(
        "[PASS] No outcome/treatment/"
        "counterfactual leakage"
    )

    print(
        f"Modeling features: {len(FEATURES)}"
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    treatment_rate = df.loc[
        df["treatment"] == 1,
        "recovered"
    ].mean()

    control_rate = df.loc[
        df["treatment"] == 0,
        "recovered"
    ].mean()

    observed_lift = (
        treatment_rate
        - control_rate
    )

    print()
    print("=" * 70)
    print("EXPERIMENT BASELINE")
    print("=" * 70)

    print(
        f"Treatment recovery rate: "
        f"{treatment_rate:.2%}"
    )

    print(
        f"Control recovery rate:   "
        f"{control_rate:.2%}"
    )

    print(
        f"Observed recovery lift:  "
        f"{observed_lift:.2%}"
    )

    # --------------------------------------------------------
    # REPEATED VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING V4 REPEATED HELD-OUT VALIDATION")
    print("=" * 70)

    print(
        f"Validation runs: {N_VALIDATION_RUNS}"
    )

    results = []

    for i in range(N_VALIDATION_RUNS):

        seed = 100 + i

        result = run_validation(
            df,
            seed,
            save_predictions=(
                i == N_VALIDATION_RUNS - 1
            ),
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V4 STABILITY SUMMARY")
    print("=" * 70)

    mean_qini = results_df[
        "qini"
    ].mean()

    std_qini = results_df[
        "qini"
    ].std()

    positive_qini_rate = (
        results_df["qini"] > 0
    ).mean()

    mean_uplift = results_df[
        "mean_probability_uplift"
    ].mean()

    mean_economic_uplift = (
        results_df[
            "mean_economic_uplift"
        ].mean()
    )

    mean_intervention_rate = (
        results_df[
            "intervention_rate"
        ].mean()
    )

    mean_efficiency_gain = (
        results_df[
            "efficiency_gain"
        ].mean()
    )

    mean_vs_no_intervention = (
        results_df[
            "targeting_vs_no_intervention"
        ].mean()
    )

    mean_vs_retry_all = (
        results_df[
            "targeting_vs_retry_all"
        ].mean()
    )

    positive_efficiency_runs = (
        results_df[
            "efficiency_gain"
        ] > 0
    ).sum()

    print(
        f"Validation runs: {N_VALIDATION_RUNS}"
    )

    print(
        f"Mean QINI: {mean_qini:,.2f}"
    )

    print(
        f"QINI std dev: {std_qini:,.2f}"
    )

    print(
        f"Positive QINI runs: "
        f"{int((results_df['qini'] > 0).sum())}"
        f"/{N_VALIDATION_RUNS}"
    )

    print(
        f"Mean probability uplift: "
        f"{mean_uplift:.2%}"
    )

    print(
        f"Mean economic uplift: ₹"
        f"{mean_economic_uplift:,.2f}"
    )

    print(
        f"Mean intervention rate: "
        f"{mean_intervention_rate:.2%}"
    )

    print()
    print(
        "TARGETING VS NO_INTERVENTION"
    )

    print(
        f"Mean economic advantage: ₹"
        f"{mean_vs_no_intervention:,.2f}"
    )

    print()
    print(
        "TARGETING EFFICIENCY VS RETRY_ALL"
    )

    print(
        f"Mean efficiency gain/intervention: ₹"
        f"{mean_efficiency_gain:,.2f}"
    )

    print(
        f"Runs with better efficiency: "
        f"{positive_efficiency_runs}"
        f"/{N_VALIDATION_RUNS}"
    )

    print()
    print(
        "TOTAL VALUE VS RETRY_ALL"
    )

    print(
        f"Mean total-value difference: ₹"
        f"{mean_vs_retry_all:,.2f}"
    )

    # --------------------------------------------------------
    # VERDICT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL V4 ECONOMIC UPLIFT VERDICT")
    print("=" * 70)

    if (
        positive_qini_rate >= 0.80
        and mean_vs_no_intervention > 0
        and mean_efficiency_gain > 0
    ):

        verdict = (
            "TARGETING_EFFICIENCY_SUPPORTED"
        )

        print(
            "[PASS] Causal targeting shows "
            "stable positive economic efficiency."
        )

    elif (
        positive_qini_rate >= 0.80
        and mean_vs_no_intervention > 0
    ):

        verdict = (
            "POSITIVE_UPLIFT_BUT_RETRY_ALL_MORE_EFFICIENT"
        )

        print(
            "[WARN] Targeting is causally useful "
            "but RETRY_ALL remains more efficient."
        )

    else:

        verdict = (
            "TARGETING_NOT_SUPPORTED"
        )

        print(
            "[WARN] Economic targeting is "
            "not sufficiently stable."
        )

    print()
    print(
        f"FINAL VERDICT: {verdict}"
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "validation_runs": N_VALIDATION_RUNS,
                "mean_qini": mean_qini,
                "qini_std": std_qini,
                "positive_qini_rate": (
                    positive_qini_rate
                ),
                "mean_probability_uplift": (
                    mean_uplift
                ),
                "mean_economic_uplift": (
                    mean_economic_uplift
                ),
                "mean_intervention_rate": (
                    mean_intervention_rate
                ),
                "mean_efficiency_gain": (
                    mean_efficiency_gain
                ),
                "positive_efficiency_runs": (
                    positive_efficiency_runs
                ),
                "mean_vs_no_intervention": (
                    mean_vs_no_intervention
                ),
                "mean_vs_retry_all_total_value": (
                    mean_vs_retry_all
                ),
                "verdict": verdict,
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("V4 COMPLETE")
    print("=" * 70)

    print(
        f"[PASS] Validation results:\n"
        f"{VALIDATION_FILE}"
    )

    print(
        f"[PASS] Predictions:\n"
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"[PASS] Policy results:\n"
        f"{POLICY_FILE}"
    )

    print(
        f"[PASS] Summary:\n"
        f"{SUMMARY_FILE}"
    )

    print()
    print(
        f"STATUS: {verdict}"
    )


if __name__ == "__main__":
    main()

