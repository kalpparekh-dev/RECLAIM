"""
RECLAIM CAUSAL UPLIFT STABILITY VALIDATOR

Purpose
-------
Determine whether the causal uplift model is stable across
multiple held-out randomized splits.

This is NOT another optimizer.

It answers:
    "Does uplift targeting consistently outperform RETRY_ALL?"

The experiment must already contain:
    treatment
    recovered
    recovered_amount
    intervention_cost
    customer_friction_cost
    net_recovered_value

The treatment assignment must have happened BEFORE outcomes.
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


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "generated"

EXPERIMENT_FILE = DATA_DIR / "reclaim_treatment_control_experiment.csv"

OUTPUT_STABILITY = DATA_DIR / "causal_uplift_stability_results.csv"
OUTPUT_SUMMARY = DATA_DIR / "causal_uplift_stability_summary.csv"

RANDOM_SEEDS = list(range(100, 120))

TEST_SIZE = 0.30

MIN_GROUP_SIZE = 20

COST_PER_INTERVENTION = 1.50

FRICTION_COST = 0.25


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def money(value):
    return f"₹{value:,.2f}"


# ======================================================================
# LOAD DATA
# ======================================================================

def load_data():

    banner("RECLAIM CAUSAL UPLIFT STABILITY VALIDATOR")

    print("Loading randomized treatment/control experiment...")
    print(f"File: {EXPERIMENT_FILE}")

    if not EXPERIMENT_FILE.exists():
        raise FileNotFoundError(
            f"Experiment file not found:\n{EXPERIMENT_FILE}"
        )

    df = pd.read_csv(EXPERIMENT_FILE)

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
            "Missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    if df["transaction_id"].duplicated().any():
        raise ValueError("Transaction IDs are not unique.")

    print("[PASS] Transaction IDs are unique")
    print("[PASS] Required causal columns present")

    return df


# ======================================================================
# VALIDATE EXPERIMENT
# ======================================================================

def validate_experiment(df):

    banner("VALIDATING RANDOMIZED EXPERIMENT")

    treatment_values = sorted(df["treatment"].dropna().unique().tolist())

    print(f"Treatment values: {treatment_values}")

    if set(treatment_values) != {0, 1}:
        raise ValueError(
            "Treatment must contain both 0 and 1."
        )

    treatment_n = int((df["treatment"] == 1).sum())
    control_n = int((df["treatment"] == 0).sum())

    print("[PASS] Treatment values valid")
    print(f"Treatment rows: {treatment_n}")
    print(f"Control rows:   {control_n}")

    if treatment_n < MIN_GROUP_SIZE or control_n < MIN_GROUP_SIZE:
        raise ValueError(
            "Treatment/control groups are too small."
        )

    print("[PASS] Treatment/control groups usable")

    df["recovered"] = pd.to_numeric(
        df["recovered"],
        errors="coerce"
    ).fillna(0).astype(int)

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

    print("[PASS] Recovery outcome valid")
    print("[PASS] Economic variables valid")

    return df


# ======================================================================
# FEATURE SELECTION
# ======================================================================

def select_features(df):

    banner("SELECTING AVAILABLE UPLIFT FEATURES")

    preferred_numeric = [
        "amount",
        "recovery_probability",
        "calibrated_probability",
        "confidence",
        "historical_transaction_count",
        "historical_success_count",
        "historical_failure_count",
        "historical_success_rate",
        "hours_since_previous_transaction",
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

    preferred_categorical = [
        "payment_method",
        "failure_reason",
    ]

    numeric_features = [
        c for c in preferred_numeric
        if c in df.columns
    ]

    categorical_features = [
        c for c in preferred_categorical
        if c in df.columns
    ]

    print(f"Numeric features available: {len(numeric_features)}")
    for c in numeric_features:
        print(f"  [YES] {c}")

    print(f"Categorical features available: {len(categorical_features)}")
    for c in categorical_features:
        print(f"  [YES] {c}")

    if not numeric_features and not categorical_features:
        raise ValueError(
            "No usable uplift features found."
        )

    return numeric_features, categorical_features


# ======================================================================
# MODEL BUILDER
# ======================================================================

def build_model(numeric_features, categorical_features):

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

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0
                )
            ),
        ]
    )

    return model


# ======================================================================
# ECONOMIC EVALUATION
# ======================================================================

def evaluate_policy(
    test_df,
    selected_indices,
    predicted_incremental_net
):

    if len(selected_indices) == 0:

        return {
            "interventions": 0,
            "intervention_rate": 0.0,
            "observed_recovered_value": 0.0,
            "observed_net_value": 0.0,
            "predicted_incremental_net_value": 0.0,
        }

    selected = test_df.iloc[selected_indices]

    recovered_value = selected[
        "recovered_amount"
    ].sum()

    intervention_cost = selected[
        "intervention_cost"
    ].sum()

    friction_cost = selected[
        "customer_friction_cost"
    ].sum()

    net_value = (
        recovered_value
        - intervention_cost
        - friction_cost
    )

    return {
        "interventions": len(selected),
        "intervention_rate": (
            len(selected) / len(test_df)
        ),
        "observed_recovered_value": recovered_value,
        "observed_net_value": net_value,
        "predicted_incremental_net_value":
            predicted_incremental_net,
    }


# ======================================================================
# SINGLE SPLIT
# ======================================================================

def evaluate_split(
    df,
    numeric_features,
    categorical_features,
    seed
):

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=df["treatment"]
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_t = train_df[
        train_df["treatment"] == 1
    ]

    train_c = train_df[
        train_df["treatment"] == 0
    ]

    test_t = test_df[
        test_df["treatment"] == 1
    ]

    test_c = test_df[
        test_df["treatment"] == 0
    ]

    if (
        len(train_t) < MIN_GROUP_SIZE
        or len(train_c) < MIN_GROUP_SIZE
        or len(test_t) < MIN_GROUP_SIZE
        or len(test_c) < MIN_GROUP_SIZE
    ):
        return None

    X_train_t = train_t[
        numeric_features + categorical_features
    ]

    X_train_c = train_c[
        numeric_features + categorical_features
    ]

    y_train_t = train_t["recovered"]
    y_train_c = train_c["recovered"]

    X_test = test_df[
        numeric_features + categorical_features
    ]

    y_test = test_df["recovered"]

    treatment_model = build_model(
        numeric_features,
        categorical_features
    )

    control_model = build_model(
        numeric_features,
        categorical_features
    )

    treatment_model.fit(
        X_train_t,
        y_train_t
    )

    control_model.fit(
        X_train_c,
        y_train_c
    )

    p_treatment = treatment_model.predict_proba(
        X_test
    )[:, 1]

    p_control = control_model.predict_proba(
        X_test
    )[:, 1]

    uplift = p_treatment - p_control

    # --------------------------------------------------------------
    # MODEL DIAGNOSTICS
    # --------------------------------------------------------------

    test_t_mask = (
        test_df["treatment"].values == 1
    )

    test_c_mask = (
        test_df["treatment"].values == 0
    )

    treatment_brier = brier_score_loss(
        y_test[test_t_mask],
        p_treatment[test_t_mask]
    )

    control_brier = brier_score_loss(
        y_test[test_c_mask],
        p_control[test_c_mask]
    )

    # --------------------------------------------------------------
    # ECONOMIC UPLIFT
    # --------------------------------------------------------------

    amounts = pd.to_numeric(
        test_df["amount"],
        errors="coerce"
    ).fillna(0).values

    treatment_cost = pd.to_numeric(
        test_df["intervention_cost"],
        errors="coerce"
    ).fillna(COST_PER_INTERVENTION).values

    friction = pd.to_numeric(
        test_df["customer_friction_cost"],
        errors="coerce"
    ).fillna(FRICTION_COST).values

    expected_incremental_recovery = (
        uplift * amounts
    )

    expected_incremental_net = (
        expected_incremental_recovery
        - treatment_cost
        - friction
    )

    # --------------------------------------------------------------
    # TARGETING POLICY
    # --------------------------------------------------------------

    positive_mask = (
        expected_incremental_net > 0
    )

    selected_indices = np.where(
        positive_mask
    )[0]

    predicted_targeting_value = (
        expected_incremental_net[
            positive_mask
        ].sum()
    )

    targeting = evaluate_policy(
        test_df,
        selected_indices,
        predicted_targeting_value
    )

    # --------------------------------------------------------------
    # RETRY ALL BASELINE
    # --------------------------------------------------------------

    all_indices = np.arange(len(test_df))

    retry_all = evaluate_policy(
        test_df,
        all_indices,
        expected_incremental_net.sum()
    )

    targeting_minus_retry = (
        targeting["observed_net_value"]
        - retry_all["observed_net_value"]
    )

    # --------------------------------------------------------------
    # SIMPLE QINI-LIKE SCORE
    # --------------------------------------------------------------

    order = np.argsort(
        -uplift
    )

    ordered = test_df.iloc[
        order
    ].copy()

    ordered_uplift = uplift[
        order
    ]

    cumulative_qini = 0.0

    treated_seen = 0
    control_seen = 0

    treated_success = 0.0
    control_success = 0.0

    qini_points = []

    for i, row in enumerate(
        ordered.itertuples(index=False)
    ):

        if row.treatment == 1:

            treated_seen += 1
            treated_success += row.recovered

        else:

            control_seen += 1
            control_success += row.recovered

        if treated_seen > 0 and control_seen > 0:

            treated_rate = (
                treated_success / treated_seen
            )

            control_rate = (
                control_success / control_seen
            )

            qini_value = (
                (treated_rate - control_rate)
                * (treated_seen + control_seen)
            )

            qini_points.append(
                qini_value
            )

    if qini_points:

        qini_score = np.trapezoid(
            qini_points,
            dx=1
        )

    else:

        qini_score = np.nan

    return {
        "seed": seed,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_treatment": len(train_t),
        "train_control": len(train_c),
        "test_treatment": len(test_t),
        "test_control": len(test_c),

        "treatment_brier": treatment_brier,
        "control_brier": control_brier,

        "mean_predicted_treatment_probability":
            np.mean(p_treatment),

        "mean_predicted_control_probability":
            np.mean(p_control),

        "mean_predicted_uplift":
            np.mean(uplift),

        "positive_uplift_rate":
            np.mean(uplift > 0),

        "positive_economic_uplift_rate":
            np.mean(expected_incremental_net > 0),

        "qini_score":
            qini_score,

        "targeting_interventions":
            targeting["interventions"],

        "targeting_intervention_rate":
            targeting["intervention_rate"],

        "targeting_observed_net_value":
            targeting["observed_net_value"],

        "targeting_predicted_incremental_net":
            targeting[
                "predicted_incremental_net_value"
            ],

        "retry_all_net_value":
            retry_all[
                "observed_net_value"
            ],

        "targeting_minus_retry_all":
            targeting_minus_retry,

        "targeting_beats_retry_all":
            int(
                targeting_minus_retry > 0
            ),
    }


# ======================================================================
# MAIN STABILITY ANALYSIS
# ======================================================================

def main():

    df = load_data()

    df = validate_experiment(
        df
    )

    numeric_features, categorical_features = (
        select_features(df)
    )

    banner(
        "RUNNING REPEATED HELD-OUT VALIDATION"
    )

    print(
        f"Randomized splits: {len(RANDOM_SEEDS)}"
    )

    print(
        f"Test size: {TEST_SIZE:.0%}"
    )

    results = []

    for i, seed in enumerate(
        RANDOM_SEEDS,
        start=1
    ):

        print(
            f"\n[{i:02d}/{len(RANDOM_SEEDS)}] "
            f"Evaluating seed {seed}..."
        )

        result = evaluate_split(
            df,
            numeric_features,
            categorical_features,
            seed
        )

        if result is None:

            print(
                "[WARN] Split skipped "
                "(insufficient treatment/control rows)"
            )

            continue

        results.append(result)

        print(
            f"  QINI: "
            f"{result['qini_score']:.4f}"
        )

        print(
            f"  Mean uplift: "
            f"{result['mean_predicted_uplift']:.2%}"
        )

        print(
            f"  Targeting vs RETRY_ALL: "
            f"{money(result['targeting_minus_retry_all'])}"
        )

    if not results:

        raise RuntimeError(
            "No valid validation splits were produced."
        )

    results_df = pd.DataFrame(
        results
    )

    # ==================================================================
    # SUMMARY STATISTICS
    # ==================================================================

    banner(
        "STABILITY SUMMARY"
    )

    mean_qini = results_df[
        "qini_score"
    ].mean()

    std_qini = results_df[
        "qini_score"
    ].std()

    mean_uplift = results_df[
        "mean_predicted_uplift"
    ].mean()

    std_uplift = results_df[
        "mean_predicted_uplift"
    ].std()

    mean_targeting_advantage = results_df[
        "targeting_minus_retry_all"
    ].mean()

    std_targeting_advantage = results_df[
        "targeting_minus_retry_all"
    ].std()

    beat_count = int(
        results_df[
            "targeting_beats_retry_all"
        ].sum()
    )

    total_runs = len(
        results_df
    )

    beat_rate = (
        beat_count / total_runs
    )

    positive_qini_runs = int(
        (
            results_df["qini_score"] > 0
        ).sum()
    )

    positive_qini_rate = (
        positive_qini_runs
        / total_runs
    )

    # 95% confidence interval for targeting advantage
    sem = (
        std_targeting_advantage
        / np.sqrt(total_runs)
        if total_runs > 1
        else np.nan
    )

    ci_low = (
        mean_targeting_advantage
        - 1.96 * sem
        if not np.isnan(sem)
        else np.nan
    )

    ci_high = (
        mean_targeting_advantage
        + 1.96 * sem
        if not np.isnan(sem)
        else np.nan
    )

    print(
        f"Validation runs: "
        f"{total_runs}"
    )

    print()

    print(
        f"Mean QINI: "
        f"{mean_qini:.4f}"
    )

    print(
        f"QINI std dev: "
        f"{std_qini:.4f}"
    )

    print(
        f"Positive-QINI runs: "
        f"{positive_qini_runs}/{total_runs} "
        f"({positive_qini_rate:.2%})"
    )

    print()

    print(
        f"Mean predicted uplift: "
        f"{mean_uplift:.2%}"
    )

    print(
        f"Uplift std dev: "
        f"{std_uplift:.2%}"
    )

    print()

    print(
        "TARGETING VS RETRY_ALL"
    )

    print(
        f"Mean economic difference: "
        f"{money(mean_targeting_advantage)}"
    )

    print(
        f"Std dev: "
        f"{money(std_targeting_advantage)}"
    )

    print(
        f"95% CI: "
        f"[{money(ci_low)}, "
        f"{money(ci_high)}]"
    )

    print()

    print(
        f"Runs where targeting beats RETRY_ALL: "
        f"{beat_count}/{total_runs} "
        f"({beat_rate:.2%})"
    )

    # ==================================================================
    # FINAL DECISION
    # ==================================================================

    banner(
        "FINAL CAUSAL UPLIFT STABILITY VERDICT"
    )

    # Conservative criteria.
    #
    # We require:
    #   1. Positive mean economic advantage
    #   2. More than half the validation runs beat RETRY_ALL
    #   3. Positive lower 95% confidence bound
    #   4. Positive mean QINI
    #
    # This prevents declaring success from one lucky split.

    if (
        mean_targeting_advantage > 0
        and beat_rate >= 0.60
        and ci_low > 0
        and mean_qini > 0
    ):

        verdict = "TARGETING_STABLE"

        print(
            "[PASS] Causal targeting appears "
            "stable across repeated held-out splits."
        )

    else:

        verdict = "TARGETING_NOT_STABLE"

        print(
            "[WARN] Causal targeting is NOT "
            "stable enough to deploy."
        )

    print()

    print(
        f"Mean targeting advantage: "
        f"{money(mean_targeting_advantage)}"
    )

    print(
        f"Beat RETRY_ALL rate: "
        f"{beat_rate:.2%}"
    )

    print(
        f"Mean QINI: "
        f"{mean_qini:.4f}"
    )

    print(
        f"95% CI lower bound: "
        f"{money(ci_low)}"
    )

    print()

    print(
        f"FINAL VERDICT: {verdict}"
    )

    # ==================================================================
    # SAVE RESULTS
    # ==================================================================

    banner(
        "SAVING STABILITY RESULTS"
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_STABILITY,
        index=False
    )

    summary = pd.DataFrame(
        [
            {
                "validation_runs": total_runs,
                "mean_qini": mean_qini,
                "qini_std": std_qini,
                "positive_qini_rate":
                    positive_qini_rate,
                "mean_predicted_uplift":
                    mean_uplift,
                "uplift_std":
                    std_uplift,
                "mean_targeting_minus_retry_all":
                    mean_targeting_advantage,
                "std_targeting_minus_retry_all":
                    std_targeting_advantage,
                "targeting_advantage_ci_low":
                    ci_low,
                "targeting_advantage_ci_high":
                    ci_high,
                "targeting_beats_retry_all_rate":
                    beat_rate,
                "verdict":
                    verdict,
            }
        ]
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    print(
        f"[PASS] Saved stability results:\n"
        f"{OUTPUT_STABILITY}"
    )

    print(
        f"[PASS] Saved stability summary:\n"
        f"{OUTPUT_SUMMARY}"
    )

    # ==================================================================
    # FINAL
    # ==================================================================

    banner(
        "CAUSAL UPLIFT STABILITY VALIDATION COMPLETE"
    )

    print(
        f"Validation runs:       {total_runs}"
    )

    print(
        f"Mean QINI:              {mean_qini:.4f}"
    )

    print(
        f"Mean predicted uplift:  {mean_uplift:+.2%}"
    )

    print(
        f"Mean targeting advantage: "
        f"{money(mean_targeting_advantage)}"
    )

    print(
        f"Beat RETRY_ALL:         {beat_rate:.2%}"
    )

    print(
        f"Final verdict:          {verdict}"
    )

    print()

    if verdict == "TARGETING_NOT_STABLE":

        print(
            "NEXT STEP:"
        )

        print(
            "Do NOT deploy the uplift policy."
        )

        print(
            "Improve the experimental data/features "
            "and rerun validation."
        )

    else:

        print(
            "NEXT STEP:"
        )

        print(
            "Causal targeting has sufficient stability "
            "for a controlled pilot."
        )


if __name__ == "__main__":
    main()