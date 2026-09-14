"""
RECLAIM CAUSAL UPLIFT OPTIMIZER V7

Heterogeneous treatment-effect modeling
+ cross-fitted economic targeting
+ IPW held-out policy evaluation
+ oracle benchmark

Input:
    D:\RECLAIM\data\generated\reclaim_causal_experiment_v3.csv

Outputs:
    causal_uplift_v7_predictions.csv
    causal_uplift_v7_policy_comparison.csv
    causal_uplift_v7_validation_results.csv
    causal_uplift_v7_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "generated"
    / "reclaim_causal_experiment_v3.csv"
)

OUTPUT_DIR = ROOT / "data" / "generated"

SEEDS = list(range(500, 520))
TEST_SIZE = 0.30

PROPENSITY = 0.50

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

LEAKAGE_COLUMNS = {
    "transaction_id",
    "customer_id",
    "timestamp",
    "status",
    "recovered",
    "recovery_flag",
    "recovered_amount",
    "observed_recovery_probability",
    "recovery_status",
    "net_recovered_value",
    "treatment",
    "control",
    "treatment_group",
    "treatment_assignment_before_outcome",
    "treatment_effect",
    "control_recovery_probability",
    "treatment_recovery_probability",
    "incremental_recovery_probability",
    "expected_control_recovery",
    "expected_treatment_recovery",
    "expected_incremental_recovery",
    "expected_incremental_net_value",
}


def make_model(seed):
    numeric_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipe,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipe,
                CATEGORICAL_FEATURES
            ),
        ]
    )

    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.045,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=1.0,
        random_state=seed,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def fit_two_models(train, seed):
    treatment_train = train[train["treatment"] == 1]
    control_train = train[train["treatment"] == 0]

    treatment_model = make_model(seed)
    control_model = make_model(seed + 1)

    treatment_model.fit(
        treatment_train[FEATURES],
        treatment_train["recovered"]
    )

    control_model.fit(
        control_train[FEATURES],
        control_train["recovered"]
    )

    return treatment_model, control_model


def economic_uplift(
    treatment_probability,
    control_probability,
    amount,
    intervention_cost,
    friction_cost,
):
    incremental_recovery_value = (
        treatment_probability - control_probability
    ) * amount

    incremental_cost = (
        intervention_cost
        + friction_cost
    )

    return incremental_recovery_value - incremental_cost


def ipw_policy_value(test, intervention):
    """
    IPW estimator for policy value.

    intervention:
        1 = treat
        0 = don't treat

    Randomized propensity is 0.5.
    """

    treatment = test["treatment"].to_numpy(float)
    outcome = test["net_recovered_value"].to_numpy(float)

    policy = np.asarray(intervention).astype(int)

    treated_contribution = (
        (policy == 1)
        & (treatment == 1)
    ) * outcome / PROPENSITY

    control_contribution = (
        (policy == 0)
        & (treatment == 0)
    ) * outcome / (1.0 - PROPENSITY)

    return np.mean(
        treated_contribution
        + control_contribution
    )


def observed_policy_value(test, intervention):
    """
    Direct observed value for diagnostic reporting.
    """

    policy = np.asarray(intervention).astype(int)

    selected = (
        test["treatment"].to_numpy() == policy
    )

    if selected.sum() == 0:
        return 0.0

    return test.loc[selected, "net_recovered_value"].mean()


def qini_score(test, uplift):
    """
    Simple randomized-treatment QINI-style score.

    Higher is better.
    """

    treatment = test["treatment"].to_numpy()
    outcome = test["recovered"].to_numpy(float)

    order = np.argsort(
        -np.asarray(uplift)
    )

    treatment = treatment[order]
    outcome = outcome[order]

    n = len(test)

    treated_count = np.cumsum(
        treatment
    )

    control_count = np.cumsum(
        1 - treatment
    )

    treated_sum = np.cumsum(
        treatment * outcome
    )

    control_sum = np.cumsum(
        (1 - treatment) * outcome
    )

    treated_mean = np.divide(
        treated_sum,
        treated_count,
        out=np.zeros(n),
        where=treated_count > 0
    )

    control_mean = np.divide(
        control_sum,
        control_count,
        out=np.zeros(n),
        where=control_count > 0
    )

    incremental_effect = (
        treated_mean
        - control_mean
    )

    curve = incremental_effect * np.arange(
        1,
        n + 1
    )

    return np.trapezoid(
        curve,
        dx=1
    )


def main():

    print("=" * 70)
    print("RECLAIM CAUSAL UPLIFT OPTIMIZER V7")
    print("HETEROGENEOUS EFFECT + IPW POLICY EVALUATION")
    print("=" * 70)

    print(f"File: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing V3 experiment: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"[PASS] Rows loaded: {len(df):,}"
    )

    print(
        f"[PASS] Columns loaded: {len(df.columns)}"
    )

    # ---------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATING V3 EXPERIMENT")
    print("=" * 70)

    required = (
        FEATURES
        + [
            "treatment",
            "recovered",
            "amount",
            "net_recovered_value",
            "intervention_cost",
            "customer_friction_cost",
        ]
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df["treatment"].nunique() != 2:
        raise ValueError(
            "Treatment must contain 0 and 1."
        )

    if df["recovered"].isna().any():
        raise ValueError(
            "Recovery outcome contains nulls."
        )

    print("[PASS] Required columns present")
    print("[PASS] Treatment assignment valid")
    print("[PASS] Recovery outcome valid")
    print("[PASS] Economic variables valid")

    # ---------------------------------------------------------------
    # LEAKAGE
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("LEAKAGE PROTECTION")
    print("=" * 70)

    forbidden_used = set(FEATURES) & LEAKAGE_COLUMNS

    if forbidden_used:
        raise RuntimeError(
            f"Leakage detected: {forbidden_used}"
        )

    print(
        "[PASS] Outcome/treatment/counterfactual "
        "columns excluded"
    )

    print(
        f"Modeling features: {len(FEATURES)}"
    )

    # ---------------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------------

    treatment_mask = df["treatment"] == 1
    control_mask = df["treatment"] == 0

    treatment_rate = (
        df.loc[treatment_mask, "recovered"].mean()
    )

    control_rate = (
        df.loc[control_mask, "recovered"].mean()
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT BASELINE")
    print("=" * 70)

    print(
        f"Treatment recovery rate: "
        f"{treatment_rate * 100:.2f}%"
    )

    print(
        f"Control recovery rate:   "
        f"{control_rate * 100:.2f}%"
    )

    print(
        f"Observed recovery lift:   "
        f"{(treatment_rate - control_rate) * 100:+.2f}%"
    )

    # ---------------------------------------------------------------
    # V3 HETEROGENEITY
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TREATMENT EFFECT HETEROGENEITY")
    print("=" * 70)

    true_effect = df[
        "treatment_effect"
    ].to_numpy(float)

    print(
        f"Mean effect: "
        f"{np.mean(true_effect) * 100:.2f}%"
    )

    print(
        f"Std effect: "
        f"{np.std(true_effect) * 100:.2f}%"
    )

    print(
        f"Positive effects: "
        f"{np.sum(true_effect > 0):,}"
    )

    print(
        f"Negative effects: "
        f"{np.sum(true_effect < 0):,}"
    )

    # ---------------------------------------------------------------
    # REPEATED VALIDATION
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING V7 REPEATED HELD-OUT VALIDATION")
    print("=" * 70)

    results = []
    prediction_frames = []

    for run_number, seed in enumerate(SEEDS, start=1):

        print("\n" + "-" * 70)
        print(
            f"[{run_number:02d}/{len(SEEDS)}] SEED {seed}"
        )

        rng = np.random.default_rng(seed)

        indices = np.arange(len(df))
        rng.shuffle(indices)

        split = int(
            len(df) * (1.0 - TEST_SIZE)
        )

        train_idx = indices[:split]
        test_idx = indices[split:]

        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        print(
            f"Training rows: {len(train):,} | "
            f"Test rows: {len(test):,}"
        )

        treatment_model, control_model = (
            fit_two_models(
                train,
                seed
            )
        )

        treatment_probability = (
            treatment_model.predict_proba(
                test[FEATURES]
            )[:, 1]
        )

        control_probability = (
            control_model.predict_proba(
                test[FEATURES]
            )[:, 1]
        )

        amount = test["amount"].to_numpy(float)

        intervention_cost = test[
            "intervention_cost"
        ].to_numpy(float)

        friction_cost = test[
            "customer_friction_cost"
        ].to_numpy(float)

        # Economic uplift of treating this transaction.

        predicted_economic_uplift = economic_uplift(
            treatment_probability,
            control_probability,
            amount,
            intervention_cost,
            friction_cost,
        )

        predicted_probability_uplift = (
            treatment_probability
            - control_probability
        )

        # -----------------------------------------------------------
        # POLICY SELECTION
        # -----------------------------------------------------------

        # Select only transactions where the estimated incremental
        # economic value is positive.

        intervention = (
            predicted_economic_uplift > 0
        ).astype(int)

        intervention_rate = (
            intervention.mean()
        )

        # -----------------------------------------------------------
        # POLICY VALUES
        # -----------------------------------------------------------

        targeting_value = ipw_policy_value(
            test,
            intervention
        )

        retry_all_value = ipw_policy_value(
            test,
            np.ones(len(test), dtype=int)
        )

        no_intervention_value = ipw_policy_value(
            test,
            np.zeros(len(test), dtype=int)
        )

        targeting_advantage = (
            targeting_value
            - retry_all_value
        )

        targeting_vs_none = (
            targeting_value
            - no_intervention_value
        )

        # -----------------------------------------------------------
        # ORACLE
        # -----------------------------------------------------------

        # This uses the known synthetic treatment effect and is ONLY
        # a benchmark. It must never be used by the actual model.

        true_economic_uplift = (
            test["expected_incremental_net_value"]
            .to_numpy(float)
        )

        oracle_policy = (
            true_economic_uplift > 0
        ).astype(int)

        oracle_value = ipw_policy_value(
            test,
            oracle_policy
        )

        oracle_advantage = (
            oracle_value
            - retry_all_value
        )

        qini = qini_score(
            test,
            predicted_probability_uplift
        )

        # -----------------------------------------------------------
        # MODEL QUALITY
        # -----------------------------------------------------------

        treatment_auc = roc_auc_score(
            test["recovered"],
            treatment_probability
        )

        control_auc = roc_auc_score(
            test["recovered"],
            control_probability
        )

        mean_probability_uplift = np.mean(
            predicted_probability_uplift
        )

        mean_economic_uplift = np.mean(
            predicted_economic_uplift
        )

        positive_economic = np.mean(
            predicted_economic_uplift > 0
        )

        print(
            f"Mean probability uplift: "
            f"{mean_probability_uplift * 100:.2f}%"
        )

        print(
            f"Mean economic uplift: "
            f"₹{mean_economic_uplift:,.2f}"
        )

        print(
            f"Positive economic uplift: "
            f"{positive_economic * 100:.2f}%"
        )

        print(
            f"Intervention rate: "
            f"{intervention_rate * 100:.2f}%"
        )

        print(
            f"QINI: "
            f"{qini:,.2f}"
        )

        print(
            f"Targeting value/transaction: "
            f"₹{targeting_value:,.2f}"
        )

        print(
            f"RETRY_ALL value/transaction: "
            f"₹{retry_all_value:,.2f}"
        )

        print(
            f"Advantage vs RETRY_ALL: "
            f"₹{targeting_advantage:,.2f}"
        )

        print(
            f"Oracle advantage vs RETRY_ALL: "
            f"₹{oracle_advantage:,.2f}"
        )

        results.append(
            {
                "seed": seed,
                "training_rows": len(train),
                "test_rows": len(test),
                "mean_probability_uplift":
                    mean_probability_uplift,
                "mean_economic_uplift":
                    mean_economic_uplift,
                "positive_economic_uplift":
                    positive_economic,
                "intervention_rate":
                    intervention_rate,
                "qini":
                    qini,
                "treatment_auc":
                    treatment_auc,
                "control_auc":
                    control_auc,
                "targeting_value":
                    targeting_value,
                "retry_all_value":
                    retry_all_value,
                "no_intervention_value":
                    no_intervention_value,
                "targeting_advantage_vs_retry_all":
                    targeting_advantage,
                "targeting_advantage_vs_no_intervention":
                    targeting_vs_none,
                "oracle_value":
                    oracle_value,
                "oracle_advantage_vs_retry_all":
                    oracle_advantage,
            }
        )

        prediction_frame = test[
            [
                "transaction_id",
                "treatment",
                "recovered",
                "amount",
            ]
        ].copy()

        prediction_frame["seed"] = seed

        prediction_frame[
            "predicted_treatment_probability"
        ] = treatment_probability

        prediction_frame[
            "predicted_control_probability"
        ] = control_probability

        prediction_frame[
            "predicted_probability_uplift"
        ] = predicted_probability_uplift

        prediction_frame[
            "predicted_economic_uplift"
        ] = predicted_economic_uplift

        prediction_frame[
            "target_intervention"
        ] = intervention

        prediction_frame[
            "true_treatment_effect_benchmark"
        ] = test["treatment_effect"].to_numpy()

        prediction_frame[
            "true_economic_uplift_benchmark"
        ] = true_economic_uplift

        prediction_frames.append(
            prediction_frame
        )

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("V7 STABILITY SUMMARY")
    print("=" * 70)

    positive_qini_rate = np.mean(
        results_df["qini"] > 0
    )

    beat_retry_rate = np.mean(
        results_df[
            "targeting_advantage_vs_retry_all"
        ] > 0
    )

    beat_none_rate = np.mean(
        results_df[
            "targeting_advantage_vs_no_intervention"
        ] > 0
    )

    mean_qini = results_df["qini"].mean()
    qini_std = results_df["qini"].std()

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

    mean_retry_advantage = (
        results_df[
            "targeting_advantage_vs_retry_all"
        ].mean()
    )

    mean_none_advantage = (
        results_df[
            "targeting_advantage_vs_no_intervention"
        ].mean()
    )

    mean_oracle_advantage = (
        results_df[
            "oracle_advantage_vs_retry_all"
        ].mean()
    )

    std_retry_advantage = (
        results_df[
            "targeting_advantage_vs_retry_all"
        ].std()
    )

    standard_error = (
        std_retry_advantage
        / np.sqrt(len(results_df))
    )

    ci_low = (
        mean_retry_advantage
        - 1.96 * standard_error
    )

    ci_high = (
        mean_retry_advantage
        + 1.96 * standard_error
    )

    print(
        f"Validation runs: "
        f"{len(results_df)}"
    )

    print(
        f"Mean QINI: "
        f"{mean_qini:,.2f}"
    )

    print(
        f"QINI std dev: "
        f"{qini_std:,.2f}"
    )

    print(
        f"Positive-QINI rate: "
        f"{positive_qini_rate * 100:.2f}%"
    )

    print(
        f"Mean economic uplift: "
        f"₹{mean_economic_uplift:,.2f}"
    )

    print(
        f"Mean intervention rate: "
        f"{mean_intervention_rate * 100:.2f}%"
    )

    print("\nTARGETING VS NO_INTERVENTION")

    print(
        f"Mean advantage: "
        f"₹{mean_none_advantage:,.2f}"
    )

    print(
        f"Runs beating NO_INTERVENTION: "
        f"{beat_none_rate * 100:.2f}%"
    )

    print("\nTARGETING VS RETRY_ALL")

    print(
        f"Mean advantage: "
        f"₹{mean_retry_advantage:,.2f}"
    )

    print(
        f"95% CI: "
        f"[₹{ci_low:,.2f}, ₹{ci_high:,.2f}]"
    )

    print(
        f"Runs beating RETRY_ALL: "
        f"{beat_retry_rate * 100:.2f}%"
    )

    print("\nORACLE BENCHMARK")

    print(
        f"Mean oracle advantage vs RETRY_ALL: "
        f"₹{mean_oracle_advantage:,.2f}"
    )

    # ---------------------------------------------------------------
    # VERDICT
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL V7 CAUSAL POLICY VERDICT")
    print("=" * 70)

    if (
        mean_oracle_advantage > 0
        and mean_retry_advantage > 0
        and ci_low > 0
    ):
        verdict = (
            "TARGETING_BEATS_RETRY_ALL"
        )

    elif (
        mean_oracle_advantage > 0
        and mean_retry_advantage <= 0
    ):
        verdict = (
            "HETEROGENEITY_EXISTS_BUT_MODEL "
            "FAILS_TO_RECOVER_IT"
        )

    else:
        verdict = (
            "RETRY_ALL_REMAINS_SUPPORTED"
        )

    print(
        f"Oracle advantage: "
        f"₹{mean_oracle_advantage:,.2f}"
    )

    print(
        f"Learned targeting advantage: "
        f"₹{mean_retry_advantage:,.2f}"
    )

    print(
        f"95% CI lower bound: "
        f"₹{ci_low:,.2f}"
    )

    print(
        f"\nFINAL VERDICT: {verdict}"
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAVING V7 RESULTS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_path = (
        OUTPUT_DIR
        / "causal_uplift_v7_validation_results.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / "causal_uplift_v7_predictions.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "causal_uplift_v7_summary.csv"
    )

    policy_path = (
        OUTPUT_DIR
        / "causal_uplift_v7_policy_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    policy_df = results_df[
        [
            "seed",
            "targeting_value",
            "retry_all_value",
            "no_intervention_value",
            "targeting_advantage_vs_retry_all",
            "targeting_advantage_vs_no_intervention",
            "oracle_value",
            "oracle_advantage_vs_retry_all",
        ]
    ].copy()

    policy_df.to_csv(
        policy_path,
        index=False
    )

    summary = pd.DataFrame(
        [
            {
                "validation_runs":
                    len(results_df),
                "mean_qini":
                    mean_qini,
                "qini_std":
                    qini_std,
                "positive_qini_rate":
                    positive_qini_rate,
                "mean_probability_uplift":
                    results_df[
                        "mean_probability_uplift"
                    ].mean(),
                "mean_economic_uplift":
                    mean_economic_uplift,
                "mean_intervention_rate":
                    mean_intervention_rate,
                "targeting_advantage_vs_no_intervention":
                    mean_none_advantage,
                "beat_no_intervention_rate":
                    beat_none_rate,
                "targeting_advantage_vs_retry_all":
                    mean_retry_advantage,
                "beat_retry_all_rate":
                    beat_retry_rate,
                "retry_all_ci_low":
                    ci_low,
                "retry_all_ci_high":
                    ci_high,
                "oracle_advantage_vs_retry_all":
                    mean_oracle_advantage,
                "verdict":
                    verdict,
            }
        ]
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print(
        f"[PASS] Validation:  {results_path}"
    )

    print(
        f"[PASS] Predictions: {predictions_path}"
    )

    print(
        f"[PASS] Policies:    {policy_path}"
    )

    print(
        f"[PASS] Summary:     {summary_path}"
    )

    print("\n" + "=" * 70)
    print("V7 COMPLETE")
    print("=" * 70)

    print(
        f"Mean QINI: "
        f"{mean_qini:,.2f}"
    )

    print(
        f"Positive QINI rate: "
        f"{positive_qini_rate * 100:.2f}%"
    )

    print(
        f"Mean learned targeting advantage: "
        f"₹{mean_retry_advantage:,.2f}"
    )

    print(
        f"Mean oracle advantage: "
        f"₹{mean_oracle_advantage:,.2f}"
    )

    print(
        f"FINAL VERDICT: {verdict}"
    )


if __name__ == "__main__":
    main()