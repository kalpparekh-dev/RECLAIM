"""
RECLAIM CAUSAL UPLIFT OPTIMIZER V6

Decision-theoretic causal policy selection.

Goals:
1. Protect against leakage.
2. Estimate treatment/control probabilities using held-out data.
3. Convert probability uplift into economic uplift.
4. Compare selective targeting against:
      - NO_INTERVENTION
      - RETRY_ALL
5. Use repeated held-out validation.
6. Determine whether selective targeting has a realistic path
   to beating RETRY_ALL.
7. Detect when the experiment itself implies that RETRY_ALL
   is the optimal policy.

This version intentionally does NOT force a selective policy
when the experimental data indicate that treatment is beneficial
across essentially the entire population.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "reclaim_causal_experiment_v2.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"

N_RUNS = 20
TEST_SIZE = 0.30

SEEDS = list(range(300, 300 + N_RUNS))

# Conservative economic thresholds.
# A selective intervention must have enough predicted
# incremental economic value to justify targeting.
ECONOMIC_THRESHOLDS = [
    0,
    10,
    25,
    50,
    75,
    100,
    150,
    200,
]

# Policy coverage levels.
TOP_PERCENTS = [
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

RANDOM_STATE = 42


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
    "treatment_assignment_before_outcome",

    "recovered",
    "recovery_flag",
    "recovered_amount",
    "observed_recovery_probability",
    "recovery_status",

    "net_recovered_value",
    "customer_friction_cost",

    "control_recovery_probability",
    "treatment_recovery_probability",
    "incremental_recovery_probability",

    "expected_control_recovery",
    "expected_treatment_recovery",
    "expected_incremental_recovery",
    "expected_incremental_net_value",

    "treatment_effect",
}


# ============================================================
# UTILITY
# ============================================================

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def money(x):
    return f"₹{x:,.2f}"


def validate_dataset(df):

    required = [
        "transaction_id",
        "treatment",
        "recovered",
        "amount",
        "intervention_cost",
        "customer_friction_cost",
        "net_recovered_value",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found.")

    if not set(df["treatment"].unique()).issubset({0, 1}):
        raise ValueError("Treatment must contain only 0/1.")

    if df["recovered"].isna().any():
        raise ValueError("Recovered contains null values.")

    print("[PASS] Dataset validation complete")


# ============================================================
# MODEL
# ============================================================

def build_model():

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
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
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )

    model = LogisticRegression(
        max_iter=1500,
        C=0.5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ============================================================
# ECONOMIC EFFECT
# ============================================================

def calculate_economic_uplift(
    treatment_probability,
    control_probability,
    amount,
    intervention_cost,
    friction_cost,
):
    """
    Incremental expected net value from intervention.

    Treatment:
        probability of recovery * amount
        - intervention cost
        - friction

    Control:
        probability of recovery * amount
    """

    treatment_value = (
        treatment_probability * amount
        - intervention_cost
        - friction_cost
    )

    control_value = (
        control_probability * amount
    )

    return treatment_value - control_value


# ============================================================
# POLICY EVALUATION
# ============================================================

def evaluate_policy(
    test_df,
    score,
    policy_name,
    mask,
):

    evaluated = test_df.copy()

    evaluated["policy_selected"] = mask

    # Actual economic outcome under the policy.
    #
    # For selected rows:
    #     use actual net recovered value
    #
    # For non-selected rows:
    #     no intervention => recovered value is zero
    #
    # This preserves the held-out policy evaluation structure.
    evaluated["policy_net_value"] = np.where(
        mask,
        evaluated["net_recovered_value"],
        0.0,
    )

    total_value = evaluated["policy_net_value"].sum()

    n = len(evaluated)

    mean_value = total_value / n

    intervention_count = int(mask.sum())

    intervention_rate = (
        intervention_count / n
        if n > 0
        else 0
    )

    return {
        "policy": policy_name,
        "interventions": intervention_count,
        "intervention_rate": intervention_rate,
        "total_value": total_value,
        "value_per_transaction": mean_value,
    }


# ============================================================
# EXPERIMENT-LEVEL TREATMENT EFFECT CHECK
# ============================================================

def inspect_generated_effect(df):

    section("EXPERIMENT-LEVEL CAUSAL STRUCTURE")

    if "treatment_effect" not in df.columns:
        print("[INFO] treatment_effect column unavailable.")
        return None

    effect = pd.to_numeric(
        df["treatment_effect"],
        errors="coerce",
    )

    positive = (effect > 0).mean()
    negative = (effect < 0).mean()
    zero = (effect == 0).mean()

    print(
        f"Mean treatment effect: {effect.mean():.4%}"
    )

    print(
        f"Positive treatment-effect rows: "
        f"{(effect > 0).sum():,} "
        f"({positive:.2%})"
    )

    print(
        f"Negative treatment-effect rows: "
        f"{(effect < 0).sum():,} "
        f"({negative:.2%})"
    )

    print(
        f"Zero treatment-effect rows: "
        f"{(effect == 0).sum():,} "
        f"({zero:.2%})"
    )

    if negative == 0:
        print()
        print(
            "[IMPORTANT] No negative generated treatment effects."
        )
        print(
            "[IMPORTANT] Selective targeting has no obvious"
        )
        print(
            "            causal subgroup advantage over RETRY_ALL."
        )

    return {
        "positive_rate": positive,
        "negative_rate": negative,
        "zero_rate": zero,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings("ignore")

    section("RECLAIM CAUSAL UPLIFT OPTIMIZER V6")

    print(
        "Decision-theoretic economic targeting + "
        "repeated held-out validation"
    )

    print(f"File: {DATA_FILE}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Experiment file not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"[PASS] Rows loaded: {len(df):,}")
    print(f"[PASS] Columns loaded: {len(df.columns):,}")

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    section("VALIDATING DATA")

    validate_dataset(df)

    # --------------------------------------------------------
    # LEAKAGE
    # --------------------------------------------------------

    section("LEAKAGE PROTECTION")

    leakage_found = []

    for col in FEATURES:
        if col in LEAKAGE_COLUMNS:
            leakage_found.append(col)

    if leakage_found:
        raise ValueError(
            f"Leakage features detected: {leakage_found}"
        )

    missing_features = [
        c for c in FEATURES
        if c not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing modeling features: {missing_features}"
        )

    print(
        "[PASS] No outcome/treatment/counterfactual leakage"
    )

    print(
        f"Modeling features: {len(FEATURES)}"
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    section("EXPERIMENT BASELINE")

    treatment_df = df[df["treatment"] == 1]
    control_df = df[df["treatment"] == 0]

    treatment_rate = treatment_df["recovered"].mean()
    control_rate = control_df["recovered"].mean()

    observed_lift = treatment_rate - control_rate

    print(
        f"Treatment recovery rate: {treatment_rate:.2%}"
    )

    print(
        f"Control recovery rate:   {control_rate:.2%}"
    )

    print(
        f"Observed recovery lift:  {observed_lift:.2%}"
    )

    print(
        f"Treatment mean net:      "
        f"{money(treatment_df['net_recovered_value'].mean())}"
    )

    print(
        f"Control mean net:        "
        f"{money(control_df['net_recovered_value'].mean())}"
    )

    effect_info = inspect_generated_effect(df)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    section("RUNNING V6 REPEATED HELD-OUT VALIDATION")

    print(f"Validation runs: {N_RUNS}")
    print(f"Test size: {TEST_SIZE:.0%}")

    results = []
    predictions = []

    for run_number, seed in enumerate(SEEDS, start=1):

        print()
        print("-" * 70)
        print(
            f"[{run_number:02d}/{N_RUNS}] SEED {seed}"
        )

        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=df["treatment"],
        )

        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        print(
            f"Training rows: {len(train):,} | "
            f"Test rows: {len(test):,}"
        )

        train_treatment = train[
            train["treatment"] == 1
        ]

        train_control = train[
            train["treatment"] == 0
        ]

        # ----------------------------------------------------
        # TRAIN TREATMENT MODEL
        # ----------------------------------------------------

        treatment_model = build_model()

        treatment_model.fit(
            train_treatment[FEATURES],
            train_treatment["recovered"],
        )

        # ----------------------------------------------------
        # TRAIN CONTROL MODEL
        # ----------------------------------------------------

        control_model = build_model()

        control_model.fit(
            train_control[FEATURES],
            train_control["recovered"],
        )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        treatment_probability = treatment_model.predict_proba(
            test[FEATURES]
        )[:, 1]

        control_probability = control_model.predict_proba(
            test[FEATURES]
        )[:, 1]

        probability_uplift = (
            treatment_probability
            - control_probability
        )

        # ----------------------------------------------------
        # ECONOMIC UPLIFT
        # ----------------------------------------------------

        economic_uplift = calculate_economic_uplift(
            treatment_probability,
            control_probability,
            test["amount"].to_numpy(),
            test["intervention_cost"].to_numpy(),
            test["customer_friction_cost"].to_numpy(),
        )

        test = test.copy()

        test["predicted_treatment_probability"] = (
            treatment_probability
        )

        test["predicted_control_probability"] = (
            control_probability
        )

        test["predicted_probability_uplift"] = (
            probability_uplift
        )

        test["predicted_economic_uplift"] = (
            economic_uplift
        )

        print(
            f"Mean probability uplift: "
            f"{probability_uplift.mean():.2%}"
        )

        print(
            f"Mean economic uplift: "
            f"{money(economic_uplift.mean())}"
        )

        print(
            f"Positive economic uplift: "
            f"{(economic_uplift > 0).mean():.2%}"
        )

        # ----------------------------------------------------
        # POLICY SEARCH
        # ----------------------------------------------------

        policy_candidates = []

        # NO INTERVENTION
        no_mask = np.zeros(
            len(test),
            dtype=bool,
        )

        no_policy = evaluate_policy(
            test,
            economic_uplift,
            "NO_INTERVENTION",
            no_mask,
        )

        policy_candidates.append(no_policy)

        # RETRY ALL
        all_mask = np.ones(
            len(test),
            dtype=bool,
        )

        retry_all = evaluate_policy(
            test,
            economic_uplift,
            "RETRY_ALL",
            all_mask,
        )

        policy_candidates.append(retry_all)

        # TOP-K POLICIES
        order = np.argsort(
            -economic_uplift
        )

        for percent in TOP_PERCENTS:

            k = max(
                1,
                int(len(test) * percent / 100),
            )

            mask = np.zeros(
                len(test),
                dtype=bool,
            )

            mask[order[:k]] = True

            policy = evaluate_policy(
                test,
                economic_uplift,
                f"TOP_{percent}",
                mask,
            )

            policy_candidates.append(policy)

        # ECONOMIC THRESHOLDS
        for threshold in ECONOMIC_THRESHOLDS:

            mask = (
                economic_uplift >= threshold
            )

            policy = evaluate_policy(
                test,
                economic_uplift,
                f"ECONOMIC_{threshold}",
                mask,
            )

            policy_candidates.append(policy)

        policy_df = pd.DataFrame(
            policy_candidates
        )

        # ----------------------------------------------------
        # SELECT BEST POLICY USING MODEL VALUE
        # ----------------------------------------------------

        # Expected value of policy:
        #
        # selected rows:
        #     predicted economic uplift
        #
        # non-selected rows:
        #     0
        #
        policy_expected_values = []

        for _, row in policy_df.iterrows():

            name = row["policy"]

            if name == "NO_INTERVENTION":

                expected_value = 0.0

            elif name == "RETRY_ALL":

                expected_value = (
                    economic_uplift.sum()
                )

            elif name.startswith("TOP_"):

                percent = int(
                    name.split("_")[1]
                )

                k = max(
                    1,
                    int(len(test) * percent / 100),
                )

                expected_value = (
                    economic_uplift[order[:k]].sum()
                )

            elif name.startswith("ECONOMIC_"):

                threshold = float(
                    name.split("_")[1]
                )

                expected_value = (
                    economic_uplift[
                        economic_uplift >= threshold
                    ].sum()
                )

            else:

                expected_value = -np.inf

            policy_expected_values.append(
                expected_value
            )

        policy_df["predicted_total_uplift"] = (
            policy_expected_values
        )

        best_idx = (
            policy_df["predicted_total_uplift"]
            .idxmax()
        )

        selected = policy_df.loc[
            best_idx
        ]

        selected_policy = selected["policy"]

        selected_mask = None

        if selected_policy == "NO_INTERVENTION":

            selected_mask = no_mask

        elif selected_policy == "RETRY_ALL":

            selected_mask = all_mask

        elif selected_policy.startswith("TOP_"):

            percent = int(
                selected_policy.split("_")[1]
            )

            k = max(
                1,
                int(len(test) * percent / 100),
            )

            selected_mask = np.zeros(
                len(test),
                dtype=bool,
            )

            selected_mask[order[:k]] = True

        elif selected_policy.startswith(
            "ECONOMIC_"
        ):

            threshold = float(
                selected_policy.split("_")[1]
            )

            selected_mask = (
                economic_uplift >= threshold
            )

        selected_evaluation = evaluate_policy(
            test,
            economic_uplift,
            selected_policy,
            selected_mask,
        )

        # ----------------------------------------------------
        # QINI-LIKE RANKING SCORE
        # ----------------------------------------------------

        ranked = test.sort_values(
            "predicted_economic_uplift",
            ascending=False,
        ).copy()

        ranked["rank"] = np.arange(
            1,
            len(ranked) + 1,
        )

        # Observed incremental value proxy.
        #
        # Because only one potential outcome is observed,
        # this is evaluated through randomized treatment/control
        # groups.
        #
        treated = ranked[
            ranked["treatment"] == 1
        ]

        control = ranked[
            ranked["treatment"] == 0
        ]

        if len(treated) > 0 and len(control) > 0:

            qini = (
                treated["net_recovered_value"].sum()
                - (
                    len(treated)
                    / len(control)
                    * control[
                        "net_recovered_value"
                    ].sum()
                )
            )

        else:

            qini = 0.0

        # ----------------------------------------------------
        # RETRY ALL COMPARISON
        # ----------------------------------------------------

        selected_value = (
            selected_evaluation[
                "total_value"
            ]
        )

        retry_value = (
            retry_all[
                "total_value"
            ]
        )

        no_value = (
            no_policy[
                "total_value"
            ]
        )

        advantage_retry = (
            selected_value
            - retry_value
        )

        advantage_no = (
            selected_value
            - no_value
        )

        print(
            f"Selected policy: {selected_policy}"
        )

        print(
            f"Intervention rate: "
            f"{selected_evaluation['intervention_rate']:.2%}"
        )

        print(
            f"Mean economic uplift: "
            f"{money(economic_uplift.mean())}"
        )

        print(
            f"QINI: {qini:,.2f}"
        )

        print(
            f"Target value/transaction: "
            f"{money(selected_value / len(test))}"
        )

        print(
            f"RETRY_ALL value/transaction: "
            f"{money(retry_value / len(test))}"
        )

        print(
            f"Advantage vs RETRY_ALL: "
            f"{money(advantage_retry / len(test))}"
        )

        results.append(
            {
                "seed": seed,
                "selected_policy": selected_policy,
                "intervention_rate":
                    selected_evaluation[
                        "intervention_rate"
                    ],
                "mean_probability_uplift":
                    probability_uplift.mean(),
                "mean_economic_uplift":
                    economic_uplift.mean(),
                "positive_economic_uplift":
                    (
                        economic_uplift > 0
                    ).mean(),
                "qini": qini,
                "target_value":
                    selected_value,
                "retry_all_value":
                    retry_value,
                "advantage_vs_retry_all":
                    advantage_retry,
                "advantage_vs_no_intervention":
                    advantage_no,
            }
        )

        predictions.append(
            test[
                [
                    "transaction_id",
                    "treatment",
                    "recovered",
                    "amount",
                    "predicted_treatment_probability",
                    "predicted_control_probability",
                    "predicted_probability_uplift",
                    "predicted_economic_uplift",
                ]
            ]
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    results_df = pd.DataFrame(results)

    section("V6 STABILITY SUMMARY")

    mean_qini = results_df["qini"].mean()
    std_qini = results_df["qini"].std()

    positive_qini_rate = (
        results_df["qini"] > 0
    ).mean()

    mean_probability_uplift = (
        results_df[
            "mean_probability_uplift"
        ].mean()
    )

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
            "advantage_vs_retry_all"
        ].mean()
    )

    mean_no_advantage = (
        results_df[
            "advantage_vs_no_intervention"
        ].mean()
    )

    retry_win_rate = (
        results_df[
            "advantage_vs_retry_all"
        ] > 0
    ).mean()

    no_win_rate = (
        results_df[
            "advantage_vs_no_intervention"
        ] > 0
    ).mean()

    retry_advantage_per_transaction = (
        results_df[
            "advantage_vs_retry_all"
        ]
        / (
            len(df) * TEST_SIZE
        )
    )

    mean_advantage_per_transaction = (
        retry_advantage_per_transaction.mean()
    )

    std_advantage_per_transaction = (
        retry_advantage_per_transaction.std()
    )

    standard_error = (
        std_advantage_per_transaction
        / np.sqrt(N_RUNS)
    )

    ci_lower = (
        mean_advantage_per_transaction
        - 1.96 * standard_error
    )

    ci_upper = (
        mean_advantage_per_transaction
        + 1.96 * standard_error
    )

    print(
        f"Validation runs: {N_RUNS}"
    )

    print(
        f"Mean QINI: {mean_qini:,.2f}"
    )

    print(
        f"QINI std dev: {std_qini:,.2f}"
    )

    print(
        f"Positive-QINI runs: "
        f"{positive_qini_rate:.2%}"
    )

    print(
        f"Mean probability uplift: "
        f"{mean_probability_uplift:.2%}"
    )

    print(
        f"Mean economic uplift: "
        f"{money(mean_economic_uplift)}"
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
        f"Mean advantage/transaction: "
        f"{money(mean_no_advantage / (
            len(df) * TEST_SIZE
        ))}"
    )

    print(
        f"Runs beating NO_INTERVENTION: "
        f"{no_win_rate:.2%}"
    )

    print()
    print(
        "TARGETING VS RETRY_ALL"
    )

    print(
        f"Mean advantage/transaction: "
        f"{money(mean_advantage_per_transaction)}"
    )

    print(
        f"95% CI: "
        f"[{money(ci_lower)}, "
        f"{money(ci_upper)}]"
    )

    print(
        f"Runs beating RETRY_ALL: "
        f"{retry_win_rate:.2%}"
    )

    # ========================================================
    # POLICY STABILITY
    # ========================================================

    section("POLICY STABILITY")

    print(
        results_df[
            "selected_policy"
        ].value_counts()
    )

    # ========================================================
    # DECISION LOGIC
    # ========================================================

    section("FINAL V6 ECONOMIC POLICY VERDICT")

    negative_effect_rate = 0.0

    if effect_info is not None:
        negative_effect_rate = (
            effect_info["negative_rate"]
        )

    # Strong condition:
    #
    # If:
    #   - almost all treatment effects are positive
    #   - selective targeting never beats retry-all
    #   - confidence interval is not clearly positive
    #
    # then there is no evidence that targeting
    # should replace RETRY_ALL.
    #
    if (
        negative_effect_rate < 0.01
        and retry_win_rate == 0
        and ci_upper <= 0
    ):

        verdict = (
            "RETRY_ALL_SUPPORTED_BY_CURRENT_EXPERIMENT"
        )

        print(
            "[PASS] Current experiment supports RETRY_ALL."
        )

        print(
            "[INFO] There is insufficient evidence that"
        )

        print(
            "       selective targeting improves total value."
        )

    elif retry_win_rate >= 0.70 and ci_lower > 0:

        verdict = (
            "SELECTIVE_TARGETING_SUPPORTED"
        )

        print(
            "[PASS] Selective targeting consistently"
        )

        print(
            "       beats RETRY_ALL."
        )

    elif retry_win_rate > 0:

        verdict = (
            "TARGETING_PROMISING_BUT_UNCERTAIN"
        )

        print(
            "[WARN] Targeting shows some advantage,"
        )

        print(
            "       but evidence is not yet sufficient."
        )

    else:

        verdict = (
            "TARGETING_NOT_SUPPORTED"
        )

        print(
            "[WARN] No evidence that targeting"
        )

        print(
            "       improves over RETRY_ALL."
        )

    print()
    print(
        f"FINAL VERDICT: {verdict}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    section("SAVING V6 RESULTS")

    validation_file = (
        OUTPUT_DIR
        / "causal_uplift_v6_validation_results.csv"
    )

    predictions_file = (
        OUTPUT_DIR
        / "causal_uplift_v6_predictions.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "causal_uplift_v6_summary.csv"
    )

    results_df.to_csv(
        validation_file,
        index=False,
    )

    all_predictions = pd.concat(
        predictions,
        ignore_index=True,
    )

    all_predictions.to_csv(
        predictions_file,
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "validation_runs": N_RUNS,
                "mean_qini": mean_qini,
                "qini_std": std_qini,
                "positive_qini_rate":
                    positive_qini_rate,
                "mean_probability_uplift":
                    mean_probability_uplift,
                "mean_economic_uplift":
                    mean_economic_uplift,
                "mean_intervention_rate":
                    mean_intervention_rate,
                "retry_all_win_rate":
                    retry_win_rate,
                "no_intervention_win_rate":
                    no_win_rate,
                "mean_advantage_vs_retry_all":
                    mean_advantage_per_transaction,
                "retry_all_ci_lower":
                    ci_lower,
                "retry_all_ci_upper":
                    ci_upper,
                "negative_treatment_effect_rate":
                    negative_effect_rate,
                "verdict":
                    verdict,
            }
        ]
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print(
        f"[PASS] Validation: {validation_file}"
    )

    print(
        f"[PASS] Predictions: {predictions_file}"
    )

    print(
        f"[PASS] Summary: {summary_file}"
    )

    section("V6 COMPLETE")

    print(
        f"Mean QINI: {mean_qini:,.2f}"
    )

    print(
        f"Positive QINI rate: "
        f"{positive_qini_rate:.2%}"
    )

    print(
        f"Mean economic uplift: "
        f"{money(mean_economic_uplift)}"
    )

    print(
        f"Beat RETRY_ALL rate: "
        f"{retry_win_rate:.2%}"
    )

    print(
        f"Mean advantage vs RETRY_ALL: "
        f"{money(mean_advantage_per_transaction)}"
    )

    print(
        f"95% CI: "
        f"[{money(ci_lower)}, {money(ci_upper)}]"
    )

    print(
        f"FINAL VERDICT: {verdict}"
    )


if __name__ == "__main__":
    main()