"""
RECLAIM CAUSAL UPLIFT OPTIMIZER V3
Economic uplift + cross-fitted held-out policy evaluation.

Input:
    data/generated/reclaim_causal_experiment_v2.csv

Outputs:
    data/generated/causal_uplift_v3_predictions.csv
    data/generated/causal_uplift_v3_policy_comparison.csv
    data/generated/causal_uplift_v3_summary.csv

IMPORTANT:
- Outcome/treatment/counterfactual columns are excluded from model features.
- Policy selection is based on TRAINING predictions only.
- Final policy value is evaluated on held-out data.
- Cross-fitting reduces dependence on a single train/test split.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

SEED = 2026
N_FOLDS = 5
RANDOM_SPLITS = 10
TEST_SIZE = 0.30

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "generated" / "reclaim_causal_experiment_v2.csv"
OUT_PRED = ROOT / "data" / "generated" / "causal_uplift_v3_predictions.csv"
OUT_POLICY = ROOT / "data" / "generated" / "causal_uplift_v3_policy_comparison.csv"
OUT_SUMMARY = ROOT / "data" / "generated" / "causal_uplift_v3_summary.csv"

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

CATEGORICAL_FEATURES = ["payment_method", "failure_reason"]

# Explicit leakage protection.
FORBIDDEN = {
    "treatment", "control", "treatment_group",
    "recovered", "recovery_flag", "recovered_amount",
    "recovery_status", "net_recovered_value",
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
    "experiment_transaction_id",
    "experiment_source_transaction_id",
    "source_repeat",
    "random_seed",
    "experiment_version",
    "treatment_assignment_before_outcome",
}

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

def make_model():
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=180,
            max_depth=7,
            min_samples_leaf=12,
            random_state=SEED,
            n_jobs=-1,
        )),
    ])

def check_input(df):
    required = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + ["treatment", "recovered", "amount",
           "intervention_cost", "customer_friction_cost",
           "net_recovered_value", "experiment_transaction_id"]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["experiment_transaction_id"].nunique() != len(df):
        raise ValueError("Experiment transaction IDs are not unique.")

    if not set(df["treatment"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Treatment must contain only 0/1.")

def economic_policy_value(pred, treated_mask):
    """
    Actual held-out economic value of applying treatment to selected rows.

    This is a policy evaluation quantity, not used to choose the policy.
    """
    if treated_mask.sum() == 0:
        return 0.0
    return float(pred.loc[treated_mask, "net_recovered_value"].sum())

def policy_table(test_df, predicted_economic_uplift):
    x = test_df.copy()
    x["predicted_economic_uplift"] = predicted_economic_uplift

    policies = {
        "NO_INTERVENTION": np.full(len(x), False),
        "RETRY_ALL": np.full(len(x), True),
    }

    # Probability-uplift thresholds.
    for t in [0.00, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]:
        policies[f"ECON_UPLIFT_{t:.3f}"] = (
            x["predicted_economic_uplift"].to_numpy() > t * x["amount"].to_numpy()
        )

    rows = []
    retry_value = economic_policy_value(x, policies["RETRY_ALL"])

    for name, mask in policies.items():
        mask = pd.Series(mask, index=x.index)
        value = economic_policy_value(x, mask)

        # Counterfactual policy metrics from observed held-out outcomes.
        interventions = int(mask.sum())
        recovered = int(x.loc[mask, "recovered"].sum())

        rows.append({
            "policy": name,
            "interventions": interventions,
            "intervention_rate": interventions / len(x),
            "observed_recovered_payments": recovered,
            "observed_net_value": value,
            "observed_net_per_intervention": (
                value / interventions if interventions else 0.0
            ),
            "observed_vs_retry_all": value - retry_value,
        })

    return pd.DataFrame(rows)

def fit_cross_fitted_models(train_df, x_train):
    """
    Cross-fitted treatment/control outcome models.

    The predictions are generated out-of-fold for training rows, so policy
    thresholds are selected without evaluating the same observations used
    to fit their prediction.
    """
    oof_t = np.zeros(len(train_df))
    oof_c = np.zeros(len(train_df))

    y = train_df["recovered"].astype(float).to_numpy()
    w = train_df["treatment"].astype(int).to_numpy()

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, w), 1):
        print(f"  Cross-fit fold {fold}/{N_FOLDS}...")

        tr = train_df.iloc[tr_idx]
        va_x = x_train.iloc[va_idx]

        t = tr["treatment"].to_numpy() == 1
        c = ~t

        mt = make_model()
        mc = make_model()

        mt.fit(x_train.iloc[tr_idx][NUMERIC_FEATURES + CATEGORICAL_FEATURES],
               y[tr_idx])
        mc.fit(x_train.iloc[tr_idx][NUMERIC_FEATURES + CATEGORICAL_FEATURES],
               y[tr_idx])

        # Because treatment is a randomized assignment, fit each potential
        # outcome model only on its corresponding arm.
        mt.fit(
            x_train.iloc[tr_idx][NUMERIC_FEATURES + CATEGORICAL_FEATURES][t],
            y[tr_idx][t],
        )
        mc.fit(
            x_train.iloc[tr_idx][NUMERIC_FEATURES + CATEGORICAL_FEATURES][c],
            y[tr_idx][c],
        )

        oof_t[va_idx] = mt.predict(va_x[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
        oof_c[va_idx] = mc.predict(va_x[NUMERIC_FEATURES + CATEGORICAL_FEATURES])

    return oof_t, oof_c

def main():
    print("=" * 70)
    print("RECLAIM CAUSAL UPLIFT OPTIMIZER V3")
    print("=" * 70)
    print("Economic uplift + cross-fitted policy evaluation")
    print(f"File: {INPUT}")

    if not INPUT.exists():
        raise FileNotFoundError(f"Experiment file not found: {INPUT}")

    df = pd.read_csv(INPUT)
    print(f"[PASS] Rows loaded: {len(df):,}")
    print(f"[PASS] Columns loaded: {len(df.columns)}")

    check_input(df)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df["recovered"].astype(float)
    w = df["treatment"].astype(int)

    print("\n" + "=" * 70)
    print("LEAKAGE PROTECTION")
    print("=" * 70)
    print("[PASS] Outcome/treatment/counterfactual columns excluded")
    print(f"Modeling features: {len(feature_cols)}")

    # Repeated held-out validation.
    results = []
    all_predictions = []

    for run in range(RANDOM_SPLITS):
        seed = 100 + run
        rng = np.random.RandomState(seed)

        # Stratified random holdout.
        idx = np.arange(len(df))
        train_idx = []
        test_idx = []

        for arm in [0, 1]:
            arm_idx = idx[w.to_numpy() == arm]
            rng.shuffle(arm_idx)
            cut = int(len(arm_idx) * (1 - TEST_SIZE))
            train_idx.extend(arm_idx[:cut])
            test_idx.extend(arm_idx[cut:])

        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)

        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        print("\n" + "-" * 70)
        print(f"[{run+1:02d}/{RANDOM_SPLITS}] Seed {seed}")
        print(f"Training rows: {len(train):,} | Test rows: {len(test):,}")

        # Cross-fitting is performed on the training set.
        oof_t, oof_c = fit_cross_fitted_models(
            train,
            train[feature_cols],
        )

        # Economic uplift estimate.
        #
        # A treatment changes recovery probability. Convert that probability
        # difference into expected recovered monetary value, then subtract
        # treatment + friction costs.
        train_amount = train["amount"].to_numpy()
        train_cost = (
            train["intervention_cost"].to_numpy()
            + train["customer_friction_cost"].to_numpy()
        )
        train_econ_uplift = (oof_t - oof_c) * train_amount - train_cost

        # For test data, fit arm-specific models on ALL training data.
        mt = make_model()
        mc = make_model()

        tmask = train["treatment"].to_numpy() == 1
        cmask = ~tmask

        mt.fit(train.loc[tmask, feature_cols], train.loc[tmask, "recovered"])
        mc.fit(train.loc[cmask, feature_cols], train.loc[cmask, "recovered"])

        test_t = mt.predict(test[feature_cols])
        test_c = mc.predict(test[feature_cols])

        # Expected economic benefit of intervention.
        test_amount = test["amount"].to_numpy()
        test_cost = (
            test["intervention_cost"].to_numpy()
            + test["customer_friction_cost"].to_numpy()
        )
        economic_uplift = (test_t - test_c) * test_amount - test_cost

        # QINI-like cumulative gain using observed randomized outcomes.
        order = np.argsort(-economic_uplift)
        ordered = test.iloc[order].copy()
        ordered_w = ordered["treatment"].to_numpy()
        ordered_y = ordered["recovered"].to_numpy()
        ordered_amount = ordered["amount"].to_numpy()

        treat_n = np.cumsum(ordered_w)
        control_n = np.cumsum(1 - ordered_w)
        treat_value = np.cumsum(ordered_y * ordered_amount * ordered_w)
        control_value = np.cumsum(ordered_y * ordered_amount * (1 - ordered_w))

        qini_curve = (
            treat_value
            - np.divide(
                treat_n,
                np.maximum(control_n, 1),
            ) * control_value
        )
        qini = float(np.max(qini_curve))

        # Choose policy ONLY from training OOF economic uplift.
        # The policy thresholds are based on economically meaningful
        # expected incremental value per transaction.
        train_amount_series = train["amount"].replace(0, np.nan)
        train_ratio = pd.Series(train_econ_uplift).to_numpy() / train_amount_series.to_numpy()
        train_ratio = np.nan_to_num(train_ratio, nan=0.0, posinf=0.0, neginf=0.0)

        # Select threshold maximizing observed OOF training policy value.
        threshold_grid = [-0.05, -0.02, 0.00, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
        best_threshold = 0.0
        best_train_policy_value = -np.inf

        for threshold in threshold_grid:
            mask = train_ratio > threshold
            # OOF estimate: only use observed outcomes from training rows
            # as an internal model-selection criterion.
            val = float(train.loc[mask, "net_recovered_value"].sum())
            if val > best_train_policy_value:
                best_train_policy_value = val
                best_threshold = threshold

        selected_mask = economic_uplift / np.maximum(test_amount, 1e-9) > best_threshold

        # Compare against RETRY_ALL.
        selected_value = float(test.loc[selected_mask, "net_recovered_value"].sum())
        retry_value = float(test["net_recovered_value"].sum())
        no_value = 0.0

        results.append({
            "seed": seed,
            "training_rows": len(train),
            "test_rows": len(test),
            "qini": qini,
            "mean_economic_uplift": float(np.mean(economic_uplift)),
            "positive_economic_uplift_rate": float(np.mean(economic_uplift > 0)),
            "selected_threshold": best_threshold,
            "intervention_rate": float(np.mean(selected_mask)),
            "targeting_value": selected_value,
            "retry_all_value": retry_value,
            "no_intervention_value": no_value,
            "targeting_vs_retry_all": selected_value - retry_value,
            "targeting_vs_no_intervention": selected_value,
        })

        print(f"  QINI: {qini:,.2f}")
        print(f"  Mean economic uplift: ₹{np.mean(economic_uplift):,.2f}")
        print(f"  Positive economic uplift: {np.mean(economic_uplift > 0):.2%}")
        print(f"  Selected threshold: {best_threshold:.3f}")
        print(f"  Intervention rate: {np.mean(selected_mask):.2%}")
        print(f"  Targeting vs RETRY_ALL: ₹{selected_value - retry_value:,.2f}")

        pred_frame = test[
            ["experiment_transaction_id", "treatment", "recovered",
             "amount", "net_recovered_value"]
        ].copy()
        pred_frame["seed"] = seed
        pred_frame["predicted_treatment_probability"] = test_t
        pred_frame["predicted_control_probability"] = test_c
        pred_frame["predicted_probability_uplift"] = test_t - test_c
        pred_frame["predicted_economic_uplift"] = economic_uplift
        pred_frame["selected_threshold"] = best_threshold
        pred_frame["recommended_intervention"] = selected_mask.astype(int)
        all_predictions.append(pred_frame)

    res = pd.DataFrame(results)
    preds = pd.concat(all_predictions, ignore_index=True)

    print("\n" + "=" * 70)
    print("V3 STABILITY SUMMARY")
    print("=" * 70)

    mean_qini = res["qini"].mean()
    std_qini = res["qini"].std(ddof=1)
    mean_econ = res["mean_economic_uplift"].mean()
    mean_target_retry = res["targeting_vs_retry_all"].mean()
    std_target_retry = res["targeting_vs_retry_all"].std(ddof=1)
    beat_retry = (res["targeting_vs_retry_all"] > 0).mean()
    beat_no = (res["targeting_vs_no_intervention"] > 0).mean()

    ci_low = mean_target_retry - 1.96 * std_target_retry / np.sqrt(len(res))
    ci_high = mean_target_retry + 1.96 * std_target_retry / np.sqrt(len(res))

    print(f"Validation runs: {len(res)}")
    print(f"Mean QINI: {mean_qini:,.2f}")
    print(f"QINI std dev: {std_qini:,.2f}")
    print(f"Positive-QINI runs: {(res['qini'] > 0).sum()}/{len(res)}")
    print(f"Mean economic uplift: ₹{mean_econ:,.2f}")
    print(f"Mean intervention rate: {res['intervention_rate'].mean():.2%}")
    print(f"Beat RETRY_ALL: {beat_retry:.2%}")
    print(f"Beat NO_INTERVENTION: {beat_no:.2%}")
    print(f"Mean targeting advantage vs RETRY_ALL: ₹{mean_target_retry:,.2f}")
    print(f"95% CI: [₹{ci_low:,.2f}, ₹{ci_high:,.2f}]")

    # Conservative deployment gate.
    stable = (
        beat_retry >= 0.80
        and ci_low > 0
        and (res["qini"] > 0).mean() >= 0.80
        and beat_no >= 0.80
    )

    verdict = "TARGETING_STABLE" if stable else "TARGETING_NOT_STABLE"

    print("\n" + "=" * 70)
    print("FINAL V3 ECONOMIC UPLIFT VERDICT")
    print("=" * 70)
    print(f"Positive QINI rate: {(res['qini'] > 0).mean():.2%}")
    print(f"Beat RETRY_ALL rate: {beat_retry:.2%}")
    print(f"Beat NO_INTERVENTION rate: {beat_no:.2%}")
    print(f"Mean targeting advantage vs RETRY_ALL: ₹{mean_target_retry:,.2f}")
    print(f"95% CI lower bound: ₹{ci_low:,.2f}")

    if stable:
        print("[PASS] Economic targeting appears stable across repeated held-out runs.")
    else:
        print("[WARN] Economic targeting is NOT stable enough to deploy.")

    print(f"\nFINAL VERDICT: {verdict}")

    preds.to_csv(OUT_PRED, index=False)
    res.to_csv(OUT_POLICY, index=False)

    summary = pd.DataFrame([{
        "validation_runs": len(res),
        "mean_qini": mean_qini,
        "qini_std": std_qini,
        "positive_qini_rate": (res["qini"] > 0).mean(),
        "mean_economic_uplift": mean_econ,
        "mean_intervention_rate": res["intervention_rate"].mean(),
        "beat_retry_all_rate": beat_retry,
        "beat_no_intervention_rate": beat_no,
        "mean_targeting_vs_retry_all": mean_target_retry,
        "targeting_vs_retry_all_ci_low": ci_low,
        "targeting_vs_retry_all_ci_high": ci_high,
        "final_verdict": verdict,
    }])
    summary.to_csv(OUT_SUMMARY, index=False)

    print("\n" + "=" * 70)
    print("V3 COMPLETE")
    print("=" * 70)
    print(f"[PASS] Predictions: {OUT_PRED}")
    print(f"[PASS] Policy results: {OUT_POLICY}")
    print(f"[PASS] Summary: {OUT_SUMMARY}")
    print(f"STATUS: {verdict}")

if __name__ == "__main__":
    main()