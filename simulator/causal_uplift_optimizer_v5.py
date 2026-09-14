"""
RECLAIM CAUSAL UPLIFT OPTIMIZER V5
Economic net-value uplift + constrained targeting + IPW held-out evaluation.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "generated" / "reclaim_causal_experiment_v2.csv"
OUT_DIR = PROJECT_ROOT / "data" / "generated"

SEEDS = list(range(200, 220))  # 20 repeated validations
TEST_SIZE = 0.30
CROSSFIT_FOLDS = 5

NUMERIC_FEATURES = [
    "amount","historical_transaction_count","historical_success_count",
    "historical_failure_count","historical_success_rate",
    "hours_since_previous_transaction","previous_transaction_success",
    "transactions_last_7d","successful_payments_last_7d",
    "failed_payments_last_7d","avg_amount_last_7d","failure_rate_last_7d",
    "amount_vs_recent_average","payment_momentum","failure_streak",
    "hours_since_last_success","hours_since_last_failure","method_share",
    "recoverability_score","model_confidence","base_recovery_probability",
    "recovery_probability","calibrated_probability","confidence",
]
CATEGORICAL_FEATURES = ["payment_method", "failure_reason"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LEAKAGE = {
    "treatment","control","treatment_group","recovered","recovery_flag",
    "recovered_amount","observed_recovery_probability","recovery_status",
    "net_recovered_value","control_recovery_probability",
    "treatment_recovery_probability","incremental_recovery_probability",
    "expected_control_recovery","expected_treatment_recovery",
    "expected_incremental_recovery","expected_incremental_net_value",
    "treatment_effect","treatment_assignment_before_outcome",
    "experiment_transaction_id","experiment_source_transaction_id",
    "source_repeat","experiment_version","random_seed",
}

BUDGET_RATES = [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00]


def make_model(seed):
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
    ])
    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=8,
        min_samples_leaf=20,
        max_features=0.75,
        n_jobs=-1,
        random_state=seed,
    )
    return Pipeline([("prep", prep), ("model", model)])


def ipw_policy_value(df, policy, propensity=0.5):
    a = df["treatment"].to_numpy(dtype=int)
    y = df["net_recovered_value"].to_numpy(dtype=float)
    p = np.asarray(policy, dtype=int)
    selected = (a == p).astype(float)
    prob = np.where(p == 1, propensity, 1.0 - propensity)
    return float(np.mean(selected * y / prob))


def qini_score(df, uplift):
    x = df[["treatment","net_recovered_value"]].copy()
    x["uplift"] = np.asarray(uplift)
    x = x.sort_values("uplift", ascending=False).reset_index(drop=True)
    a = x["treatment"].to_numpy(dtype=int)
    y = x["net_recovered_value"].to_numpy(dtype=float)
    tn = np.cumsum(a)
    cn = np.cumsum(1-a)
    ts = np.cumsum(y*a)
    cs = np.cumsum(y*(1-a))
    tm = np.divide(ts, tn, out=np.zeros_like(ts), where=tn>0)
    cm = np.divide(cs, cn, out=np.zeros_like(cs), where=cn>0)
    curve = tn * (tm-cm)
    return float(np.trapezoid(np.nan_to_num(curve), dx=1.0) / len(x))


def crossfit(train, seed):
    train = train.reset_index(drop=True)
    X = train[FEATURES]
    y = train["net_recovered_value"].to_numpy(float)
    a = train["treatment"].to_numpy(int)
    pt = np.zeros(len(train))
    pc = np.zeros(len(train))
    kf = KFold(CROSSFIT_FOLDS, shuffle=True, random_state=seed)

    for fold, (fit_idx, val_idx) in enumerate(kf.split(train), 1):
        ti = fit_idx[a[fit_idx] == 1]
        ci = fit_idx[a[fit_idx] == 0]
        mt = make_model(seed + fold*31 + 1)
        mc = make_model(seed + fold*31 + 2)
        mt.fit(X.iloc[ti], y[ti])
        mc.fit(X.iloc[ci], y[ci])
        pt[val_idx] = mt.predict(X.iloc[val_idx])
        pc[val_idx] = mc.predict(X.iloc[val_idx])
    return pt, pc, pt-pc


def fit_final(train, seed):
    X = train[FEATURES]
    y = train["net_recovered_value"].to_numpy(float)
    a = train["treatment"].to_numpy(int)
    mt = make_model(seed+7001)
    mc = make_model(seed+7002)
    mt.fit(X.iloc[a==1], y[a==1])
    mc.fit(X.iloc[a==0], y[a==0])
    return mt, mc


def policy_grid(df, uplift):
    rows = []
    n = len(df)
    no = ipw_policy_value(df, np.zeros(n, dtype=int))
    retry = ipw_policy_value(df, np.ones(n, dtype=int))

    rows.append(["NO_INTERVENTION",0.0,no,no-retry])
    rows.append(["RETRY_ALL",1.0,retry,0.0])

    order = np.argsort(-uplift)
    for rate in BUDGET_RATES:
        k = max(1, int(round(rate*n)))
        p = np.zeros(n, dtype=int)
        p[order[:k]] = 1
        value = ipw_policy_value(df,p)
        rows.append([
            f"TOP_{int(rate*100)}", k/n, value, value-retry
        ])

    for threshold in [0,25,50,100,150,250,500]:
        p = (uplift >= threshold).astype(int)
        if p.sum() == 0:
            continue
        value = ipw_policy_value(df,p)
        rows.append([
            f"ECONOMIC_GE_{threshold}", p.mean(), value, value-retry
        ])

    return pd.DataFrame(rows, columns=[
        "policy","intervention_rate",
        "policy_value_per_transaction","advantage_vs_retry_per_transaction"
    ])


def main():
    print("="*70)
    print("RECLAIM CAUSAL UPLIFT OPTIMIZER V5")
    print("="*70)
    print("Economic net-value uplift + constrained targeting + IPW evaluation")
    print(f"File: {DATA_FILE}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_csv(DATA_FILE)
    print(f"[PASS] Rows loaded: {len(df):,}")
    print(f"[PASS] Columns loaded: {len(df.columns)}")

    required = FEATURES + [
        "transaction_id","treatment","recovered","net_recovered_value"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["transaction_id"].astype(str).duplicated().any():
        raise ValueError("Transaction IDs are not unique.")
    if not set(df["treatment"].unique()).issubset({0,1}):
        raise ValueError("Treatment must be 0/1.")
    if df["net_recovered_value"].isna().any():
        raise ValueError("Null net_recovered_value found.")

    print("\n"+"="*70)
    print("LEAKAGE PROTECTION")
    print("="*70)
    if any(c in LEAKAGE for c in FEATURES):
        raise ValueError("Leakage feature detected.")
    print("[PASS] No outcome/treatment/counterfactual leakage")
    print(f"Modeling features: {len(FEATURES)}")

    tr = df[df.treatment==1]
    ct = df[df.treatment==0]
    print("\n"+"="*70)
    print("EXPERIMENT BASELINE")
    print("="*70)
    print(f"Treatment recovery rate: {tr.recovered.mean():.2%}")
    print(f"Control recovery rate:   {ct.recovered.mean():.2%}")
    print(f"Observed recovery lift:  {tr.recovered.mean()-ct.recovered.mean():.2%}")
    print(f"Treatment mean net:      ₹{tr.net_recovered_value.mean():,.2f}")
    print(f"Control mean net:        ₹{ct.net_recovered_value.mean():,.2f}")

    validations, predictions, policies = [], [], []

    for run, seed in enumerate(SEEDS, 1):
        print("\n"+"-"*70)
        print(f"[{run:02d}/{len(SEEDS)}] SEED {seed}")

        train, test = train_test_split(
            df, test_size=TEST_SIZE, random_state=seed,
            stratify=df["treatment"]
        )
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        print(f"Training rows: {len(train):,} | Test rows: {len(test):,}")
        _, _, train_uplift = crossfit(train, seed)

        # Policy is selected only from cross-fitted training predictions.
        train_grid = policy_grid(train, train_uplift)
        candidates = train_grid[
            ~train_grid.policy.isin(["NO_INTERVENTION","RETRY_ALL"])
        ]
        best = candidates.sort_values(
            "policy_value_per_transaction", ascending=False
        ).iloc[0]

        model_t, model_c = fit_final(train, seed)
        pt = model_t.predict(test[FEATURES])
        pc = model_c.predict(test[FEATURES])
        econ_uplift = pt-pc

        # Translate the training-selected policy to the untouched test set.
        if str(best.policy).startswith("TOP_"):
            rate = float(best.intervention_rate)
            k = max(1, int(round(rate*len(test))))
            order = np.argsort(-econ_uplift)
            test_policy = np.zeros(len(test), dtype=int)
            test_policy[order[:k]] = 1
        else:
            threshold = float(str(best.policy).split("_")[-1])
            test_policy = (econ_uplift >= threshold).astype(int)

        grid = policy_grid(test, econ_uplift)
        no_value = ipw_policy_value(test, np.zeros(len(test), dtype=int))
        retry_value = ipw_policy_value(test, np.ones(len(test), dtype=int))
        target_value = ipw_policy_value(test, test_policy)

        advantage_retry = target_value-retry_value
        advantage_no = target_value-no_value
        qini = qini_score(test, econ_uplift)

        print(f"Cross-fitted selected policy: {best.policy}")
        print(f"Mean economic uplift: ₹{econ_uplift.mean():,.2f}")
        print(f"Positive economic uplift: {(econ_uplift>0).mean():.2%}")
        print(f"Intervention rate: {test_policy.mean():.2%}")
        print(f"QINI: {qini:,.2f}")
        print(f"Target value/transaction: ₹{target_value:,.2f}")
        print(f"RETRY_ALL value/transaction: ₹{retry_value:,.2f}")
        print(f"Advantage vs RETRY_ALL: ₹{advantage_retry:,.2f}")

        validations.append({
            "seed":seed,
            "train_rows":len(train),
            "test_rows":len(test),
            "selected_policy":best.policy,
            "intervention_rate":test_policy.mean(),
            "mean_economic_uplift":econ_uplift.mean(),
            "positive_economic_uplift_rate":(econ_uplift>0).mean(),
            "qini":qini,
            "target_value_per_transaction":target_value,
            "retry_all_value_per_transaction":retry_value,
            "no_intervention_value_per_transaction":no_value,
            "advantage_vs_retry_all_per_transaction":advantage_retry,
            "advantage_vs_no_intervention_per_transaction":advantage_no,
            "total_advantage_vs_retry_all":advantage_retry*len(test),
            "total_advantage_vs_no_intervention":advantage_no*len(test),
        })

        grid["seed"] = seed
        grid["selected_policy"] = best.policy
        policies.append(grid)

        for i in range(len(test)):
            predictions.append({
                "seed":seed,
                "transaction_id":test.iloc[i]["transaction_id"],
                "treatment":int(test.iloc[i]["treatment"]),
                "observed_net_value":float(test.iloc[i]["net_recovered_value"]),
                "predicted_treatment_net":float(pt[i]),
                "predicted_control_net":float(pc[i]),
                "predicted_economic_uplift":float(econ_uplift[i]),
                "selected_for_intervention":int(test_policy[i]),
            })

    v = pd.DataFrame(validations)
    p = pd.DataFrame(predictions)
    g = pd.concat(policies, ignore_index=True)

    mean_qini = v.qini.mean()
    qini_std = v.qini.std(ddof=1)
    positive_qini = (v.qini>0).mean()
    mean_uplift = v.mean_economic_uplift.mean()
    mean_rate = v.intervention_rate.mean()

    mean_retry = v.advantage_vs_retry_all_per_transaction.mean()
    mean_no = v.advantage_vs_no_intervention_per_transaction.mean()
    beat_retry = (v.advantage_vs_retry_all_per_transaction>0).mean()
    beat_no = (v.advantage_vs_no_intervention_per_transaction>0).mean()

    se = v.advantage_vs_retry_all_per_transaction.std(ddof=1)/np.sqrt(len(v))
    ci_low = mean_retry-1.96*se
    ci_high = mean_retry+1.96*se

    print("\n"+"="*70)
    print("V5 STABILITY SUMMARY")
    print("="*70)
    print(f"Validation runs: {len(SEEDS)}")
    print(f"Mean QINI: {mean_qini:,.2f}")
    print(f"QINI std dev: {qini_std:,.2f}")
    print(f"Positive-QINI runs: {positive_qini:.2%}")
    print(f"Mean economic uplift: ₹{mean_uplift:,.2f}")
    print(f"Mean intervention rate: {mean_rate:.2%}")
    print()
    print("TARGETING VS NO_INTERVENTION")
    print(f"Mean advantage/transaction: ₹{mean_no:,.2f}")
    print(f"Runs beating NO_INTERVENTION: {beat_no:.2%}")
    print()
    print("TARGETING VS RETRY_ALL")
    print(f"Mean advantage/transaction: ₹{mean_retry:,.2f}")
    print(f"95% CI: [₹{ci_low:,.2f}, ₹{ci_high:,.2f}]")
    print(f"Runs beating RETRY_ALL: {beat_retry:.2%}")

    print("\n"+"="*70)
    print("POLICY STABILITY")
    print("="*70)
    print(v.selected_policy.value_counts().to_string())

    # Strict rule: positive QINI, beats no-intervention, and statistically
    # positive advantage over RETRY_ALL with controlled intervention rate.
    if (
        positive_qini >= 0.90 and beat_no >= 0.90 and
        beat_retry >= 0.80 and ci_low > 0 and mean_rate <= 0.50
    ):
        verdict = "TARGETING_STABLE_FOR_FUTURE_RANDOMIZED_PILOT"
    elif positive_qini >= 0.90 and beat_no >= 0.90:
        verdict = "POSITIVE_TARGETING_BUT_RETRY_ALL_NOT_BEATEN"
    else:
        verdict = "TARGETING_NOT_STABLE"

    print("\n"+"="*70)
    print("FINAL V5 ECONOMIC UPLIFT VERDICT")
    print("="*70)
    print(f"Positive QINI rate: {positive_qini:.2%}")
    print(f"Beat NO_INTERVENTION rate: {beat_no:.2%}")
    print(f"Beat RETRY_ALL rate: {beat_retry:.2%}")
    print(f"Mean advantage vs RETRY_ALL: ₹{mean_retry:,.2f}")
    print(f"95% CI lower bound: ₹{ci_low:,.2f}")
    print(f"\nFINAL VERDICT: {verdict}")

    validation_path = OUT_DIR/"causal_uplift_v5_validation_results.csv"
    predictions_path = OUT_DIR/"causal_uplift_v5_predictions.csv"
    policy_path = OUT_DIR/"causal_uplift_v5_policy_comparison.csv"
    summary_path = OUT_DIR/"causal_uplift_v5_summary.csv"

    v.to_csv(validation_path,index=False)
    p.to_csv(predictions_path,index=False)
    g.to_csv(policy_path,index=False)

    pd.DataFrame([{
        "validation_runs":len(SEEDS),
        "mean_qini":mean_qini,
        "qini_std":qini_std,
        "positive_qini_rate":positive_qini,
        "mean_economic_uplift":mean_uplift,
        "mean_intervention_rate":mean_rate,
        "beat_no_intervention_rate":beat_no,
        "beat_retry_all_rate":beat_retry,
        "mean_advantage_vs_retry_all_per_transaction":mean_retry,
        "retry_all_ci_low":ci_low,
        "retry_all_ci_high":ci_high,
        "mean_advantage_vs_no_intervention_per_transaction":mean_no,
        "verdict":verdict,
    }]).to_csv(summary_path,index=False)

    print("\n"+"="*70)
    print("V5 COMPLETE")
    print("="*70)
    print(f"[PASS] Validation:  {validation_path}")
    print(f"[PASS] Predictions: {predictions_path}")
    print(f"[PASS] Policies:    {policy_path}")
    print(f"[PASS] Summary:     {summary_path}")
    print(f"STATUS: {verdict}")


if __name__ == "__main__":
    main()