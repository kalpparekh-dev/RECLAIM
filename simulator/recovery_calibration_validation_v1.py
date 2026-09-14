from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"
PRED = DATA / "recovery_model_predictions.csv"
OUT = DATA / "recovery_outcomes.csv"
RESULTS = DATA / "recovery_calibration_validation_v1_results.csv"
SUMMARY = DATA / "recovery_calibration_validation_v1_summary.csv"

SEEDS = range(4200, 4220)
TRAIN_RATE = 0.70
MIN_P = 0.05
MAX_P = 0.95

def banner(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)

def load_data():
    if not PRED.exists():
        raise FileNotFoundError(PRED)
    if not OUT.exists():
        raise FileNotFoundError(OUT)
    p = pd.read_csv(PRED)
    o = pd.read_csv(OUT)
    for c in ["transaction_id", "recovery_probability"]:
        if c not in p:
            raise ValueError(f"Missing prediction column: {c}")
    if "transaction_id" not in o:
        raise ValueError("Outcome file missing transaction_id")
    if "recovered" in o:
        y = pd.to_numeric(o["recovered"], errors="coerce").fillna(0).astype(int)
    elif "recovery_status" in o:
        y = o["recovery_status"].astype(str).str.upper().eq("RECOVERED").astype(int)
    else:
        raise ValueError("Outcome file must contain recovered or recovery_status")
    if p.transaction_id.duplicated().any() or o.transaction_id.duplicated().any():
        raise ValueError("Transaction IDs must be unique")
    df = p[["transaction_id", "recovery_probability"]].merge(
        pd.DataFrame({"transaction_id": o.transaction_id, "recovery_flag": y}),
        on="transaction_id", how="inner", validate="one_to_one")
    if len(df) != len(p):
        raise ValueError("Prediction population changed after merge")
    x = pd.to_numeric(df.recovery_probability, errors="coerce")
    if x.isna().any() or ((x < 0) | (x > 1)).any():
        raise ValueError("Recovery probabilities must be in [0,1]")
    if df.recovery_flag.nunique() < 2:
        raise ValueError("Both recovery classes are required")
    df.recovery_probability = x.astype(float)
    return df

def run(df, seed):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(len(idx) * TRAIN_RATE)
    tr, te = idx[:cut], idx[cut:]
    x = df.recovery_probability.to_numpy(float)
    y = df.recovery_flag.to_numpy(int)
    xt, yt, xv, yv = x[tr], y[tr], x[te], y[te]

    raw = np.clip(xv, 1e-6, 1 - 1e-6)

    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(xt.reshape(-1, 1), yt)
    lp = np.clip(lr.predict_proba(xv.reshape(-1, 1))[:, 1], MIN_P, MAX_P)

    iso = IsotonicRegression(y_min=MIN_P, y_max=MAX_P, out_of_bounds="clip")
    iso.fit(xt, yt)
    ip = np.clip(iso.predict(xv), MIN_P, MAX_P)

    rb = brier_score_loss(yv, raw)
    lb = brier_score_loss(yv, lp)
    ib = brier_score_loss(yv, ip)

    return {
        "seed": seed, "train_rows": len(tr), "test_rows": len(te),
        "raw_brier": rb, "logistic_brier": lb, "isotonic_brier": ib,
        "raw_log_loss": log_loss(yv, raw),
        "logistic_log_loss": log_loss(yv, lp),
        "isotonic_log_loss": log_loss(yv, ip),
        "raw_auc": roc_auc_score(yv, raw),
        "logistic_auc": roc_auc_score(yv, lp),
        "isotonic_auc": roc_auc_score(yv, ip),
        "logistic_better_than_raw": lb < rb,
        "isotonic_better_than_raw": ib < rb,
        "selected_calibration": "LOGISTIC" if lb <= ib else "ISOTONIC",
    }

def main():
    banner("RECLAIM RECOVERY CALIBRATION VALIDATION V1")
    print("Calibration quality gate only")
    print("V10.2 TOP_RATE 5% remains untouched")
    print("No production targeting policy is generated")

    df = load_data()
    print(f"[PASS] Population: {len(df):,}")
    print(f"[PASS] Observed recovery rate: {df.recovery_flag.mean():.2%}")

    banner("REPEATED HELD-OUT CALIBRATION VALIDATION")
    rows = []
    for i, seed in enumerate(SEEDS, 1):
        r = run(df, seed)
        rows.append(r)
        print(f"[{i:02d}/20] seed={seed} | raw={r['raw_brier']:.6f} | "
              f"logistic={r['logistic_brier']:.6f} | "
              f"isotonic={r['isotonic_brier']:.6f} | "
              f"selected={r['selected_calibration']}")

    val = pd.DataFrame(rows)
    raw = val.raw_brier.mean()
    logi = val.logistic_brier.mean()
    iso = val.isotonic_brier.mean()
    logi_rate = val.logistic_better_than_raw.mean()
    iso_rate = val.isotonic_better_than_raw.mean()
    logi_selected = (val.selected_calibration == "LOGISTIC").mean()
    improvement = (raw - logi) / raw

    banner("CALIBRATION STABILITY SUMMARY")
    print(f"Validation runs: {len(val)}")
    print(f"Mean raw Brier:      {raw:.6f}")
    print(f"Mean logistic Brier: {logi:.6f}")
    print(f"Mean isotonic Brier: {iso:.6f}")
    print(f"Logistic relative improvement: {improvement:.2%}")
    print(f"Logistic beats raw: {logi_rate:.2%}")
    print(f"Isotonic beats raw: {iso_rate:.2%}")
    print(f"Logistic selected: {logi_selected:.2%}")

    if logi < raw and logi_rate >= 0.80:
        verdict = "CALIBRATION_STABILITY_SUPPORTED"
    elif logi < raw:
        verdict = "CALIBRATION_PROMISING_BUT_REQUIRES_REVIEW"
    else:
        verdict = "CALIBRATION_NOT_SUPPORTED"

    banner("INTEGRATION GOVERNANCE")
    print("V10.2 policy: TOP_RATE 5.00%")
    print("V10.2 target population: 500 / 10,000")
    print("Calibration changes V10.2 targeting: NO")
    print("Calibration changes economic layer: NO")
    print("Oracle fields used: NO")
    print("Threshold optimizer promoted to production: NO")
    print("Production policy generated: NO")
    print(f"FINAL VERDICT: {verdict}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    val.to_csv(RESULTS, index=False)
    pd.DataFrame([{
        "version": "CALIBRATION_V1",
        "validation_runs": len(val),
        "population": len(df),
        "mean_raw_brier": raw,
        "mean_logistic_brier": logi,
        "mean_isotonic_brier": iso,
        "logistic_relative_brier_improvement": improvement,
        "logistic_better_rate": logi_rate,
        "isotonic_better_rate": iso_rate,
        "logistic_selected_rate": logi_selected,
        "calibration_verdict": verdict,
        "v10_2_policy_preserved": True,
        "v10_2_target_rate": 0.05,
        "production_policy_changed": False,
        "oracle_fields_used": False,
    }]).to_csv(SUMMARY, index=False)

    print(f"[PASS] Results: {RESULTS}")
    print(f"[PASS] Summary: {SUMMARY}")
    print("CALIBRATION VALIDATION V1 COMPLETE")

if __name__ == "__main__":
    main()
