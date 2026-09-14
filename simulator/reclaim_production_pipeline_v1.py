from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# RECLAIM PRODUCTION PIPELINE V1
# V10.2 -> PRODUCTION DECISIONS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "generated" / "causal_uplift_v10_2_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"

VALIDATED_POLICY_TYPE = "TOP_RATE"
VALIDATED_TARGET_RATE = 0.05


def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def validate_schema(df):
    required = [
        "predicted_treatment_probability",
        "predicted_control_probability",
        "predicted_probability_uplift",
        "selected_policy_type",
        "selected_policy_parameter",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required V10.2 prediction columns: "
            f"{missing}\nAvailable columns: {list(df.columns)}"
        )

    print("[PASS] V10.2 prediction schema validated")


def clean_numeric(df):
    numeric_cols = [
        "predicted_treatment_probability",
        "predicted_control_probability",
        "predicted_probability_uplift",
        "selected_policy_parameter",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isna().any().any():
        bad = df[numeric_cols].isna().sum()
        raise ValueError(f"Invalid numeric values found:\n{bad}")

    return df


def build_production_policy(df):
    """
    Production policy:
      1. Rank by predicted probability uplift.
      2. Target exactly the validated top 5%.
      3. Do not use oracle/evaluation columns for the decision.
    """
    n = len(df)
    k = max(1, int(np.ceil(n * VALIDATED_TARGET_RATE)))

    # Stable ranking makes ties deterministic.
    order = np.argsort(
        -df["predicted_probability_uplift"].to_numpy(),
        kind="stable"
    )

    target = np.zeros(n, dtype=bool)
    target[order[:k]] = True

    out = df.copy()

    out["production_rank"] = 0
    out.loc[out.index[order], "production_rank"] = np.arange(1, n + 1)

    out["production_policy_type"] = VALIDATED_POLICY_TYPE
    out["production_target_rate"] = VALIDATED_TARGET_RATE

    out["production_decision"] = np.where(
        target,
        "TARGET",
        "RETRY_ALL"
    )

    out["production_intervention_selected"] = target.astype(int)

    out["production_priority"] = np.select(
        [
            target & (out["production_rank"] <= max(1, int(n * 0.01))),
            target,
        ],
        [
            "HIGH",
            "TARGET",
        ],
        default="RETRY_ALL"
    )

    return out


def main():
    header("RECLAIM PRODUCTION PIPELINE V1")

    print("Input:", INPUT_FILE)
    print("Validated policy:", VALIDATED_POLICY_TYPE)
    print("Validated target rate:", f"{VALIDATED_TARGET_RATE:.2%}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nV10.2 predictions file not found:\n{INPUT_FILE}\n\n"
            "Copy causal_uplift_v10_2_predictions.csv into "
            "data/generated/ and run again."
        )

    df = pd.read_csv(INPUT_FILE)
    print("[PASS] Rows loaded:", len(df))
    print("[PASS] Columns loaded:", len(df.columns))

    validate_schema(df)
    df = clean_numeric(df)

    header("BUILDING PRODUCTION POLICY")
    decisions = build_production_policy(df)

    target_count = int(decisions["production_intervention_selected"].sum())
    target_rate = target_count / len(decisions)

    print("[PASS] Ranking generated")
    print("[PASS] Top-rate policy applied")
    print("[PASS] Oracle/evaluation columns not used for decisions")
    print("Target transactions:", target_count)
    print("Total transactions:", len(decisions))
    print("Actual intervention rate:", f"{target_rate:.2%}")
    print("Mean predicted uplift:", f"{decisions['predicted_probability_uplift'].mean():.2%}")
    print(
        "Mean predicted uplift among TARGET:",
        f"{decisions.loc[decisions.production_intervention_selected == 1, 'predicted_probability_uplift'].mean():.2%}"
    )
    print(
        "Mean predicted uplift among RETRY_ALL:",
        f"{decisions.loc[decisions.production_intervention_selected == 0, 'predicted_probability_uplift'].mean():.2%}"
    )

    output_file = OUTPUT_DIR / "reclaim_production_decisions_v1.csv"
    decisions.to_csv(output_file, index=False)

    header("PRODUCTION PIPELINE V1 COMPLETE")
    print("[PASS] Production decisions:", output_file)
    print()
    print("Decision flow:")
    print("V10.2 predictions")
    print("      -> uplift ranking")
    print("      -> validated 5% target rate")
    print("      -> TARGET / RETRY_ALL")
    print("      -> production decision CSV")


if __name__ == "__main__":
    main()