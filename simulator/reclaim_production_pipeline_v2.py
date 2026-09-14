from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# RECLAIM PRODUCTION PIPELINE V2
# DECISION INTELLIGENCE LAYER
#
# V10.2 validated policy:
#   TOP_RATE = 5%
#
# Input:
#   causal_uplift_v10_2_predictions.csv
#
# Output:
#   reclaim_production_decisions_v2.csv
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
        raise ValueError(
            "Invalid numeric values found:\n"
            f"{df[numeric_cols].isna().sum()}"
        )

    return df


def assign_priority(rank, target_count):
    """
    Priority is based on position inside the validated top-5% capacity.

    Top 1% of all transactions -> HIGH
    Remaining targeted transactions -> TARGET
    Everything outside targeting capacity -> RETRY_ALL
    """
    high_cutoff = max(1, int(np.ceil(target_count * 0.20)))

    if rank <= high_cutoff:
        return "HIGH"
    return "TARGET"


def build_decision_intelligence(df):
    out = df.copy()
    n = len(out)

    # --------------------------------------------------------
    # 1. Rank by predicted uplift
    # --------------------------------------------------------
    uplift = out["predicted_probability_uplift"].to_numpy(float)

    order = np.argsort(-uplift, kind="stable")

    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)

    out["production_rank"] = rank

    # --------------------------------------------------------
    # 2. Apply validated V10.2 TOP_RATE policy
    # --------------------------------------------------------
    target_count = max(
        1,
        int(np.ceil(n * VALIDATED_TARGET_RATE))
    )

    target = rank <= target_count

    out["production_policy_type"] = VALIDATED_POLICY_TYPE
    out["production_target_rate"] = VALIDATED_TARGET_RATE
    out["production_capacity"] = target_count

    out["production_intervention_selected"] = target.astype(int)

    out["production_decision"] = np.where(
        target,
        "TARGET",
        "RETRY_ALL"
    )

    # --------------------------------------------------------
    # 3. Priority
    # --------------------------------------------------------
    high_cutoff = max(
        1,
        int(np.ceil(target_count * 0.20))
    )

    out["production_priority"] = np.where(
        target & (rank <= high_cutoff),
        "HIGH",
        np.where(
            target,
            "TARGET",
            "RETRY_ALL"
        )
    )

    # --------------------------------------------------------
    # 4. Decision reason
    # --------------------------------------------------------
    out["decision_reason"] = np.where(
        target,
        np.where(
            rank <= high_cutoff,
            "Top predicted uplift segment; highest targeting priority",
            "Inside validated top-5% targeting capacity"
        ),
        "Outside validated targeting capacity; retain RETRY_ALL policy"
    )

    # --------------------------------------------------------
    # 5. Uplift segment
    # --------------------------------------------------------
    out["uplift_segment"] = pd.qcut(
        out["predicted_probability_uplift"].rank(
            method="first"
        ),
        q=10,
        labels=[
            "BOTTOM_10",
            "P10_P20",
            "P20_P30",
            "P30_P40",
            "P40_P50",
            "P50_P60",
            "P60_P70",
            "P70_P80",
            "P80_P90",
            "TOP_10",
        ]
    )

    # --------------------------------------------------------
    # 6. Probability gap
    # --------------------------------------------------------
    out["treatment_control_probability_gap"] = (
        out["predicted_treatment_probability"]
        - out["predicted_control_probability"]
    )

    # --------------------------------------------------------
    # 7. Production score
    #
    # This is a ranking/interpretability score, not a new
    # causal model and not used to override the validated
    # 5% policy.
    # --------------------------------------------------------
    out["production_uplift_score"] = (
        out["predicted_probability_uplift"] * 100.0
    )

    # --------------------------------------------------------
    # 8. Compact operational reason
    # --------------------------------------------------------
    out["operator_action"] = np.where(
        target,
        "Prioritize intervention",
        "Continue standard retry process"
    )

    return out


def print_summary(out):
    n = len(out)
    target = out["production_intervention_selected"].astype(bool)

    target_count = int(target.sum())
    actual_rate = target_count / n

    header("DECISION INTELLIGENCE SUMMARY")

    print("Total transactions:", n)
    print("Target transactions:", target_count)
    print("Retry-all transactions:", n - target_count)
    print("Actual intervention rate:", f"{actual_rate:.2%}")

    print(
        "Mean predicted uplift:",
        f"{out['predicted_probability_uplift'].mean():.2%}"
    )

    if target_count:
        print(
            "TARGET mean predicted uplift:",
            f"{out.loc[target, 'predicted_probability_uplift'].mean():.2%}"
        )

    if (~target).any():
        print(
            "RETRY_ALL mean predicted uplift:",
            f"{out.loc[~target, 'predicted_probability_uplift'].mean():.2%}"
        )

    print("\nPriority distribution:")
    print(out["production_priority"].value_counts().to_string())

    print("\nUplift segment distribution:")
    print(out["uplift_segment"].value_counts().sort_index().to_string())


def main():
    header("RECLAIM PRODUCTION PIPELINE V2")
    print("Decision Intelligence Layer")
    print("Input:", INPUT_FILE)
    print("Validated policy:", VALIDATED_POLICY_TYPE)
    print("Validated target rate:", f"{VALIDATED_TARGET_RATE:.2%}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nV10.2 predictions file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("[PASS] Rows loaded:", len(df))
    print("[PASS] Columns loaded:", len(df.columns))

    validate_schema(df)
    df = clean_numeric(df)

    header("BUILDING DECISION INTELLIGENCE")
    out = build_decision_intelligence(df)

    print("[PASS] Uplift ranking generated")
    print("[PASS] Validated 5% policy applied")
    print("[PASS] Priority generated")
    print("[PASS] Decision reason generated")
    print("[PASS] Uplift segments generated")
    print("[PASS] Treatment/control probability gap generated")
    print("[PASS] Operator action generated")

    print_summary(out)

    output_file = OUTPUT_DIR / "reclaim_production_decisions_v2.csv"
    out.to_csv(output_file, index=False)

    header("PRODUCTION PIPELINE V2 COMPLETE")
    print("[PASS] Output:", output_file)

    print("\nOperational flow:")
    print("V10.2 predictions")
    print("      -> uplift ranking")
    print("      -> validated 5% capacity")
    print("      -> TARGET / RETRY_ALL")
    print("      -> priority")
    print("      -> decision reason")
    print("      -> operator action")


if __name__ == "__main__":
    main()