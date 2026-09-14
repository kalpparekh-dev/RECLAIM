from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# RECLAIM PRODUCTION PIPELINE V3
# ECONOMIC DECISION LAYER
#
# V10.2 validated policy:
#   TOP_RATE = 5%
#
# Production rule:
#   Top 5% by predicted uplift -> TARGET
#   Remaining 95% -> RETRY_ALL
#
# IMPORTANT:
# Oracle/evaluation columns are never used to make production
# decisions.
#
# V3 adds a transparent production-side economic score using
# only non-oracle fields available in the V10.2 prediction file.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "causal_uplift_v10_2_predictions.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"

VALIDATED_POLICY_TYPE = "TOP_RATE"
VALIDATED_TARGET_RATE = 0.05

# Transparent economic scoring assumptions.
# These are deliberately separated from V10.2 validation.
BASE_VALUE_RUPEES = 1500.0
VALUE_PER_UPLIFT_POINT = 10.0


def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def money(x):
    return f"₹{x:,.2f}"


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


def build_rank_and_policy(out):
    n = len(out)

    uplift = out["predicted_probability_uplift"].to_numpy(float)

    order = np.argsort(-uplift, kind="stable")

    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)

    out["production_rank"] = rank

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

    high_cutoff = max(
        1,
        int(np.ceil(target_count * 0.20))
    )

    out["production_priority"] = np.where(
        target & (rank <= high_cutoff),
        "HIGH",
        np.where(target, "TARGET", "RETRY_ALL")
    )

    return out, target, target_count, high_cutoff


def build_economic_layer(out, target, target_count, high_cutoff):
    """
    Production-side economic proxy.

    We do NOT use:
      oracle_control_value
      oracle_treatment_value
      oracle_incremental_value

    We use predicted uplift only, converting probability uplift
    into an interpretable expected incremental recovery value.

    Formula:
        expected_incremental_value =
            predicted_uplift * BASE_VALUE_RUPEES

    The additional score is a ranking aid and does not override
    the validated 5% targeting policy.
    """

    uplift = out["predicted_probability_uplift"].to_numpy(float)

    # Predicted uplift is a probability difference:
    # 0.309 means approximately 30.9 percentage points.
    expected_incremental_value = (
        np.clip(uplift, 0.0, None) * BASE_VALUE_RUPEES
    )

    out["expected_incremental_recovery_value"] = (
        expected_incremental_value
    )

    # A simple normalized economic score.
    out["economic_opportunity_score"] = (
        np.clip(uplift, 0.0, None) * 100.0
    )

    out["economic_value_band"] = pd.cut(
        out["expected_incremental_recovery_value"],
        bins=[-np.inf, 25, 50, 100, 200, 500, np.inf],
        labels=[
            "VERY_LOW",
            "LOW",
            "MODERATE",
            "HIGH",
            "VERY_HIGH",
            "EXCEPTIONAL",
        ],
        include_lowest=True,
    )

    # Economic priority is descriptive. The validated policy still
    # controls TARGET vs RETRY_ALL.
    out["economic_priority"] = np.where(
        target & (out["production_rank"] <= high_cutoff),
        "HIGH_VALUE_TARGET",
        np.where(
            target,
            "TARGET_VALUE",
            np.where(
                out["expected_incremental_recovery_value"] >= 100,
                "HIGH_POTENTIAL_RETRY_ALL",
                "STANDARD_RETRY_ALL",
            ),
        ),
    )

    out["economic_decision_reason"] = np.where(
        target,
        np.where(
            out["production_rank"] <= high_cutoff,
            "High predicted uplift and inside highest-priority targeting capacity",
            "Predicted uplift places transaction inside validated 5% targeting capacity",
        ),
        np.where(
            out["expected_incremental_recovery_value"] >= 100,
            "Economic potential is notable but transaction is outside validated targeting capacity",
            "Expected incremental value is below the production targeting priority",
        ),
    )

    # Operator-facing action remains constrained by the validated policy.
    out["economic_operator_action"] = np.where(
        target & (out["production_rank"] <= high_cutoff),
        "Prioritize intervention first",
        np.where(
            target,
            "Queue for intervention",
            "Continue standard retry process",
        ),
    )

    return out


def add_segments(out):
    # Deciles are descriptive only.
    out["uplift_segment"] = pd.qcut(
        out["predicted_probability_uplift"].rank(method="first"),
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
        ],
    )

    out["treatment_control_probability_gap"] = (
        out["predicted_treatment_probability"]
        - out["predicted_control_probability"]
    )

    return out


def print_summary(out, target):
    n = len(out)
    target_count = int(target.sum())
    retry_count = n - target_count

    target_value = (
        out.loc[target, "expected_incremental_recovery_value"].mean()
        if target_count
        else 0.0
    )

    retry_value = (
        out.loc[~target, "expected_incremental_recovery_value"].mean()
        if retry_count
        else 0.0
    )

    total_target_opportunity = (
        out.loc[target, "expected_incremental_recovery_value"].sum()
    )

    header("ECONOMIC DECISION SUMMARY")

    print("Total transactions:", n)
    print("TARGET transactions:", target_count)
    print("RETRY_ALL transactions:", retry_count)
    print("Actual intervention rate:", f"{target_count / n:.2%}")

    print(
        "Mean predicted uplift:",
        f"{out['predicted_probability_uplift'].mean():.2%}",
    )

    print(
        "TARGET mean predicted uplift:",
        f"{out.loc[target, 'predicted_probability_uplift'].mean():.2%}",
    )

    print(
        "RETRY_ALL mean predicted uplift:",
        f"{out.loc[~target, 'predicted_probability_uplift'].mean():.2%}",
    )

    print(
        "TARGET mean expected incremental value:",
        money(target_value),
    )

    print(
        "RETRY_ALL mean expected incremental value:",
        money(retry_value),
    )

    print(
        "Estimated TARGET opportunity:",
        money(total_target_opportunity),
    )

    print("\nEconomic value bands:")
    print(
        out["economic_value_band"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nEconomic priority:")
    print(
        out["economic_priority"]
        .value_counts()
        .to_string()
    )


def main():
    header("RECLAIM PRODUCTION PIPELINE V3")
    print("Economic Decision Layer")
    print("Input:", INPUT_FILE)
    print("Validated policy:", VALIDATED_POLICY_TYPE)
    print("Validated target rate:", f"{VALIDATED_TARGET_RATE:.2%}")
    print("Economic base value:", money(BASE_VALUE_RUPEES))

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nV10.2 predictions file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("[PASS] Rows loaded:", len(df))
    print("[PASS] Columns loaded:", len(df.columns))

    validate_schema(df)
    df = clean_numeric(df)

    header("BUILDING ECONOMIC DECISION LAYER")

    out = df.copy()

    out, target, target_count, high_cutoff = build_rank_and_policy(out)

    print("[PASS] Uplift ranking generated")
    print("[PASS] Validated 5% policy applied")
    print("[PASS] TARGET / RETRY_ALL decisions generated")

    out = build_economic_layer(
        out,
        target,
        target_count,
        high_cutoff,
    )

    print("[PASS] Expected incremental recovery value generated")
    print("[PASS] Economic opportunity score generated")
    print("[PASS] Economic value bands generated")
    print("[PASS] Economic priority generated")
    print("[PASS] Economic decision reasons generated")

    out = add_segments(out)

    print("[PASS] Uplift segments generated")
    print("[PASS] Treatment/control probability gap generated")

    print_summary(out, target)

    output_file = (
        OUTPUT_DIR
        / "reclaim_production_decisions_v3.csv"
    )

    out.to_csv(output_file, index=False)

    header("PRODUCTION PIPELINE V3 COMPLETE")
    print("[PASS] Output:", output_file)

    print("\nDecision flow:")
    print("V10.2 predictions")
    print("      -> validated uplift ranking")
    print("      -> validated 5% targeting")
    print("      -> TARGET / RETRY_ALL")
    print("      -> expected incremental value")
    print("      -> economic priority")
    print("      -> operator action")


if __name__ == "__main__":
    main()