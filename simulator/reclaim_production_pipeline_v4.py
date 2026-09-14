from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

INPUT_FILE = Path(r"D:\RECLAIM\data\generated\reclaim_production_decisions_v3.csv")
OUTPUT_DIR = Path(r"D:\RECLAIM\data\generated")

CAPACITY = 500
ACTION_RATES = {
    "HIGH_INTENSITY": 0.10,
    "STANDARD_TARGET": 0.20,
    "LOW_INTENSITY": 0.35,
    "MONITOR": 0.35,
}

def header(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)

def money(x):
    return f"₹{x:,.2f}"

def first(df, names):
    return next((x for x in names if x in df.columns), None)

def validate_schema(df):
    required = {
        "predicted_probability_uplift",
        "selected_policy",
        "expected_incremental_recovery_value",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required V3 columns: {sorted(missing)}\n"
            f"Available: {list(df.columns)}"
        )

    print("[PASS] V3 decision schema validated")

def build_action_layer(df):
    out = df.copy()

    uplift_col = "predicted_probability_uplift"
    value_col = "expected_incremental_recovery_value"

    out["action_rank"] = (
        out[value_col].rank(method="first", ascending=False).astype(int)
    )

    target_mask = out["selected_policy"].eq("TARGET")

    # Capacity is already validated by V3. V4 assigns different operational
    # intensities only inside the TARGET population.
    target = out.loc[target_mask].copy()
    target = target.sort_values(
        [value_col, uplift_col], ascending=[False, False]
    )

    n = len(target)
    if n == 0:
        raise ValueError("No TARGET transactions found in V3 output.")

    # Use quantile-style action tiers while respecting the available target
    # capacity. This is an allocation layer, not a re-selection of the 5%.
    q1 = max(1, int(np.ceil(n * ACTION_RATES["HIGH_INTENSITY"])))
    q2 = max(q1, int(np.ceil(n * (ACTION_RATES["HIGH_INTENSITY"] +
                                  ACTION_RATES["STANDARD_TARGET"]))))
    q3 = max(q2, int(np.ceil(n * (ACTION_RATES["HIGH_INTENSITY"] +
                                  ACTION_RATES["STANDARD_TARGET"] +
                                  ACTION_RATES["LOW_INTENSITY"]))))

    actions = np.full(n, "MONITOR", dtype=object)
    actions[:q1] = "HIGH_INTENSITY"
    actions[q1:q2] = "STANDARD_TARGET"
    actions[q2:q3] = "LOW_INTENSITY"

    out["recommended_action"] = "RETRY_ALL"
    out["action_intensity"] = "NONE"
    out["action_reason"] = np.where(
        target_mask,
        "Selected by validated 5% uplift policy",
        "Outside validated target capacity"
    )

    out.loc[target.index, "recommended_action"] = actions
    out.loc[target.index, "action_intensity"] = actions

    # Transparent reason for each operational tier.
    reasons = {
        "HIGH_INTENSITY": "Top economic-value target; allocate strongest recovery action",
        "STANDARD_TARGET": "High-value target; allocate standard recovery action",
        "LOW_INTENSITY": "Targeted opportunity; allocate lower-cost recovery action",
        "MONITOR": "Targeted but lower relative value; monitor for capacity-efficient follow-up",
    }

    for action, reason in reasons.items():
        mask = out["recommended_action"].eq(action)
        out.loc[mask, "action_reason"] = reason

    # Estimated action cost is deliberately a transparent planning proxy.
    # It is NOT used to alter the validated 5% targeting policy.
    cost_map = {
        "HIGH_INTENSITY": 120.0,
        "STANDARD_TARGET": 80.0,
        "LOW_INTENSITY": 45.0,
        "MONITOR": 10.0,
        "RETRY_ALL": 25.0,
    }
    out["estimated_action_cost"] = (
        out["recommended_action"].map(cost_map).astype(float)
    )
    out["estimated_net_action_value"] = (
        pd.to_numeric(out[value_col], errors="coerce")
        - out["estimated_action_cost"]
    )

    return out

def main():
    header("RECLAIM PRODUCTION PIPELINE V4")
    print("Action Allocation / Capacity Optimization")
    print("Input:", INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print("[PASS] Rows loaded:", len(df))
    print("[PASS] Columns loaded:", len(df.columns))
    validate_schema(df)

    header("BUILDING ACTION ALLOCATION LAYER")
    out = build_action_layer(df)

    target = out["selected_policy"].eq("TARGET")
    retry = ~target

    print("[PASS] Validated target population preserved")
    print("[PASS] Action ranking generated")
    print("[PASS] Capacity-aware action allocation generated")
    print("[PASS] Action intensity generated")
    print("[PASS] Action reasons generated")
    print("[PASS] Transparent action-cost proxy generated")
    print("[PASS] Estimated net action value generated")

    header("ACTION ALLOCATION SUMMARY")
    print("Total transactions:", len(out))
    print("Validated TARGET capacity:", int(target.sum()))
    print("RETRY_ALL:", int(retry.sum()))
    print("Actual intervention rate:", f"{target.mean():.2%}")

    print("\nRecommended action distribution:")
    print(out["recommended_action"].value_counts().to_string())

    print("\nTARGET economic value by action:")
    for action in ["HIGH_INTENSITY", "STANDARD_TARGET", "LOW_INTENSITY", "MONITOR"]:
        x = out.loc[out["recommended_action"].eq(action), "expected_incremental_recovery_value"]
        if len(x):
            print(
                f"{action:18s} | rows={len(x):4d} | "
                f"mean={money(x.mean())} | total={money(x.sum())}"
            )

    target_out = out.loc[target]
    print("\nTotal TARGET expected incremental value:",
          money(target_out["expected_incremental_recovery_value"].sum()))
    print("Total estimated action cost:",
          money(target_out["estimated_action_cost"].sum()))
    print("Estimated net action value:",
          money(target_out["estimated_net_action_value"].sum()))

    # Sanity checks
    if int(target.sum()) != CAPACITY:
        raise ValueError(
            f"Validated capacity changed unexpectedly: {int(target.sum())} != {CAPACITY}"
        )
    if not out.loc[retry, "recommended_action"].eq("RETRY_ALL").all():
        raise ValueError("Non-target rows received an operational target action.")

    output = OUTPUT_DIR / "reclaim_production_decisions_v4.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    header("PRODUCTION PIPELINE V4 COMPLETE")
    print("[PASS] Output:", output)
    print("\nOperational flow:")
    print("V10.2 predictions")
    print("      -> validated 5% targeting")
    print("      -> economic decision layer V3")
    print("      -> capacity-aware action allocation")
    print("      -> action intensity")
    print("      -> estimated action cost")
    print("      -> estimated net action value")
    print("\nIMPORTANT:")
    print("V4 does not change the validated 5% targeting population.")
    print("Action costs are transparent planning proxies, not validated causal estimates.")

if __name__ == "__main__":
    main()