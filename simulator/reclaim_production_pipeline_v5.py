
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

INPUT_FILE = Path(
    r"D:\RECLAIM\data\generated\reclaim_production_decisions_v4.csv"
)
OUTPUT_DIR = Path(r"D:\RECLAIM\data\generated")

EXPECTED_TARGET_RATE = 0.05
EXPECTED_CAPACITY = 500
TOLERANCE = 1e-9


def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def money(x):
    return f"₹{x:,.2f}"


def fail(message):
    raise ValueError("[FAIL] " + message)


def require_columns(df):
    required = [
        "transaction_id",
        "customer_id",
        "predicted_treatment_probability",
        "predicted_control_probability",
        "predicted_probability_uplift",
        "selected_policy_type",
        "selected_policy_parameter",
        "selected_policy",
        "intervention_selected",
        "production_rank",
        "production_policy_type",
        "production_target_rate",
        "production_capacity",
        "production_intervention_selected",
        "production_decision",
        "production_priority",
        "expected_incremental_recovery_value",
        "economic_opportunity_score",
        "economic_value_band",
        "economic_priority",
        "economic_decision_reason",
        "economic_operator_action",
        "uplift_segment",
        "treatment_control_probability_gap",
        "action_rank",
        "recommended_action",
        "action_intensity",
        "action_reason",
        "estimated_action_cost",
        "estimated_net_action_value",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        fail(f"Missing required V4 columns: {missing}")

    print("[PASS] V4 decision schema validated")


def numeric_check(df, column):
    values = pd.to_numeric(df[column], errors="coerce")

    if values.isna().any():
        fail(f"Column contains invalid numeric values: {column}")

    return values


def audit_probability_ranges(df):
    header("PROBABILITY & UPLIFT GUARDRAILS")

    probability_columns = [
        "predicted_treatment_probability",
        "predicted_control_probability",
    ]

    for column in probability_columns:
        values = numeric_check(df, column)

        if ((values < 0) | (values > 1)).any():
            fail(f"{column} contains values outside [0,1]")

        print(f"[PASS] {column} within [0,1]")

    uplift = numeric_check(df, "predicted_probability_uplift")

    expected_gap = (
        df["predicted_treatment_probability"]
        - df["predicted_control_probability"]
    )

    mismatch = ~np.isclose(
        uplift.to_numpy(float),
        expected_gap.to_numpy(float),
        atol=1e-8,
        rtol=1e-8,
    )

    if mismatch.any():
        fail(
            "predicted_probability_uplift does not match "
            "treatment probability - control probability"
        )

    print("[PASS] Probability uplift arithmetic consistent")


def audit_policy(df):
    header("VALIDATED POLICY GUARDRAILS")

    policy_types = df["production_policy_type"].dropna().astype(str).unique()

    if len(policy_types) != 1 or policy_types[0] != "TOP_RATE":
        fail(f"Unexpected production policy: {policy_types}")

    print("[PASS] Production policy = TOP_RATE")

    target_rates = numeric_check(df, "production_target_rate")

    if not np.allclose(target_rates, EXPECTED_TARGET_RATE):
        fail(
            f"Production target rate is not consistently "
            f"{EXPECTED_TARGET_RATE:.2%}"
        )

    print("[PASS] Validated target rate = 5.00%")

    capacities = numeric_check(df, "production_capacity")

    if not np.allclose(capacities, EXPECTED_CAPACITY):
        fail(
            f"Production capacity is not consistently "
            f"{EXPECTED_CAPACITY}"
        )

    print("[PASS] Validated capacity = 500")


def audit_target_population(df):
    header("TARGET POPULATION INTEGRITY")

    target_mask = df["production_decision"].eq("TARGET")
    retry_mask = df["production_decision"].eq("RETRY_ALL")

    target_count = int(target_mask.sum())
    retry_count = int(retry_mask.sum())

    unexpected = ~(target_mask | retry_mask)

    if unexpected.any():
        fail(
            "production_decision contains values other than "
            "TARGET / RETRY_ALL"
        )

    if target_count != EXPECTED_CAPACITY:
        fail(
            f"TARGET population is {target_count}, "
            f"expected {EXPECTED_CAPACITY}"
        )

    expected_rate = target_count / len(df)

    if not np.isclose(
        expected_rate,
        EXPECTED_TARGET_RATE,
        atol=TOLERANCE,
    ):
        fail(
            f"Actual intervention rate {expected_rate:.4%} "
            f"does not equal validated 5%"
        )

    print(f"[PASS] TARGET population = {target_count}")
    print(f"[PASS] RETRY_ALL population = {retry_count}")
    print(f"[PASS] Actual intervention rate = {expected_rate:.2%}")


def audit_intervention_flags(df):
    header("INTERVENTION FLAG CONSISTENCY")

    decision_flag = df["production_decision"].eq("TARGET").astype(int)
    production_flag = numeric_check(
        df,
        "production_intervention_selected",
    ).astype(int)

    selected_flag = numeric_check(
        df,
        "intervention_selected",
    ).astype(int)

    if not np.array_equal(
        decision_flag.to_numpy(),
        production_flag.to_numpy(),
    ):
        fail(
            "production_intervention_selected does not match "
            "production_decision"
        )

    if not np.array_equal(
        decision_flag.to_numpy(),
        selected_flag.to_numpy(),
    ):
        fail(
            "intervention_selected does not match "
            "production_decision"
        )

    print("[PASS] Production intervention flags consistent")
    print("[PASS] V10.2 intervention flags preserved")


def audit_ranking(df):
    header("RANKING INTEGRITY")

    ranks = numeric_check(df, "production_rank")

    expected = np.arange(1, len(df) + 1)

    if not np.array_equal(
        np.sort(ranks.astype(int).to_numpy()),
        expected,
    ):
        fail("production_rank is not a complete 1..N ranking")

    sorted_df = df.sort_values(
        "predicted_probability_uplift",
        ascending=False,
        kind="stable",
    )

    sorted_ranks = sorted_df["production_rank"].to_numpy()

    if not np.array_equal(sorted_ranks, expected):
        fail(
            "production_rank does not correspond to descending "
            "predicted uplift"
        )

    print("[PASS] Production ranking is complete")
    print("[PASS] Production ranking follows predicted uplift")


def audit_economic_layer(df):
    header("ECONOMIC LAYER GUARDRAILS")

    recovery_value = numeric_check(
        df,
        "expected_incremental_recovery_value",
    )

    opportunity = numeric_check(
        df,
        "economic_opportunity_score",
    )

    action_cost = numeric_check(
        df,
        "estimated_action_cost",
    )

    net_value = numeric_check(
        df,
        "estimated_net_action_value",
    )

    expected_opportunity_score = (
    df["predicted_probability_uplift"] * 100.0
)

if not np.allclose(
    df["economic_opportunity_score"].to_numpy(dtype=float),
    expected_opportunity_score.to_numpy(dtype=float),
    rtol=1e-6,
    atol=1e-6,
):
    fail(
        "economic_opportunity_score is inconsistent with "
        "the V3 uplift-derived calculation"
    )
    print("[PASS] Economic opportunity arithmetic consistent")

    expected_net = recovery_value - action_cost

    if not np.allclose(
        net_value.to_numpy(float),
        expected_net.to_numpy(float),
        atol=1e-6,
        rtol=1e-6,
    ):
        fail(
            "estimated_net_action_value does not equal "
            "expected_incremental_recovery_value - estimated_action_cost"
        )

    print("[PASS] Net action value arithmetic consistent")

    target_values = recovery_value[
        df["production_decision"].eq("TARGET")
    ]

    if target_values.empty:
        fail("No TARGET rows available for economic audit")

    print(
        "[PASS] TARGET economic values present:",
        len(target_values),
    )


def audit_action_layer(df):
    header("ACTION ALLOCATION GUARDRAILS")

    target = df["production_decision"].eq("TARGET")

    retry = df["production_decision"].eq("RETRY_ALL")

    retry_actions = df.loc[retry, "recommended_action"].astype(str)

    if not retry_actions.eq("RETRY_ALL").all():
        fail(
            "RETRY_ALL rows contain unexpected recommended actions"
        )

    print("[PASS] RETRY_ALL action assignment consistent")

    target_actions = df.loc[
        target,
        "recommended_action",
    ].astype(str)

    allowed = {
        "MONITOR",
        "LOW_INTENSITY",
        "STANDARD_TARGET",
        "HIGH_INTENSITY",
    }

    invalid = ~target_actions.isin(allowed)

    if invalid.any():
        fail(
            "TARGET rows contain invalid recommended_action values"
        )

    print("[PASS] TARGET action allocation uses approved action set")

    costs = numeric_check(df, "estimated_action_cost")

    if (costs < 0).any():
        fail("Negative action costs detected")

    print("[PASS] Action costs are non-negative")


def audit_duplicates(df):
    header("IDENTITY & DATA INTEGRITY")

    if df["transaction_id"].isna().any():
        fail("Missing transaction_id values")

    duplicate_transactions = df["transaction_id"].duplicated().sum()

    if duplicate_transactions:
        fail(
            f"Duplicate transaction_id values detected: "
            f"{duplicate_transactions}"
        )

    if df["customer_id"].isna().any():
        fail("Missing customer_id values")

    print("[PASS] Transaction IDs are complete and unique")
    print("[PASS] Customer IDs are complete")


def audit_extremes(df):
    header("EXTREME-VALUE AUDIT")

    uplift = numeric_check(
        df,
        "predicted_probability_uplift",
    )

    recovery_value = numeric_check(
        df,
        "expected_incremental_recovery_value",
    )

    if not np.isfinite(uplift).all():
        fail("Non-finite uplift values detected")

    if not np.isfinite(recovery_value).all():
        fail(
            "Non-finite expected incremental recovery values detected"
        )

    print("[PASS] Uplift values are finite")
    print("[PASS] Economic values are finite")

    print(
        "Uplift range:",
        f"{uplift.min():.4f}",
        "to",
        f"{uplift.max():.4f}",
    )

    print(
        "Economic value range:",
        money(recovery_value.min()),
        "to",
        money(recovery_value.max()),
    )


def build_audit_report(df):
    target = df["production_decision"].eq("TARGET")

    retry = df["production_decision"].eq("RETRY_ALL")

    uplift = numeric_check(
        df,
        "predicted_probability_uplift",
    )

    recovery_value = numeric_check(
        df,
        "expected_incremental_recovery_value",
    )

    action_cost = numeric_check(
        df,
        "estimated_action_cost",
    )

    net_value = numeric_check(
        df,
        "estimated_net_action_value",
    )

    target_value = recovery_value[target]
    target_cost = action_cost[target]
    target_net = net_value[target]

    report = pd.DataFrame(
        [
            {
                "pipeline_version": "V5",
                "input_version": "V4",
                "total_transactions": len(df),
                "target_transactions": int(target.sum()),
                "retry_all_transactions": int(retry.sum()),
                "validated_target_rate": EXPECTED_TARGET_RATE,
                "actual_target_rate": float(target.mean()),
                "validated_capacity": EXPECTED_CAPACITY,
                "mean_predicted_uplift": float(uplift.mean()),
                "target_mean_predicted_uplift": float(
                    uplift[target].mean()
                ),
                "retry_all_mean_predicted_uplift": float(
                    uplift[retry].mean()
                ),
                "target_expected_incremental_value": float(
                    target_value.mean()
                ),
                "total_target_expected_incremental_value": float(
                    target_value.sum()
                ),
                "total_target_action_cost": float(
                    target_cost.sum()
                ),
                "total_target_estimated_net_value": float(
                    target_net.sum()
                ),
                "max_predicted_uplift": float(uplift.max()),
                "min_predicted_uplift": float(uplift.min()),
                "audit_status": "PASS",
                "production_gate": "GO",
            }
        ]
    )

    return report


def main():
    header("RECLAIM PRODUCTION PIPELINE V5")
    print("Production Audit + Guardrails")
    print("Input:", INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print("[PASS] Rows loaded:", len(df))
    print("[PASS] Columns loaded:", len(df.columns))

    require_columns(df)

    audit_duplicates(df)
    audit_probability_ranges(df)
    audit_policy(df)
    audit_target_population(df)
    audit_intervention_flags(df)
    audit_ranking(df)
    audit_economic_layer(df)
    audit_action_layer(df)
    audit_extremes(df)

    header("PRODUCTION AUDIT SUMMARY")

    target = df["production_decision"].eq("TARGET")

    uplift = df["predicted_probability_uplift"].astype(float)
    recovery_value = df[
        "expected_incremental_recovery_value"
    ].astype(float)
    action_cost = df["estimated_action_cost"].astype(float)
    net_value = df["estimated_net_action_value"].astype(float)

    print("Total transactions:", len(df))
    print("TARGET transactions:", int(target.sum()))
    print(
        "RETRY_ALL transactions:",
        int((~target).sum()),
    )
    print(
        "Actual intervention rate:",
        f"{target.mean():.2%}",
    )

    print(
        "Mean predicted uplift:",
        f"{uplift.mean():.2%}",
    )

    print(
        "TARGET mean predicted uplift:",
        f"{uplift[target].mean():.2%}",
    )

    print(
        "RETRY_ALL mean predicted uplift:",
        f"{uplift[~target].mean():.2%}",
    )

    print(
        "TARGET expected incremental value:",
        money(recovery_value[target].mean()),
    )

    print(
        "TARGET action cost:",
        money(action_cost[target].sum()),
    )

    print(
        "TARGET estimated net value:",
        money(net_value[target].sum()),
    )

    audit_report = build_audit_report(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audit_path = (
        OUTPUT_DIR
        / "reclaim_production_audit_v5.csv"
    )

    output_path = (
        OUTPUT_DIR
        / "reclaim_production_decisions_v5.csv"
    )

    # V5 is an audit layer.
    # Preserve the V4 decision data exactly and append audit metadata.
    out = df.copy()

    out["v5_audit_status"] = "PASS"
    out["v5_production_gate"] = "GO"
    out["v5_validated_target_rate"] = EXPECTED_TARGET_RATE
    out["v5_validated_capacity"] = EXPECTED_CAPACITY

    audit_report.to_csv(
        audit_path,
        index=False,
    )

    out.to_csv(
        output_path,
        index=False,
    )

    header("PRODUCTION PIPELINE V5 COMPLETE")

    print("[PASS] All production guardrails passed")
    print("[PASS] V4 decisions preserved")
    print("[PASS] Validated 5% targeting preserved")
    print("[PASS] Capacity = 500 preserved")
    print("[PASS] Economic arithmetic validated")
    print("[PASS] Action allocation validated")
    print("[PASS] Production gate: GO")

    print("\nOutputs:")
    print("[PASS] Audit:", audit_path)
    print("[PASS] Decisions:", output_path)

    print("\nOperational flow:")
    print("V10.2 causal uplift")
    print("      -> production V1")
    print("      -> decision intelligence V2")
    print("      -> economic decision V3")
    print("      -> action allocation V4")
    print("      -> production audit + guardrails V5")
    print("      -> GO / HOLD")


if __name__ == "__main__":
    main()

