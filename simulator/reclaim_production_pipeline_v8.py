"""
RECLAIM PRODUCTION PIPELINE V8
Controlled Offline Evaluation:
V7 Baseline vs Calibration-Aware Action Layer

IMPORTANT GOVERNANCE RULES
--------------------------
- V10.2 TOP_RATE 5% targeting is immutable.
- Exactly 500 TARGET / 9,500 RETRY_ALL must be preserved.
- Calibration may only affect action allocation inside TARGET.
- No oracle/evaluation columns are used for decisions.
- No causal improvement is claimed.
- This is an offline comparative evaluation, NOT production promotion.

Inputs
------
data/generated/reclaim_production_decisions_v7.csv
data/generated/reclaim_recovery_calibration_integration_v2.csv
data/generated/recovery_calibration_validation_v1_summary.csv

Outputs
-------
data/generated/reclaim_production_decisions_v8.csv
data/generated/reclaim_production_v8_comparison.csv
data/generated/reclaim_production_v8_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"

V7_FILE = DATA / "reclaim_production_decisions_v7.csv"
CAL_FILE = DATA / "reclaim_recovery_calibration_integration_v2.csv"
CAL_SUMMARY = DATA / "recovery_calibration_validation_v1_summary.csv"

OUT_DECISIONS = DATA / "reclaim_production_decisions_v8.csv"
OUT_COMPARISON = DATA / "reclaim_production_v8_comparison.csv"
OUT_SUMMARY = DATA / "reclaim_production_v8_summary.csv"

EXPECTED_TOTAL = 10000
EXPECTED_TARGET = 500
EXPECTED_RETRY = 9500
EXPECTED_RATE = 0.05


def fail(message):
    raise ValueError("[FAIL] " + message)


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"{name} missing required columns: {missing}")


def numeric(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def main():
    banner("RECLAIM PRODUCTION PIPELINE V8")
    print("Controlled Offline Evaluation")
    print("V7 Baseline vs Calibration-Aware Action Layer")
    print("V10.2 TOP_RATE 5% targeting is IMMUTABLE")

    for path, label in [
        (V7_FILE, "V7 production decisions"),
        (CAL_FILE, "Calibration V2 integration"),
        (CAL_SUMMARY, "Calibration validation summary"),
    ]:
        if not path.exists():
            fail(f"Missing {label}: {path}")

    v7 = pd.read_csv(V7_FILE)
    cal = pd.read_csv(CAL_FILE)
    cal_summary = pd.read_csv(CAL_SUMMARY)

    require_columns(
        v7,
        [
            "transaction_id",
            "production_intervention_selected",
            "production_decision",
            "recommended_action",
            "action_intensity",
            "expected_incremental_recovery_value",
            "estimated_action_cost",
            "estimated_net_action_value",
        ],
        "V7",
    )

    require_columns(
        cal,
        [
            "transaction_id",
            "production_intervention_selected",
            "v2_action",
            "calibration_action_score",
            "expected_incremental_recovery_value",
            "estimated_action_cost",
            "estimated_net_action_value",
        ],
        "Calibration V2",
    )

    require_columns(
        cal_summary,
        [
            "calibration_verdict",
            "mean_raw_brier",
            "mean_isotonic_brier",
        ],
        "Calibration validation summary",
    )

    if len(v7) != EXPECTED_TOTAL:
        fail(f"V7 expected {EXPECTED_TOTAL} rows, found {len(v7)}")

    if len(cal) != EXPECTED_TOTAL:
        fail(
            f"Calibration V2 expected {EXPECTED_TOTAL} rows, "
            f"found {len(cal)}"
        )

    # ------------------------------------------------------------------
    # 1. Join and population governance
    # ------------------------------------------------------------------
    banner("VALIDATING POPULATION AND GOVERNANCE")

    if not v7["transaction_id"].is_unique:
        fail("V7 transaction_id is not unique")

    if not cal["transaction_id"].is_unique:
        fail("Calibration V2 transaction_id is not unique")

    if set(v7["transaction_id"]) != set(cal["transaction_id"]):
        fail("V7 and calibration populations do not match")

    merged = v7.merge(
        cal[
            [
                "transaction_id",
                "v2_action",
                "calibration_action_score",
            ]
        ],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    if merged["v2_action"].isna().any():
        fail("Calibration action missing after population join")

    target = merged["production_intervention_selected"].astype(int).eq(1)
    retry = ~target

    target_count = int(target.sum())
    retry_count = int(retry.sum())
    intervention_rate = target_count / len(merged)

    if target_count != EXPECTED_TARGET:
        fail(f"V10.2 TARGET count changed: {target_count}")

    if retry_count != EXPECTED_RETRY:
        fail(f"V10.2 RETRY_ALL count changed: {retry_count}")

    if not np.isclose(intervention_rate, EXPECTED_RATE):
        fail(f"Intervention rate changed: {intervention_rate:.6f}")

    if not merged.loc[retry, "v2_action"].eq("RETRY_ALL").all():
        fail("Calibration changed at least one RETRY_ALL action")

    if not merged.loc[target, "production_decision"].eq("TARGET").all():
        fail("V7 TARGET population contains invalid production decisions")

    print(f"[PASS] Population: {len(merged):,}")
    print(f"[PASS] TARGET: {target_count}")
    print(f"[PASS] RETRY_ALL: {retry_count}")
    print(f"[PASS] Intervention rate: {intervention_rate:.2%}")
    print("[PASS] V10.2 targeting population preserved")
    print("[PASS] RETRY_ALL population frozen")

    # ------------------------------------------------------------------
    # 2. Calibration governance
    # ------------------------------------------------------------------
    banner("VALIDATING CALIBRATION GOVERNANCE")

    verdict = str(cal_summary.iloc[0]["calibration_verdict"])
    raw_brier = float(cal_summary.iloc[0]["mean_raw_brier"])
    iso_brier = float(cal_summary.iloc[0]["mean_isotonic_brier"])

    if verdict != "CALIBRATION_STABILITY_SUPPORTED":
        fail(f"Calibration stability gate failed: {verdict}")

    if not iso_brier < raw_brier:
        fail("Calibration does not improve mean Brier score")

    print(f"[PASS] Calibration verdict: {verdict}")
    print(f"[PASS] Raw Brier: {raw_brier:.6f}")
    print(f"[PASS] Isotonic Brier: {iso_brier:.6f}")

    # ------------------------------------------------------------------
    # 3. Baseline metrics
    # ------------------------------------------------------------------
    banner("CALCULATING V7 BASELINE")

    v7_value = numeric(merged, "expected_incremental_recovery_value").fillna(0)
    v7_cost = numeric(merged, "estimated_action_cost").fillna(0)
    v7_net = numeric(merged, "estimated_net_action_value").fillna(0)

    # V7 financial metrics are evaluated over the validated TARGET
    # population because RETRY_ALL has no production intervention.
    baseline_value = float(v7_value[target].sum())
    baseline_cost = float(v7_cost[target].sum())
    baseline_net = float(v7_net[target].sum())

    baseline_net_per_target = (
        baseline_net / target_count if target_count else 0.0
    )

    print(f"V7 expected value: ₹{baseline_value:,.2f}")
    print(f"V7 action cost:    ₹{baseline_cost:,.2f}")
    print(f"V7 estimated net:  ₹{baseline_net:,.2f}")
    print(f"V7 net / TARGET:   ₹{baseline_net_per_target:,.2f}")

    # ------------------------------------------------------------------
    # 4. Calibration-aware action comparison
    # ------------------------------------------------------------------
    banner("BUILDING CONTROLLED CALIBRATION COMPARISON")

    # IMPORTANT:
    # The calibration V2 score is an action-ranking signal only.
    # It does not replace causal uplift and does not change TARGET.
    #
    # For a controlled comparison, the V8 action allocation inherits the
    # existing V7 economic quantities. We only compare how the same 500
    # TARGET rows are partitioned into action categories.

    merged["v8_action"] = merged["v2_action"]

    # Reconciliation: calibration-aware layer cannot modify the economic
    # totals unless a separate validated action-cost/value model exists.
    merged["v8_expected_value"] = v7_value
    merged["v8_action_cost"] = v7_cost
    merged["v8_estimated_net_value"] = v7_net

    if not merged.loc[retry, "v8_action"].eq("RETRY_ALL").all():
        fail("V8 changed RETRY_ALL")

    if int(merged["v8_action"].ne("RETRY_ALL").sum()) != EXPECTED_TARGET:
        fail("V8 changed intervention population")

    # ------------------------------------------------------------------
    # 5. Action distribution comparison
    # ------------------------------------------------------------------
    banner("ACTION DISTRIBUTION COMPARISON")

    baseline_actions = merged.loc[target, "recommended_action"].value_counts()
    calibrated_actions = merged.loc[target, "v8_action"].value_counts()

    all_actions = sorted(
        set(baseline_actions.index).union(calibrated_actions.index)
    )

    comparison_rows = []

    for action in all_actions:
        b = int(baseline_actions.get(action, 0))
        c = int(calibrated_actions.get(action, 0))
        comparison_rows.append(
            {
                "action": action,
                "v7_baseline_rows": b,
                "v8_calibrated_rows": c,
                "row_change": c - b,
            }
        )
        print(
            f"{action:<30} "
            f"V7={b:>4} | V8={c:>4} | Δ={c-b:+5}"
        )

    # ------------------------------------------------------------------
    # 6. Financial reconciliation
    # ------------------------------------------------------------------
    banner("FINANCIAL RECONCILIATION")

    v8_value_total = float(merged.loc[target, "v8_expected_value"].sum())
    v8_cost_total = float(merged.loc[target, "v8_action_cost"].sum())
    v8_net_total = float(merged.loc[target, "v8_estimated_net_value"].sum())

    financial_error = v8_value_total - v8_cost_total - v8_net_total
    baseline_error = baseline_value - baseline_cost - baseline_net

    # Existing V7 numbers may use a different reporting convention.
    # Therefore V8 does not force a synthetic equality if the source
    # itself is not algebraically represented that way. It reports both.
    print(f"V7 reported net:    ₹{baseline_net:,.2f}")
    print(f"V8 reported net:    ₹{v8_net_total:,.2f}")
    print(f"V7 reconciliation:  ₹{baseline_error:,.10f}")
    print(f"V8 reconciliation:  ₹{financial_error:,.10f}")

    if not np.isfinite(financial_error):
        fail("V8 financial reconciliation is non-finite")

    # ------------------------------------------------------------------
    # 7. Comparative decision
    # ------------------------------------------------------------------
    banner("CONTROLLED EVALUATION VERDICT")

    # Because V8 deliberately holds the financial model constant, it cannot
    # legitimately claim financial improvement. The only evidence available
    # here is action redistribution within the immutable TARGET population.
    action_distribution_changed = any(
        row["row_change"] != 0 for row in comparison_rows
    )

    if action_distribution_changed:
        verdict_v8 = "CALIBRATION_ACTION_LAYER_REQUIRES_OUTCOME_TEST"
    else:
        verdict_v8 = "CALIBRATION_ACTION_LAYER_NEUTRAL"

    print("Targeting population changed: NO")
    print("RETRY_ALL changed: NO")
    print("Financial assumptions changed: NO")
    print("Causal uplift changed: NO")
    print("Oracle/evaluation fields used: NO")
    print(
        "Calibration changes action allocation:",
        "YES" if action_distribution_changed else "NO",
    )
    print("Financial improvement claimed: NO")
    print(f"FINAL VERDICT: {verdict_v8}")

    # ------------------------------------------------------------------
    # 8. Output production
    # ------------------------------------------------------------------
    banner("WRITING V8 OUTPUTS")

    merged["v8_governance_targeting_changed"] = False
    merged["v8_governance_retry_all_changed"] = False
    merged["v8_calibration_used_for_action_layer"] = True
    merged["v8_financial_improvement_claimed"] = False
    merged["v8_verdict"] = verdict_v8

    merged.to_csv(OUT_DECISIONS, index=False)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df["target_population"] = target_count
    comparison_df["intervention_rate"] = intervention_rate
    comparison_df.to_csv(OUT_COMPARISON, index=False)

    summary = pd.DataFrame(
        [
            {
                "pipeline_version": "V8",
                "evaluation_type": "CONTROLLED_OFFLINE_COMPARISON",
                "total_transactions": len(merged),
                "target_transactions": target_count,
                "retry_all_transactions": retry_count,
                "intervention_rate": intervention_rate,
                "v7_expected_value": baseline_value,
                "v7_action_cost": baseline_cost,
                "v7_estimated_net_value": baseline_net,
                "v8_expected_value": v8_value_total,
                "v8_action_cost": v8_cost_total,
                "v8_estimated_net_value": v8_net_total,
                "v7_net_per_target": baseline_net_per_target,
                "calibration_raw_brier": raw_brier,
                "calibration_isotonic_brier": iso_brier,
                "calibration_verdict": verdict,
                "targeting_changed": False,
                "retry_all_changed": False,
                "causal_uplift_changed": False,
                "economic_assumptions_changed": False,
                "oracle_fields_used": False,
                "action_distribution_changed": action_distribution_changed,
                "financial_improvement_claimed": False,
                "final_verdict": verdict_v8,
            }
        ]
    )

    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"[PASS] Decisions:  {OUT_DECISIONS}")
    print(f"[PASS] Comparison: {OUT_COMPARISON}")
    print(f"[PASS] Summary:    {OUT_SUMMARY}")

    banner("RECLAIM PRODUCTION PIPELINE V8 COMPLETE")
    print(f"FINAL VERDICT: {verdict_v8}")
    print("V10.2 5% targeting remains untouched.")


if __name__ == "__main__":
    main()
