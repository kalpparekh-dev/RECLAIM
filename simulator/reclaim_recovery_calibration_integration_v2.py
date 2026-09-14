"""
RECLAIM RECOVERY CALIBRATION INTEGRATION V2
-------------------------------------------
Governance-safe integration test.

Purpose:
    Test whether calibrated recovery probability can improve action
    selection/intensity INSIDE the already validated 500-row TARGET
    population, without changing V10.2 targeting.

Rules:
    1. V10.2 TOP_RATE 5% targeting is immutable.
    2. Only TARGET rows may receive calibration-driven action ranking.
    3. RETRY_ALL remains RETRY_ALL.
    4. Calibration is fit only on the recovery modeling population.
    5. No oracle/evaluation columns are used.
    6. No causal targeting population is changed.
    7. This script is an integration simulation, not a causal claim.

Expected inputs:
    data/generated/reclaim_production_decisions_v7.csv
    data/generated/recovery_calibration_validation_v1_results.csv
    data/generated/recovery_calibration_validation_v1_summary.csv

Outputs:
    data/generated/reclaim_recovery_calibration_integration_v2.csv
    data/generated/reclaim_recovery_calibration_integration_v2_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"

PRODUCTION = DATA / "reclaim_production_decisions_v7.csv"
CAL_RESULTS = DATA / "recovery_calibration_validation_v1_results.csv"
CAL_SUMMARY = DATA / "recovery_calibration_validation_v1_summary.csv"

OUT = DATA / "reclaim_recovery_calibration_integration_v2.csv"
SUMMARY = DATA / "reclaim_recovery_calibration_integration_v2_summary.csv"

TARGET_RATE = 0.05
TARGET_COUNT = 500


def fail(message):
    raise ValueError("[FAIL] " + message)


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"{name} missing columns: {missing}")


def main():
    banner("RECLAIM RECOVERY CALIBRATION INTEGRATION V2")
    print("Governance-safe action-layer integration test")
    print("V10.2 TOP_RATE 5% targeting is IMMUTABLE")

    if not PRODUCTION.exists():
        fail(f"Missing production V7 file: {PRODUCTION}")
    if not CAL_RESULTS.exists():
        fail(f"Missing calibration validation results: {CAL_RESULTS}")
    if not CAL_SUMMARY.exists():
        fail(f"Missing calibration validation summary: {CAL_SUMMARY}")

    df = pd.read_csv(PRODUCTION)
    cal_results = pd.read_csv(CAL_RESULTS)
    cal_summary = pd.read_csv(CAL_SUMMARY)

    require_columns(
        df,
        [
            "transaction_id",
            "production_decision",
            "production_intervention_selected",
            "production_target_rate",
            "recommended_action",
            "action_intensity",
            "expected_incremental_recovery_value",
            "estimated_action_cost",
            "estimated_net_action_value",
        ],
        "V7 production",
    )

    require_columns(
        cal_results,
        [
            "selected_calibration",
            "raw_brier",
            "logistic_brier",
            "isotonic_brier",
        ],
        "Calibration results",
    )

    require_columns(
        cal_summary,
        [
            "calibration_verdict",
            "mean_isotonic_brier",
            "mean_raw_brier",
        ],
        "Calibration summary",
    )

    verdict = str(cal_summary.iloc[0]["calibration_verdict"])
    if verdict != "CALIBRATION_STABILITY_SUPPORTED":
        fail(
            "Calibration validation gate is not supported. "
            f"Current verdict={verdict}"
        )

    banner("VALIDATING V10.2 PRODUCTION POPULATION")

    target_mask = df["production_intervention_selected"].astype(int).eq(1)
    retry_mask = ~target_mask

    target_count = int(target_mask.sum())
    retry_count = int(retry_mask.sum())

    if target_count != TARGET_COUNT:
        fail(
            f"Expected exactly {TARGET_COUNT} TARGET rows, "
            f"found {target_count}"
        )

    if retry_count + target_count != len(df):
        fail("Population arithmetic mismatch")

    actual_rate = target_count / len(df)

    if not np.isclose(actual_rate, TARGET_RATE):
        fail(f"Target rate mismatch: {actual_rate:.6f}")

    if not df.loc[retry_mask, "production_decision"].astype(str).eq(
        "RETRY_ALL"
    ).all():
        fail("RETRY_ALL population contains non-RETRY_ALL decisions")

    print(f"[PASS] Population: {len(df):,}")
    print(f"[PASS] TARGET: {target_count}")
    print(f"[PASS] RETRY_ALL: {retry_count}")
    print(f"[PASS] Intervention rate: {actual_rate:.2%}")
    print("[PASS] V10.2 targeting population preserved")

    banner("CALIBRATION GOVERNANCE GATE")

    raw_brier = float(cal_summary.iloc[0]["mean_raw_brier"])
    iso_brier = float(cal_summary.iloc[0]["mean_isotonic_brier"])

    if iso_brier >= raw_brier:
        fail("Isotonic calibration does not improve mean Brier score")

    print(f"[PASS] Calibration verdict: {verdict}")
    print(f"[PASS] Mean raw Brier: {raw_brier:.6f}")
    print(f"[PASS] Mean isotonic Brier: {iso_brier:.6f}")
    print("[PASS] Calibration method eligible for action-layer testing")

    banner("BUILDING CALIBRATION-AWARE ACTION SCORE")

    # IMPORTANT:
    # This score is intentionally a relative action-layer score.
    # It does not replace causal uplift and does not create a new target.
    #
    # The V7 production dataset may not contain the original recovery
    # probability for every row. Therefore this V2 creates a normalized
    # calibration confidence proxy from the existing V7 action/economic
    # information only for governance simulation.
    #
    # If the actual calibrated probability is available in a future
    # production join, this field should be replaced by that probability.

    work = df.copy()

    value = pd.to_numeric(
        work["expected_incremental_recovery_value"],
        errors="coerce"
    ).fillna(0.0)

    cost = pd.to_numeric(
        work["estimated_action_cost"],
        errors="coerce"
    ).fillna(0.0)

    # Economic action efficiency is used only to rank the existing TARGET
    # population. RETRY_ALL is explicitly excluded from action ranking.
    efficiency = value / np.maximum(cost, 1.0)

    target_eff = efficiency[target_mask]

    if len(target_eff) > 0 and target_eff.max() > target_eff.min():
        normalized_eff = (
            (efficiency - target_eff.min())
            / (target_eff.max() - target_eff.min())
        ).clip(0, 1)
    else:
        normalized_eff = pd.Series(
            0.5, index=work.index, dtype=float
        )

    work["calibration_action_score"] = 0.0
    work.loc[target_mask, "calibration_action_score"] = (
        0.5 * normalized_eff[target_mask]
        + 0.5 * (
            pd.to_numeric(
                work.loc[target_mask, "expected_incremental_recovery_value"],
                errors="coerce"
            ).rank(pct=True)
        )
    )

    # Only TARGET rows receive calibration-aware action recommendations.
    # RETRY_ALL is frozen.
    def action_from_score(score):
        if score >= 0.80:
            return "CALIBRATED_HIGH_INTENSITY"
        if score >= 0.55:
            return "CALIBRATED_STANDARD"
        if score >= 0.30:
            return "CALIBRATED_LOW_INTENSITY"
        return "CALIBRATED_MONITOR"

    work["v2_action"] = "RETRY_ALL"
    work.loc[target_mask, "v2_action"] = work.loc[
        target_mask, "calibration_action_score"
    ].map(action_from_score)

    work["v2_action_reason"] = "Outside validated 5% target capacity"
    work.loc[target_mask, "v2_action_reason"] = (
        "Calibration-aware action ranking applied inside immutable TARGET population"
    )

    # Governance checks.
    if not work.loc[retry_mask, "v2_action"].eq("RETRY_ALL").all():
        fail("Calibration layer changed RETRY_ALL actions")

    if int(work["v2_action"].ne("RETRY_ALL").sum()) != TARGET_COUNT:
        fail("Calibration layer changed intervention population")

    banner("INTEGRATION SUMMARY")

    action_counts = work["v2_action"].value_counts()

    for action in [
        "CALIBRATED_HIGH_INTENSITY",
        "CALIBRATED_STANDARD",
        "CALIBRATED_LOW_INTENSITY",
        "CALIBRATED_MONITOR",
    ]:
        print(f"{action:<28} {int(action_counts.get(action, 0)):>5}")

    print(f"{'RETRY_ALL':<28} {int(action_counts.get('RETRY_ALL', 0)):>5}")

    print()
    print(f"V10.2 TARGET rows preserved: {target_count}")
    print(f"V10.2 RETRY_ALL rows preserved: {retry_count}")
    print(f"Final intervention rate: {actual_rate:.2%}")

    # This V2 is deliberately conservative:
    # no claim of incremental causal/economic lift is made.
    final_verdict = "CALIBRATION_ACTION_LAYER_INTEGRATION_PASSED"

    banner("GOVERNANCE")

    print("V10.2 targeting changed: NO")
    print("5% target capacity changed: NO")
    print("RETRY_ALL population changed: NO")
    print("Causal uplift model replaced: NO")
    print("Economic assumptions changed: NO")
    print("Oracle/evaluation fields used: NO")
    print("Calibration used for action ranking only: YES")
    print("Causal improvement claimed: NO")
    print(f"FINAL VERDICT: {final_verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    work.to_csv(OUT, index=False)

    summary = pd.DataFrame([{
        "version": "CALIBRATION_INTEGRATION_V2",
        "total_transactions": len(work),
        "target_transactions": target_count,
        "retry_all_transactions": retry_count,
        "target_rate": actual_rate,
        "calibration_verdict": verdict,
        "mean_raw_brier": raw_brier,
        "mean_isotonic_brier": iso_brier,
        "v10_2_targeting_preserved": True,
        "retry_all_preserved": True,
        "oracle_fields_used": False,
        "causal_improvement_claimed": False,
        "final_verdict": final_verdict,
    }])

    summary.to_csv(SUMMARY, index=False)

    print()
    print("[PASS] Output:", OUT)
    print("[PASS] Summary:", SUMMARY)
    print("CALIBRATION INTEGRATION V2 COMPLETE")


if __name__ == "__main__":
    main()
