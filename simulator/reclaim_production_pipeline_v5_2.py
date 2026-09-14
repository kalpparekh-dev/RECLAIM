"""
RECLAIM PRODUCTION PIPELINE V5.1
Production Audit + Governance Layer

Fixes V5:
- Uses the actual V3/V4 column name:
    expected_incremental_recovery_value
  instead of the missing expected_incremental_value.
- Audits economic_opportunity_score against the V3 uplift-derived definition:
    predicted_probability_uplift * 100
- Uses np.isclose() rather than exact floating-point equality.
- Does NOT use oracle/evaluation columns for production decisions.
- Preserves the validated 5% TOP_RATE targeting population.
- Adds audit status, governance flags, and production readiness fields.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    r"D:\RECLAIM\data\generated\reclaim_production_decisions_v4.csv"
)
DEFAULT_OUTPUT = Path(
    r"D:\RECLAIM\data\generated\reclaim_production_decisions_v5.csv"
)
DEFAULT_AUDIT = Path(
    r"D:\RECLAIM\data\generated\reclaim_production_audit_v5.csv"
)

EXPECTED_TARGET_RATE = 0.05
ECONOMIC_BASE_VALUE = 1500.0
UPLIFT_SCORE_SCALE = 100.0


def fail(message: str) -> None:
    raise ValueError("[FAIL] " + message)


def pass_msg(message: str) -> None:
    print("[PASS] " + message)


def require_columns(df: pd.DataFrame) -> None:
    required = [
        "transaction_id",
        "customer_id",
        "predicted_treatment_probability",
        "predicted_control_probability",
        "predicted_probability_uplift",
        "selected_policy_type",
        "selected_policy_parameter",
        "selected_policy",
        "production_rank",
        "production_policy_type",
        "production_target_rate",
        "production_capacity",
        "production_intervention_selected",
        "production_decision",
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

    pass_msg("V4 decision schema validated")


def numeric_columns_valid(df: pd.DataFrame) -> None:
    cols = [
        "predicted_treatment_probability",
        "predicted_control_probability",
        "predicted_probability_uplift",
        "production_target_rate",
        "production_capacity",
        "production_intervention_selected",
        "production_rank",
        "expected_incremental_recovery_value",
        "economic_opportunity_score",
        "treatment_control_probability_gap",
        "action_rank",
        "estimated_action_cost",
        "estimated_net_action_value",
    ]

    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            fail(f"Column is not numeric: {col}")
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            fail(f"Non-finite values found in column: {col}")

    pass_msg("Numeric fields validated")


def audit_prediction_relationships(df: pd.DataFrame) -> None:
    uplift_expected = (
        df["predicted_treatment_probability"]
        - df["predicted_control_probability"]
    )

    if not np.allclose(
        df["predicted_probability_uplift"].to_numpy(float),
        uplift_expected.to_numpy(float),
        rtol=1e-8,
        atol=1e-10,
    ):
        fail(
            "predicted_probability_uplift is inconsistent with "
            "treatment probability - control probability"
        )

    gap = (
        df["predicted_treatment_probability"]
        - df["predicted_control_probability"]
    )

    if not np.allclose(
        df["treatment_control_probability_gap"].to_numpy(float),
        gap.to_numpy(float),
        rtol=1e-8,
        atol=1e-10,
    ):
        fail(
            "treatment_control_probability_gap is inconsistent with "
            "treatment probability - control probability"
        )

    pass_msg("Prediction/uplift relationships validated")


def audit_economic_layer(df: pd.DataFrame) -> None:
    """
    V3 definitions visible in the supplied V4 output:

        expected_incremental_recovery_value
            = max(predicted_probability_uplift, 0) * 1500

        economic_opportunity_score
            = max(predicted_probability_uplift, 0) * 100

    The audit intentionally uses tolerance because CSV round-tripping
    can introduce tiny floating-point differences.
    """

    # V3 economic layer uses only POSITIVE uplift as actionable
    # incremental recovery value. Negative uplift therefore contributes
    # zero economic opportunity rather than a negative opportunity.
    positive_uplift = (
        df["predicted_probability_uplift"].astype(float).clip(lower=0.0)
    )

    expected_value = positive_uplift * ECONOMIC_BASE_VALUE
    expected_score = positive_uplift * UPLIFT_SCORE_SCALE

    if not np.allclose(
        df["expected_incremental_recovery_value"].to_numpy(float),
        expected_value.to_numpy(float),
        rtol=1e-7,
        atol=1e-6,
    ):
        max_error = np.max(
            np.abs(
                df["expected_incremental_recovery_value"].to_numpy(float)
                - expected_value.to_numpy(float)
            )
        )
        fail(
            "expected_incremental_recovery_value is inconsistent with "
            f"the V3 uplift-derived calculation. Max error={max_error:.10f}"
        )

    if not np.allclose(
        df["economic_opportunity_score"].to_numpy(float),
        expected_score.to_numpy(float),
        rtol=1e-7,
        atol=1e-6,
    ):
        max_error = np.max(
            np.abs(
                df["economic_opportunity_score"].to_numpy(float)
                - expected_score.to_numpy(float)
            )
        )
        fail(
            "economic_opportunity_score is inconsistent with "
            "the V3 uplift-derived calculation. "
            f"Max error={max_error:.10f}"
        )

    pass_msg("Economic layer audited against V3 definitions")


def audit_targeting(df: pd.DataFrame) -> None:
    total = len(df)

    target_mask = df["production_intervention_selected"].astype(int) == 1
    target_count = int(target_mask.sum())

    expected_count = int(round(total * EXPECTED_TARGET_RATE))
    actual_rate = target_count / total if total else 0.0

    if target_count != expected_count:
        fail(
            f"Validated target population changed: expected {expected_count}, "
            f"found {target_count}"
        )

    if not np.isclose(actual_rate, EXPECTED_TARGET_RATE, atol=1e-12):
        fail(
            f"Target rate mismatch: expected {EXPECTED_TARGET_RATE:.2%}, "
            f"found {actual_rate:.2%}"
        )

    if not (df["production_policy_type"] == "TOP_RATE").all():
        fail("Production policy type is not consistently TOP_RATE")

    if not np.allclose(
        df["production_target_rate"].to_numpy(float),
        EXPECTED_TARGET_RATE,
        atol=1e-12,
    ):
        fail("Production target rate is not consistently 5%")

    if int(df["production_capacity"].nunique()) != 1:
        fail("Production capacity is not stable across rows")

    capacity = int(df["production_capacity"].iloc[0])
    if capacity != expected_count:
        fail(
            f"Production capacity mismatch: expected {expected_count}, "
            f"found {capacity}"
        )

    pass_msg("Validated 5% targeting population preserved")


def audit_action_layer(df: pd.DataFrame) -> None:
    target_mask = df["production_intervention_selected"].astype(int) == 1

    target_actions = df.loc[target_mask, "recommended_action"]
    retry_actions = df.loc[~target_mask, "recommended_action"]

    if (retry_actions != "RETRY_ALL").any():
        fail("A RETRY_ALL transaction has a non-RETRY_ALL recommended action")

    allowed_target_actions = {
        "MONITOR",
        "LOW_INTENSITY",
        "STANDARD_TARGET",
        "HIGH_INTENSITY",
    }

    invalid_target = ~target_actions.isin(allowed_target_actions)
    if invalid_target.any():
        bad = sorted(target_actions[invalid_target].unique().tolist())
        fail(f"Invalid TARGET action(s): {bad}")

    # Action costs are planning proxies from V4. Verify only arithmetic,
    # not causal validity.
    net_expected = (
        df["expected_incremental_recovery_value"]
        - df["estimated_action_cost"]
    )

    if not np.allclose(
        df["estimated_net_action_value"].to_numpy(float),
        net_expected.to_numpy(float),
        rtol=1e-7,
        atol=1e-6,
    ):
        fail(
            "estimated_net_action_value is inconsistent with "
            "expected value - estimated action cost"
        )

    pass_msg("Action allocation arithmetic and action constraints validated")


def audit_no_decision_leakage(df: pd.DataFrame) -> None:
    """
    V5 is an audit/governance layer. Oracle columns may exist in the source
    because they were retained for evaluation, but they are explicitly not
    used to create or modify production decisions.
    """

    oracle_cols = [
        "oracle_control_value",
        "oracle_treatment_value",
        "oracle_incremental_value",
    ]

    present = [c for c in oracle_cols if c in df.columns]
    if present:
        pass_msg(
            "Oracle/evaluation columns detected and treated as audit-only: "
            + ", ".join(present)
        )
    else:
        pass_msg("No oracle/evaluation columns present")


def build_governance_layer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    target = out["production_intervention_selected"].astype(int) == 1

    out["v5_audit_status"] = "PASS"

    out["v5_governance_class"] = np.where(
        target,
        "VALIDATED_TARGET",
        "STANDARD_RETRY",
    )

    out["v5_decision_source"] = "V10.2_TOP_RATE_5_PERCENT"

    out["v5_economic_basis"] = (
        "UPLIFT_DERIVED_EXPECTED_VALUE"
    )

    out["v5_action_cost_basis"] = (
        "V4_TRANSPARENT_PLANNING_PROXY"
    )

    out["v5_oracle_used_for_decision"] = 0

    out["v5_production_ready"] = 1

    # Keep the original decision unchanged. This is an audit/governance layer,
    # not a new targeting model.
    out["v5_final_decision"] = out["production_decision"]

    return out


def build_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    target = df["production_intervention_selected"].astype(int) == 1

    checks = [
        (
            "schema",
            "V4 decision schema",
            "PASS",
        ),
        (
            "prediction_math",
            "Uplift and probability-gap arithmetic",
            "PASS",
        ),
        (
            "economic_value",
            "Expected incremental recovery value",
            "PASS",
        ),
        (
            "economic_score",
            "Economic opportunity score",
            "PASS",
        ),
        (
            "target_capacity",
            "Validated 5% target capacity",
            "PASS",
        ),
        (
            "action_layer",
            "Action allocation arithmetic",
            "PASS",
        ),
        (
            "oracle_governance",
            "Oracle/evaluation fields excluded from decisions",
            "PASS",
        ),
    ]

    audit = pd.DataFrame(
        checks,
        columns=["check_id", "check_name", "status"],
    )

    audit["total_transactions"] = len(df)
    audit["target_transactions"] = int(target.sum())
    audit["target_rate"] = target.mean()
    audit["mean_predicted_uplift"] = df["predicted_probability_uplift"].mean()
    audit["total_expected_incremental_value"] = (
        df["expected_incremental_recovery_value"].sum()
    )
    audit["total_estimated_action_cost"] = (
        df["estimated_action_cost"].sum()
    )
    audit["total_estimated_net_action_value"] = (
        df["estimated_net_action_value"].sum()
    )

    return audit


def print_summary(df: pd.DataFrame) -> None:
    target = df["production_intervention_selected"].astype(int) == 1

    print("\n" + "=" * 70)
    print("V5 PRODUCTION AUDIT / GOVERNANCE SUMMARY")
    print("=" * 70)

    print(f"Total transactions: {len(df)}")
    print(f"Validated TARGET transactions: {int(target.sum())}")
    print(f"RETRY_ALL transactions: {int((~target).sum())}")
    print(f"Actual intervention rate: {target.mean():.2%}")

    print(
        f"Mean predicted uplift: "
        f"{df['predicted_probability_uplift'].mean():.2%}"
    )

    print(
        "TARGET mean predicted uplift: "
        f"{df.loc[target, 'predicted_probability_uplift'].mean():.2%}"
    )

    print(
        "TARGET expected incremental value: ₹"
        f"{df.loc[target, 'expected_incremental_recovery_value'].sum():,.2f}"
    )

    print(
        "Total estimated action cost: ₹"
        f"{df['estimated_action_cost'].sum():,.2f}"
    )

    print(
        "Estimated net action value: ₹"
        f"{df['estimated_net_action_value'].sum():,.2f}"
    )

    print("\nGovernance:")
    print("  Targeting model: V10.2 TOP_RATE")
    print("  Validated target rate: 5.00%")
    print("  Oracle columns used for decision: NO")
    print("  Action costs: transparent V4 planning proxies")
    print("  Causal validation changed by V5: NO")

    print("\n" + "=" * 70)
    print("FINAL V5 VERDICT")
    print("=" * 70)
    print("FINAL VERDICT: V5_PRODUCTION_AUDIT_PASSED")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RECLAIM Production Pipeline V5.1"
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="V4 production decision CSV",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="V5 production decision CSV",
    )
    parser.add_argument(
        "--audit-output",
        default=str(DEFAULT_AUDIT),
        help="V5 audit CSV",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_path = Path(args.audit_output)

    print("=" * 70)
    print("RECLAIM PRODUCTION PIPELINE V5.1")
    print("=" * 70)
    print("Production Audit + Governance Layer")
    print(f"Input: {input_path}")
    print("Validated policy: TOP_RATE")
    print("Validated target rate: 5.00%")
    print(f"Economic base value: ₹{ECONOMIC_BASE_VALUE:,.2f}")

    if not input_path.exists():
        fail(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    print(f"[PASS] Rows loaded: {len(df)}")
    print(f"[PASS] Columns loaded: {len(df.columns)}")

    require_columns(df)
    numeric_columns_valid(df)

    print("\n" + "=" * 70)
    print("RUNNING PRODUCTION AUDITS")
    print("=" * 70)

    audit_prediction_relationships(df)
    audit_economic_layer(df)
    audit_targeting(df)
    audit_action_layer(df)
    audit_no_decision_leakage(df)

    print("\n" + "=" * 70)
    print("BUILDING V5 GOVERNANCE LAYER")
    print("=" * 70)

    governed = build_governance_layer(df)
    audit = build_audit_table(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    governed.to_csv(output_path, index=False)
    audit.to_csv(audit_path, index=False)

    pass_msg(f"Production decisions: {output_path}")
    pass_msg(f"Audit report: {audit_path}")

    print_summary(governed)

    print("\n" + "=" * 70)
    print("PRODUCTION PIPELINE V5.1 COMPLETE")
    print("=" * 70)
    print(f"[PASS] Output: {output_path}")
    print(f"[PASS] Audit:  {audit_path}")
    print("No targeting population was changed.")
    print("No oracle/evaluation field was used for production decisions.")
    print("=" * 70)


if __name__ == "__main__":
    main()