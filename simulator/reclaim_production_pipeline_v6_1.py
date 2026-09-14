"""
RECLAIM PRODUCTION PIPELINE V6.1
Financial Reconciliation + Production Monitoring

Design principles:
- V10.2 TOP_RATE 5% remains the validated production policy.
- V5 production economic fields are treated as supplied production values.
- estimated_action_cost is the authoritative V4/V5 transparent planning proxy.
- No action costs are hard-coded.
- Oracle fields are audit/evaluation-only and are never used for decisions.
- V6.1 does not retrain, re-rank, resize, or otherwise change the targeting policy.
- Financial totals are planning/reconciliation metrics, not causal estimates.

Inputs
------
D:\\RECLAIM\\data\\generated\\reclaim_production_decisions_v5.csv

Outputs
-------
D:\\RECLAIM\\data\\generated\\reclaim_production_decisions_v6_1.csv
D:\\RECLAIM\\data\\generated\\reclaim_production_financial_reconciliation_v6_1.csv
D:\\RECLAIM\\data\\generated\\reclaim_production_monitoring_v6_1.csv

Usage
-----
python reclaim_production_pipeline_v6_1.py
python reclaim_production_pipeline_v6_1.py --input <path> --output-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_POLICY = "TOP_RATE"
EXPECTED_TARGET_RATE = 0.05
EXPECTED_ROWS = 10_000
EXPECTED_TARGET_ROWS = 500
DEFAULT_INPUT = Path(r"D:\RECLAIM\data\generated\reclaim_production_decisions_v5.csv")
DEFAULT_OUTPUT_DIR = Path(r"D:\RECLAIM\data\generated")

NUMERIC_TOL = 1e-6
RATE_TOL = 1e-9


def fail(message: str) -> None:
    raise ValueError(f"[FAIL] {message}")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"Missing required columns: {missing}")


def as_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            bad = int(converted.isna().sum())
            fail(f"Non-numeric/invalid values in {col}: {bad} rows")
        df[col] = converted


def validate_numeric_fields(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            fail(f"Numeric validation failed for {col}: non-numeric/NaN values found")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            fail(f"Numeric validation failed for {col}: non-finite values found")


def single_value(df: pd.DataFrame, col: str):
    vals = df[col].dropna().unique()
    if len(vals) != 1:
        fail(f"{col} is not constant across the production artifact: {len(vals)} distinct values")
    return vals[0]


def validate_policy(df: pd.DataFrame) -> None:
    if "production_policy_type" in df.columns:
        policies = df["production_policy_type"].dropna().astype(str).str.upper().unique()
        if len(policies) != 1 or policies[0] != EXPECTED_POLICY:
            fail(f"Production policy is not {EXPECTED_POLICY}: {policies}")

    if "production_target_rate" in df.columns:
        rates = pd.to_numeric(df["production_target_rate"], errors="coerce")
        if rates.isna().any():
            fail("production_target_rate contains invalid values")
        if not np.allclose(rates.to_numpy(), EXPECTED_TARGET_RATE, atol=RATE_TOL, rtol=0):
            fail("Validated production target rate is not 5.00%")

    if "selected_policy_type" in df.columns:
        policies = df["selected_policy_type"].dropna().astype(str).str.upper().unique()
        if len(policies) and (len(policies) != 1 or policies[0] != EXPECTED_POLICY):
            fail(f"Selected policy type is not {EXPECTED_POLICY}: {policies}")


def validate_target_population(df: pd.DataFrame) -> tuple[int, int]:
    if "production_decision" not in df.columns:
        fail("production_decision is required to validate the production population")

    decisions = df["production_decision"].astype(str).str.upper()
    target_count = int((decisions == "TARGET").sum())
    retry_count = int((decisions == "RETRY_ALL").sum())

    if target_count != EXPECTED_TARGET_ROWS:
        fail(
            f"Target population changed: expected {EXPECTED_TARGET_ROWS} TARGET rows, "
            f"found {target_count}"
        )

    if retry_count != EXPECTED_ROWS - EXPECTED_TARGET_ROWS:
        fail(
            f"RETRY_ALL population changed: expected {EXPECTED_ROWS - EXPECTED_TARGET_ROWS} rows, "
            f"found {retry_count}"
        )

    actual_rate = target_count / len(df)
    if not np.isclose(actual_rate, EXPECTED_TARGET_RATE, atol=RATE_TOL, rtol=0):
        fail(f"Actual intervention rate is {actual_rate:.6%}, expected 5.00%")

    if "production_intervention_selected" in df.columns:
        selected = pd.to_numeric(df["production_intervention_selected"], errors="coerce")
        expected = (decisions == "TARGET").astype(int)
        if selected.isna().any():
            fail("production_intervention_selected contains invalid values")
        if not np.array_equal(selected.to_numpy(dtype=int), expected.to_numpy(dtype=int)):
            fail("production_intervention_selected is inconsistent with production_decision")

    return target_count, retry_count


def validate_action_allocation(df: pd.DataFrame, target_count: int) -> None:
    if "recommended_action" not in df.columns:
        return

    actions = df["recommended_action"].astype(str).str.upper()
    target_mask = df["production_decision"].astype(str).str.upper().eq("TARGET")
    retry_mask = df["production_decision"].astype(str).str.upper().eq("RETRY_ALL")

    # V6.1 deliberately does not impose action counts or action prices.
    # It only checks that target/retry assignment is coherent.
    if (target_mask & actions.eq("RETRY_ALL")).any():
        fail("A TARGET row has recommended_action=RETRY_ALL")

    if (retry_mask & ~actions.eq("RETRY_ALL")).any():
        fail("A RETRY_ALL row has a non-RETRY_ALL recommended_action")

    if int(target_mask.sum()) != target_count:
        fail("Target allocation arithmetic is inconsistent")


def validate_economic_reconciliation(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "expected_incremental_recovery_value",
        "estimated_action_cost",
        "estimated_net_action_value",
    ]
    require_columns(df, required)
    as_numeric(df, required)
    validate_numeric_fields(df, required)

    if (df["estimated_action_cost"] < 0).any():
        fail("estimated_action_cost contains negative planning costs")

    # This is the only financial reconstruction performed by V6.1.
    # It reconciles existing production fields; it does not recreate V3 economics.
    calculated_net = (
        df["expected_incremental_recovery_value"] - df["estimated_action_cost"]
    )
    supplied_net = df["estimated_net_action_value"]

    error = calculated_net - supplied_net
    max_abs_error = float(np.abs(error).max())

    if max_abs_error > NUMERIC_TOL:
        fail(
            "Financial reconciliation failed: "
            "expected_incremental_recovery_value - estimated_action_cost "
            "does not equal estimated_net_action_value. "
            f"Max absolute error = {max_abs_error:.12f}"
        )

    reconciliation = pd.DataFrame(
        {
            "transaction_id": (
                df["transaction_id"] if "transaction_id" in df.columns
                else pd.Series(np.arange(1, len(df) + 1), index=df.index)
            ),
            "production_decision": df["production_decision"],
            "recommended_action": (
                df["recommended_action"]
                if "recommended_action" in df.columns
                else pd.Series("UNSPECIFIED", index=df.index)
            ),
            "expected_incremental_recovery_value": df[
                "expected_incremental_recovery_value"
            ],
            "estimated_action_cost": df["estimated_action_cost"],
            "estimated_net_action_value": df["estimated_net_action_value"],
            "reconciled_net_action_value": calculated_net,
            "reconciliation_error": error,
            "reconciliation_status": np.where(
                np.abs(error) <= NUMERIC_TOL, "PASS", "FAIL"
            ),
        },
        index=df.index,
    )

    return reconciliation


def build_monitoring(
    df: pd.DataFrame,
    reconciliation: pd.DataFrame,
    target_count: int,
    retry_count: int,
) -> pd.DataFrame:
    decisions = df["production_decision"].astype(str).str.upper()
    target_mask = decisions.eq("TARGET")
    retry_mask = decisions.eq("RETRY_ALL")

    def sum_for(mask: pd.Series, col: str) -> float:
        return float(df.loc[mask, col].sum())

    def mean_for(mask: pd.Series, col: str) -> float:
        return float(df.loc[mask, col].mean()) if int(mask.sum()) else float("nan")

    rows = [
        ("ALL", len(df)),
        ("TARGET", target_count),
        ("RETRY_ALL", retry_count),
    ]

    monitoring = pd.DataFrame(
        {
            "scope": [x[0] for x in rows],
            "rows": [x[1] for x in rows],
            "target_rate": [
                target_count / len(df),
                1.0,
                0.0,
            ],
            "mean_predicted_uplift": [
                mean_for(pd.Series(True, index=df.index), "predicted_probability_uplift")
                if "predicted_probability_uplift" in df.columns else np.nan,
                mean_for(target_mask, "predicted_probability_uplift")
                if "predicted_probability_uplift" in df.columns else np.nan,
                mean_for(retry_mask, "predicted_probability_uplift")
                if "predicted_probability_uplift" in df.columns else np.nan,
            ],
            "expected_incremental_recovery_value_total": [
                sum_for(pd.Series(True, index=df.index), "expected_incremental_recovery_value"),
                sum_for(target_mask, "expected_incremental_recovery_value"),
                sum_for(retry_mask, "expected_incremental_recovery_value"),
            ],
            "estimated_action_cost_total": [
                sum_for(pd.Series(True, index=df.index), "estimated_action_cost"),
                sum_for(target_mask, "estimated_action_cost"),
                sum_for(retry_mask, "estimated_action_cost"),
            ],
            "estimated_net_action_value_total": [
                sum_for(pd.Series(True, index=df.index), "estimated_net_action_value"),
                sum_for(target_mask, "estimated_net_action_value"),
                sum_for(retry_mask, "estimated_net_action_value"),
            ],
            "reconciliation_max_abs_error": [
                float(np.abs(reconciliation["reconciliation_error"]).max()),
                float(np.abs(reconciliation.loc[target_mask, "reconciliation_error"]).max()),
                float(np.abs(reconciliation.loc[retry_mask, "reconciliation_error"]).max()),
            ],
            "financial_metric_type": [
                "planning/reconciliation",
                "planning/reconciliation",
                "planning/reconciliation",
            ],
        }
    )

    return monitoring


def add_v6_1_audit_columns(
    df: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    out["v6_1_financial_reconciliation_status"] = reconciliation[
        "reconciliation_status"
    ].to_numpy()
    out["v6_1_reconciliation_error"] = reconciliation[
        "reconciliation_error"
    ].to_numpy()
    out["v6_1_oracle_used_for_decision"] = False
    out["v6_1_target_population_changed"] = False
    out["v6_1_targeting_model_changed"] = False
    out["v6_1_economic_assumptions_changed"] = False
    out["v6_1_action_cost_source"] = "V4/V5 supplied estimated_action_cost"
    out["v6_1_financial_metric_type"] = "planning/reconciliation"

    return out


def run(input_path: Path, output_dir: Path) -> None:
    print("=" * 72)
    print("RECLAIM PRODUCTION PIPELINE V6.1")
    print("Financial Reconciliation + Production Monitoring")
    print("=" * 72)

    if not input_path.exists():
        fail(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if len(df) != EXPECTED_ROWS:
        fail(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")

    required = [
        "production_decision",
        "expected_incremental_recovery_value",
        "estimated_action_cost",
        "estimated_net_action_value",
    ]
    require_columns(df, required)

    print(f"[PASS] Input loaded: {len(df):,} rows, {len(df.columns)} columns")

    validate_policy(df)
    print("[PASS] Validated policy preserved: TOP_RATE 5.00%")

    numeric_candidates = [
        c for c in [
            "predicted_treatment_probability",
            "predicted_control_probability",
            "predicted_probability_uplift",
            "production_target_rate",
            "production_capacity",
            "expected_incremental_recovery_value",
            "economic_opportunity_score",
            "treatment_control_probability_gap",
            "action_rank",
            "estimated_action_cost",
            "estimated_net_action_value",
        ] if c in df.columns
    ]
    as_numeric(df, numeric_candidates)
    validate_numeric_fields(df, numeric_candidates)
    print("[PASS] Numeric fields validated")

    target_count, retry_count = validate_target_population(df)
    print(
        f"[PASS] Target population preserved: "
        f"{target_count:,} TARGET / {retry_count:,} RETRY_ALL"
    )

    validate_action_allocation(df, target_count)
    print("[PASS] Action allocation remains coherent without hard-coded action costs")

    reconciliation = validate_economic_reconciliation(df)
    print("[PASS] Financial reconciliation validated")
    print(
        f"       max |reconciliation error| = "
        f"{np.abs(reconciliation['reconciliation_error']).max():.12f}"
    )

    decisions_v6_1 = add_v6_1_audit_columns(df, reconciliation)
    monitoring = build_monitoring(
        df, reconciliation, target_count, retry_count
    )

    # Oracle fields are copied only because they already exist in the production
    # artifact; no V6.1 decision logic references them.
    oracle_cols = [
        c for c in [
            "oracle_control_value",
            "oracle_treatment_value",
            "oracle_incremental_value",
        ] if c in df.columns
    ]
    print(
        "[PASS] Oracle fields treated as audit-only: "
        + (", ".join(oracle_cols) if oracle_cols else "none present")
    )

    decisions_path = output_dir / "reclaim_production_decisions_v6_1.csv"
    financial_path = output_dir / "reclaim_production_financial_reconciliation_v6_1.csv"
    monitoring_path = output_dir / "reclaim_production_monitoring_v6_1.csv"

    decisions_v6_1.to_csv(decisions_path, index=False)
    reconciliation.to_csv(financial_path, index=False)
    monitoring.to_csv(monitoring_path, index=False)

    print("[PASS] Output files written:")
    print(f"       {decisions_path}")
    print(f"       {financial_path}")
    print(f"       {monitoring_path}")

    total_expected = float(df["expected_incremental_recovery_value"].sum())
    total_cost = float(df["estimated_action_cost"].sum())
    total_net = float(df["estimated_net_action_value"].sum())

    target_expected = float(
        df.loc[df["production_decision"].astype(str).str.upper().eq("TARGET"),
               "expected_incremental_recovery_value"].sum()
    )
    target_cost = float(
        df.loc[df["production_decision"].astype(str).str.upper().eq("TARGET"),
               "estimated_action_cost"].sum()
    )
    target_net = float(
        df.loc[df["production_decision"].astype(str).str.upper().eq("TARGET"),
               "estimated_net_action_value"].sum()
    )

    print()
    print("FINANCIAL RECONCILIATION")
    print(f"  All expected incremental recovery value: ₹{total_expected:,.2f}")
    print(f"  All estimated action cost:              ₹{total_cost:,.2f}")
    print(f"  All estimated net action value:         ₹{total_net:,.2f}")
    print(f"  TARGET expected incremental value:      ₹{target_expected:,.2f}")
    print(f"  TARGET estimated action cost:           ₹{target_cost:,.2f}")
    print(f"  TARGET estimated net action value:      ₹{target_net:,.2f}")

    print()
    print("GOVERNANCE")
    print("  Targeting model: V10.2 TOP_RATE")
    print("  Validated target rate: 5.00%")
    print(f"  Target population: {target_count:,} / {len(df):,}")
    print("  Oracle fields used for decisions: NO")
    print("  Targeting changed by V6.1: NO")
    print("  Economic assumptions changed by V6.1: NO")
    print("  Action costs hard-coded by V6.1: NO")
    print("  Financial totals: planning/reconciliation metrics")
    print()
    print("FINAL VERDICT:")
    print("V6_1_FINANCIAL_RECONCILIATION_PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RECLAIM V6.1 financial reconciliation and monitoring"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to reclaim_production_decisions_v5.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for V6.1 output CSV files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output_dir)