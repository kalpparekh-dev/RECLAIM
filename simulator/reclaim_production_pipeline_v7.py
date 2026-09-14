"""
RECLAIM PRODUCTION PIPELINE V7
Production Monitoring + Control

Purpose
-------
V7 is a monitoring/control layer over the already validated RECLAIM
production policy. It DOES NOT retrain the model, change the target rate,
change the economic assumptions, or use oracle fields for decisions.

Validated production invariants:
    Policy:       V10.2 TOP_RATE
    Target rate:  5%
    Population:   10,000 transactions
    TARGET:       500
    RETRY_ALL:    9,500

V7 reads the V6.1 production artifact and produces monitoring/control
artifacts. All financial figures remain planning/reconciliation metrics.

Input
-----
D:\\RECLAIM\\data\\generated\\reclaim_production_decisions_v6_1.csv

Outputs
-------
D:\\RECLAIM\\data\\generated\\reclaim_production_decisions_v7.csv
D:\\RECLAIM\\data\\generated\\reclaim_production_monitoring_v7.csv
D:\\RECLAIM\\data\\generated\\reclaim_production_control_audit_v7.csv

Usage
-----
python reclaim_production_pipeline_v7.py
python reclaim_production_pipeline_v7.py --input <path> --output-dir <dir>
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
EXPECTED_RETRY_ROWS = 9_500

NUMERIC_TOL = 1e-6
RATE_TOL = 1e-9

DEFAULT_INPUT = Path(
    r"D:\RECLAIM\data\generated\reclaim_production_decisions_v6_1.csv"
)
DEFAULT_OUTPUT_DIR = Path(r"D:\RECLAIM\data\generated")


def fail(message: str) -> None:
    raise ValueError("[FAIL] " + message)


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"Missing required columns: {missing}")


def numeric_check(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            fail(f"Numeric validation failed for {col}: NaN/invalid values")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            fail(f"Numeric validation failed for {col}: non-finite values")


def validate_schema(df: pd.DataFrame) -> None:
    required = [
        "production_decision",
        "production_policy_type",
        "production_target_rate",
        "predicted_probability_uplift",
        "expected_incremental_recovery_value",
        "estimated_action_cost",
        "estimated_net_action_value",
        "v6_1_financial_reconciliation_status",
        "v6_1_oracle_used_for_decision",
        "v6_1_target_population_changed",
        "v6_1_targeting_model_changed",
        "v6_1_economic_assumptions_changed",
    ]
    require_columns(df, required)


def validate_policy(df: pd.DataFrame) -> None:
    policies = (
        df["production_policy_type"]
        .astype(str)
        .str.upper()
        .dropna()
        .unique()
    )

    if len(policies) != 1 or policies[0] != EXPECTED_POLICY:
        fail(f"Policy drift detected: {policies}")

    rates = pd.to_numeric(df["production_target_rate"], errors="coerce")
    if not np.allclose(
        rates.to_numpy(),
        EXPECTED_TARGET_RATE,
        atol=RATE_TOL,
        rtol=0,
    ):
        fail("Target-rate drift detected: expected 5.00%")

    print("[PASS] V10.2 TOP_RATE 5.00% policy preserved")


def validate_population(df: pd.DataFrame) -> tuple[int, int]:
    decisions = df["production_decision"].astype(str).str.upper()

    target = int((decisions == "TARGET").sum())
    retry = int((decisions == "RETRY_ALL").sum())

    if len(df) != EXPECTED_ROWS:
        fail(f"Population size changed: expected {EXPECTED_ROWS}, found {len(df)}")

    if target != EXPECTED_TARGET_ROWS:
        fail(
            f"TARGET population drift: expected {EXPECTED_TARGET_ROWS}, "
            f"found {target}"
        )

    if retry != EXPECTED_RETRY_ROWS:
        fail(
            f"RETRY_ALL population drift: expected {EXPECTED_RETRY_ROWS}, "
            f"found {retry}"
        )

    rate = target / len(df)
    if not np.isclose(rate, EXPECTED_TARGET_RATE, atol=RATE_TOL, rtol=0):
        fail(f"Intervention rate drift: {rate:.6%}")

    print(
        f"[PASS] Population preserved: "
        f"{target:,} TARGET / {retry:,} RETRY_ALL / {rate:.2%}"
    )

    return target, retry


def validate_v6_1_governance(df: pd.DataFrame) -> None:
    status = df["v6_1_financial_reconciliation_status"].astype(str).str.upper()
    if not (status == "PASS").all():
        fail("V6.1 financial reconciliation is not PASS for every row")

    for col in [
        "v6_1_oracle_used_for_decision",
        "v6_1_target_population_changed",
        "v6_1_targeting_model_changed",
        "v6_1_economic_assumptions_changed",
    ]:
        values = df[col].astype(str).str.upper()
        if col == "v6_1_oracle_used_for_decision":
            allowed = {"FALSE", "0", "NO"}
        else:
            allowed = {"FALSE", "0", "NO"}

        if not values.isin(allowed).all():
            fail(f"V6.1 governance violation detected in {col}")

    print("[PASS] V6.1 governance status preserved")


def validate_financial_reconciliation(df: pd.DataFrame) -> float:
    expected = pd.to_numeric(
        df["expected_incremental_recovery_value"], errors="coerce"
    )
    cost = pd.to_numeric(df["estimated_action_cost"], errors="coerce")
    net = pd.to_numeric(df["estimated_net_action_value"], errors="coerce")

    error = expected - cost - net
    max_error = float(np.abs(error).max())

    if max_error > NUMERIC_TOL:
        fail(
            "Financial reconciliation drift detected. "
            f"Maximum absolute error = {max_error:.12f}"
        )

    print(
        "[PASS] Financial reconciliation preserved: "
        f"max error = {max_error:.12f}"
    )

    return max_error


def validate_actions(df: pd.DataFrame, target_count: int) -> None:
    if "recommended_action" not in df.columns:
        fail("recommended_action is required for V7 monitoring")

    decisions = df["production_decision"].astype(str).str.upper()
    actions = df["recommended_action"].astype(str).str.upper()

    target_mask = decisions.eq("TARGET")
    retry_mask = decisions.eq("RETRY_ALL")

    if (target_mask & actions.eq("RETRY_ALL")).any():
        fail("TARGET row has RETRY_ALL action")

    if (retry_mask & ~actions.eq("RETRY_ALL")).any():
        fail("RETRY_ALL row has non-RETRY_ALL action")

    if int(target_mask.sum()) != target_count:
        fail("Action allocation does not match TARGET population")

    print("[PASS] Action allocation remains coherent")


def validate_uplift(df: pd.DataFrame) -> None:
    uplift = pd.to_numeric(df["predicted_probability_uplift"], errors="coerce")

    if uplift.isna().any():
        fail("predicted_probability_uplift contains invalid values")

    # Monitoring only: do not alter ranking or decisions.
    target_mask = df["production_decision"].astype(str).str.upper().eq("TARGET")
    retry_mask = df["production_decision"].astype(str).str.upper().eq("RETRY_ALL")

    overall = float(uplift.mean())
    target_mean = float(uplift.loc[target_mask].mean())
    retry_mean = float(uplift.loc[retry_mask].mean())

    print(f"[PASS] Uplift monitored: overall={overall:.4%}, "
          f"TARGET={target_mean:.4%}, RETRY_ALL={retry_mean:.4%}")


def build_monitoring(
    df: pd.DataFrame,
    target_count: int,
    retry_count: int,
) -> pd.DataFrame:
    decisions = df["production_decision"].astype(str).str.upper()
    target_mask = decisions.eq("TARGET")
    retry_mask = decisions.eq("RETRY_ALL")
    all_mask = pd.Series(True, index=df.index)

    def mean(mask: pd.Series, col: str) -> float:
        return float(df.loc[mask, col].mean())

    def total(mask: pd.Series, col: str) -> float:
        return float(df.loc[mask, col].sum())

    scopes = [
        ("ALL", all_mask, len(df)),
        ("TARGET", target_mask, target_count),
        ("RETRY_ALL", retry_mask, retry_count),
    ]

    records = []

    for scope, mask, count in scopes:
        records.append(
            {
                "monitoring_scope": scope,
                "rows": count,
                "rate": count / len(df),
                "mean_predicted_uplift": mean(
                    mask, "predicted_probability_uplift"
                ),
                "expected_incremental_recovery_value_total": total(
                    mask, "expected_incremental_recovery_value"
                ),
                "estimated_action_cost_total": total(
                    mask, "estimated_action_cost"
                ),
                "estimated_net_action_value_total": total(
                    mask, "estimated_net_action_value"
                ),
                "financial_metric_type": "planning/reconciliation",
            }
        )

    return pd.DataFrame(records)


def build_action_monitoring(df: pd.DataFrame) -> pd.DataFrame:
    if "recommended_action" not in df.columns:
        return pd.DataFrame()

    group = (
        df.groupby(
            df["recommended_action"].astype(str).str.upper(),
            dropna=False,
        )
        .agg(
            rows=("recommended_action", "size"),
            mean_predicted_uplift=("predicted_probability_uplift", "mean"),
            expected_incremental_recovery_value_total=(
                "expected_incremental_recovery_value",
                "sum",
            ),
            estimated_action_cost_total=("estimated_action_cost", "sum"),
            estimated_net_action_value_total=(
                "estimated_net_action_value",
                "sum",
            ),
        )
        .reset_index()
        .rename(columns={"recommended_action": "action"})
    )

    group["financial_metric_type"] = "planning/reconciliation"
    return group


def build_control_audit(
    df: pd.DataFrame,
    max_financial_error: float,
) -> pd.DataFrame:
    decisions = df["production_decision"].astype(str).str.upper()

    checks = [
        {
            "control_id": "POLICY",
            "control": "V10.2 TOP_RATE policy",
            "status": "PASS",
            "observed_value": EXPECTED_POLICY,
            "expected_value": EXPECTED_POLICY,
            "severity": "INFO",
        },
        {
            "control_id": "TARGET_RATE",
            "control": "Validated target rate",
            "status": "PASS",
            "observed_value": "5.00%",
            "expected_value": "5.00%",
            "severity": "INFO",
        },
        {
            "control_id": "TARGET_POPULATION",
            "control": "TARGET population",
            "status": "PASS",
            "observed_value": int(decisions.eq("TARGET").sum()),
            "expected_value": EXPECTED_TARGET_ROWS,
            "severity": "INFO",
        },
        {
            "control_id": "RETRY_ALL_POPULATION",
            "control": "RETRY_ALL population",
            "status": "PASS",
            "observed_value": int(decisions.eq("RETRY_ALL").sum()),
            "expected_value": EXPECTED_RETRY_ROWS,
            "severity": "INFO",
        },
        {
            "control_id": "FINANCIAL_RECONCILIATION",
            "control": "V6.1 financial reconciliation",
            "status": "PASS" if max_financial_error <= NUMERIC_TOL else "FAIL",
            "observed_value": max_financial_error,
            "expected_value": f"<= {NUMERIC_TOL}",
            "severity": "CRITICAL",
        },
        {
            "control_id": "ORACLE_DECISION_USAGE",
            "control": "Oracle fields used for production decisions",
            "status": "PASS",
            "observed_value": "NO",
            "expected_value": "NO",
            "severity": "CRITICAL",
        },
        {
            "control_id": "TARGETING_CHANGE",
            "control": "V7 changed targeting",
            "status": "PASS",
            "observed_value": "NO",
            "expected_value": "NO",
            "severity": "CRITICAL",
        },
        {
            "control_id": "ECONOMIC_CHANGE",
            "control": "V7 changed economic assumptions",
            "status": "PASS",
            "observed_value": "NO",
            "expected_value": "NO",
            "severity": "CRITICAL",
        },
    ]

    return pd.DataFrame(checks)


def add_v7_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # V7 is deliberately observational/control-only.
    out["v7_monitoring_layer"] = "PRODUCTION_MONITORING_CONTROL"
    out["v7_targeting_model"] = "V10.2_TOP_RATE"
    out["v7_target_rate"] = EXPECTED_TARGET_RATE
    out["v7_targeting_changed"] = False
    out["v7_economic_assumptions_changed"] = False
    out["v7_oracle_used_for_decision"] = False
    out["v7_financial_metrics_are_causal_estimates"] = False
    out["v7_financial_metric_type"] = "planning/reconciliation"

    return out


def run(input_path: Path, output_dir: Path) -> None:
    print("=" * 72)
    print("RECLAIM PRODUCTION PIPELINE V7")
    print("Production Monitoring + Control")
    print("=" * 72)

    if not input_path.exists():
        fail(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    validate_schema(df)
    print(f"[PASS] Input loaded: {len(df):,} rows, {len(df.columns)} columns")

    numeric_columns = [
        "production_target_rate",
        "predicted_probability_uplift",
        "expected_incremental_recovery_value",
        "estimated_action_cost",
        "estimated_net_action_value",
    ]
    numeric_check(df, numeric_columns)
    print("[PASS] Numeric monitoring fields validated")

    validate_policy(df)
    target_count, retry_count = validate_population(df)
    validate_v6_1_governance(df)
    max_financial_error = validate_financial_reconciliation(df)
    validate_actions(df, target_count)
    validate_uplift(df)

    monitoring = build_monitoring(df, target_count, retry_count)
    action_monitoring = build_action_monitoring(df)
    if not action_monitoring.empty:
        monitoring = pd.concat(
            [
                monitoring,
                action_monitoring.rename(
                    columns={"action": "monitoring_scope"}
                ),
            ],
            ignore_index=True,
            sort=False,
        )

    control_audit = build_control_audit(df, max_financial_error)
    if not (control_audit["status"] == "PASS").all():
        fail("One or more V7 control checks failed")

    decisions_v7 = add_v7_metadata(df)

    decisions_path = output_dir / "reclaim_production_decisions_v7.csv"
    monitoring_path = output_dir / "reclaim_production_monitoring_v7.csv"
    audit_path = output_dir / "reclaim_production_control_audit_v7.csv"

    decisions_v7.to_csv(decisions_path, index=False)
    monitoring.to_csv(monitoring_path, index=False)
    control_audit.to_csv(audit_path, index=False)

    print("[PASS] Output files written:")
    print(f"       {decisions_path}")
    print(f"       {monitoring_path}")
    print(f"       {audit_path}")

    decisions = df["production_decision"].astype(str).str.upper()
    target_mask = decisions.eq("TARGET")
    retry_mask = decisions.eq("RETRY_ALL")

    print()
    print("V7 MONITORING SUMMARY")
    print(f"  Transactions:              {len(df):,}")
    print(f"  TARGET:                    {target_count:,}")
    print(f"  RETRY_ALL:                 {retry_count:,}")
    print(f"  Intervention rate:         {target_count / len(df):.2%}")
    print(
        f"  Mean uplift overall:       "
        f"{df['predicted_probability_uplift'].mean():.4%}"
    )
    print(
        f"  Mean uplift TARGET:        "
        f"{df.loc[target_mask, 'predicted_probability_uplift'].mean():.4%}"
    )
    print(
        f"  Mean uplift RETRY_ALL:     "
        f"{df.loc[retry_mask, 'predicted_probability_uplift'].mean():.4%}"
    )
    print(
        f"  Expected value total:      "
        f"₹{df['expected_incremental_recovery_value'].sum():,.2f}"
    )
    print(
        f"  Action cost total:         "
        f"₹{df['estimated_action_cost'].sum():,.2f}"
    )
    print(
        f"  Estimated net value total: "
        f"₹{df['estimated_net_action_value'].sum():,.2f}"
    )
    print(f"  Max financial error:       ₹{max_financial_error:.12f}")

    print()
    print("GOVERNANCE")
    print("  Targeting model: V10.2 TOP_RATE")
    print("  Validated target rate: 5.00%")
    print(f"  Target population: {target_count:,} / {len(df):,}")
    print("  Oracle fields used for decisions: NO")
    print("  V7 changed targeting: NO")
    print("  V7 changed economic assumptions: NO")
    print("  Financial metrics: planning/reconciliation only")

    print()
    print("FINAL VERDICT:")
    print("V7_PRODUCTION_MONITORING_CONTROL_PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RECLAIM V7 production monitoring and control"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="V6.1 production decisions CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output_dir)