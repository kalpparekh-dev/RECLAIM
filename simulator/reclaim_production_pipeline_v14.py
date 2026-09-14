
"""
========================================================================
RECLAIM PRODUCTION PIPELINE V14
Production Outcome Ingestion + Evaluation Readiness
========================================================================

Purpose
-------
V14 is a governance-safe outcome ingestion layer.

It DOES NOT:
- change V10.2 targeting
- change the 5% target capacity
- change V7/V8 decisions
- fabricate outcomes
- create artificial transaction IDs
- treat calibration predictions as production outcomes
- claim V7 vs V8 improvement without legitimate matched outcomes

It DOES:
- load the immutable V13 production cohort
- load legitimate production outcomes
- normalize the outcome schema
- perform exact transaction_id linkage
- reject duplicate outcome IDs
- calculate cohort coverage
- calculate TARGET / RETRY_ALL coverage
- evaluate outcomes only when sufficient legitimate observations exist
- produce explicit readiness verdicts

Expected files
--------------
Input:
    D:\\RECLAIM\\data\\generated\\reclaim_production_cohort_v13.csv
    D:\\RECLAIM\\data\\generated\\reclaim_production_decisions_v13.csv

Preferred outcome file:
    D:\\RECLAIM\\data\\generated\\recovery_outcomes.csv

Outputs:
    D:\\RECLAIM\\data\\generated\\reclaim_production_decisions_v14.csv
    D:\\RECLAIM\\data\\generated\\reclaim_production_cohort_v14.csv
    D:\\RECLAIM\\data\\generated\\reclaim_production_outcome_capture_v14.csv
    D:\\RECLAIM\\data\\generated\\reclaim_production_v14_summary.csv
    D:\\RECLAIM\\data\\generated\\reclaim_production_v14_audit.csv
    D:\\RECLAIM\\data\\generated\\reclaim_production_v14_comparison.csv
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(r"D:\RECLAIM")
GENERATED_DIR = BASE_DIR / "data" / "generated"

V13_COHORT = GENERATED_DIR / "reclaim_production_cohort_v13.csv"
V13_DECISIONS = GENERATED_DIR / "reclaim_production_decisions_v13.csv"

# Legitimate production outcome source.
# Change this path only when the real production outcome file is supplied.
OUTCOME_SOURCE = GENERATED_DIR / "recovery_outcomes.csv"

V14_DECISIONS = GENERATED_DIR / "reclaim_production_decisions_v14.csv"
V14_COHORT = GENERATED_DIR / "reclaim_production_cohort_v14.csv"
V14_CAPTURE = GENERATED_DIR / "reclaim_production_outcome_capture_v14.csv"
V14_SUMMARY = GENERATED_DIR / "reclaim_production_v14_summary.csv"
V14_AUDIT = GENERATED_DIR / "reclaim_production_v14_audit.csv"
V14_COMPARISON = GENERATED_DIR / "reclaim_production_v14_comparison.csv"

EXPECTED_POPULATION = 10_000
TARGET_CAPACITY = 500
TARGET_RATE = 0.05

# V14 deliberately uses a conservative readiness gate.
#
# No production comparison is permitted unless:
#   1. exact transaction IDs match,
#   2. both TARGET and RETRY_ALL have observations,
#   3. each group has at least MIN_GROUP_MATCHES observations.
#
# This is a readiness threshold, NOT a claim that this is statistically
# sufficient for every scientific analysis.
MIN_GROUP_MATCHES = 30

EPS = 1e-9


# ======================================================================
# UTILITIES
# ======================================================================

def fail(message: str) -> None:
    raise ValueError("[FAIL] " + message)


def pass_msg(message: str) -> None:
    print(f"[PASS] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"{label} not found: {path}")


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"{label} missing required columns: {missing}")


def numeric_series(
    df: pd.DataFrame,
    column: str,
    label: str,
) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")

    if values.isna().any():
        fail(f"{label}: non-numeric values found in `{column}`")

    if not np.isfinite(values.to_numpy(dtype=float)).all():
        fail(f"{label}: non-finite values found in `{column}`")

    return values


def normalize_transaction_id(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
    )


def derive_recovered(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Derive a strict binary observed outcome.

    Priority:
        recovered
        recovered_amount
        recovery_status

    No prediction field is accepted here.
    """

    if "recovered" in df.columns:
        recovered = pd.to_numeric(df["recovered"], errors="coerce")

        if recovered.isna().any():
            fail("Outcome `recovered` contains non-numeric values.")

        unique_values = set(recovered.astype(int).unique())

        if not unique_values.issubset({0, 1}):
            fail(
                "Outcome `recovered` must contain only 0/1 values."
            )

        return recovered.astype(int)

    if "recovered_amount" in df.columns:
        amount = pd.to_numeric(
            df["recovered_amount"],
            errors="coerce",
        )

        if amount.isna().any():
            fail(
                "Outcome `recovered_amount` contains non-numeric values."
            )

        return (amount > 0).astype(int)

    if "recovery_status" in df.columns:
        status = (
            df["recovery_status"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        recovered_values = {
            "recovered",
            "success",
            "successful",
            "paid",
            "completed",
            "true",
            "1",
        }

        return status.isin(recovered_values).astype(int)

    fail(
        "Outcome source must contain one of: "
        "`recovered`, `recovered_amount`, or `recovery_status`."
    )


def normalize_outcome_schema(
    outcomes: pd.DataFrame,
) -> pd.DataFrame:

    require_columns(
        outcomes,
        ["transaction_id"],
        "Production outcome source",
    )

    result = outcomes.copy()

    result["transaction_id"] = normalize_transaction_id(
        result["transaction_id"]
    )

    if (result["transaction_id"] == "").any():
        fail("Production outcomes contain blank transaction IDs.")

    if result["transaction_id"].duplicated().any():
        duplicate_count = int(
            result["transaction_id"].duplicated().sum()
        )

        fail(
            f"Production outcome source contains "
            f"{duplicate_count} duplicate transaction IDs."
        )

    result["recovered"] = derive_recovered(result)

    if "recovered_amount" in result.columns:
        recovered_amount = pd.to_numeric(
            result["recovered_amount"],
            errors="coerce",
        )

        if recovered_amount.isna().any():
            fail(
                "Production outcomes contain invalid "
                "`recovered_amount` values."
            )

        if (recovered_amount < -EPS).any():
            fail(
                "Production outcomes contain negative "
                "`recovered_amount` values."
            )

        result["recovered_amount"] = recovered_amount.clip(lower=0)

    elif "amount" in result.columns:
        amount = pd.to_numeric(
            result["amount"],
            errors="coerce",
        )

        if amount.isna().any():
            fail(
                "Production outcomes contain invalid `amount` values."
            )

        if (amount < -EPS).any():
            fail(
                "Production outcomes contain negative `amount` values."
            )

        result["amount"] = amount

        result["recovered_amount"] = np.where(
            result["recovered"].eq(1),
            amount,
            0.0,
        )

    else:
        fail(
            "Production outcome source requires either "
            "`recovered_amount` or `amount`."
        )

    result["recovered_amount"] = result[
        "recovered_amount"
    ].astype(float)

    # Strict consistency check:
    # recovered=0 cannot have positive recovered amount.
    inconsistent_zero = (
        result["recovered"].eq(0)
        & (result["recovered_amount"] > EPS)
    )

    if inconsistent_zero.any():
        fail(
            "Production outcome source contains rows with "
            "`recovered=0` but positive `recovered_amount`."
        )

    return result


def validate_production_population(
    cohort: pd.DataFrame,
    decisions: pd.DataFrame,
) -> None:

    require_columns(
        cohort,
        [
            "transaction_id",
            "production_decision",
            "production_intervention_selected",
        ],
        "V13 cohort",
    )

    require_columns(
        decisions,
        [
            "transaction_id",
            "production_decision",
            "production_intervention_selected",
        ],
        "V13 decisions",
    )

    cohort["transaction_id"] = normalize_transaction_id(
        cohort["transaction_id"]
    )

    decisions["transaction_id"] = normalize_transaction_id(
        decisions["transaction_id"]
    )

    if cohort["transaction_id"].duplicated().any():
        fail("V13 cohort transaction IDs are not unique.")

    if decisions["transaction_id"].duplicated().any():
        fail("V13 decisions transaction IDs are not unique.")

    if len(cohort) != EXPECTED_POPULATION:
        fail(
            f"V13 cohort has {len(cohort)} rows; "
            f"expected {EXPECTED_POPULATION}."
        )

    if len(decisions) != EXPECTED_POPULATION:
        fail(
            f"V13 decisions has {len(decisions)} rows; "
            f"expected {EXPECTED_POPULATION}."
        )

    cohort_ids = set(cohort["transaction_id"])
    decision_ids = set(decisions["transaction_id"])

    if cohort_ids != decision_ids:
        fail(
            "V13 cohort and V13 decisions do not contain "
            "the identical transaction population."
        )

    target_count = int(
        (decisions["production_decision"] == "TARGET").sum()
    )

    retry_count = int(
        (decisions["production_decision"] == "RETRY_ALL").sum()
    )

    if target_count != TARGET_CAPACITY:
        fail(
            f"TARGET population changed: {target_count}; "
            f"expected {TARGET_CAPACITY}."
        )

    if retry_count != EXPECTED_POPULATION - TARGET_CAPACITY:
        fail(
            f"RETRY_ALL population changed: {retry_count}; "
            f"expected {EXPECTED_POPULATION - TARGET_CAPACITY}."
        )

    intervention = pd.to_numeric(
        decisions["production_intervention_selected"],
        errors="coerce",
    )

    if intervention.isna().any():
        fail(
            "`production_intervention_selected` contains "
            "non-numeric values."
        )

    intervention_rate = float(intervention.mean())

    if not math.isclose(
        intervention_rate,
        TARGET_RATE,
        rel_tol=0,
        abs_tol=EPS,
    ):
        fail(
            f"Intervention rate changed: "
            f"{intervention_rate:.6f}; expected {TARGET_RATE:.6f}."
        )

    if "production_policy_type" in decisions.columns:
        policy_values = (
            decisions["production_policy_type"]
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )

        if policy_values != ["TOP_RATE"]:
            fail(
                "Production policy is not exclusively TOP_RATE: "
                f"{policy_values}"
            )

    if "production_target_rate" in decisions.columns:
        rates = numeric_series(
            decisions,
            "production_target_rate",
            "V13 decisions",
        )

        if not np.all(
            np.isclose(
                rates.to_numpy(),
                TARGET_RATE,
                rtol=0,
                atol=EPS,
            )
        ):
            fail("V13 production target rate is not 5.00%.")

    pass_msg(
        "V10.2 TOP_RATE 5% production population preserved"
    )


# ======================================================================
# OUTCOME LINKAGE
# ======================================================================

def build_outcome_linkage(
    cohort: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:

    production = cohort[
        [
            "transaction_id",
            "production_decision",
            "production_intervention_selected",
        ]
    ].copy()

    production["transaction_id"] = normalize_transaction_id(
        production["transaction_id"]
    )

    outcome_cols = [
        "transaction_id",
        "recovered",
        "recovered_amount",
    ]

    outcome_subset = outcomes[outcome_cols].copy()

    merged = production.merge(
        outcome_subset,
        on="transaction_id",
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    merged["outcome_matched"] = (
        merged["_merge"] == "both"
    ).astype(int)

    merged["observed_outcome_available"] = (
        merged["outcome_matched"] == 1
    ).astype(int)

    merged["recovered"] = pd.to_numeric(
        merged["recovered"],
        errors="coerce",
    )

    merged["recovered_amount"] = pd.to_numeric(
        merged["recovered_amount"],
        errors="coerce",
    )

    merged.drop(columns=["_merge"], inplace=True)

    return merged


def calculate_metrics(
    linkage: pd.DataFrame,
) -> dict:

    matched = linkage[
        linkage["outcome_matched"].eq(1)
    ].copy()

    target = linkage[
        linkage["production_decision"].eq("TARGET")
    ]

    retry = linkage[
        linkage["production_decision"].eq("RETRY_ALL")
    ]

    target_matched = target[
        target["outcome_matched"].eq(1)
    ]

    retry_matched = retry[
        retry["outcome_matched"].eq(1)
    ]

    matched_count = len(matched)
    target_matched_count = len(target_matched)
    retry_matched_count = len(retry_matched)

    overall_match_rate = (
        matched_count / len(linkage)
        if len(linkage)
        else 0.0
    )

    target_match_rate = (
        target_matched_count / len(target)
        if len(target)
        else 0.0
    )

    retry_match_rate = (
        retry_matched_count / len(retry)
        if len(retry)
        else 0.0
    )

    if matched_count:
        observed_recovery_rate = float(
            matched["recovered"].mean()
        )

        observed_recovered_value = float(
            matched["recovered_amount"].sum()
        )

        recovered_rows = int(
            matched["recovered"].sum()
        )
    else:
        observed_recovery_rate = np.nan
        observed_recovered_value = 0.0
        recovered_rows = 0

    if target_matched_count:
        target_recovery_rate = float(
            target_matched["recovered"].mean()
        )

        target_recovered_value = float(
            target_matched["recovered_amount"].sum()
        )
    else:
        target_recovery_rate = np.nan
        target_recovered_value = 0.0

    if retry_matched_count:
        retry_recovery_rate = float(
            retry_matched["recovered"].mean()
        )

        retry_recovered_value = float(
            retry_matched["recovered_amount"].sum()
        )
    else:
        retry_recovery_rate = np.nan
        retry_recovered_value = 0.0

    return {
        "production_transactions": len(linkage),
        "target_transactions": len(target),
        "retry_all_transactions": len(retry),
        "matched_production_outcomes": matched_count,
        "target_matched_outcomes": target_matched_count,
        "retry_all_matched_outcomes": retry_matched_count,
        "overall_match_rate": overall_match_rate,
        "target_match_rate": target_match_rate,
        "retry_all_match_rate": retry_match_rate,
        "observed_recovered_rows": recovered_rows,
        "observed_recovery_rate": observed_recovery_rate,
        "observed_recovered_value": observed_recovered_value,
        "target_observed_recovery_rate": target_recovery_rate,
        "target_observed_recovered_value": target_recovered_value,
        "retry_all_observed_recovery_rate": retry_recovery_rate,
        "retry_all_observed_recovered_value": retry_recovered_value,
    }


# ======================================================================
# READINESS GATE
# ======================================================================

def determine_evaluation_readiness(
    metrics: dict,
) -> tuple[str, str]:

    matched = metrics["matched_production_outcomes"]
    target_matched = metrics["target_matched_outcomes"]
    retry_matched = metrics["retry_all_matched_outcomes"]

    if matched == 0:
        return (
            "OUTCOME_CAPTURE_INSUFFICIENT_NO_PRODUCTION_MATCH",
            "No legitimate production transaction IDs matched "
            "the immutable V10.2 cohort.",
        )

    if target_matched == 0:
        return (
            "OUTCOME_CAPTURE_INSUFFICIENT_NO_TARGET_MATCH",
            "Matched production outcomes exist, but none belong "
            "to the TARGET cohort.",
        )

    if retry_matched == 0:
        return (
            "OUTCOME_CAPTURE_INSUFFICIENT_NO_RETRY_ALL_MATCH",
            "Matched production outcomes exist, but none belong "
            "to the RETRY_ALL cohort.",
        )

    if target_matched < MIN_GROUP_MATCHES:
        return (
            "OUTCOME_CAPTURE_INSUFFICIENT_TARGET_COHORT",
            f"Only {target_matched} TARGET outcomes matched; "
            f"minimum readiness threshold is {MIN_GROUP_MATCHES}.",
        )

    if retry_matched < MIN_GROUP_MATCHES:
        return (
            "OUTCOME_CAPTURE_INSUFFICIENT_RETRY_ALL_COHORT",
            f"Only {retry_matched} RETRY_ALL outcomes matched; "
            f"minimum readiness threshold is {MIN_GROUP_MATCHES}.",
        )

    return (
        "PRODUCTION_OUTCOME_EVALUATION_READY",
        "Both production cohorts have sufficient exact-ID "
        "outcome coverage for controlled evaluation.",
    )


# ======================================================================
# V7 / V8 COMPARISON
# ======================================================================

def build_comparison(
    linkage: pd.DataFrame,
    readiness: str,
) -> pd.DataFrame:

    rows = []

    target = linkage[
        linkage["production_decision"].eq("TARGET")
        & linkage["outcome_matched"].eq(1)
    ]

    retry = linkage[
        linkage["production_decision"].eq("RETRY_ALL")
        & linkage["outcome_matched"].eq(1)
    ]

    if readiness != "PRODUCTION_OUTCOME_EVALUATION_READY":
        rows.append(
            {
                "evaluation_status": readiness,
                "comparison": "V7_vs_V8",
                "target_matched": len(target),
                "retry_all_matched": len(retry),
                "target_recovery_rate": np.nan,
                "retry_all_recovery_rate": np.nan,
                "recovery_rate_difference": np.nan,
                "target_recovered_value": 0.0,
                "retry_all_recovered_value": 0.0,
                "financial_improvement_claim": "NO",
            }
        )

        return pd.DataFrame(rows)

    target_rate = float(target["recovered"].mean())
    retry_rate = float(retry["recovered"].mean())

    rows.append(
        {
            "evaluation_status": readiness,
            "comparison": "TARGET_vs_RETRY_ALL_observed_outcomes",
            "target_matched": len(target),
            "retry_all_matched": len(retry),
            "target_recovery_rate": target_rate,
            "retry_all_recovery_rate": retry_rate,
            "recovery_rate_difference": target_rate - retry_rate,
            "target_recovered_value": float(
                target["recovered_amount"].sum()
            ),
            "retry_all_recovered_value": float(
                retry["recovered_amount"].sum()
            ),
            "financial_improvement_claim": "NO",
        }
    )

    return pd.DataFrame(rows)


# ======================================================================
# AUDIT
# ======================================================================

def build_audit(
    metrics: dict,
    readiness: str,
) -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "audit_item": "V10.2 targeting immutable",
                "status": "PASS",
                "detail": "5% TOP_RATE population preserved.",
            },
            {
                "audit_item": "Artificial transaction IDs",
                "status": "PASS",
                "detail": "No artificial transaction IDs created.",
            },
            {
                "audit_item": "Artificial outcomes",
                "status": "PASS",
                "detail": "No artificial production outcomes created.",
            },
            {
                "audit_item": "Calibration predictions",
                "status": "PASS",
                "detail": (
                    "Calibration predictions are not treated "
                    "as observed production outcomes."
                ),
            },
            {
                "audit_item": "Exact transaction linkage",
                "status": "PASS",
                "detail": (
                    "Production outcomes are linked only by exact "
                    "transaction_id."
                ),
            },
            {
                "audit_item": "Matched production outcomes",
                "status": "INFO",
                "detail": str(
                    metrics["matched_production_outcomes"]
                ),
            },
            {
                "audit_item": "Evaluation readiness",
                "status": (
                    "PASS"
                    if readiness
                    == "PRODUCTION_OUTCOME_EVALUATION_READY"
                    else "WAIT"
                ),
                "detail": readiness,
            },
            {
                "audit_item": "V7 vs V8 improvement claim",
                "status": "PASS",
                "detail": (
                    "NO CLAIM until legitimate production "
                    "outcome evaluation is ready."
                ),
            },
        ]
    )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    print("=" * 72)
    print("RECLAIM PRODUCTION PIPELINE V14")
    print("Production Outcome Ingestion + Evaluation Readiness")
    print("V10.2 TOP_RATE 5% targeting is IMMUTABLE")
    print("=" * 72)

    # --------------------------------------------------------------
    # LOAD V13
    # --------------------------------------------------------------

    section("LOADING IMMUTABLE V13 PRODUCTION COHORT")

    require_file(V13_COHORT, "V13 cohort")
    require_file(V13_DECISIONS, "V13 decisions")

    cohort = pd.read_csv(V13_COHORT)
    decisions = pd.read_csv(V13_DECISIONS)

    print(
        f"[PASS] V13 cohort loaded: "
        f"{len(cohort):,} rows, {len(cohort.columns)} columns"
    )

    print(
        f"[PASS] V13 decisions loaded: "
        f"{len(decisions):,} rows, {len(decisions.columns)} columns"
    )

    # --------------------------------------------------------------
    # POPULATION VALIDATION
    # --------------------------------------------------------------

    section("VALIDATING IMMUTABLE PRODUCTION POPULATION")

    validate_production_population(
        cohort,
        decisions,
    )

    pass_msg(
        f"Population: {len(cohort):,}"
    )

    # --------------------------------------------------------------
    # LOAD LEGITIMATE OUTCOMES
    # --------------------------------------------------------------

    section("LOADING LEGITIMATE PRODUCTION OUTCOMES")

    require_file(
        OUTCOME_SOURCE,
        "Legitimate production outcome source",
    )

    raw_outcomes = pd.read_csv(OUTCOME_SOURCE)

    print(
        f"[PASS] Outcome source loaded: "
        f"{len(raw_outcomes):,} rows, "
        f"{len(raw_outcomes.columns)} columns"
    )

    outcomes = normalize_outcome_schema(
        raw_outcomes
    )

    pass_msg(
        "Legitimate production outcome schema normalized"
    )

    print(
        f"[PASS] Normalized outcomes: "
        f"{len(outcomes):,}"
    )

    # --------------------------------------------------------------
    # LINKAGE
    # --------------------------------------------------------------

    section("BUILDING EXACT TRANSACTION-ID OUTCOME LINKAGE")

    linkage = build_outcome_linkage(
        cohort,
        outcomes,
    )

    matched_count = int(
        linkage["outcome_matched"].sum()
    )

    if matched_count == 0:
        warn(
            "Zero production transaction IDs have "
            "legitimate observed outcomes."
        )
    else:
        pass_msg(
            f"Matched legitimate production outcomes: "
            f"{matched_count:,}"
        )

    # --------------------------------------------------------------
    # METRICS
    # --------------------------------------------------------------

    section("CALCULATING PRODUCTION OUTCOME METRICS")

    metrics = calculate_metrics(linkage)

    print(
        f"Production transactions:      "
        f"{metrics['production_transactions']:,}"
    )

    print(
        f"TARGET transactions:          "
        f"{metrics['target_transactions']:,}"
    )

    print(
        f"RETRY_ALL transactions:       "
        f"{metrics['retry_all_transactions']:,}"
    )

    print(
        f"Matched production outcomes:  "
        f"{metrics['matched_production_outcomes']:,}"
    )

    print(
        f"TARGET matched outcomes:      "
        f"{metrics['target_matched_outcomes']:,}"
    )

    print(
        f"RETRY_ALL matched outcomes:   "
        f"{metrics['retry_all_matched_outcomes']:,}"
    )

    print(
        f"Overall match rate:            "
        f"{metrics['overall_match_rate']:.2%}"
    )

    print(
        f"TARGET match rate:             "
        f"{metrics['target_match_rate']:.2%}"
    )

    print(
        f"RETRY_ALL match rate:          "
        f"{metrics['retry_all_match_rate']:.2%}"
    )

    if np.isnan(metrics["observed_recovery_rate"]):
        print(
            "Observed recovery rate:       N/A"
        )
    else:
        print(
            f"Observed recovery rate:       "
            f"{metrics['observed_recovery_rate']:.2%}"
        )

    print(
        f"Observed recovered value:      "
        f"₹{metrics['observed_recovered_value']:,.2f}"
    )

    # --------------------------------------------------------------
    # READINESS
    # --------------------------------------------------------------

    section("EVALUATION READINESS GATE")

    readiness, readiness_reason = (
        determine_evaluation_readiness(metrics)
    )

    if readiness == "PRODUCTION_OUTCOME_EVALUATION_READY":
        pass_msg(readiness_reason)
    else:
        warn(readiness_reason)

    # --------------------------------------------------------------
    # CONTROLLED COMPARISON
    # --------------------------------------------------------------

    section("BUILDING CONTROLLED V7 / V8 COMPARISON")

    comparison = build_comparison(
        linkage,
        readiness,
    )

    if readiness == "PRODUCTION_OUTCOME_EVALUATION_READY":
        pass_msg(
            "Production outcome coverage is sufficient "
            "for controlled comparison."
        )
    else:
        warn(
            "V7 vs V8 production improvement is NOT claimed."
        )

    # --------------------------------------------------------------
    # OUTPUT DATASETS
    # --------------------------------------------------------------

    section("BUILDING V14 OUTPUTS")

    decisions_v14 = decisions.copy()

    # Add linkage information without changing production decisions.
    linkage_for_merge = linkage[
        [
            "transaction_id",
            "outcome_matched",
            "observed_outcome_available",
            "recovered",
            "recovered_amount",
        ]
    ].copy()

    decisions_v14 = decisions_v14.merge(
        linkage_for_merge,
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    cohort_v14 = cohort.copy()

    cohort_v14 = cohort_v14.merge(
        linkage_for_merge,
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    capture = linkage.copy()

    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "pipeline_version": "V14",
                "production_transactions": metrics[
                    "production_transactions"
                ],
                "target_transactions": metrics[
                    "target_transactions"
                ],
                "retry_all_transactions": metrics[
                    "retry_all_transactions"
                ],
                "target_rate": TARGET_RATE,
                "matched_production_outcomes": metrics[
                    "matched_production_outcomes"
                ],
                "target_matched_outcomes": metrics[
                    "target_matched_outcomes"
                ],
                "retry_all_matched_outcomes": metrics[
                    "retry_all_matched_outcomes"
                ],
                "overall_match_rate": metrics[
                    "overall_match_rate"
                ],
                "target_match_rate": metrics[
                    "target_match_rate"
                ],
                "retry_all_match_rate": metrics[
                    "retry_all_match_rate"
                ],
                "observed_recovered_rows": metrics[
                    "observed_recovered_rows"
                ],
                "observed_recovery_rate": metrics[
                    "observed_recovery_rate"
                ],
                "observed_recovered_value": metrics[
                    "observed_recovered_value"
                ],
                "target_observed_recovery_rate": metrics[
                    "target_observed_recovery_rate"
                ],
                "retry_all_observed_recovery_rate": metrics[
                    "retry_all_observed_recovery_rate"
                ],
                "readiness_verdict": readiness,
                "readiness_reason": readiness_reason,
                "targeting_changed": "NO",
                "retry_all_changed": "NO",
                "oracle_fields_used": "NO",
                "artificial_ids_created": "NO",
                "artificial_outcomes_created": "NO",
                "calibration_predictions_used_as_outcomes": "NO",
                "financial_improvement_claim": "NO",
            }
        ]
    )

    audit = build_audit(
        metrics,
        readiness,
    )

    # --------------------------------------------------------------
    # WRITE
    # --------------------------------------------------------------

    decisions_v14.to_csv(
        V14_DECISIONS,
        index=False,
    )

    cohort_v14.to_csv(
        V14_COHORT,
        index=False,
    )

    capture.to_csv(
        V14_CAPTURE,
        index=False,
    )

    summary.to_csv(
        V14_SUMMARY,
        index=False,
    )

    audit.to_csv(
        V14_AUDIT,
        index=False,
    )

    comparison.to_csv(
        V14_COMPARISON,
        index=False,
    )

    pass_msg(
        f"Decisions:  {V14_DECISIONS}"
    )

    pass_msg(
        f"Cohort:     {V14_COHORT}"
    )

    pass_msg(
        f"Capture:    {V14_CAPTURE}"
    )

    pass_msg(
        f"Summary:    {V14_SUMMARY}"
    )

    pass_msg(
        f"Audit:      {V14_AUDIT}"
    )

    pass_msg(
        f"Comparison: {V14_COMPARISON}"
    )

    # --------------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------------

    section("FINAL V14 VERDICT")

    print(
        f"FINAL VERDICT: {readiness}"
    )

    print()
    print(
        "V10.2 TOP_RATE 5% targeting remains untouched."
    )

    print(
        "No artificial transaction IDs were created."
    )

    print(
        "No artificial production outcomes were created."
    )

    print(
        "Calibration predictions are not treated as "
        "production outcomes."
    )

    if readiness == "PRODUCTION_OUTCOME_EVALUATION_READY":
        print()
        print(
            "NEXT STEP:"
        )
        print(
            "Run the controlled V7 vs V8 outcome comparison "
            "using only exact-ID matched production outcomes."
        )
    else:
        print()
        print(
            "NEXT REQUIRED STEP:"
        )
        print(
            "Capture legitimate observed outcomes whose "
            "`transaction_id` exactly matches the immutable "
            "V10.2 production cohort."
        )

        print(
            "V7 vs V8 improvement must remain unclaimed until "
            "sufficient matched production outcomes exist."
        )

    print()
    print("=" * 72)
    print("RECLAIM PRODUCTION PIPELINE V14 COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 72)
        print("V14 FAILED")
        print("=" * 72)
        print(str(exc))
        sys.exit(1)

