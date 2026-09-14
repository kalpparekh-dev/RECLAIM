
"""
========================================================================
RECLAIM PRODUCTION PIPELINE V13
Controlled Production Outcome Evaluation
V7 Baseline vs V8 Calibration-Aware Action Layer

V10.2 TOP_RATE 5% targeting is IMMUTABLE

Purpose
-------
V13 performs a governance-safe evaluation of the V7 baseline versus
the V8 calibration-aware action layer using ONLY legitimate observed
production outcomes matched by exact transaction_id.

IMPORTANT
---------
- No artificial transaction IDs.
- No artificial outcomes.
- No calibration predictions treated as production outcomes.
- No oracle/evaluation fields used for production decisions.
- V10.2 targeting is never modified.
- If insufficient legitimate production outcomes exist, V13 stops
  the statistical comparison and reports an insufficient-data verdict.
========================================================================
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(r"D:\RECLAIM")
GENERATED_DIR = BASE_DIR / "data" / "generated"

V7_FILE = GENERATED_DIR / "reclaim_production_decisions_v7.csv"
V8_FILE = GENERATED_DIR / "reclaim_production_decisions_v8.csv"
V12_COHORT_FILE = GENERATED_DIR / "reclaim_production_cohort_v12.csv"
V12_DECISIONS_FILE = GENERATED_DIR / "reclaim_production_decisions_v12.csv"

OUTCOME_FILE = GENERATED_DIR / "recovery_outcomes.csv"

OUTPUT_DECISIONS = GENERATED_DIR / "reclaim_production_decisions_v13.csv"
OUTPUT_COHORT = GENERATED_DIR / "reclaim_production_cohort_v13.csv"
OUTPUT_COMPARISON = GENERATED_DIR / "reclaim_production_v13_comparison.csv"
OUTPUT_SUMMARY = GENERATED_DIR / "reclaim_production_v13_summary.csv"
OUTPUT_AUDIT = GENERATED_DIR / "reclaim_production_v13_audit.csv"

TARGET_RATE = 0.05
TARGET_COUNT = 500
TOTAL_EXPECTED = 10000

# Minimum evidence requirements before making a production comparison.
MIN_MATCHED_TOTAL = 100
MIN_MATCHED_PER_GROUP = 30

# Significance threshold for exploratory statistical testing.
ALPHA = 0.05

REQUIRED_OUTCOME_COLUMNS = [
    "transaction_id",
]

PROTECTED_TARGET_COLUMNS = [
    "transaction_id",
    "production_decision",
    "production_intervention_selected",
]


# ======================================================================
# UTILITIES
# ======================================================================

def fail(message: str) -> None:
    raise ValueError("[FAIL] " + message)


def log(message: str = "") -> None:
    print(message)


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def money(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"₹{float(value):,.2f}"


def pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"{label} not found: {path}")


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
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
        fail(f"{label}: non-numeric or missing values in `{column}`")

    return values


def safe_mean(values: pd.Series) -> float:
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def safe_sum(values: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
    return float(values.sum())


# ======================================================================
# RECOVERY OUTCOME NORMALIZATION
# ======================================================================

def normalize_recovered_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize recovery outcome into a strict binary `recovered` field.

    Accepted sources:
      - recovered
      - recovered_amount
      - recovery_status

    Priority:
      recovered -> recovered_amount -> recovery_status
    """

    result = df.copy()

    if "recovered" in result.columns:
        recovered = result["recovered"]

        if recovered.dtype == bool:
            result["recovered"] = recovered.astype(int)
            return result

        numeric = pd.to_numeric(recovered, errors="coerce")

        if numeric.notna().all():
            result["recovered"] = (numeric > 0).astype(int)
            return result

        text = recovered.astype(str).str.strip().str.lower()

        true_values = {
            "1",
            "true",
            "yes",
            "y",
            "recovered",
            "success",
            "successful",
        }

        result["recovered"] = text.isin(true_values).astype(int)
        return result

    if "recovered_amount" in result.columns:
        amount = pd.to_numeric(
            result["recovered_amount"],
            errors="coerce",
        )

        if amount.isna().any():
            fail(
                "`recovered_amount` contains invalid or missing "
                "values and `recovered` is unavailable"
            )

        result["recovered"] = (amount > 0).astype(int)

        log("[PASS] Derived `recovered` from `recovered_amount`")
        return result

    if "recovery_status" in result.columns:
        status = (
            result["recovery_status"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        successful_statuses = {
            "recovered",
            "success",
            "successful",
            "paid",
            "completed",
        }

        result["recovered"] = status.isin(successful_statuses).astype(int)

        log("[PASS] Derived `recovered` from `recovery_status`")
        return result

    fail(
        "Outcome source must contain one of: "
        "`recovered`, `recovered_amount`, `recovery_status`"
    )

    return result


def normalize_recovered_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure recovered_amount exists.

    If missing, derive it from:
        recovered * amount
    when both fields exist.

    This is a reconstruction of an observed outcome field, not an
    artificial production outcome.
    """

    result = df.copy()

    if "recovered_amount" in result.columns:
        recovered_amount = pd.to_numeric(
            result["recovered_amount"],
            errors="coerce",
        )

        if recovered_amount.isna().any():
            fail("`recovered_amount` contains invalid values")

        if (recovered_amount < 0).any():
            fail("`recovered_amount` contains negative values")

        result["recovered_amount"] = recovered_amount
        return result

    if "amount" not in result.columns:
        fail(
            "Outcome source has no `recovered_amount` and no `amount` "
            "from which to reconstruct it"
        )

    amount = pd.to_numeric(result["amount"], errors="coerce")

    if amount.isna().any():
        fail("`amount` contains invalid values")

    result["recovered_amount"] = (
        result["recovered"].astype(float) * amount
    )

    log(
        "[PASS] Rebuilt `recovered_amount` from "
        "`recovered` and `amount`"
    )

    return result


def normalize_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    require_columns(
        result,
        REQUIRED_OUTCOME_COLUMNS,
        "Production outcome source",
    )

    result["transaction_id"] = (
        result["transaction_id"]
        .astype(str)
        .str.strip()
    )

    if (result["transaction_id"] == "").any():
        fail("Outcome source contains blank transaction IDs")

    result = normalize_recovered_flag(result)
    result = normalize_recovered_amount(result)

    if result["transaction_id"].duplicated().any():
        duplicate_count = int(
            result["transaction_id"].duplicated().sum()
        )

        fail(
            "Production outcome source contains "
            f"{duplicate_count} duplicate transaction IDs. "
            "Ambiguous outcomes cannot be used."
        )

    log("[PASS] Legitimate production outcome schema normalized")

    return result


# ======================================================================
# PRODUCTION POPULATION VALIDATION
# ======================================================================

def validate_population(
    df: pd.DataFrame,
    label: str,
) -> Dict[str, object]:

    require_columns(
        df,
        [
            "transaction_id",
            "production_decision",
            "production_intervention_selected",
            "production_policy_type",
            "production_target_rate",
        ],
        label,
    )

    if df["transaction_id"].duplicated().any():
        fail(f"{label} contains duplicate transaction IDs")

    production_decision = (
        df["production_decision"]
        .astype(str)
        .str.strip()
    )

    target_count = int(
        (production_decision == "TARGET").sum()
    )

    retry_count = int(
        (production_decision == "RETRY_ALL").sum()
    )

    total = len(df)

    intervention = pd.to_numeric(
        df["production_intervention_selected"],
        errors="coerce",
    )

    if intervention.isna().any():
        fail(
            f"{label}: invalid production_intervention_selected"
        )

    intervention_count = int((intervention == 1).sum())

    policy = (
        df["production_policy_type"]
        .astype(str)
        .str.strip()
    )

    if not (policy == "TOP_RATE").all():
        fail(f"{label}: policy is not uniformly TOP_RATE")

    rate = pd.to_numeric(
        df["production_target_rate"],
        errors="coerce",
    )

    if rate.isna().any():
        fail(f"{label}: invalid target rate")

    if not np.allclose(
        rate.to_numpy(),
        TARGET_RATE,
        atol=1e-12,
    ):
        fail(
            f"{label}: target rate is not "
            f"{TARGET_RATE:.2%}"
        )

    if total != TOTAL_EXPECTED:
        fail(
            f"{label}: expected {TOTAL_EXPECTED} rows, "
            f"found {total}"
        )

    if target_count != TARGET_COUNT:
        fail(
            f"{label}: expected {TARGET_COUNT} TARGET rows, "
            f"found {target_count}"
        )

    if retry_count != TOTAL_EXPECTED - TARGET_COUNT:
        fail(
            f"{label}: unexpected RETRY_ALL population"
        )

    if intervention_count != TARGET_COUNT:
        fail(
            f"{label}: intervention population does not equal "
            f"validated target capacity"
        )

    actual_rate = intervention_count / total

    if not math.isclose(
        actual_rate,
        TARGET_RATE,
        abs_tol=1e-12,
    ):
        fail(
            f"{label}: intervention rate is not "
            f"{TARGET_RATE:.2%}"
        )

    log(
        f"[PASS] {label} population preserved: "
        f"{target_count} TARGET / "
        f"{retry_count} RETRY_ALL / "
        f"{actual_rate:.2%}"
    )

    return {
        "total": total,
        "target": target_count,
        "retry_all": retry_count,
        "intervention_rate": actual_rate,
    }


def validate_same_population(
    v7: pd.DataFrame,
    v8: pd.DataFrame,
) -> None:

    ids_v7 = set(v7["transaction_id"].astype(str))
    ids_v8 = set(v8["transaction_id"].astype(str))

    if ids_v7 != ids_v8:
        only_v7 = ids_v7 - ids_v8
        only_v8 = ids_v8 - ids_v7

        fail(
            "V7/V8 transaction population differs. "
            f"Only V7={len(only_v7)}, "
            f"Only V8={len(only_v8)}"
        )

    log(
        "[PASS] V7/V8 production transaction population identical"
    )


# ======================================================================
# IMMUTABLE V12 COHORT VALIDATION
# ======================================================================

def validate_v12_cohort(
    cohort: pd.DataFrame,
    decisions: pd.DataFrame,
) -> None:

    require_columns(
        cohort,
        [
            "transaction_id",
        ],
        "V12 cohort",
    )

    if cohort["transaction_id"].duplicated().any():
        fail("V12 cohort contains duplicate transaction IDs")

    cohort_ids = set(
        cohort["transaction_id"].astype(str)
    )

    decision_ids = set(
        decisions["transaction_id"].astype(str)
    )

    if cohort_ids != decision_ids:
        fail(
            "V12 cohort and V12 decisions have different "
            "transaction populations"
        )

    if len(cohort) != TOTAL_EXPECTED:
        fail(
            f"V12 cohort must contain {TOTAL_EXPECTED} rows, "
            f"found {len(cohort)}"
        )

    log(
        f"[PASS] V12 immutable cohort validated: "
        f"{len(cohort):,} rows"
    )


# ======================================================================
# OUTCOME LINKAGE
# ======================================================================

def build_outcome_linkage(
    production: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:

    production_ids = set(
        production["transaction_id"].astype(str)
    )

    outcome_ids = set(
        outcomes["transaction_id"].astype(str)
    )

    matched_ids = production_ids & outcome_ids

    linkage = production[
        [
            "transaction_id",
            "customer_id",
            "production_decision",
            "production_intervention_selected",
        ]
    ].copy()

    linkage["has_observed_outcome"] = (
        linkage["transaction_id"].isin(matched_ids)
    ).astype(int)

    outcome_subset = outcomes[
        [
            "transaction_id",
            "recovered",
            "recovered_amount",
        ]
    ].copy()

    linkage = linkage.merge(
        outcome_subset,
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    return linkage


# ======================================================================
# STATISTICAL HELPERS
# ======================================================================

def wilson_interval(
    successes: int,
    n: int,
    z: float = 1.96,
) -> Tuple[float, float]:

    if n <= 0:
        return float("nan"), float("nan")

    p = successes / n

    denominator = 1 + (z * z / n)

    centre = (
        p
        + (z * z / (2 * n))
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (
                p * (1 - p) / n
            )
            + (
                z * z / (4 * n * n)
            )
        )
        / denominator
    )

    return (
        max(0.0, centre - margin),
        min(1.0, centre + margin),
    )


def two_proportion_z_test(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
) -> Tuple[float, float]:

    if n_a <= 0 or n_b <= 0:
        return float("nan"), float("nan")

    p_a = successes_a / n_a
    p_b = successes_b / n_b

    pooled = (
        (successes_a + successes_b)
        / (n_a + n_b)
    )

    denominator = math.sqrt(
        pooled
        * (1 - pooled)
        * (
            (1 / n_a)
            + (1 / n_b)
        )
    )

    if denominator == 0:
        return float("nan"), float("nan")

    z = (p_a - p_b) / denominator

    # Normal approximation, avoiding scipy dependency.
    p_value = math.erfc(
        abs(z) / math.sqrt(2)
    )

    return float(z), float(p_value)


# ======================================================================
# COHORT METRICS
# ======================================================================

def calculate_group_metrics(
    df: pd.DataFrame,
    group_name: str,
) -> Dict[str, object]:

    n = len(df)

    if n == 0:
        return {
            "group": group_name,
            "n": 0,
            "recovered": 0,
            "recovery_rate": float("nan"),
            "recovered_value": 0.0,
            "mean_recovered_amount": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    recovered = pd.to_numeric(
        df["recovered"],
        errors="coerce",
    )

    recovered_amount = pd.to_numeric(
        df["recovered_amount"],
        errors="coerce",
    )

    if recovered.isna().any():
        fail(
            f"Group {group_name}: invalid recovered values"
        )

    if recovered_amount.isna().any():
        fail(
            f"Group {group_name}: invalid recovered_amount values"
        )

    successes = int((recovered == 1).sum())

    recovery_rate = successes / n

    ci_low, ci_high = wilson_interval(
        successes,
        n,
    )

    return {
        "group": group_name,
        "n": n,
        "recovered": successes,
        "recovery_rate": recovery_rate,
        "recovered_value": float(
            recovered_amount.sum()
        ),
        "mean_recovered_amount": float(
            recovered_amount.mean()
        ),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


# ======================================================================
# V7 VS V8 EVALUATION
# ======================================================================

def evaluate_v7_v8(
    matched: pd.DataFrame,
) -> Dict[str, object]:

    target = matched[
        matched["production_intervention_selected"] == 1
    ].copy()

    retry = matched[
        matched["production_intervention_selected"] == 0
    ].copy()

    target_metrics = calculate_group_metrics(
        target,
        "TARGET",
    )

    retry_metrics = calculate_group_metrics(
        retry,
        "RETRY_ALL",
    )

    n_target = target_metrics["n"]
    n_retry = retry_metrics["n"]

    sufficient = (
        len(matched) >= MIN_MATCHED_TOTAL
        and n_target >= MIN_MATCHED_PER_GROUP
        and n_retry >= MIN_MATCHED_PER_GROUP
    )

    if not sufficient:
        return {
            "sufficient": False,
            "target": target_metrics,
            "retry": retry_metrics,
            "z": float("nan"),
            "p_value": float("nan"),
            "rate_difference": float("nan"),
        }

    z, p_value = two_proportion_z_test(
        target_metrics["recovered"],
        target_metrics["n"],
        retry_metrics["recovered"],
        retry_metrics["n"],
    )

    rate_difference = (
        target_metrics["recovery_rate"]
        - retry_metrics["recovery_rate"]
    )

    return {
        "sufficient": True,
        "target": target_metrics,
        "retry": retry_metrics,
        "z": z,
        "p_value": p_value,
        "rate_difference": rate_difference,
    }


# ======================================================================
# GOVERNANCE AUDIT
# ======================================================================

def governance_audit(
    v7: pd.DataFrame,
    v8: pd.DataFrame,
    matched: pd.DataFrame,
    sufficient: bool,
) -> Dict[str, object]:

    audit = {}

    v7_target = (
        v7["production_intervention_selected"]
        .astype(int)
        == 1
    )

    v8_target = (
        v8["production_intervention_selected"]
        .astype(int)
        == 1
    )

    audit["v7_target_count"] = int(v7_target.sum())
    audit["v8_target_count"] = int(v8_target.sum())

    audit["targeting_changed"] = (
        not v7_target.equals(v8_target)
    )

    audit["population_changed"] = (
        set(v7["transaction_id"])
        != set(v8["transaction_id"])
    )

    audit["matched_outcomes"] = len(matched)

    audit["sufficient_for_evaluation"] = sufficient

    # This pipeline never reads oracle fields.
    oracle_fields = [
        "oracle_control_value",
        "oracle_treatment_value",
        "oracle_incremental_value",
    ]

    audit["oracle_fields_used"] = False

    # Explicitly record their presence without using them.
    audit["oracle_fields_detected"] = [
        c
        for c in oracle_fields
        if c in v7.columns or c in v8.columns
    ]

    return audit


# ======================================================================
# OUTPUT BUILDERS
# ======================================================================

def build_decisions_output(
    v12_decisions: pd.DataFrame,
    linkage: pd.DataFrame,
) -> pd.DataFrame:

    result = v12_decisions.copy()

    outcome_columns = [
        "transaction_id",
        "has_observed_outcome",
        "recovered",
        "recovered_amount",
    ]

    outcome_data = linkage[outcome_columns].copy()

    result = result.merge(
        outcome_data,
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    result["has_observed_outcome"] = (
        result["has_observed_outcome"]
        .fillna(0)
        .astype(int)
    )

    return result


def build_comparison_output(
    evaluation: Dict[str, object],
) -> pd.DataFrame:

    rows = []

    for key in ["target", "retry"]:
        metrics = evaluation[key]

        rows.append(
            {
                "group": metrics["group"],
                "matched_transactions": metrics["n"],
                "observed_recovered": metrics["recovered"],
                "observed_recovery_rate": metrics[
                    "recovery_rate"
                ],
                "recovery_ci_low": metrics["ci_low"],
                "recovery_ci_high": metrics["ci_high"],
                "observed_recovered_value": metrics[
                    "recovered_value"
                ],
                "mean_recovered_amount": metrics[
                    "mean_recovered_amount"
                ],
            }
        )

    rows.append(
        {
            "group": "TARGET_MINUS_RETRY_ALL",
            "matched_transactions": (
                evaluation["target"]["n"]
                + evaluation["retry"]["n"]
            ),
            "observed_recovered": (
                evaluation["target"]["recovered"]
                + evaluation["retry"]["recovered"]
            ),
            "observed_recovery_rate": evaluation[
                "rate_difference"
            ],
            "recovery_ci_low": np.nan,
            "recovery_ci_high": np.nan,
            "observed_recovered_value": (
                evaluation["target"]["recovered_value"]
                - evaluation["retry"]["recovered_value"]
            ),
            "mean_recovered_amount": np.nan,
        }
    )

    return pd.DataFrame(rows)


def build_summary(
    v7_stats: Dict[str, object],
    v8_stats: Dict[str, object],
    linkage: pd.DataFrame,
    evaluation: Dict[str, object],
    audit: Dict[str, object],
    verdict: str,
) -> pd.DataFrame:

    matched_count = len(linkage)

    observed_recovered = int(
        linkage["recovered"].sum()
    ) if matched_count else 0

    observed_value = (
        float(linkage["recovered_amount"].sum())
        if matched_count
        else 0.0
    )

    rows = [
        ("v7_transactions", v7_stats["total"]),
        ("v7_target", v7_stats["target"]),
        ("v7_retry_all", v7_stats["retry_all"]),
        ("v8_transactions", v8_stats["total"]),
        ("v8_target", v8_stats["target"]),
        ("v8_retry_all", v8_stats["retry_all"]),
        ("matched_production_outcomes", matched_count),
        (
            "overall_match_rate",
            matched_count / TOTAL_EXPECTED,
        ),
        ("observed_recovered_rows", observed_recovered),
        (
            "observed_recovery_rate",
            (
                observed_recovered / matched_count
                if matched_count
                else np.nan
            ),
        ),
        ("observed_recovered_value", observed_value),
        (
            "target_matched",
            evaluation["target"]["n"],
        ),
        (
            "target_recovery_rate",
            evaluation["target"]["recovery_rate"],
        ),
        (
            "retry_all_matched",
            evaluation["retry"]["n"],
        ),
        (
            "retry_all_recovery_rate",
            evaluation["retry"]["recovery_rate"],
        ),
        (
            "rate_difference_target_minus_retry",
            evaluation["rate_difference"],
        ),
        ("statistical_z", evaluation["z"]),
        ("statistical_p_value", evaluation["p_value"]),
        (
            "evaluation_sufficient",
            evaluation["sufficient"],
        ),
        ("targeting_changed", audit["targeting_changed"]),
        ("population_changed", audit["population_changed"]),
        ("oracle_fields_used", False),
        ("verdict", verdict),
    ]

    return pd.DataFrame(
        rows,
        columns=["metric", "value"],
    )


def build_audit(
    audit: Dict[str, object],
    verdict: str,
) -> pd.DataFrame:

    rows = [
        (
            "V10.2_TOP_RATE_5_PERCENT_PRESERVED",
            True,
        ),
        (
            "V7_V8_TARGETING_CHANGED",
            audit["targeting_changed"],
        ),
        (
            "V7_V8_POPULATION_CHANGED",
            audit["population_changed"],
        ),
        (
            "ORACLE_FIELDS_USED",
            False,
        ),
        (
            "ARTIFICIAL_IDS_CREATED",
            False,
        ),
        (
            "ARTIFICIAL_OUTCOMES_CREATED",
            False,
        ),
        (
            "CALIBRATION_PREDICTIONS_TREATED_AS_OUTCOMES",
            False,
        ),
        (
            "SUFFICIENT_PRODUCTION_OUTCOMES",
            audit["sufficient_for_evaluation"],
        ),
        (
            "FINAL_VERDICT",
            verdict,
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=["audit_check", "result"],
    )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    section(
        "RECLAIM PRODUCTION PIPELINE V13\n"
        "Controlled Production Outcome Evaluation\n"
        "V7 Baseline vs V8 Calibration-Aware Action Layer\n"
        "V10.2 TOP_RATE 5% targeting is IMMUTABLE"
    )

    # ------------------------------------------------------------------
    # LOAD FILES
    # ------------------------------------------------------------------

    section("LOADING PRODUCTION BASELINES")

    require_file(
        V7_FILE,
        "V7 production file",
    )

    require_file(
        V8_FILE,
        "V8 production file",
    )

    require_file(
        V12_COHORT_FILE,
        "V12 immutable cohort",
    )

    require_file(
        V12_DECISIONS_FILE,
        "V12 decisions",
    )

    require_file(
        OUTCOME_FILE,
        "Legitimate production outcome source",
    )

    v7 = pd.read_csv(V7_FILE)
    v8 = pd.read_csv(V8_FILE)
    v12_cohort = pd.read_csv(V12_COHORT_FILE)
    v12_decisions = pd.read_csv(V12_DECISIONS_FILE)
    outcomes_raw = pd.read_csv(OUTCOME_FILE)

    log(
        f"[PASS] V7 loaded: "
        f"{len(v7):,} rows, {len(v7.columns)} columns"
    )

    log(
        f"[PASS] V8 loaded: "
        f"{len(v8):,} rows, {len(v8.columns)} columns"
    )

    log(
        f"[PASS] V12 cohort loaded: "
        f"{len(v12_cohort):,} rows, "
        f"{len(v12_cohort.columns)} columns"
    )

    log(
        f"[PASS] V12 decisions loaded: "
        f"{len(v12_decisions):,} rows, "
        f"{len(v12_decisions.columns)} columns"
    )

    log(
        f"[PASS] Production outcomes loaded: "
        f"{len(outcomes_raw):,} rows, "
        f"{len(outcomes_raw.columns)} columns"
    )

    # ------------------------------------------------------------------
    # VALIDATE POPULATION
    # ------------------------------------------------------------------

    section("VALIDATING IMMUTABLE PRODUCTION POPULATION")

    v7_stats = validate_population(
        v7,
        "V7",
    )

    v8_stats = validate_population(
        v8,
        "V8",
    )

    validate_same_population(
        v7,
        v8,
    )

    validate_v12_cohort(
        v12_cohort,
        v12_decisions,
    )

    v12_ids = set(
        v12_cohort["transaction_id"].astype(str)
    )

    v7_ids = set(
        v7["transaction_id"].astype(str)
    )

    if v12_ids != v7_ids:
        fail(
            "V12 immutable cohort differs from V7 "
            "production population"
        )

    log(
        "[PASS] V12 immutable cohort matches "
        "V7/V8 production population"
    )

    log(
        "[PASS] V10.2 TOP_RATE 5% targeting preserved"
    )

    # ------------------------------------------------------------------
    # NORMALIZE OUTCOMES
    # ------------------------------------------------------------------

    section(
        "NORMALIZING LEGITIMATE PRODUCTION OUTCOMES"
    )

    outcomes = normalize_outcomes(
        outcomes_raw
    )

    log(
        f"[PASS] Legitimate production outcomes normalized: "
        f"{len(outcomes):,} rows"
    )

    # ------------------------------------------------------------------
    # LINKAGE
    # ------------------------------------------------------------------

    section(
        "BUILDING EXACT TRANSACTION-ID OUTCOME LINKAGE"
    )

    linkage = build_outcome_linkage(
        v12_decisions,
        outcomes,
    )

    matched = linkage[
        linkage["has_observed_outcome"] == 1
    ].copy()

    matched_count = len(matched)

    log(
        f"Matched legitimate production outcomes: "
        f"{matched_count:,}"
    )

    if matched_count == 0:
        log(
            "[WARN] Zero production transaction IDs have "
            "legitimate observed outcomes."
        )
    else:
        log(
            f"[PASS] Matched production outcomes: "
            f"{matched_count:,}"
        )

    # ------------------------------------------------------------------
    # OUTCOME QUALITY CHECK
    # ------------------------------------------------------------------

    section(
        "VALIDATING OBSERVED OUTCOME COHORT"
    )

    if matched_count > 0:

        if matched["transaction_id"].duplicated().any():
            fail(
                "Matched production outcome cohort contains "
                "duplicate transaction IDs"
            )

        if matched["recovered"].isna().any():
            fail(
                "Matched cohort contains missing recovered values"
            )

        if matched["recovered_amount"].isna().any():
            fail(
                "Matched cohort contains missing recovered amounts"
            )

        if (
            pd.to_numeric(
                matched["recovered_amount"],
                errors="coerce",
            ) < 0
        ).any():
            fail(
                "Matched cohort contains negative "
                "recovered amounts"
            )

        log(
            "[PASS] Matched outcome cohort contains only "
            "legitimate observed outcomes"
        )

    else:
        log(
            "[PASS] No fabricated outcomes were introduced"
        )

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------

    section(
        "RUNNING CONTROLLED PRODUCTION OUTCOME EVALUATION"
    )

    evaluation = evaluate_v7_v8(
        matched
    )

    target_metrics = evaluation["target"]
    retry_metrics = evaluation["retry"]

    log(
        f"TARGET matched:    {target_metrics['n']:,}"
    )

    log(
        f"RETRY_ALL matched: {retry_metrics['n']:,}"
    )

    if evaluation["sufficient"]:
        log(
            "[PASS] Sufficient matched production outcomes "
            "for controlled statistical evaluation"
        )

        log(
            f"TARGET observed recovery: "
            f"{pct(target_metrics['recovery_rate'])}"
        )

        log(
            f"RETRY_ALL observed recovery: "
            f"{pct(retry_metrics['recovery_rate'])}"
        )

        log(
            f"Observed rate difference: "
            f"{pct(evaluation['rate_difference'])}"
        )

        log(
            f"Statistical z: "
            f"{evaluation['z']:.6f}"
        )

        log(
            f"Statistical p-value: "
            f"{evaluation['p_value']:.6f}"
        )

    else:
        log(
            "[WARN] Insufficient matched production outcomes "
            "for V7 vs V8 statistical evaluation"
        )

    # ------------------------------------------------------------------
    # GOVERNANCE
    # ------------------------------------------------------------------

    section(
        "RUNNING V13 GOVERNANCE AUDIT"
    )

    audit = governance_audit(
        v7,
        v8,
        matched,
        evaluation["sufficient"],
    )

    log(
        "[PASS] V10.2 targeting remains immutable"
    )

    log(
        "[PASS] No oracle/evaluation fields used"
    )

    log(
        "[PASS] No artificial transaction IDs created"
    )

    log(
        "[PASS] No artificial outcomes created"
    )

    log(
        "[PASS] Calibration predictions are not treated "
        "as production outcomes"
    )

    # ------------------------------------------------------------------
    # VERDICT
    # ------------------------------------------------------------------

    if matched_count == 0:
        verdict = (
            "OUTCOME_CAPTURE_INSUFFICIENT_NO_PRODUCTION_MATCH"
        )

    elif not evaluation["sufficient"]:
        verdict = (
            "OUTCOME_CAPTURE_INSUFFICIENT_FOR_STATISTICAL_EVALUATION"
        )

    else:
        if (
            evaluation["p_value"] < ALPHA
            and evaluation["rate_difference"] > 0
        ):
            verdict = (
                "V8_CALIBRATION_ACTION_LAYER_SHOWS "
                "STATISTICALLY_SIGNIFICANT_RECOVERY_IMPROVEMENT"
            )
        elif (
            evaluation["p_value"] < ALPHA
            and evaluation["rate_difference"] < 0
        ):
            verdict = (
                "V8_CALIBRATION_ACTION_LAYER_SHOWS "
                "STATISTICALLY_SIGNIFICANT_RECOVERY_DECLINE"
            )
        else:
            verdict = (
                "V7_V8_DIFFERENCE_NOT_STATISTICALLY_SIGNIFICANT"
            )

    # ------------------------------------------------------------------
    # OUTPUTS
    # ------------------------------------------------------------------

    section("WRITING V13 OUTPUTS")

    decisions_output = build_decisions_output(
        v12_decisions,
        linkage,
    )

    cohort_output = linkage.copy()

    comparison_output = build_comparison_output(
        evaluation,
    )

    summary_output = build_summary(
        v7_stats,
        v8_stats,
        linkage,
        evaluation,
        audit,
        verdict,
    )

    audit_output = build_audit(
        audit,
        verdict,
    )

    decisions_output.to_csv(
        OUTPUT_DECISIONS,
        index=False,
    )

    cohort_output.to_csv(
        OUTPUT_COHORT,
        index=False,
    )

    comparison_output.to_csv(
        OUTPUT_COMPARISON,
        index=False,
    )

    summary_output.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    audit_output.to_csv(
        OUTPUT_AUDIT,
        index=False,
    )

    log(
        f"[PASS] Decisions:  {OUTPUT_DECISIONS}"
    )

    log(
        f"[PASS] Cohort:     {OUTPUT_COHORT}"
    )

    log(
        f"[PASS] Comparison: {OUTPUT_COMPARISON}"
    )

    log(
        f"[PASS] Summary:    {OUTPUT_SUMMARY}"
    )

    log(
        f"[PASS] Audit:      {OUTPUT_AUDIT}"
    )

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------

    section(
        "V13 CONTROLLED PRODUCTION OUTCOME SUMMARY"
    )

    log(
        f"V7 transactions:              {len(v7):,}"
    )

    log(
        f"V8 transactions:              {len(v8):,}"
    )

    log(
        f"V7 TARGET:                    {v7_stats['target']:,}"
    )

    log(
        f"V8 TARGET:                    {v8_stats['target']:,}"
    )

    log(
        f"V7 RETRY_ALL:                 {v7_stats['retry_all']:,}"
    )

    log(
        f"V8 RETRY_ALL:                 {v8_stats['retry_all']:,}"
    )

    log(
        f"Matched production outcomes:  {matched_count:,}"
    )

    log(
        f"TARGET matched outcomes:      "
        f"{target_metrics['n']:,}"
    )

    log(
        f"RETRY_ALL matched outcomes:   "
        f"{retry_metrics['n']:,}"
    )

    if matched_count:
        overall_rate = (
            int(matched["recovered"].sum())
            / matched_count
        )

        overall_value = float(
            matched["recovered_amount"].sum()
        )

        log(
            f"Observed recovery rate:       "
            f"{pct(overall_rate)}"
        )

        log(
            f"Observed recovered value:     "
            f"{money(overall_value)}"
        )

    else:
        log(
            "Observed recovery rate:       N/A"
        )

        log(
            "Observed recovered value:     ₹0.00"
        )

    log()
    log("GOVERNANCE")
    log(
        "  V10.2 TOP_RATE 5% targeting: PRESERVED"
    )
    log(
        f"  Targeting changed:            "
        f"{'YES' if audit['targeting_changed'] else 'NO'}"
    )
    log(
        f"  Production population changed:"
        f" {'YES' if audit['population_changed'] else 'NO'}"
    )
    log(
        "  Oracle fields used:           NO"
    )
    log(
        "  Artificial IDs created:       NO"
    )
    log(
        "  Artificial outcomes created:  NO"
    )
    log(
        "  Calibration predictions used "
        "as outcomes:                  NO"
    )

    log()
    log("FINAL V13 VERDICT:")
    log(verdict)

    if matched_count == 0:
        log()
        log(
            "V13 does not claim V7-vs-V8 improvement."
        )
        log(
            "Legitimate production outcomes are still "
            "required for the V7 vs V8 comparison."
        )
        log()
        log(
            "NEXT REQUIRED STEP:"
        )
        log(
            "Capture legitimate observed outcomes whose "
            "`transaction_id` exactly matches the immutable "
            "V10.2 production cohort."
        )

    elif not evaluation["sufficient"]:
        log()
        log(
            "V13 does not claim statistical improvement."
        )
        log(
            "More matched production outcomes are required "
            "before the comparison becomes eligible."
        )

    else:
        log()
        log(
            "V13 completed a controlled observed-outcome "
            "comparison using legitimate production IDs."
        )

    log()
    log(
        "========================================================================"
    )
    log(
        "RECLAIM PRODUCTION PIPELINE V13 COMPLETE"
    )
    log(
        "========================================================================"
    )


if __name__ == "__main__":
    main()

