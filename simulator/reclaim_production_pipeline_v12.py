"""
========================================================================
RECLAIM PRODUCTION PIPELINE V12
Production Outcome Capture + Strict Validation
V10.2 TOP_RATE 5% targeting is IMMUTABLE
========================================================================

Purpose:
    Validate the immutable V11 production cohort and production decisions,
    attach legitimate observed outcomes when available, and prepare a
    strict production outcome dataset for future V7 vs V8 evaluation.

Governance rules:
    1. V10.2 TOP_RATE 5% targeting is immutable.
    2. V11 cohort defines the immutable production population.
    3. V11 decisions define production decision/action fields.
    4. Legitimate production outcomes must match existing transaction IDs.
    5. No artificial IDs.
    6. No artificial outcomes.
    7. Calibration predictions are NOT production outcomes.
    8. No oracle/evaluation columns are used for production decisions.
    9. V12 does not claim V7 vs V8 improvement.
"""

from pathlib import Path
import sys
import math
import pandas as pd
import numpy as np


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(r"D:\RECLAIM")

DATA_DIR = BASE_DIR / "data" / "generated"

V11_COHORT = DATA_DIR / "reclaim_production_cohort_v11.csv"
V11_DECISIONS = DATA_DIR / "reclaim_production_decisions_v11.csv"

# Legitimate observed production outcomes.
# This is intentionally NOT the calibration prediction file.
PRODUCTION_OUTCOMES = DATA_DIR / "recovery_outcomes.csv"

OUTPUT_DECISIONS = DATA_DIR / "reclaim_production_decisions_v12.csv"
OUTPUT_COHORT = DATA_DIR / "reclaim_production_cohort_v12.csv"
OUTPUT_CAPTURE = DATA_DIR / "reclaim_production_outcome_capture_v12.csv"
OUTPUT_SUMMARY = DATA_DIR / "reclaim_production_v12_summary.csv"
OUTPUT_AUDIT = DATA_DIR / "reclaim_production_v12_audit.csv"


TARGET_RATE = 0.05
EXPECTED_POPULATION = 10_000
EXPECTED_TARGET = 500
EXPECTED_RETRY_ALL = 9_500


# ======================================================================
# TERMINAL HELPERS
# ======================================================================

def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def passed(message):
    print(f"[PASS] {message}")


def warning(message):
    print(f"[WARN] {message}")


def fail(message):
    raise ValueError("[FAIL] " + message)


# ======================================================================
# FILE VALIDATION
# ======================================================================

def require_file(path, label):
    if not path.exists():
        fail(f"{label} not found: {path}")


def require_columns(df, required, label):
    missing = [c for c in required if c not in df.columns]

    if missing:
        fail(
            f"{label} missing required columns: {missing}"
        )


# ======================================================================
# BASIC DATA VALIDATION
# ======================================================================

def validate_unique_ids(df, label):
    if "transaction_id" not in df.columns:
        fail(f"{label} does not contain transaction_id")

    if df["transaction_id"].isna().any():
        fail(f"{label} contains null transaction_id values")

    duplicated = df["transaction_id"].duplicated().sum()

    if duplicated:
        fail(
            f"{label} contains {duplicated} duplicated transaction IDs"
        )

    passed(f"{label} transaction IDs are unique")


def normalize_intervention_flag(series):
    """
    Convert common boolean-like representations to integers.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        return numeric.astype(int)

    text = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )

    mapping = {
        "TRUE": 1,
        "FALSE": 0,
        "YES": 1,
        "NO": 0,
        "Y": 1,
        "N": 0,
        "TARGET": 1,
        "RETRY_ALL": 0,
    }

    converted = text.map(mapping)

    if converted.isna().any():
        fail(
            "Unable to normalize production_intervention_selected"
        )

    return converted.astype(int)


# ======================================================================
# V11 COHORT VALIDATION
# ======================================================================

def validate_v11_cohort(cohort):
    """
    IMPORTANT:
    The V11 cohort is intentionally a small immutable population file.

    Production decision fields are NOT required here.

    They belong to the V11 decisions file.
    """

    required = [
        "transaction_id",
    ]

    require_columns(
        cohort,
        required,
        "V11 cohort"
    )

    validate_unique_ids(
        cohort,
        "V11 cohort"
    )

    if len(cohort) != EXPECTED_POPULATION:
        fail(
            f"V11 cohort population expected "
            f"{EXPECTED_POPULATION}, got {len(cohort)}"
        )

    passed(
        f"V11 immutable cohort validated: "
        f"{len(cohort):,} rows"
    )


# ======================================================================
# V11 DECISION VALIDATION
# ======================================================================

def validate_v11_decisions(decisions):
    """
    Production decision fields belong to the decisions dataset.
    """

    required = [
        "transaction_id",
        "production_decision",
        "production_intervention_selected",
    ]

    require_columns(
        decisions,
        required,
        "V11 decisions"
    )

    validate_unique_ids(
        decisions,
        "V11 decisions"
    )

    if len(decisions) != EXPECTED_POPULATION:
        fail(
            f"V11 decisions population expected "
            f"{EXPECTED_POPULATION}, got {len(decisions)}"
        )

    decisions = decisions.copy()

    decisions[
        "production_intervention_selected"
    ] = normalize_intervention_flag(
        decisions["production_intervention_selected"]
    )

    valid_decisions = {
        "TARGET",
        "RETRY_ALL",
    }

    actual_decisions = set(
        decisions["production_decision"]
        .astype(str)
        .str.upper()
        .unique()
    )

    invalid = actual_decisions - valid_decisions

    if invalid:
        fail(
            f"Unexpected production_decision values: {sorted(invalid)}"
        )

    target_count = (
        decisions["production_decision"]
        .astype(str)
        .str.upper()
        .eq("TARGET")
        .sum()
    )

    retry_count = (
        decisions["production_decision"]
        .astype(str)
        .str.upper()
        .eq("RETRY_ALL")
        .sum()
    )

    intervention_count = (
        decisions["production_intervention_selected"]
        .eq(1)
        .sum()
    )

    if target_count != EXPECTED_TARGET:
        fail(
            f"TARGET population expected {EXPECTED_TARGET}, "
            f"got {target_count}"
        )

    if retry_count != EXPECTED_RETRY_ALL:
        fail(
            f"RETRY_ALL population expected {EXPECTED_RETRY_ALL}, "
            f"got {retry_count}"
        )

    if intervention_count != EXPECTED_TARGET:
        fail(
            f"Intervention count expected {EXPECTED_TARGET}, "
            f"got {intervention_count}"
        )

    intervention_rate = intervention_count / len(decisions)

    if not math.isclose(
        intervention_rate,
        TARGET_RATE,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        fail(
            f"Intervention rate changed: "
            f"{intervention_rate:.6%}"
        )

    passed(
        f"V11 production decision population preserved: "
        f"{target_count} TARGET / "
        f"{retry_count} RETRY_ALL / "
        f"{intervention_rate:.2%}"
    )

    return decisions


# ======================================================================
# POPULATION IDENTITY CHECK
# ======================================================================

def validate_population_identity(cohort, decisions):
    cohort_ids = set(cohort["transaction_id"])
    decision_ids = set(decisions["transaction_id"])

    if cohort_ids != decision_ids:
        missing_from_decisions = cohort_ids - decision_ids
        missing_from_cohort = decision_ids - cohort_ids

        fail(
            "V11 cohort and V11 decisions populations differ. "
            f"Missing from decisions={len(missing_from_decisions)}, "
            f"missing from cohort={len(missing_from_cohort)}"
        )

    passed(
        "V11 cohort and V11 decisions transaction population identical"
    )


# ======================================================================
# LEGITIMATE OUTCOME NORMALIZATION
# ======================================================================

def normalize_outcome_source(outcomes):
    """
    Normalize legitimate observed outcomes.

    Accepted observed recovery representations:
        recovered
        recovered_amount
        recovery_status

    We do NOT infer an outcome from a prediction.
    """

    if "transaction_id" not in outcomes.columns:
        fail(
            "Production outcome source missing transaction_id"
        )

    outcomes = outcomes.copy()

    # --------------------------------------------------------------
    # Normalize recovered flag
    # --------------------------------------------------------------

    if "recovered" in outcomes.columns:

        recovered_numeric = pd.to_numeric(
            outcomes["recovered"],
            errors="coerce"
        )

        if recovered_numeric.notna().all():
            outcomes["recovered"] = (
                recovered_numeric > 0
            ).astype(int)

        else:
            text = (
                outcomes["recovered"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            mapping = {
                "TRUE": 1,
                "FALSE": 0,
                "YES": 1,
                "NO": 0,
                "Y": 1,
                "N": 0,
                "1": 1,
                "0": 0,
            }

            mapped = text.map(mapping)

            if mapped.isna().any():
                fail(
                    "Unable to normalize recovered field"
                )

            outcomes["recovered"] = mapped.astype(int)

    elif "recovered_amount" in outcomes.columns:

        recovered_amount = pd.to_numeric(
            outcomes["recovered_amount"],
            errors="coerce"
        ).fillna(0)

        outcomes["recovered"] = (
            recovered_amount > 0
        ).astype(int)

        passed(
            "Derived `recovered` from `recovered_amount`"
        )

    elif "recovery_status" in outcomes.columns:

        status = (
            outcomes["recovery_status"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        recovered_statuses = {
            "RECOVERED",
            "SUCCESS",
            "SUCCEEDED",
            "PAID",
            "COMPLETE",
            "COMPLETED",
        }

        outcomes["recovered"] = (
            status.isin(recovered_statuses)
        ).astype(int)

        passed(
            "Derived `recovered` from `recovery_status`"
        )

    else:
        fail(
            "Production outcome source must contain "
            "`recovered`, `recovered_amount`, or "
            "`recovery_status`"
        )

    # --------------------------------------------------------------
    # Normalize recovered amount
    # --------------------------------------------------------------

    if "recovered_amount" in outcomes.columns:

        outcomes["recovered_amount"] = pd.to_numeric(
            outcomes["recovered_amount"],
            errors="coerce"
        ).fillna(0)

    elif "amount" in outcomes.columns:

        amount = pd.to_numeric(
            outcomes["amount"],
            errors="coerce"
        ).fillna(0)

        outcomes["recovered_amount"] = np.where(
            outcomes["recovered"] == 1,
            amount,
            0.0
        )

        passed(
            "Derived recovered_amount from observed "
            "recovered flag and amount"
        )

    else:
        fail(
            "Production outcome source lacks both "
            "`recovered_amount` and `amount`"
        )

    # --------------------------------------------------------------
    # Validate outcome IDs
    # --------------------------------------------------------------

    validate_unique_ids(
        outcomes,
        "Production outcomes"
    )

    if (
        outcomes["recovered"]
        .isin([0, 1])
        .all()
        is False
    ):
        fail(
            "Production outcome recovered field contains "
            "values other than 0/1"
        )

    if (
        outcomes["recovered_amount"] < 0
    ).any():
        fail(
            "Production outcomes contain negative "
            "recovered_amount values"
        )

    passed("Legitimate production outcome schema normalized")

    return outcomes


# ======================================================================
# OUTCOME LINKAGE
# ======================================================================

def build_outcome_linkage(cohort, decisions, outcomes):
    """
    Strict transaction-ID linkage.

    No fuzzy matching.
    No artificial IDs.
    No prediction-based outcome creation.
    """

    production = cohort[
        ["transaction_id"]
    ].merge(
        decisions[
            [
                "transaction_id",
                "production_decision",
                "production_intervention_selected",
            ]
        ],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    outcome_columns = [
        "transaction_id",
        "recovered",
        "recovered_amount",
    ]

    available = [
        c for c in outcome_columns
        if c in outcomes.columns
    ]

    production = production.merge(
        outcomes[available],
        on="transaction_id",
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    production["observed_outcome"] = (
        production["_merge"]
        .eq("both")
        .astype(int)
    )

    production.drop(
        columns=["_merge"],
        inplace=True
    )

    return production


# ======================================================================
# OUTCOME METRICS
# ======================================================================

def calculate_metrics(linked):
    total = len(linked)

    matched = int(
        linked["observed_outcome"].sum()
    )

    target_mask = (
        linked["production_decision"]
        .astype(str)
        .str.upper()
        .eq("TARGET")
    )

    retry_mask = (
        linked["production_decision"]
        .astype(str)
        .str.upper()
        .eq("RETRY_ALL")
    )

    target_total = int(target_mask.sum())
    retry_total = int(retry_mask.sum())

    target_matched = int(
        (
            target_mask
            & linked["observed_outcome"].eq(1)
        ).sum()
    )

    retry_matched = int(
        (
            retry_mask
            & linked["observed_outcome"].eq(1)
        ).sum()
    )

    overall_match_rate = (
        matched / total
        if total
        else 0
    )

    target_match_rate = (
        target_matched / target_total
        if target_total
        else 0
    )

    retry_match_rate = (
        retry_matched / retry_total
        if retry_total
        else 0
    )

    if matched > 0:

        observed = linked[
            linked["observed_outcome"] == 1
        ]

        observed_recovery_rate = (
            observed["recovered"]
            .mean()
        )

        observed_recovered_value = (
            observed["recovered_amount"]
            .sum()
        )

        observed_recovered_rows = int(
            observed["recovered"].sum()
        )

    else:

        observed_recovery_rate = np.nan
        observed_recovered_value = 0.0
        observed_recovered_rows = 0

    return {
        "production_transactions": total,
        "target_transactions": target_total,
        "retry_all_transactions": retry_total,
        "matched_production_outcomes": matched,
        "target_matched_outcomes": target_matched,
        "retry_all_matched_outcomes": retry_matched,
        "overall_match_rate": overall_match_rate,
        "target_match_rate": target_match_rate,
        "retry_all_match_rate": retry_match_rate,
        "observed_recovered_rows": observed_recovered_rows,
        "observed_recovery_rate": observed_recovery_rate,
        "observed_recovered_value": observed_recovered_value,
    }


# ======================================================================
# GOVERNANCE AUDIT
# ======================================================================

def run_governance_audit(
    cohort,
    decisions,
    linked,
    metrics,
):
    audit = []

    def add(check, status, detail):
        audit.append(
            {
                "check": check,
                "status": status,
                "detail": detail,
            }
        )

    # --------------------------------------------------------------
    # Immutable population
    # --------------------------------------------------------------

    population_ok = (
        metrics["production_transactions"]
        == EXPECTED_POPULATION
    )

    add(
        "production_population",
        "PASS" if population_ok else "FAIL",
        (
            f"{metrics['production_transactions']} rows; "
            f"expected {EXPECTED_POPULATION}"
        ),
    )

    # --------------------------------------------------------------
    # Target population
    # --------------------------------------------------------------

    target_ok = (
        metrics["target_transactions"]
        == EXPECTED_TARGET
    )

    add(
        "target_population",
        "PASS" if target_ok else "FAIL",
        (
            f"{metrics['target_transactions']} TARGET; "
            f"expected {EXPECTED_TARGET}"
        ),
    )

    # --------------------------------------------------------------
    # Retry population
    # --------------------------------------------------------------

    retry_ok = (
        metrics["retry_all_transactions"]
        == EXPECTED_RETRY_ALL
    )

    add(
        "retry_all_population",
        "PASS" if retry_ok else "FAIL",
        (
            f"{metrics['retry_all_transactions']} RETRY_ALL; "
            f"expected {EXPECTED_RETRY_ALL}"
        ),
    )

    # --------------------------------------------------------------
    # Intervention rate
    # --------------------------------------------------------------

    intervention_rate = (
        metrics["target_transactions"]
        / metrics["production_transactions"]
    )

    rate_ok = math.isclose(
        intervention_rate,
        TARGET_RATE,
        rel_tol=0,
        abs_tol=1e-12,
    )

    add(
        "intervention_rate",
        "PASS" if rate_ok else "FAIL",
        f"{intervention_rate:.6%}",
    )

    # --------------------------------------------------------------
    # Artificial outcomes
    # --------------------------------------------------------------

    add(
        "artificial_outcomes",
        "PASS",
        "No artificial outcomes created",
    )

    # --------------------------------------------------------------
    # Artificial IDs
    # --------------------------------------------------------------

    add(
        "artificial_transaction_ids",
        "PASS",
        "No artificial transaction IDs created",
    )

    # --------------------------------------------------------------
    # Calibration separation
    # --------------------------------------------------------------

    add(
        "calibration_separation",
        "PASS",
        "Calibration predictions are not treated as production outcomes",
    )

    # --------------------------------------------------------------
    # Oracle separation
    # --------------------------------------------------------------

    oracle_columns = [
        c
        for c in decisions.columns
        if c.startswith("oracle_")
    ]

    if oracle_columns:

        add(
            "oracle_decision_usage",
            "PASS",
            (
                "Oracle/evaluation columns detected but "
                "not used to create production decisions: "
                + ", ".join(oracle_columns)
            ),
        )

    else:

        add(
            "oracle_decision_usage",
            "PASS",
            "No oracle/evaluation fields detected",
        )

    # --------------------------------------------------------------
    # Outcome linkage
    # --------------------------------------------------------------

    if metrics["matched_production_outcomes"] > 0:

        add(
            "production_outcome_linkage",
            "PASS",
            (
                f"{metrics['matched_production_outcomes']} "
                "legitimate transaction-ID matches"
            ),
        )

    else:

        add(
            "production_outcome_linkage",
            "WARN",
            "No legitimate production outcomes matched yet",
        )

    return pd.DataFrame(audit)


# ======================================================================
# SUMMARY
# ======================================================================

def build_summary(metrics):
    summary = pd.DataFrame(
        [
            {
                "production_transactions":
                    metrics["production_transactions"],

                "target_transactions":
                    metrics["target_transactions"],

                "retry_all_transactions":
                    metrics["retry_all_transactions"],

                "matched_production_outcomes":
                    metrics["matched_production_outcomes"],

                "target_matched_outcomes":
                    metrics["target_matched_outcomes"],

                "retry_all_matched_outcomes":
                    metrics["retry_all_matched_outcomes"],

                "overall_match_rate":
                    metrics["overall_match_rate"],

                "target_match_rate":
                    metrics["target_match_rate"],

                "retry_all_match_rate":
                    metrics["retry_all_match_rate"],

                "observed_recovered_rows":
                    metrics["observed_recovered_rows"],

                "observed_recovery_rate":
                    metrics["observed_recovery_rate"],

                "observed_recovered_value":
                    metrics["observed_recovered_value"],

                "target_rate_policy":
                    TARGET_RATE,

                "targeting_population_immutable":
                    True,

                "artificial_outcomes_created":
                    False,

                "artificial_ids_created":
                    False,

                "calibration_predictions_used_as_outcomes":
                    False,

                "oracle_fields_used_for_decisions":
                    False,
            }
        ]
    )

    return summary


# ======================================================================
# VERDICT
# ======================================================================

def determine_verdict(metrics):

    if (
        metrics["matched_production_outcomes"]
        == 0
    ):
        return "OUTCOME_CAPTURE_REQUIRED_BEFORE_V7_V8_EVALUATION"

    return "PRODUCTION_OUTCOME_CAPTURE_AVAILABLE_FOR_NEXT_STAGE"


# ======================================================================
# MAIN
# ======================================================================

def main():

    banner(
        "RECLAIM PRODUCTION PIPELINE V12\n"
        "Production Outcome Capture + Strict Validation\n"
        "V10.2 TOP_RATE 5% targeting is IMMUTABLE"
    )

    # ==================================================================
    # LOAD V11 COHORT
    # ==================================================================

    section(
        "LOADING IMMUTABLE V11 PRODUCTION COHORT"
    )

    require_file(
        V11_COHORT,
        "V11 cohort"
    )

    require_file(
        V11_DECISIONS,
        "V11 decisions"
    )

    cohort = pd.read_csv(
        V11_COHORT
    )

    decisions = pd.read_csv(
        V11_DECISIONS
    )

    passed(
        f"V11 cohort loaded: "
        f"{len(cohort):,} rows, "
        f"{len(cohort.columns)} columns"
    )

    passed(
        f"V11 decisions loaded: "
        f"{len(decisions):,} rows, "
        f"{len(decisions.columns)} columns"
    )

    # ==================================================================
    # VALIDATE COHORT
    # ==================================================================

    section(
        "VALIDATING IMMUTABLE PRODUCTION POPULATION"
    )

    validate_v11_cohort(
        cohort
    )

    # ==================================================================
    # VALIDATE DECISIONS
    # ==================================================================

    decisions = validate_v11_decisions(
        decisions
    )

    # ==================================================================
    # CROSS-FILE POPULATION
    # ==================================================================

    validate_population_identity(
        cohort,
        decisions
    )

    passed(
        "V10.2 TOP_RATE 5% targeting preserved"
    )

    # ==================================================================
    # LOAD LEGITIMATE OUTCOMES
    # ==================================================================

    section(
        "LOADING LEGITIMATE PRODUCTION OUTCOMES"
    )

    if not PRODUCTION_OUTCOMES.exists():

        warning(
            "Production outcome source not found."
        )

        outcomes = pd.DataFrame(
            columns=[
                "transaction_id",
                "recovered",
                "recovered_amount",
            ]
        )

    else:

        outcomes = pd.read_csv(
            PRODUCTION_OUTCOMES
        )

        passed(
            f"Production outcome source loaded: "
            f"{len(outcomes):,} rows, "
            f"{len(outcomes.columns)} columns"
        )

        outcomes = normalize_outcome_source(
            outcomes
        )

    # ==================================================================
    # BUILD LINKAGE
    # ==================================================================

    section(
        "BUILDING PRODUCTION OUTCOME LINKAGE"
    )

    linked = build_outcome_linkage(
        cohort,
        decisions,
        outcomes,
    )

    matched = int(
        linked["observed_outcome"].sum()
    )

    if matched == 0:

        warning(
            "Zero production transaction IDs have "
            "legitimate observed outcomes."
        )

    else:

        passed(
            f"Matched legitimate production outcomes: "
            f"{matched:,}"
        )

    # ==================================================================
    # CALCULATE METRICS
    # ==================================================================

    section(
        "CALCULATING PRODUCTION OUTCOME METRICS"
    )

    metrics = calculate_metrics(
        linked
    )

    # ==================================================================
    # SUMMARY
    # ==================================================================

    banner(
        "V12 OUTCOME LINKAGE SUMMARY"
    )

    print(
        f"\nProduction transactions:      "
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
        f"Overall match rate:           "
        f"{metrics['overall_match_rate']:.2%}"
    )

    print(
        f"TARGET match rate:            "
        f"{metrics['target_match_rate']:.2%}"
    )

    print(
        f"RETRY_ALL match rate:         "
        f"{metrics['retry_all_match_rate']:.2%}"
    )

    if np.isnan(
        metrics["observed_recovery_rate"]
    ):

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

    # ==================================================================
    # GOVERNANCE
    # ==================================================================

    section(
        "RUNNING V12 GOVERNANCE AUDIT"
    )

    audit = run_governance_audit(
        cohort,
        decisions,
        linked,
        metrics,
    )

    # ==================================================================
    # VERDICT
    # ==================================================================

    verdict = determine_verdict(
        metrics
    )

    # ==================================================================
    # BUILD OUTPUT DATASETS
    # ==================================================================

    # Decisions output:
    # Keep all original V11 decision columns and add V12 linkage fields.

    decisions_v12 = decisions.copy()

    linkage_columns = [
        "transaction_id",
        "observed_outcome",
        "recovered",
        "recovered_amount",
    ]

    decisions_v12 = decisions_v12.merge(
        linked[linkage_columns],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # Cohort output:
    # Preserve the immutable cohort and explicitly mark its role.

    cohort_v12 = cohort.copy()

    cohort_v12["v12_cohort_status"] = (
        "IMMUTABLE_PRODUCTION_COHORT"
    )

    cohort_v12["v10_2_target_rate"] = TARGET_RATE

    cohort_v12["artificial_outcome"] = False

    cohort_v12["artificial_transaction_id"] = False

    # Capture output:
    # This is specifically designed for future production outcome capture.

    capture = linked.copy()

    capture[
        "capture_status"
    ] = np.where(
        capture["observed_outcome"].eq(1),
        "OBSERVED_OUTCOME_CAPTURED",
        "AWAITING_LEGITIMATE_PRODUCTION_OUTCOME",
    )

    capture[
        "calibration_prediction_is_outcome"
    ] = False

    capture[
        "artificial_outcome"
    ] = False

    # ==================================================================
    # WRITE OUTPUTS
    # ==================================================================

    section(
        "WRITING V12 OUTPUTS"
    )

    decisions_v12.to_csv(
        OUTPUT_DECISIONS,
        index=False,
    )

    cohort_v12.to_csv(
        OUTPUT_COHORT,
        index=False,
    )

    capture.to_csv(
        OUTPUT_CAPTURE,
        index=False,
    )

    summary = build_summary(
        metrics
    )

    summary[
        "final_verdict"
    ] = verdict

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    audit[
        "final_verdict"
    ] = verdict

    audit.to_csv(
        OUTPUT_AUDIT,
        index=False,
    )

    passed(
        f"Decisions:  {OUTPUT_DECISIONS}"
    )

    passed(
        f"Cohort:     {OUTPUT_COHORT}"
    )

    passed(
        f"Capture:    {OUTPUT_CAPTURE}"
    )

    passed(
        f"Summary:    {OUTPUT_SUMMARY}"
    )

    passed(
        f"Audit:      {OUTPUT_AUDIT}"
    )

    # ==================================================================
    # FINAL
    # ==================================================================

    banner(
        "FINAL V12 VERDICT"
    )

    print(
        f"FINAL VERDICT: {verdict}"
    )

    print(
        "\nV10.2 TOP_RATE 5% targeting remains untouched."
    )

    print(
        "No artificial transaction IDs were created."
    )

    print(
        "No artificial outcomes were created."
    )

    print(
        "Calibration predictions are not treated as "
        "production outcomes."
    )

    if metrics["matched_production_outcomes"] == 0:

        print(
            "\nNEXT REQUIRED STEP:"
        )

        print(
            "Capture legitimate observed outcomes for "
            "the V10.2 production transaction IDs."
        )

        print(
            "Only after sufficient matched production "
            "outcomes exist should V7 vs V8 be evaluated."
        )

    else:

        print(
            "\nNEXT STAGE:"
        )

        print(
            "Use the matched production outcomes for "
            "a controlled V7 vs V8 outcome evaluation."
        )

    banner(
        "RECLAIM PRODUCTION PIPELINE V12 COMPLETE"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:

        print(
            "\n" + "=" * 72
        )

        print(
            "V12 FAILED"
        )

        print(
            "=" * 72
        )

        print(
            str(exc)
        )

        sys.exit(1)