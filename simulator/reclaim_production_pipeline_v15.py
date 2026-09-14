"""
========================================================================
RECLAIM PRODUCTION PIPELINE V15
Production Outcome Capture Template + Evaluation Readiness Gate

Purpose:
    Prepare and validate legitimate production outcomes for the
    immutable V10.2 TOP_RATE 5% production cohort.

Governance:
    - V10.2 targeting is IMMUTABLE
    - V7/V8 production population is IMMUTABLE
    - No artificial transaction IDs
    - No artificial outcomes
    - Calibration predictions are NOT production outcomes
    - Oracle/evaluation columns are NOT used for decisions
    - V7 vs V8 improvement is NOT claimed without legitimate outcomes

V15 does NOT fabricate or infer missing outcomes.
========================================================================
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(r"D:\RECLAIM")
DATA_DIR = BASE_DIR / "data" / "generated"

V13_COHORT = DATA_DIR / "reclaim_production_cohort_v13.csv"
V14_DECISIONS = DATA_DIR / "reclaim_production_decisions_v14.csv"

# Legitimate production outcome source.
# If a new real production outcome file becomes available, replace this
# path or change OUTCOME_CANDIDATES below.
OUTCOME_CANDIDATES = [
    DATA_DIR / "recovery_outcomes.csv",
    DATA_DIR / "production_outcomes.csv",
    DATA_DIR / "reclaim_production_outcomes.csv",
]

# V15 outputs
DECISIONS_OUTPUT = DATA_DIR / "reclaim_production_decisions_v15.csv"
COHORT_OUTPUT = DATA_DIR / "reclaim_production_cohort_v15.csv"
CAPTURE_TEMPLATE_OUTPUT = DATA_DIR / "reclaim_production_outcome_capture_v15.csv"
VALIDATED_OUTCOMES_OUTPUT = DATA_DIR / "reclaim_production_validated_outcomes_v15.csv"
SUMMARY_OUTPUT = DATA_DIR / "reclaim_production_v15_summary.csv"
AUDIT_OUTPUT = DATA_DIR / "reclaim_production_v15_audit.csv"


# ======================================================================
# HELPERS
# ======================================================================

def banner(text):
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def fail(message):
    raise ValueError("[FAIL] " + message)


def warn(message):
    print("[WARN] " + message)


def passed(message):
    print("[PASS] " + message)


def require_file(path, description):
    if not path.exists():
        fail(f"{description} not found: {path}")


def normalize_bool(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, bool):
        return int(value)

    text = str(value).strip().lower()

    if text in {"1", "true", "yes", "y", "recovered", "success"}:
        return 1

    if text in {"0", "false", "no", "n", "not_recovered", "failure"}:
        return 0

    return np.nan


# ======================================================================
# OUTCOME NORMALIZATION
# ======================================================================

def normalize_outcome_schema(df):
    """
    Normalize legitimate production outcome data without inventing
    transaction IDs or outcome values.
    """

    df = df.copy()

    if "transaction_id" not in df.columns:
        fail(
            "Legitimate production outcome source must contain "
            "`transaction_id`."
        )

    # --------------------------------------------------------------
    # recovered
    # --------------------------------------------------------------

    if "recovered" not in df.columns:

        if "recovery_flag" in df.columns:
            df["recovered"] = df["recovery_flag"].apply(normalize_bool)

        elif "recovered_amount" in df.columns:
            df["recovered"] = (
                pd.to_numeric(
                    df["recovered_amount"],
                    errors="coerce"
                )
                .fillna(0)
                .gt(0)
                .astype(int)
            )

        elif "recovery_status" in df.columns:
            df["recovered"] = (
                df["recovery_status"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(
                    {
                        "recovered",
                        "success",
                        "successful",
                        "paid",
                        "complete",
                        "completed",
                    }
                )
                .astype(int)
            )

        else:
            fail(
                "Outcome source contains no usable observed recovery "
                "field (`recovered`, `recovery_flag`, `recovered_amount`, "
                "or `recovery_status`)."
            )

    else:
        df["recovered"] = df["recovered"].apply(normalize_bool)

    # --------------------------------------------------------------
    # recovered_amount
    # --------------------------------------------------------------

    if "recovered_amount" not in df.columns:

        if "amount" in df.columns:
            amount = pd.to_numeric(
                df["amount"],
                errors="coerce"
            )

            df["recovered_amount"] = np.where(
                df["recovered"] == 1,
                amount,
                0.0
            )

        elif "recovered_value" in df.columns:
            df["recovered_amount"] = pd.to_numeric(
                df["recovered_value"],
                errors="coerce"
            )

        else:
            fail(
                "Cannot establish observed `recovered_amount` from "
                "legitimate outcome data."
            )

    else:
        df["recovered_amount"] = pd.to_numeric(
            df["recovered_amount"],
            errors="coerce"
        )

    # --------------------------------------------------------------
    # Canonical numeric validation
    # --------------------------------------------------------------

    df["recovered"] = pd.to_numeric(
        df["recovered"],
        errors="coerce"
    )

    df["recovered_amount"] = pd.to_numeric(
        df["recovered_amount"],
        errors="coerce"
    )

    return df


# ======================================================================
# PRODUCTION COHORT VALIDATION
# ======================================================================

def validate_cohort(cohort):
    required = [
        "transaction_id",
        "customer_id",
        "production_decision",
        "production_intervention_selected",
    ]

    missing = [
        c for c in required
        if c not in cohort.columns
    ]

    if missing:
        fail(
            f"V13/V14 immutable cohort missing required columns: {missing}"
        )

    if cohort["transaction_id"].isna().any():
        fail("Immutable production cohort contains null transaction IDs.")

    if cohort["transaction_id"].duplicated().any():
        fail("Immutable production cohort contains duplicate transaction IDs.")

    if len(cohort) != 10000:
        fail(
            f"Expected immutable production population of 10,000; "
            f"found {len(cohort)}."
        )

    target_count = (
        cohort["production_decision"]
        .astype(str)
        .eq("TARGET")
        .sum()
    )

    retry_count = (
        cohort["production_decision"]
        .astype(str)
        .eq("RETRY_ALL")
        .sum()
    )

    intervention_rate = target_count / len(cohort)

    if target_count != 500:
        fail(
            f"Expected 500 TARGET rows; found {target_count}."
        )

    if retry_count != 9500:
        fail(
            f"Expected 9500 RETRY_ALL rows; found {retry_count}."
        )

    if not np.isclose(intervention_rate, 0.05):
        fail(
            f"Expected 5% intervention rate; "
            f"found {intervention_rate:.6%}."
        )

    passed(
        "V10.2 immutable population validated: "
        "500 TARGET / 9500 RETRY_ALL / 5.00%"
    )

    return target_count, retry_count


# ======================================================================
# OUTCOME SOURCE DISCOVERY
# ======================================================================

def discover_outcome_source():
    for path in OUTCOME_CANDIDATES:
        if path.exists():
            return path

    return None


# ======================================================================
# BUILD CAPTURE TEMPLATE
# ======================================================================

def build_capture_template(cohort):
    """
    Create a capture template containing EVERY legitimate production
    transaction ID from the immutable cohort.

    Existing outcomes are NOT fabricated.
    """

    template = cohort[
        [
            "transaction_id",
            "customer_id",
            "production_decision",
            "production_intervention_selected",
        ]
    ].copy()

    template["observed_outcome_available"] = 0
    template["recovered"] = np.nan
    template["recovered_amount"] = np.nan
    template["outcome_timestamp"] = pd.NaT
    template["outcome_source"] = ""
    template["outcome_notes"] = ""

    return template


# ======================================================================
# EXACT OUTCOME LINKAGE
# ======================================================================

def link_outcomes(cohort, outcomes):
    """
    Exact transaction-ID linkage only.

    No fuzzy matching.
    No customer-ID matching.
    No positional matching.
    No synthetic IDs.
    """

    production_ids = set(
        cohort["transaction_id"].astype(str)
    )

    outcomes = outcomes.copy()
    outcomes["transaction_id"] = (
        outcomes["transaction_id"]
        .astype(str)
        .str.strip()
    )

    matched = outcomes[
        outcomes["transaction_id"].isin(production_ids)
    ].copy()

    if matched.empty:
        warn(
            "Zero production transaction IDs have legitimate "
            "observed outcomes."
        )

        return matched

    # Multiple legitimate outcomes for the same production transaction
    # are ambiguous until explicitly resolved.
    duplicate_ids = (
        matched["transaction_id"]
        .duplicated(keep=False)
    )

    if duplicate_ids.any():
        ambiguous_ids = (
            matched.loc[
                duplicate_ids,
                "transaction_id"
            ]
            .unique()
        )

        warn(
            f"{len(ambiguous_ids)} production transaction IDs have "
            "multiple outcome records."
        )

        matched = matched[
            ~matched["transaction_id"].isin(ambiguous_ids)
        ].copy()

    return matched


# ======================================================================
# BUILD VALIDATED OUTCOME DATASET
# ======================================================================

def build_validated_outcomes(cohort, matched):
    base = cohort[
        [
            "transaction_id",
            "customer_id",
            "production_decision",
            "production_intervention_selected",
        ]
    ].copy()

    if matched.empty:
        base["observed_outcome_available"] = 0
        base["recovered"] = np.nan
        base["recovered_amount"] = np.nan

        return base

    outcome_fields = [
        "transaction_id",
        "recovered",
        "recovered_amount",
    ]

    available_fields = [
        c for c in outcome_fields
        if c in matched.columns
    ]

    merged = base.merge(
        matched[available_fields],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    merged["observed_outcome_available"] = (
        merged["recovered"].notna()
    ).astype(int)

    return merged


# ======================================================================
# METRICS
# ======================================================================

def calculate_metrics(df):
    total = len(df)

    target_mask = (
        df["production_decision"]
        .astype(str)
        .eq("TARGET")
    )

    retry_mask = (
        df["production_decision"]
        .astype(str)
        .eq("RETRY_ALL")
    )

    matched = (
        df["observed_outcome_available"]
        .eq(1)
    )

    target_matched = target_mask & matched
    retry_matched = retry_mask & matched

    matched_count = int(matched.sum())
    target_count = int(target_mask.sum())
    retry_count = int(retry_mask.sum())

    target_match_rate = (
        target_matched.sum() / target_count
        if target_count
        else 0
    )

    retry_match_rate = (
        retry_matched.sum() / retry_count
        if retry_count
        else 0
    )

    overall_match_rate = (
        matched_count / total
        if total
        else 0
    )

    observed = df.loc[matched, "recovered"]

    if len(observed) > 0:
        recovery_rate = observed.mean()
    else:
        recovery_rate = np.nan

    recovered_value = (
        pd.to_numeric(
            df.loc[matched, "recovered_amount"],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )

    return {
        "production_transactions": total,
        "target_transactions": target_count,
        "retry_all_transactions": retry_count,
        "matched_production_outcomes": matched_count,
        "target_matched_outcomes": int(target_matched.sum()),
        "retry_all_matched_outcomes": int(retry_matched.sum()),
        "overall_match_rate": overall_match_rate,
        "target_match_rate": target_match_rate,
        "retry_all_match_rate": retry_match_rate,
        "observed_recovery_rate": recovery_rate,
        "observed_recovered_value": recovered_value,
    }


# ======================================================================
# EVALUATION READINESS
# ======================================================================

def determine_readiness(metrics):
    matched = metrics["matched_production_outcomes"]

    if matched == 0:
        return (
            "OUTCOME_CAPTURE_REQUIRED_BEFORE_V7_V8_EVALUATION"
        )

    # We deliberately do not declare V7 vs V8 superior here.
    # Presence of outcomes makes the evaluation possible, but does not
    # automatically establish statistical significance.

    if metrics["target_matched_outcomes"] == 0:
        return (
            "OUTCOME_CAPTURE_INCOMPLETE_TARGET_COHORT"
        )

    if metrics["retry_all_matched_outcomes"] == 0:
        return (
            "OUTCOME_CAPTURE_INCOMPLETE_RETRY_COHORT"
        )

    return (
        "PRODUCTION_OUTCOMES_AVAILABLE_FOR_CONTROLLED_EVALUATION"
    )


# ======================================================================
# GOVERNANCE AUDIT
# ======================================================================

def build_audit(metrics, verdict):
    rows = [
        {
            "audit_item": "V10.2 targeting immutable",
            "status": "PASS",
            "value": "YES",
        },
        {
            "audit_item": "Production population changed",
            "status": "PASS",
            "value": "NO",
        },
        {
            "audit_item": "Artificial transaction IDs created",
            "status": "PASS",
            "value": "NO",
        },
        {
            "audit_item": "Artificial outcomes created",
            "status": "PASS",
            "value": "NO",
        },
        {
            "audit_item": "Calibration predictions treated as outcomes",
            "status": "PASS",
            "value": "NO",
        },
        {
            "audit_item": "Oracle fields used",
            "status": "PASS",
            "value": "NO",
        },
        {
            "audit_item": "Matched production outcomes",
            "status": "INFO",
            "value": metrics["matched_production_outcomes"],
        },
        {
            "audit_item": "Overall match rate",
            "status": "INFO",
            "value": metrics["overall_match_rate"],
        },
        {
            "audit_item": "Final readiness verdict",
            "status": "INFO",
            "value": verdict,
        },
    ]

    return pd.DataFrame(rows)


# ======================================================================
# MAIN
# ======================================================================

def main():

    banner(
        "RECLAIM PRODUCTION PIPELINE V15\n"
        "Production Outcome Capture Template + Evaluation Readiness\n"
        "V10.2 TOP_RATE 5% targeting is IMMUTABLE"
    )

    # --------------------------------------------------------------
    # Load immutable cohort
    # --------------------------------------------------------------

    banner("LOADING IMMUTABLE PRODUCTION COHORT")

    require_file(
        V13_COHORT,
        "V13 production cohort"
    )

    require_file(
        V14_DECISIONS,
        "V14 production decisions"
    )

    cohort = pd.read_csv(V13_COHORT)
    decisions = pd.read_csv(V14_DECISIONS)

    print(
        f"[PASS] V13 cohort loaded: "
        f"{len(cohort):,} rows, {len(cohort.columns)} columns"
    )

    print(
        f"[PASS] V14 decisions loaded: "
        f"{len(decisions):,} rows, {len(decisions.columns)} columns"
    )

    # --------------------------------------------------------------
    # Validate cohort
    # --------------------------------------------------------------

    banner("VALIDATING IMMUTABLE V10.2 PRODUCTION POPULATION")

    target_count, retry_count = validate_cohort(cohort)

    # --------------------------------------------------------------
    # Validate decisions population
    # --------------------------------------------------------------

    if "transaction_id" not in decisions.columns:
        fail(
            "V14 decisions missing `transaction_id`."
        )

    if decisions["transaction_id"].duplicated().any():
        fail(
            "V14 decisions contain duplicate transaction IDs."
        )

    cohort_ids = set(
        cohort["transaction_id"].astype(str)
    )

    decision_ids = set(
        decisions["transaction_id"].astype(str)
    )

    if cohort_ids != decision_ids:
        fail(
            "V13 cohort and V14 decisions have different "
            "transaction populations."
        )

    passed(
        "V13 cohort and V14 decision populations are identical"
    )

    # --------------------------------------------------------------
    # Discover outcome source
    # --------------------------------------------------------------

    banner("DISCOVERING LEGITIMATE OUTCOME SOURCE")

    outcome_source = discover_outcome_source()

    if outcome_source is None:
        warn(
            "No legitimate production outcome source was found."
        )

        outcomes = pd.DataFrame()

    else:
        print(
            f"[PASS] Outcome source found: {outcome_source}"
        )

        outcomes = pd.read_csv(outcome_source)

        print(
            f"[PASS] Outcome source loaded: "
            f"{len(outcomes):,} rows, {len(outcomes.columns)} columns"
        )

        outcomes = normalize_outcome_schema(outcomes)

        if outcomes["transaction_id"].duplicated().any():
            fail(
                "Legitimate outcome source contains duplicate "
                "transaction IDs."
            )

        passed(
            "Legitimate production outcome schema normalized"
        )

    # --------------------------------------------------------------
    # Exact linkage
    # --------------------------------------------------------------

    banner(
        "BUILDING EXACT TRANSACTION-ID OUTCOME LINKAGE"
    )

    if outcomes.empty:
        matched = pd.DataFrame()
    else:
        matched = link_outcomes(
            cohort,
            outcomes
        )

    if len(matched) > 0:
        passed(
            f"Matched legitimate production outcomes: "
            f"{len(matched):,}"
        )
    else:
        warn(
            "Zero production transaction IDs have legitimate "
            "observed outcomes."
        )

    # --------------------------------------------------------------
    # Build validated outcome dataset
    # --------------------------------------------------------------

    banner(
        "BUILDING VALIDATED PRODUCTION OUTCOME DATASET"
    )

    validated = build_validated_outcomes(
        cohort,
        matched
    )

    passed(
        "Validated production outcome dataset created "
        "without fabricating outcomes"
    )

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    banner(
        "CALCULATING PRODUCTION OUTCOME METRICS"
    )

    metrics = calculate_metrics(
        validated
    )

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
    # Readiness
    # --------------------------------------------------------------

    banner(
        "EVALUATION READINESS GATE"
    )

    verdict = determine_readiness(
        metrics
    )

    if metrics["matched_production_outcomes"] == 0:

        warn(
            "No legitimate production outcomes match the "
            "immutable V10.2 transaction cohort."
        )

        print(
            "V7 vs V8 production improvement remains UNCLAIMED."
        )

    elif (
        metrics["target_matched_outcomes"] == 0
        or metrics["retry_all_matched_outcomes"] == 0
    ):

        warn(
            "Production outcomes exist, but both required "
            "comparison cohorts are not sufficiently represented."
        )

    else:

        passed(
            "Legitimate outcomes exist for both TARGET and RETRY_ALL "
            "production cohorts."
        )

        print(
            "V7 vs V8 controlled evaluation may proceed."
        )

    # --------------------------------------------------------------
    # Build capture template
    # --------------------------------------------------------------

    banner(
        "BUILDING PRODUCTION OUTCOME CAPTURE TEMPLATE"
    )

    capture_template = build_capture_template(
        cohort
    )

    # Fill only legitimate matched outcomes if they exist.
    if not matched.empty:

        matched_lookup = matched.set_index(
            "transaction_id"
        )

        for idx in capture_template.index:

            tid = str(
                capture_template.loc[
                    idx,
                    "transaction_id"
                ]
            )

            if tid in matched_lookup.index:

                capture_template.loc[
                    idx,
                    "observed_outcome_available"
                ] = 1

                if "recovered" in matched_lookup.columns:
                    capture_template.loc[
                        idx,
                        "recovered"
                    ] = matched_lookup.loc[
                        tid,
                        "recovered"
                    ]

                if "recovered_amount" in matched_lookup.columns:
                    capture_template.loc[
                        idx,
                        "recovered_amount"
                    ] = matched_lookup.loc[
                        tid,
                        "recovered_amount"
                    ]

    passed(
        "Capture template created for all 10,000 immutable "
        "production transactions"
    )

    # --------------------------------------------------------------
    # Decisions output
    # --------------------------------------------------------------

    decisions_out = decisions.copy()

    decisions_out["v15_observed_outcome_available"] = (
        decisions_out["transaction_id"]
        .astype(str)
        .isin(
            set(
                matched["transaction_id"].astype(str)
            )
            if not matched.empty
            else set()
        )
    ).astype(int)

    decisions_out["v15_evaluation_status"] = verdict

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                **metrics,
                "v10_2_target_rate": 0.05,
                "v10_2_target_count": 500,
                "v10_2_retry_all_count": 9500,
                "v7_v8_evaluation_ready": (
                    verdict
                    == "PRODUCTION_OUTCOMES_AVAILABLE_FOR_CONTROLLED_EVALUATION"
                ),
                "final_verdict": verdict,
            }
        ]
    )

    # --------------------------------------------------------------
    # Audit
    # --------------------------------------------------------------

    audit = build_audit(
        metrics,
        verdict
    )

    # --------------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------------

    banner(
        "WRITING V15 OUTPUTS"
    )

    decisions_out.to_csv(
        DECISIONS_OUTPUT,
        index=False
    )

    cohort.to_csv(
        COHORT_OUTPUT,
        index=False
    )

    capture_template.to_csv(
        CAPTURE_TEMPLATE_OUTPUT,
        index=False
    )

    validated.to_csv(
        VALIDATED_OUTCOMES_OUTPUT,
        index=False
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False
    )

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False
    )

    passed(
        f"Decisions:  {DECISIONS_OUTPUT}"
    )

    passed(
        f"Cohort:     {COHORT_OUTPUT}"
    )

    passed(
        f"Capture:    {CAPTURE_TEMPLATE_OUTPUT}"
    )

    passed(
        f"Validated:  {VALIDATED_OUTCOMES_OUTPUT}"
    )

    passed(
        f"Summary:    {SUMMARY_OUTPUT}"
    )

    passed(
        f"Audit:      {AUDIT_OUTPUT}"
    )

    # --------------------------------------------------------------
    # Final report
    # --------------------------------------------------------------

    banner(
        "V15 PRODUCTION OUTCOME CAPTURE SUMMARY"
    )

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

    print()
    print(
        "GOVERNANCE"
    )

    print(
        "  V10.2 TOP_RATE 5% targeting: PRESERVED"
    )

    print(
        "  Artificial IDs created:       NO"
    )

    print(
        "  Artificial outcomes created:  NO"
    )

    print(
        "  Oracle fields used:           NO"
    )

    print(
        "  Calibration predictions used "
        "as production outcomes:        NO"
    )

    print()
    print(
        "FINAL V15 VERDICT:"
    )

    print(
        f"{verdict}"
    )

    # --------------------------------------------------------------
    # Next step
    # --------------------------------------------------------------

    print()

    if verdict == "OUTCOME_CAPTURE_REQUIRED_BEFORE_V7_V8_EVALUATION":

        print(
            "NEXT REQUIRED STEP:"
        )

        print(
            "Populate `reclaim_production_outcome_capture_v15.csv` "
            "with legitimate observed production outcomes."
        )

        print(
            "The `transaction_id` MUST exactly match one of the "
            "immutable V10.2 production transaction IDs."
        )

        print(
            "Do NOT create synthetic IDs."
        )

        print(
            "Do NOT copy calibration predictions into observed "
            "outcome fields."
        )

        print(
            "Do NOT use oracle/evaluation values as production outcomes."
        )

    elif verdict == "PRODUCTION_OUTCOMES_AVAILABLE_FOR_CONTROLLED_EVALUATION":

        print(
            "NEXT STEP:"
        )

        print(
            "Proceed to the controlled V7 vs V8 outcome evaluation."
        )

        print(
            "Compare observed outcomes using the exact matched "
            "production transaction IDs."
        )

        print(
            "Do not claim improvement until the statistical and "
            "economic comparison passes the predefined gates."
        )

    else:

        print(
            "NEXT REQUIRED STEP:"
        )

        print(
            "Continue production outcome capture until both TARGET "
            "and RETRY_ALL cohorts have legitimate observed outcomes."
        )

    banner(
        "RECLAIM PRODUCTION PIPELINE V15 COMPLETE"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print(
            "========================================================================"
        )
        print(
            "V15 FAILED"
        )
        print(
            "========================================================================"
        )
        print(f"[FAIL] {exc}")
        sys.exit(1)