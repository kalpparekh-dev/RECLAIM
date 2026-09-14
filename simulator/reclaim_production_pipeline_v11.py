"""
======================================================================
RECLAIM PRODUCTION PIPELINE V11
Production Outcome Capture + Cohort Evaluation
======================================================================

Purpose
-------
V11 creates a governance-safe bridge between the immutable V10.2
production population and future observed recovery outcomes.

IMPORTANT GOVERNANCE RULES
--------------------------
1. V10.2 TOP_RATE 5% targeting is immutable.
2. V7/V8 production populations must remain identical.
3. No artificial outcomes are created.
4. No transaction IDs are fabricated or rewritten.
5. Calibration outcomes are NOT treated as production outcomes.
6. Oracle/evaluation fields are audit-only.
7. V11 does not claim V7 vs V8 improvement without legitimate
   production outcomes.
8. Only genuinely observed outcomes with matching production IDs
   are eligible for production evaluation.

Inputs
------
V7:
D:\\RECLAIM\\data\\generated\\reclaim_production_decisions_v7.csv

V8:
D:\\RECLAIM\\data\\generated\\reclaim_production_decisions_v8.csv

Optional production outcome source:
D:\\RECLAIM\\data\\generated\\recovery_outcomes.csv

Calibration source is deliberately excluded from production
outcome evaluation:
D:\\RECLAIM\\data\\generated\\calibrated_recovery_predictions.csv

Outputs
-------
reclaim_production_decisions_v11.csv
reclaim_production_cohort_v11.csv
reclaim_production_outcome_capture_v11.csv
reclaim_production_v11_summary.csv
reclaim_production_v11_audit.csv
======================================================================
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

BASE = Path(r"D:\RECLAIM")

DATA = BASE / "data" / "generated"

V7_PATH = DATA / "reclaim_production_decisions_v7.csv"
V8_PATH = DATA / "reclaim_production_decisions_v8.csv"

PRODUCTION_OUTCOME_PATH = DATA / "recovery_outcomes.csv"

# Explicitly NOT used as production outcomes.
CALIBRATION_OUTCOME_PATH = (
    DATA / "calibrated_recovery_predictions.csv"
)

OUTPUT_DECISIONS = DATA / "reclaim_production_decisions_v11.csv"
OUTPUT_COHORT = DATA / "reclaim_production_cohort_v11.csv"
OUTPUT_CAPTURE = DATA / "reclaim_production_outcome_capture_v11.csv"
OUTPUT_SUMMARY = DATA / "reclaim_production_v11_summary.csv"
OUTPUT_AUDIT = DATA / "reclaim_production_v11_audit.csv"

TARGET_RATE = 0.05
EXPECTED_ROWS = 10000
EXPECTED_TARGET = 500
EXPECTED_RETRY_ALL = 9500


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def fail(message):
    raise ValueError("[FAIL] " + message)


def passed(message):
    print("[PASS] " + message)


def warned(message):
    print("[WARN] " + message)


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def require_file(path, label):
    if not path.exists():
        fail(f"{label} not found: {path}")
    passed(f"{label} found: {path}")


def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"{label} missing required columns: {missing}")


def normalize_bool(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, bool):
        return int(value)

    text = str(value).strip().lower()

    if text in {"1", "true", "yes", "y", "recovered"}:
        return 1

    if text in {"0", "false", "no", "n", "not_recovered"}:
        return 0

    return np.nan


# ---------------------------------------------------------------------
# LOAD PRODUCTION BASELINES
# ---------------------------------------------------------------------

def load_production_baselines():
    section("LOADING PRODUCTION BASELINES")

    require_file(V7_PATH, "V7 production file")
    require_file(V8_PATH, "V8 production file")

    v7 = pd.read_csv(V7_PATH)
    v8 = pd.read_csv(V8_PATH)

    print(f"[PASS] V7 loaded: {len(v7):,} rows, {len(v7.columns)} columns")
    print(f"[PASS] V8 loaded: {len(v8):,} rows, {len(v8.columns)} columns")

    return v7, v8


# ---------------------------------------------------------------------
# VALIDATE PRODUCTION POPULATION
# ---------------------------------------------------------------------

def validate_population(v7, v8):
    section("VALIDATING PRODUCTION POPULATION")

    require_columns(
        v7,
        [
            "transaction_id",
            "production_decision",
            "production_intervention_selected",
            "production_policy_type",
            "production_target_rate",
        ],
        "V7",
    )

    require_columns(
        v8,
        [
            "transaction_id",
            "production_decision",
            "production_intervention_selected",
            "production_policy_type",
            "production_target_rate",
        ],
        "V8",
    )

    if len(v7) != EXPECTED_ROWS:
        fail(f"V7 population expected {EXPECTED_ROWS}, got {len(v7)}")

    if len(v8) != EXPECTED_ROWS:
        fail(f"V8 population expected {EXPECTED_ROWS}, got {len(v8)}")

    if not v7["transaction_id"].is_unique:
        fail("V7 transaction IDs are not unique")

    if not v8["transaction_id"].is_unique:
        fail("V8 transaction IDs are not unique")

    passed("V7 transaction IDs are unique")
    passed("V8 transaction IDs are unique")

    v7_ids = set(v7["transaction_id"])
    v8_ids = set(v8["transaction_id"])

    if v7_ids != v8_ids:
        fail("V7/V8 production transaction populations differ")

    passed("V7/V8 production transaction population identical")

    for name, df in [("V7", v7), ("V8", v8)]:

        policy_values = set(
            df["production_policy_type"]
            .dropna()
            .astype(str)
            .str.upper()
        )

        if policy_values != {"TOP_RATE"}:
            fail(f"{name} policy is not exclusively TOP_RATE")

        rate_values = pd.to_numeric(
            df["production_target_rate"],
            errors="coerce"
        )

        if not np.allclose(rate_values, TARGET_RATE):
            fail(f"{name} target rate is not 5%")

        target_count = int(
            (df["production_decision"] == "TARGET").sum()
        )

        retry_count = int(
            (df["production_decision"] == "RETRY_ALL").sum()
        )

        intervention_rate = (
            df["production_intervention_selected"].sum()
            / len(df)
        )

        if target_count != EXPECTED_TARGET:
            fail(
                f"{name} TARGET population expected "
                f"{EXPECTED_TARGET}, got {target_count}"
            )

        if retry_count != EXPECTED_RETRY_ALL:
            fail(
                f"{name} RETRY_ALL population expected "
                f"{EXPECTED_RETRY_ALL}, got {retry_count}"
            )

        if not np.isclose(intervention_rate, TARGET_RATE):
            fail(
                f"{name} intervention rate is not 5%"
            )

        passed(
            f"{name} population preserved: "
            f"{target_count} TARGET / "
            f"{retry_count} RETRY_ALL / "
            f"{intervention_rate:.2%}"
        )

    passed("V10.2 TOP_RATE 5% targeting preserved")


# ---------------------------------------------------------------------
# BUILD IMMUTABLE EVALUATION COHORT
# ---------------------------------------------------------------------

def build_cohort(v7, v8):
    section("BUILDING IMMUTABLE PRODUCTION EVALUATION COHORT")

    cohort = v7[
        [
            "transaction_id",
            "customer_id",
            "production_decision",
            "production_intervention_selected",
            "production_rank",
            "production_target_rate",
        ]
    ].copy()

    cohort = cohort.rename(
        columns={
            "production_decision": "v7_decision",
            "production_intervention_selected":
                "v7_intervention_selected",
            "production_rank": "v7_production_rank",
        }
    )

    v8_subset = v8[
        [
            "transaction_id",
            "production_decision",
            "production_intervention_selected",
            "production_rank",
        ]
    ].copy()

    v8_subset = v8_subset.rename(
        columns={
            "production_decision": "v8_decision",
            "production_intervention_selected":
                "v8_intervention_selected",
            "production_rank": "v8_production_rank",
        }
    )

    cohort = cohort.merge(
        v8_subset,
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    if cohort["v8_decision"].isna().any():
        fail("Cohort contains production IDs missing from V8")

    if not (
        cohort["v7_decision"]
        == cohort["v8_decision"]
    ).all():
        fail(
            "V7/V8 decisions differ inside immutable cohort"
        )

    if not (
        cohort["v7_intervention_selected"]
        == cohort["v8_intervention_selected"]
    ).all():
        fail(
            "V7/V8 intervention selections differ"
        )

    cohort["target_group"] = np.where(
        cohort["v7_decision"] == "TARGET",
        "TARGET",
        "RETRY_ALL",
    )

    passed(
        f"Immutable production cohort created: "
        f"{len(cohort):,} rows"
    )

    passed(
        "V7/V8 production decisions identical"
    )

    return cohort


# ---------------------------------------------------------------------
# LOAD LEGITIMATE PRODUCTION OUTCOMES
# ---------------------------------------------------------------------

def load_production_outcomes():
    section("LOADING LEGITIMATE PRODUCTION OUTCOMES")

    if not PRODUCTION_OUTCOME_PATH.exists():
        warned(
            "Production outcome file not found. "
            "V11 will create an outcome-capture template."
        )

        return None

    outcomes = pd.read_csv(PRODUCTION_OUTCOME_PATH)

    print(
        f"[PASS] Production outcome source loaded: "
        f"{len(outcomes):,} rows, {len(outcomes.columns)} columns"
    )

    require_columns(
        outcomes,
        ["transaction_id"],
        "Production outcome source",
    )

    if not outcomes["transaction_id"].is_unique:
        warned(
            "Outcome source contains duplicate transaction IDs. "
            "Duplicates will be marked ambiguous and excluded."
        )

    # Normalize recovered.
    if "recovered" not in outcomes.columns:

        if "recovered_amount" in outcomes.columns:
            outcomes["recovered"] = (
                pd.to_numeric(
                    outcomes["recovered_amount"],
                    errors="coerce"
                )
                .fillna(0)
                > 0
            ).astype(int)

            passed(
                "Derived `recovered` from `recovered_amount`"
            )

        elif "recovery_flag" in outcomes.columns:
            outcomes["recovered"] = outcomes[
                "recovery_flag"
            ].apply(normalize_bool)

            passed(
                "Derived `recovered` from `recovery_flag`"
            )

        else:
            warned(
                "No observed recovery field found. "
                "Outcome records will not be usable until "
                "a legitimate recovery outcome is supplied."
            )

    if "recovered" in outcomes.columns:
        outcomes["recovered"] = outcomes[
            "recovered"
        ].apply(normalize_bool)

    if "recovered_amount" in outcomes.columns:
        outcomes["recovered_amount"] = pd.to_numeric(
            outcomes["recovered_amount"],
            errors="coerce"
        )

    return outcomes


# ---------------------------------------------------------------------
# LINK OUTCOMES TO PRODUCTION
# ---------------------------------------------------------------------

def link_outcomes(cohort, outcomes):
    section("BUILDING PRODUCTION OUTCOME LINKAGE")

    result = cohort.copy()

    result["outcome_match_status"] = "NO_OUTCOME_MATCH"
    result["observed_recovered"] = np.nan
    result["observed_recovered_amount"] = np.nan

    if outcomes is None:
        warned(
            "No production outcomes available. "
            "All cohort rows remain unmatched."
        )

        return result

    production_ids = set(result["transaction_id"])

    outcome_counts = (
        outcomes["transaction_id"]
        .value_counts()
    )

    ambiguous_ids = set(
        outcome_counts[
            outcome_counts > 1
        ].index
    )

    valid_outcomes = outcomes[
        ~outcomes["transaction_id"].isin(ambiguous_ids)
    ].copy()

    valid_outcomes = valid_outcomes[
        valid_outcomes["transaction_id"].isin(
            production_ids
        )
    ].copy()

    if len(ambiguous_ids) > 0:
        warned(
            f"Ambiguous outcome IDs excluded: "
            f"{len(ambiguous_ids):,}"
        )

    if valid_outcomes.empty:
        warned(
            "Zero production transaction IDs have "
            "legitimate observed outcomes."
        )

        return result

    outcome_columns = ["transaction_id"]

    if "recovered" in valid_outcomes.columns:
        outcome_columns.append("recovered")

    if "recovered_amount" in valid_outcomes.columns:
        outcome_columns.append("recovered_amount")

    valid_outcomes = valid_outcomes[
        outcome_columns
    ]

    result = result.merge(
        valid_outcomes,
        on="transaction_id",
        how="left",
        suffixes=("", "_outcome"),
    )

    result["observed_recovered"] = result[
        "recovered"
    ] if "recovered" in result.columns else np.nan

    result["observed_recovered_amount"] = (
        result["recovered_amount"]
        if "recovered_amount" in result.columns
        else np.nan
    )

    result.loc[
        result["transaction_id"].isin(ambiguous_ids),
        "outcome_match_status"
    ] = "AMBIGUOUS_OUTCOME"

    legitimate_match = (
        result["observed_recovered"].notna()
        | result["observed_recovered_amount"].notna()
    )

    result.loc[
        legitimate_match,
        "outcome_match_status"
    ] = "MATCHED_OBSERVED_OUTCOME"

    # Remove temporary merge columns if they exist.
    for col in [
        "recovered",
        "recovered_amount",
    ]:
        if col in result.columns:
            result.drop(columns=[col], inplace=True)

    return result


# ---------------------------------------------------------------------
# OUTCOME CAPTURE TEMPLATE
# ---------------------------------------------------------------------

def build_capture_template(cohort):
    capture = cohort[
        [
            "transaction_id",
            "customer_id",
            "target_group",
            "v7_decision",
            "v8_decision",
            "v7_intervention_selected",
            "v8_intervention_selected",
        ]
    ].copy()

    capture["outcome_recorded"] = False
    capture["recovered"] = pd.NA
    capture["recovered_amount"] = pd.NA
    capture["outcome_timestamp"] = pd.NA
    capture["outcome_source"] = pd.NA
    capture["outcome_validation_status"] = (
        "AWAITING_LEGITIMATE_PRODUCTION_OUTCOME"
    )

    return capture


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------

def calculate_evaluation(linked):
    section("CALCULATING PRODUCTION OUTCOME METRICS")

    total = len(linked)

    matched = (
        linked["outcome_match_status"]
        == "MATCHED_OBSERVED_OUTCOME"
    ).sum()

    ambiguous = (
        linked["outcome_match_status"]
        == "AMBIGUOUS_OUTCOME"
    ).sum()

    target_rows = linked[
        linked["target_group"] == "TARGET"
    ]

    retry_rows = linked[
        linked["target_group"] == "RETRY_ALL"
    ]

    target_matched = (
        target_rows["outcome_match_status"]
        == "MATCHED_OBSERVED_OUTCOME"
    ).sum()

    retry_matched = (
        retry_rows["outcome_match_status"]
        == "MATCHED_OBSERVED_OUTCOME"
    ).sum()

    observed = linked[
        linked["outcome_match_status"]
        == "MATCHED_OBSERVED_OUTCOME"
    ].copy()

    if matched > 0:

        observed_recovered = (
            pd.to_numeric(
                observed["observed_recovered"],
                errors="coerce"
            )
        )

        observed_rate = (
            observed_recovered.mean()
            if observed_recovered.notna().any()
            else np.nan
        )

        observed_value = (
            pd.to_numeric(
                observed[
                    "observed_recovered_amount"
                ],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

    else:
        observed_rate = np.nan
        observed_value = 0.0

    match_rate = matched / total

    target_match_rate = (
        target_matched / len(target_rows)
    )

    retry_match_rate = (
        retry_matched / len(retry_rows)
    )

    metrics = {
        "total_production_transactions": total,
        "target_transactions": len(target_rows),
        "retry_all_transactions": len(retry_rows),
        "matched_production_outcomes": int(matched),
        "ambiguous_outcomes": int(ambiguous),
        "overall_match_rate": match_rate,
        "target_match_rate": target_match_rate,
        "retry_all_match_rate": retry_match_rate,
        "observed_recovered_rows": int(
            observed_recovered.notna().sum()
            if matched > 0
            else 0
        ),
        "observed_recovery_rate": (
            observed_rate
            if not pd.isna(observed_rate)
            else np.nan
        ),
        "observed_recovered_value": float(
            observed_value
        ),
    }

    return metrics


# ---------------------------------------------------------------------
# GOVERNANCE AUDIT
# ---------------------------------------------------------------------

def governance_audit(metrics):
    section("RUNNING V11 GOVERNANCE AUDIT")

    audit_rows = []

    def add(control, status, detail):
        audit_rows.append(
            {
                "control": control,
                "status": status,
                "detail": detail,
            }
        )

    add(
        "V10.2 targeting",
        "PASS",
        "TOP_RATE 5% remains immutable",
    )

    add(
        "Target capacity",
        "PASS",
        "500 / 10,000 production rows",
    )

    add(
        "RETRY_ALL population",
        "PASS",
        "9,500 / 10,000 production rows",
    )

    add(
        "Artificial ID matching",
        "PASS",
        "No synthetic transaction IDs created",
    )

    add(
        "Artificial outcomes",
        "PASS",
        "No outcomes fabricated or inferred",
    )

    add(
        "Calibration cohort misuse",
        "PASS",
        "Calibration predictions are not treated as "
        "production outcomes",
    )

    add(
        "Oracle fields",
        "PASS",
        "Oracle/evaluation fields are not used "
        "for production decisions",
    )

    if metrics["matched_production_outcomes"] > 0:

        verdict = (
            "PRODUCTION_OUTCOME_COHORT_AVAILABLE_FOR_EVALUATION"
        )

        add(
            "Production outcome linkage",
            "PASS",
            (
                f"{metrics['matched_production_outcomes']:,} "
                "legitimate production outcomes matched"
            ),
        )

    else:

        verdict = (
            "OUTCOME_CAPTURE_REQUIRED_BEFORE_V7_V8_EVALUATION"
        )

        add(
            "Production outcome linkage",
            "WARN",
            "No legitimate production outcomes matched",
        )

    add(
        "V7 vs V8 improvement claim",
        "BLOCKED",
        "No improvement claim allowed until "
        "legitimate production outcomes exist",
    )

    return pd.DataFrame(audit_rows), verdict


# ---------------------------------------------------------------------
# BUILD DECISION OUTPUT
# ---------------------------------------------------------------------

def build_decisions(v7, linked):
    decisions = v7.copy()

    evaluation_columns = [
        "outcome_match_status",
        "observed_recovered",
        "observed_recovered_amount",
    ]

    decisions = decisions.merge(
        linked[
            ["transaction_id"] + evaluation_columns
        ],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    decisions["v11_outcome_evaluation_eligible"] = (
        decisions["outcome_match_status"]
        == "MATCHED_OBSERVED_OUTCOME"
    )

    decisions["v11_governance_verdict"] = np.where(
        decisions[
            "v11_outcome_evaluation_eligible"
        ],
        "OBSERVED_OUTCOME_AVAILABLE",
        "AWAITING_PRODUCTION_OUTCOME",
    )

    return decisions


# ---------------------------------------------------------------------
# WRITE SUMMARY
# ---------------------------------------------------------------------

def write_summary(metrics, verdict):
    summary = pd.DataFrame(
        [
            {
                **metrics,
                "v10_2_target_rate": TARGET_RATE,
                "v10_2_target_population": EXPECTED_TARGET,
                "v10_2_retry_all_population": EXPECTED_RETRY_ALL,
                "targeting_changed": "NO",
                "economic_assumptions_changed": "NO",
                "causal_uplift_changed": "NO",
                "oracle_fields_used": "NO",
                "artificial_outcomes": "NO",
                "artificial_id_matching": "NO",
                "production_improvement_claim_allowed": "NO",
                "final_verdict": verdict,
            }
        ]
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print(
        """
========================================================================
RECLAIM PRODUCTION PIPELINE V11
Production Outcome Capture + Cohort Evaluation
V10.2 TOP_RATE 5% targeting is IMMUTABLE
========================================================================
"""
    )

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    v7, v8 = load_production_baselines()

    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    validate_population(v7, v8)

    # -------------------------------------------------------------
    # Cohort
    # -------------------------------------------------------------

    cohort = build_cohort(v7, v8)

    # -------------------------------------------------------------
    # Outcomes
    # -------------------------------------------------------------

    outcomes = load_production_outcomes()

    # -------------------------------------------------------------
    # Linkage
    # -------------------------------------------------------------

    linked = link_outcomes(
        cohort,
        outcomes,
    )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    metrics = calculate_evaluation(linked)

    print(
        """
========================================================================
V11 OUTCOME LINKAGE SUMMARY
========================================================================
"""
    )

    print(
        f"Production transactions:      "
        f"{metrics['total_production_transactions']:,}"
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
        f"Ambiguous outcomes:           "
        f"{metrics['ambiguous_outcomes']:,}"
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

    if not pd.isna(
        metrics["observed_recovery_rate"]
    ):
        print(
            f"Observed recovery rate:       "
            f"{metrics['observed_recovery_rate']:.2%}"
        )
    else:
        print(
            "Observed recovery rate:       "
            "N/A"
        )

    print(
        f"Observed recovered value:      "
        f"₹{metrics['observed_recovered_value']:,.2f}"
    )

    # -------------------------------------------------------------
    # Governance
    # -------------------------------------------------------------

    audit, verdict = governance_audit(
        metrics
    )

    # -------------------------------------------------------------
    # Decisions
    # -------------------------------------------------------------

    decisions = build_decisions(
        v7,
        linked,
    )

    # -------------------------------------------------------------
    # Capture template
    # -------------------------------------------------------------

    capture = build_capture_template(
        cohort
    )

    if outcomes is not None:

        outcome_ids = set(
            outcomes["transaction_id"]
        )

        capture["outcome_recorded"] = (
            capture["transaction_id"]
            .isin(outcome_ids)
        )

        capture.loc[
            capture["outcome_recorded"],
            "outcome_validation_status"
        ] = "OUTCOME_SOURCE_MATCH_FOUND"

    # -------------------------------------------------------------
    # Write
    # -------------------------------------------------------------

    section("WRITING V11 OUTPUTS")

    decisions.to_csv(
        OUTPUT_DECISIONS,
        index=False,
    )

    cohort.to_csv(
        OUTPUT_COHORT,
        index=False,
    )

    capture.to_csv(
        OUTPUT_CAPTURE,
        index=False,
    )

    audit.to_csv(
        OUTPUT_AUDIT,
        index=False,
    )

    write_summary(
        metrics,
        verdict,
    )

    print(
        f"[PASS] Decisions:  {OUTPUT_DECISIONS}"
    )

    print(
        f"[PASS] Cohort:     {OUTPUT_COHORT}"
    )

    print(
        f"[PASS] Capture:    {OUTPUT_CAPTURE}"
    )

    print(
        f"[PASS] Summary:    {OUTPUT_SUMMARY}"
    )

    print(
        f"[PASS] Audit:      {OUTPUT_AUDIT}"
    )

    # -------------------------------------------------------------
    # Final
    # -------------------------------------------------------------

    section("FINAL V11 VERDICT")

    print(f"FINAL VERDICT: {verdict}")

    print()
    print(
        "V10.2 TOP_RATE 5% targeting remains untouched."
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

        print()
        print(
            "NEXT REQUIRED STEP:"
        )

        print(
            "Capture legitimate observed outcomes for "
            "the V10.2 production transaction IDs."
        )

        print(
            "Only after sufficient matched production "
            "outcomes exist should V7 vs V8 be evaluated."
        )

    print(
        """
========================================================================
RECLAIM PRODUCTION PIPELINE V11 COMPLETE
========================================================================
"""
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("[V11 FAILED]")
        print(str(exc))
        sys.exit(1)