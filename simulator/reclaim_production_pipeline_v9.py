"""
RECLAIM PRODUCTION PIPELINE V9
Controlled Outcome Evaluation

V7 Baseline vs V8 Calibration-Aware Action Layer
V10.2 TOP_RATE 5% targeting is IMMUTABLE

IMPORTANT GOVERNANCE:
- V9 does NOT modify production targeting.
- V9 does NOT modify the causal uplift model.
- V9 does NOT use oracle/evaluation columns for decisions.
- V9 evaluates observed recovery outcomes only where transaction IDs
  actually overlap.
- If there is no production overlap, V9 reports that outcome testing
  cannot establish a V7-vs-V8 production winner.
- Calibration-cohort outcomes are evaluated separately.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\RECLAIM")

DATA_DIR = BASE_DIR / "data" / "generated"

V7_FILE = DATA_DIR / "reclaim_production_decisions_v7.csv"
V8_FILE = DATA_DIR / "reclaim_production_decisions_v8.csv"

OUTCOME_FILE = DATA_DIR / "calibrated_recovery_predictions.csv"

V9_DECISIONS_FILE = DATA_DIR / "reclaim_production_decisions_v9.csv"
V9_COMPARISON_FILE = DATA_DIR / "reclaim_production_v9_comparison.csv"
V9_SUMMARY_FILE = DATA_DIR / "reclaim_production_v9_summary.csv"


EXPECTED_ROWS = 10000
TARGET_RATE = 0.05
EXPECTED_TARGET = 500
EXPECTED_RETRY_ALL = 9500


# ============================================================
# HELPERS
# ============================================================

def banner(text):
    print("=" * 72)
    print(text)
    print("=" * 72)


def section(text):
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def passed(text):
    print(f"[PASS] {text}")


def warning(text):
    print(f"[WARN] {text}")


def failed(text):
    print(f"[FAIL] {text}")


def fail(text):
    failed(text)
    raise ValueError(text)


def money(x):
    return f"₹{float(x):,.2f}"


def require_file(path):
    if not path.exists():
        fail(f"Required file not found: {path}")


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs():

    section("VALIDATING INPUTS")

    require_file(V7_FILE)
    require_file(V8_FILE)
    require_file(OUTCOME_FILE)

    v7 = pd.read_csv(V7_FILE)
    v8 = pd.read_csv(V8_FILE)
    outcomes = pd.read_csv(OUTCOME_FILE)

    passed(
        f"V7 loaded: {len(v7):,} rows, {len(v7.columns)} columns"
    )

    passed(
        f"V8 loaded: {len(v8):,} rows, {len(v8.columns)} columns"
    )

    passed(
        f"Calibration outcomes loaded: "
        f"{len(outcomes):,} rows, {len(outcomes.columns)} columns"
    )

    passed(f"Outcome source: {OUTCOME_FILE}")

    return v7, v8, outcomes


# ============================================================
# POPULATION / GOVERNANCE VALIDATION
# ============================================================

def validate_population(df, label):

    required = [
        "transaction_id",
        "production_decision",
        "production_intervention_selected",
        "production_target_rate",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        fail(
            f"{label} missing required production columns: {missing}"
        )

    rows = len(df)

    if rows != EXPECTED_ROWS:
        fail(
            f"{label} population changed: expected "
            f"{EXPECTED_ROWS:,}, got {rows:,}"
        )

    target = int(
        (df["production_decision"] == "TARGET").sum()
    )

    retry = int(
        (df["production_decision"] == "RETRY_ALL").sum()
    )

    intervention_rate = (
        df["production_intervention_selected"].mean()
    )

    if target != EXPECTED_TARGET:
        fail(
            f"{label} TARGET population mismatch: "
            f"expected {EXPECTED_TARGET}, got {target}"
        )

    if retry != EXPECTED_RETRY_ALL:
        fail(
            f"{label} RETRY_ALL population mismatch: "
            f"expected {EXPECTED_RETRY_ALL}, got {retry}"
        )

    if not np.isclose(intervention_rate, TARGET_RATE, atol=1e-12):
        fail(
            f"{label} intervention rate changed: "
            f"{intervention_rate:.6%}"
        )

    passed(
        f"{label} population preserved: "
        f"{target} TARGET / {retry} RETRY_ALL / "
        f"{intervention_rate:.2%}"
    )

    return {
        "rows": rows,
        "target": target,
        "retry_all": retry,
        "intervention_rate": intervention_rate,
    }


def validate_governance(v7, v8):

    section("VALIDATING POPULATION AND GOVERNANCE")

    v7_stats = validate_population(v7, "V7")
    v8_stats = validate_population(v8, "V8")

    v7_ids = set(v7["transaction_id"].astype(str))
    v8_ids = set(v8["transaction_id"].astype(str))

    if v7_ids != v8_ids:
        fail(
            "V7/V8 transaction populations are not identical"
        )

    passed(
        "V7/V8 production targeting population identical"
    )

    # Targeting policy
    for label, df in [("V7", v7), ("V8", v8)]:

        if "production_policy_type" in df.columns:

            policies = (
                df["production_policy_type"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if policies != ["TOP_RATE"] and set(policies) != {"TOP_RATE"}:
                fail(
                    f"{label} targeting policy is not TOP_RATE: "
                    f"{policies}"
                )

        if "production_target_rate" in df.columns:

            rates = pd.to_numeric(
                df["production_target_rate"],
                errors="coerce"
            )

            if not np.allclose(
                rates,
                TARGET_RATE,
                atol=1e-12
            ):
                fail(
                    f"{label} contains production target rates "
                    f"other than 5%"
                )

    passed("V10.2 TOP_RATE 5% targeting preserved")


# ============================================================
# OUTCOME NORMALIZATION
# ============================================================

def normalize_outcomes(outcomes):

    section("NORMALIZING RECOVERY OUTCOME SCHEMA")

    outcomes = outcomes.copy()

    if "transaction_id" not in outcomes.columns:
        fail(
            "Recovery outcome file does not contain transaction_id"
        )

    outcomes["transaction_id"] = (
        outcomes["transaction_id"]
        .astype(str)
        .str.strip()
    )

    if "recovered" not in outcomes.columns:

        if "recovered_amount" in outcomes.columns:

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

        elif "recovery_flag" in outcomes.columns:

            outcomes["recovered"] = pd.to_numeric(
                outcomes["recovery_flag"],
                errors="coerce"
            ).fillna(0).astype(int)

            passed(
                "Derived `recovered` from `recovery_flag`"
            )

        else:
            fail(
                "Recovery outcome missing `recovered`, "
                "`recovered_amount`, and `recovery_flag`"
            )

    else:

        outcomes["recovered"] = pd.to_numeric(
            outcomes["recovered"],
            errors="coerce"
        ).fillna(0)

        outcomes["recovered"] = (
            outcomes["recovered"] > 0
        ).astype(int)

    if "recovered_amount" not in outcomes.columns:

        if "amount" in outcomes.columns:

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
                "Rebuilt recovered_amount from amount/recovered"
            )

        else:
            fail(
                "Cannot rebuild recovered_amount because "
                "`amount` is missing"
            )

    outcomes["recovered_amount"] = pd.to_numeric(
        outcomes["recovered_amount"],
        errors="coerce"
    ).fillna(0)

    if outcomes["transaction_id"].duplicated().any():
        warning(
            "Duplicate outcome transaction IDs detected; "
            "keeping the first observation for controlled matching"
        )

        outcomes = (
            outcomes
            .drop_duplicates(
                subset=["transaction_id"],
                keep="first"
            )
            .copy()
        )

    passed("Recovery outcome schema normalized")
    passed("Outcome transaction IDs are unique")

    return outcomes


# ============================================================
# MATCHING
# ============================================================

def match_outcomes(v7, v8, outcomes):

    section("BUILDING CONTROLLED OUTCOME EVALUATION")

    production_ids = set(
        v7["transaction_id"]
        .astype(str)
    )

    outcome_ids = set(
        outcomes["transaction_id"]
        .astype(str)
    )

    overlap = production_ids.intersection(
        outcome_ids
    )

    matched_count = len(overlap)

    passed(
        "V7/V8 transaction population is identical"
    )

    passed(
        f"Matched observed outcomes: "
        f"{matched_count:,} / {len(production_ids):,}"
    )

    if matched_count == 0:

        warning(
            "No production transaction IDs overlap with "
            "the calibration outcome population."
        )

        warning(
            "Production V7-vs-V8 outcome comparison cannot "
            "be established from the available outcome file."
        )

        return None

    outcome_subset = outcomes[
        outcomes["transaction_id"].isin(overlap)
    ].copy()

    v7_match = v7[
        v7["transaction_id"].isin(overlap)
    ].copy()

    v8_match = v8[
        v8["transaction_id"].isin(overlap)
    ].copy()

    return v7_match, v8_match, outcome_subset


# ============================================================
# CALIBRATION COHORT EVALUATION
# ============================================================

def evaluate_calibration_cohort(outcomes):

    section("CALIBRATION COHORT OUTCOME EVALUATION")

    n = len(outcomes)

    if n == 0:
        warning("No calibration outcomes available")
        return {
            "cohort_rows": 0,
            "recovered_rows": 0,
            "recovery_rate": np.nan,
            "recovered_value": 0.0,
        }

    recovered_rows = int(
        outcomes["recovered"].sum()
    )

    recovery_rate = (
        recovered_rows / n
    )

    recovered_value = float(
        outcomes["recovered_amount"].sum()
    )

    passed(
        f"Calibration cohort rows: {n:,}"
    )

    passed(
        f"Observed recovered rows: "
        f"{recovered_rows:,}"
    )

    passed(
        f"Observed recovery rate: "
        f"{recovery_rate:.2%}"
    )

    passed(
        f"Observed recovered value: "
        f"{money(recovered_value)}"
    )

    return {
        "cohort_rows": n,
        "recovered_rows": recovered_rows,
        "recovery_rate": recovery_rate,
        "recovered_value": recovered_value,
    }


# ============================================================
# PRODUCTION OUTCOME COMPARISON
# ============================================================

def evaluate_production_outcomes(
    matched_data,
    v7,
    v8
):

    if matched_data is None:

        return {
            "matched": False,
            "matched_rows": 0,
            "v7_recovery_rate": np.nan,
            "v8_recovery_rate": np.nan,
            "v7_recovered_value": 0.0,
            "v8_recovered_value": 0.0,
            "winner": "NOT_TESTABLE",
        }

    v7_match, v8_match, outcomes = matched_data

    merged_v7 = v7_match.merge(
        outcomes[
            [
                "transaction_id",
                "recovered",
                "recovered_amount",
            ]
        ],
        on="transaction_id",
        how="inner",
    )

    merged_v8 = v8_match.merge(
        outcomes[
            [
                "transaction_id",
                "recovered",
                "recovered_amount",
            ]
        ],
        on="transaction_id",
        how="inner",
    )

    v7_recovery_rate = float(
        merged_v7["recovered"].mean()
    )

    v8_recovery_rate = float(
        merged_v8["recovered"].mean()
    )

    v7_recovered_value = float(
        merged_v7["recovered_amount"].sum()
    )

    v8_recovered_value = float(
        merged_v8["recovered_amount"].sum()
    )

    if np.isclose(
        v7_recovered_value,
        v8_recovered_value
    ):
        winner = "NO_DIFFERENCE"

    elif v8_recovered_value > v7_recovered_value:
        winner = "V8_CALIBRATION_AWARE"

    else:
        winner = "V7_BASELINE"

    return {
        "matched": True,
        "matched_rows": len(merged_v7),
        "v7_recovery_rate": v7_recovery_rate,
        "v8_recovery_rate": v8_recovery_rate,
        "v7_recovered_value": v7_recovered_value,
        "v8_recovered_value": v8_recovered_value,
        "winner": winner,
    }


# ============================================================
# BUILD V9 DECISION OUTPUT
# ============================================================

def build_v9_decisions(
    v7,
    v8,
    production_eval,
    calibration_eval
):

    section("BUILDING V9 CONTROLLED EVALUATION OUTPUT")

    df = v8.copy()

    df["v9_outcome_match_available"] = (
        production_eval["matched"]
    )

    df["v9_matched_outcome_count"] = (
        production_eval["matched_rows"]
    )

    if production_eval["matched"]:

        df["v9_evaluation_status"] = (
            "OBSERVED_OUTCOME_AVAILABLE"
        )

        df["v9_evaluation_winner"] = (
            production_eval["winner"]
        )

    else:

        df["v9_evaluation_status"] = (
            "NO_MATCHED_PRODUCTION_OUTCOME"
        )

        df["v9_evaluation_winner"] = (
            "NOT_TESTABLE"
        )

    df["v9_calibration_cohort_rows"] = (
        calibration_eval["cohort_rows"]
    )

    df["v9_calibration_observed_recovery_rate"] = (
        calibration_eval["recovery_rate"]
    )

    df["v9_governance_targeting_changed"] = False
    df["v9_governance_retry_all_changed"] = False
    df["v9_governance_causal_model_changed"] = False
    df["v9_governance_oracle_used"] = False

    return df


# ============================================================
# COMPARISON REPORT
# ============================================================

def build_comparison(
    v7,
    v8,
    production_eval
):

    rows = []

    rows.append({
        "metric": "population",
        "v7": len(v7),
        "v8": len(v8),
        "difference": len(v8) - len(v7),
    })

    rows.append({
        "metric": "target_population",
        "v7": int(
            (v7["production_decision"] == "TARGET").sum()
        ),
        "v8": int(
            (v8["production_decision"] == "TARGET").sum()
        ),
        "difference": (
            int(
                (v8["production_decision"] == "TARGET").sum()
            )
            -
            int(
                (v7["production_decision"] == "TARGET").sum()
            )
        ),
    })

    rows.append({
        "metric": "retry_all_population",
        "v7": int(
            (v7["production_decision"] == "RETRY_ALL").sum()
        ),
        "v8": int(
            (v8["production_decision"] == "RETRY_ALL").sum()
        ),
        "difference": (
            int(
                (v8["production_decision"] == "RETRY_ALL").sum()
            )
            -
            int(
                (v7["production_decision"] == "RETRY_ALL").sum()
            )
        ),
    })

    rows.append({
        "metric": "intervention_rate",
        "v7": float(
            v7["production_intervention_selected"].mean()
        ),
        "v8": float(
            v8["production_intervention_selected"].mean()
        ),
        "difference": (
            float(
                v8["production_intervention_selected"].mean()
            )
            -
            float(
                v7["production_intervention_selected"].mean()
            )
        ),
    })

    rows.append({
        "metric": "matched_production_outcomes",
        "v7": production_eval["matched_rows"],
        "v8": production_eval["matched_rows"],
        "difference": 0,
    })

    rows.append({
        "metric": "production_outcome_testable",
        "v7": production_eval["matched"],
        "v8": production_eval["matched"],
        "difference": 0,
    })

    rows.append({
        "metric": "v7_recovery_rate",
        "v7": production_eval["v7_recovery_rate"],
        "v8": np.nan,
        "difference": np.nan,
    })

    rows.append({
        "metric": "v8_recovery_rate",
        "v7": np.nan,
        "v8": production_eval["v8_recovery_rate"],
        "difference": np.nan,
    })

    rows.append({
        "metric": "v7_recovered_value",
        "v7": production_eval["v7_recovered_value"],
        "v8": np.nan,
        "difference": np.nan,
    })

    rows.append({
        "metric": "v8_recovered_value",
        "v7": np.nan,
        "v8": production_eval["v8_recovered_value"],
        "difference": np.nan,
    })

    rows.append({
        "metric": "outcome_winner",
        "v7": production_eval["winner"],
        "v8": production_eval["winner"],
        "difference": np.nan,
    })

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    v7,
    v8,
    outcomes,
    calibration_eval,
    production_eval
):

    production_testable = production_eval["matched"]

    if production_testable:
        verdict = (
            "PRODUCTION_OUTCOME_TEST_COMPLETED"
        )
    else:
        verdict = (
            "OUTCOME_TEST_INSUFFICIENT_PRODUCTION_MATCH"
        )

    summary = {

        "pipeline_version": "V9",

        "v7_rows": len(v7),
        "v8_rows": len(v8),

        "v7_target": int(
            (v7["production_decision"] == "TARGET").sum()
        ),

        "v8_target": int(
            (v8["production_decision"] == "TARGET").sum()
        ),

        "v7_retry_all": int(
            (v7["production_decision"] == "RETRY_ALL").sum()
        ),

        "v8_retry_all": int(
            (v8["production_decision"] == "RETRY_ALL").sum()
        ),

        "v7_intervention_rate": float(
            v7["production_intervention_selected"].mean()
        ),

        "v8_intervention_rate": float(
            v8["production_intervention_selected"].mean()
        ),

        "calibration_outcome_rows": len(outcomes),

        "matched_production_outcomes":
            production_eval["matched_rows"],

        "production_outcome_testable":
            production_testable,

        "calibration_cohort_recovery_rate":
            calibration_eval["recovery_rate"],

        "calibration_cohort_recovered_value":
            calibration_eval["recovered_value"],

        "v7_observed_recovery_rate":
            production_eval["v7_recovery_rate"],

        "v8_observed_recovery_rate":
            production_eval["v8_recovery_rate"],

        "v7_observed_recovered_value":
            production_eval["v7_recovered_value"],

        "v8_observed_recovered_value":
            production_eval["v8_recovered_value"],

        "outcome_winner":
            production_eval["winner"],

        "v10_2_target_rate": TARGET_RATE,

        "v10_2_target_population": EXPECTED_TARGET,

        "oracle_fields_used": False,

        "targeting_changed": False,

        "retry_all_changed": False,

        "causal_uplift_changed": False,

        "economic_assumptions_changed": False,

        "calibration_changes_targeting": False,

        "calibration_changes_action_layer_only": True,

        "financial_improvement_claimed": False,

        "verdict": verdict,
    }

    return pd.DataFrame(
        [summary]
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    v7,
    v8,
    outcomes,
    calibration_eval,
    production_eval
):

    section("V9 CONTROLLED OUTCOME SUMMARY")

    print(
        f"V7 transactions:              {len(v7):,}"
    )

    print(
        f"V8 transactions:              {len(v8):,}"
    )

    print(
        f"V7 TARGET:                    "
        f"{(v7['production_decision'] == 'TARGET').sum():,}"
    )

    print(
        f"V8 TARGET:                    "
        f"{(v8['production_decision'] == 'TARGET').sum():,}"
    )

    print(
        f"V7 RETRY_ALL:                 "
        f"{(v7['production_decision'] == 'RETRY_ALL').sum():,}"
    )

    print(
        f"V8 RETRY_ALL:                 "
        f"{(v8['production_decision'] == 'RETRY_ALL').sum():,}"
    )

    print(
        f"Intervention rate:            "
        f"{v8['production_intervention_selected'].mean():.2%}"
    )

    print(
        f"Calibration outcome rows:     "
        f"{len(outcomes):,}"
    )

    print(
        f"Matched production outcomes:  "
        f"{production_eval['matched_rows']:,}"
    )

    print(
        f"Calibration cohort recovery:  "
        f"{calibration_eval['recovery_rate']:.2%}"
    )

    print(
        f"Calibration recovered value:  "
        f"{money(calibration_eval['recovered_value'])}"
    )

    print()
    print("GOVERNANCE")
    print(
        "  V10.2 TOP_RATE 5% targeting: PRESERVED"
    )
    print(
        "  Target population changed:   NO"
    )
    print(
        "  RETRY_ALL changed:           NO"
    )
    print(
        "  Causal uplift changed:       NO"
    )
    print(
        "  Economic assumptions changed: NO"
    )
    print(
        "  Oracle fields used:          NO"
    )
    print(
        "  Financial improvement claim: NO"
    )

    print()

    if production_eval["matched"]:

        print(
            "PRODUCTION OUTCOME TEST:"
        )

        print(
            f"  V7 observed recovery: "
            f"{production_eval['v7_recovery_rate']:.2%}"
        )

        print(
            f"  V8 observed recovery: "
            f"{production_eval['v8_recovery_rate']:.2%}"
        )

        print(
            f"  V7 recovered value: "
            f"{money(production_eval['v7_recovered_value'])}"
        )

        print(
            f"  V8 recovered value: "
            f"{money(production_eval['v8_recovered_value'])}"
        )

        print(
            f"  Outcome winner: "
            f"{production_eval['winner']}"
        )

        print(
            "FINAL VERDICT: "
            "PRODUCTION_OUTCOME_TEST_COMPLETED"
        )

    else:

        print(
            "PRODUCTION OUTCOME TEST:"
        )

        print(
            "  NO MATCHED PRODUCTION OUTCOMES"
        )

        print(
            "  V7 vs V8 production outcome comparison "
            "is NOT statistically/test-wise established."
        )

        print(
            "FINAL VERDICT: "
            "OUTCOME_TEST_INSUFFICIENT_PRODUCTION_MATCH"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "RECLAIM PRODUCTION PIPELINE V9\n"
        "Controlled Outcome Evaluation\n"
        "V7 Baseline vs V8 Calibration-Aware Action Layer\n"
        "V10.2 TOP_RATE 5% targeting is IMMUTABLE"
    )

    v7, v8, outcomes = load_inputs()

    validate_governance(v7, v8)

    outcomes = normalize_outcomes(outcomes)

    matched_data = match_outcomes(
        v7,
        v8,
        outcomes
    )

    calibration_eval = evaluate_calibration_cohort(
        outcomes
    )

    production_eval = evaluate_production_outcomes(
        matched_data,
        v7,
        v8
    )

    v9_decisions = build_v9_decisions(
        v7,
        v8,
        production_eval,
        calibration_eval
    )

    comparison = build_comparison(
        v7,
        v8,
        production_eval
    )

    summary = build_summary(
        v7,
        v8,
        outcomes,
        calibration_eval,
        production_eval
    )

    section("WRITING V9 OUTPUTS")

    v9_decisions.to_csv(
        V9_DECISIONS_FILE,
        index=False
    )

    comparison.to_csv(
        V9_COMPARISON_FILE,
        index=False
    )

    summary.to_csv(
        V9_SUMMARY_FILE,
        index=False
    )

    passed(
        f"Decisions:  {V9_DECISIONS_FILE}"
    )

    passed(
        f"Comparison: {V9_COMPARISON_FILE}"
    )

    passed(
        f"Summary:    {V9_SUMMARY_FILE}"
    )

    print_summary(
        v7,
        v8,
        outcomes,
        calibration_eval,
        production_eval
    )

    section("RECLAIM PRODUCTION PIPELINE V9 COMPLETE")

    if production_eval["matched"]:

        print(
            "FINAL VERDICT: "
            "PRODUCTION_OUTCOME_TEST_COMPLETED"
        )

    else:

        print(
            "FINAL VERDICT: "
            "OUTCOME_TEST_INSUFFICIENT_PRODUCTION_MATCH"
        )

    print(
        "V10.2 5% targeting remains untouched."
    )

    print(
        "No oracle/evaluation field was used for decisions."
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:

        print()
        print("=" * 72)
        print("V9 FAILED")
        print("=" * 72)
        print(f"[FAIL] {exc}")

        sys.exit(1)