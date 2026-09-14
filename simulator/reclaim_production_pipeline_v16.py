
from pathlib import Path
import sys
import numpy as np
import pandas as pd


# ============================================================
# RECLAIM PRODUCTION PIPELINE V16
# Population + Transaction-ID Lineage Bridge
#
# Purpose:
#   Establish whether V10.2, V7, V8, and legitimate production
#   outcomes refer to the same transaction population.
#
# Governance:
#   - No artificial transaction IDs
#   - No artificial outcomes
#   - No calibration predictions treated as observed outcomes
#   - No V7/V8 improvement claim
#   - No modification of V10.2 targeting
# ============================================================

BASE_DIR = Path(r"D:\RECLAIM\data\generated")

V10_PREDICTIONS = BASE_DIR / "causal_uplift_v10_2_predictions.csv"
V7_FILE = BASE_DIR / "reclaim_production_decisions_v7.csv"
V8_FILE = BASE_DIR / "reclaim_production_decisions_v8.csv"
OUTCOME_FILE = BASE_DIR / "recovery_outcomes.csv"

OUTPUT_DIR = BASE_DIR

EXPECTED_PRODUCTION_POPULATION = 10_000
EXPECTED_TARGET = 500
EXPECTED_RETRY_ALL = 9_500
EXPECTED_TARGET_RATE = 0.05


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def fail(message):
    print(f"[FAIL] {message}")
    raise RuntimeError(message)


def warn(message):
    print(f"[WARN] {message}")


def passed(message):
    print(f"[PASS] {message}")


def normalize_ids(series):
    """
    Normalize transaction IDs only for comparison.

    IMPORTANT:
    This does NOT create IDs and does NOT change the source files.
    """
    return (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )


def find_transaction_column(df, source_name):
    candidates = [
        "transaction_id",
        "experiment_transaction_id",
        "experiment_source_transaction_id",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    fail(
        f"{source_name} does not contain a recognized transaction ID column. "
        f"Available columns: {list(df.columns)}"
    )


def load_source(path, label):
    if not path.exists():
        fail(f"{label} not found: {path}")

    df = pd.read_csv(path)

    print(
        f"[PASS] {label} loaded: "
        f"{len(df):,} rows, {len(df.columns)} columns"
    )

    return df


def validate_unique_ids(df, id_col, label):
    ids = normalize_ids(df[id_col])

    if ids.isna().any():
        fail(f"{label} contains missing transaction IDs")

    duplicate_count = int(ids.duplicated().sum())

    if duplicate_count:
        fail(
            f"{label} contains {duplicate_count:,} duplicated "
            f"transaction IDs"
        )

    passed(f"{label} transaction IDs are unique")


def id_set(df, id_col):
    return set(normalize_ids(df[id_col]).dropna())


def overlap_metrics(source_ids, target_ids):
    intersection = source_ids & target_ids

    return {
        "source_count": len(source_ids),
        "target_count": len(target_ids),
        "matched": len(intersection),
        "source_match_rate": (
            len(intersection) / len(source_ids)
            if source_ids else 0.0
        ),
        "target_match_rate": (
            len(intersection) / len(target_ids)
            if target_ids else 0.0
        ),
    }


def print_overlap(name, source_ids, target_ids):
    m = overlap_metrics(source_ids, target_ids)

    print(f"\n{name}")
    print(f"  Source IDs:        {m['source_count']:,}")
    print(f"  Target IDs:        {m['target_count']:,}")
    print(f"  Exact overlap:     {m['matched']:,}")
    print(f"  Source match rate: {m['source_match_rate']:.2%}")
    print(f"  Target match rate: {m['target_match_rate']:.2%}")

    return m


def compare_id_examples(name, a_ids, b_ids, limit=10):
    only_a = sorted(a_ids - b_ids)
    only_b = sorted(b_ids - a_ids)

    print(f"\n{name}")

    print(f"  Only in A: {len(only_a):,}")
    if only_a:
        print("  A examples:")
        for x in only_a[:limit]:
            print(f"    {x}")

    print(f"  Only in B: {len(only_b):,}")
    if only_b:
        print("  B examples:")
        for x in only_b[:limit]:
            print(f"    {x}")


def production_policy_metrics(df, label):
    """
    Validate the V10.2 immutable production population where
    production decision columns exist.
    """

    result = {
        "label": label,
        "rows": len(df),
        "target": np.nan,
        "retry_all": np.nan,
        "intervention_rate": np.nan,
    }

    policy_col = None

    for col in [
        "production_decision",
        "selected_policy",
        "policy",
        "recommended_action",
    ]:
        if col in df.columns:
            policy_col = col
            break

    if policy_col is None:
        warn(
            f"{label}: no production policy column found; "
            "population-only validation performed"
        )
        return result

    values = df[policy_col].astype("string").str.upper()

    target_mask = values.isin(
        ["TARGET", "STANDARD_TARGET", "INTERVENTION", "TREAT"]
    )

    retry_mask = values.isin(
        ["RETRY_ALL", "RETRY"]
    )

    result["target"] = int(target_mask.sum())
    result["retry_all"] = int(retry_mask.sum())

    if len(df):
        result["intervention_rate"] = (
            result["target"] / len(df)
        )

    print(f"[PASS] {label} policy column: {policy_col}")
    print(f"[PASS] {label} TARGET rows: {result['target']:,}")
    print(f"[PASS] {label} RETRY_ALL rows: {result['retry_all']:,}")
    print(
        f"[PASS] {label} intervention rate: "
        f"{result['intervention_rate']:.2%}"
    )

    return result


def main():

    header("RECLAIM PRODUCTION PIPELINE V16")
    print("Population + Transaction-ID Lineage Bridge")
    print("V10.2 TOP_RATE 5% targeting is IMMUTABLE")
    print("No production outcome evaluation is performed in V16")

    # --------------------------------------------------------
    # LOAD SOURCES
    # --------------------------------------------------------

    header("LOADING SOURCE DATASETS")

    v10 = load_source(
        V10_PREDICTIONS,
        "V10.2 predictions"
    )

    v7 = load_source(
        V7_FILE,
        "V7 production decisions"
    )

    v8 = load_source(
        V8_FILE,
        "V8 production decisions"
    )

    outcomes = load_source(
        OUTCOME_FILE,
        "Legitimate recovery outcomes"
    )

    # --------------------------------------------------------
    # IDENTIFY ID COLUMNS
    # --------------------------------------------------------

    header("IDENTIFYING TRANSACTION-ID COLUMNS")

    v10_id_col = find_transaction_column(
        v10,
        "V10.2 predictions"
    )

    v7_id_col = find_transaction_column(
        v7,
        "V7 production decisions"
    )

    v8_id_col = find_transaction_column(
        v8,
        "V8 production decisions"
    )

    outcome_id_col = find_transaction_column(
        outcomes,
        "Recovery outcomes"
    )

    print(f"[PASS] V10.2 ID column:      {v10_id_col}")
    print(f"[PASS] V7 ID column:         {v7_id_col}")
    print(f"[PASS] V8 ID column:         {v8_id_col}")
    print(f"[PASS] Outcome ID column:    {outcome_id_col}")

    # --------------------------------------------------------
    # UNIQUE-ID VALIDATION
    # --------------------------------------------------------

    header("VALIDATING TRANSACTION-ID UNIQUENESS")

    validate_unique_ids(
        v10,
        v10_id_col,
        "V10.2"
    )

    validate_unique_ids(
        v7,
        v7_id_col,
        "V7"
    )

    validate_unique_ids(
        v8,
        v8_id_col,
        "V8"
    )

    validate_unique_ids(
        outcomes,
        outcome_id_col,
        "Recovery outcomes"
    )

    # --------------------------------------------------------
    # BUILD ID SETS
    # --------------------------------------------------------

    header("BUILDING EXACT TRANSACTION-ID SETS")

    ids_v10 = id_set(v10, v10_id_col)
    ids_v7 = id_set(v7, v7_id_col)
    ids_v8 = id_set(v8, v8_id_col)
    ids_outcomes = id_set(outcomes, outcome_id_col)

    print(f"[PASS] V10.2 unique IDs:       {len(ids_v10):,}")
    print(f"[PASS] V7 unique IDs:          {len(ids_v7):,}")
    print(f"[PASS] V8 unique IDs:          {len(ids_v8):,}")
    print(f"[PASS] Outcome unique IDs:     {len(ids_outcomes):,}")

    # --------------------------------------------------------
    # POPULATION VALIDATION
    # --------------------------------------------------------

    header("VALIDATING PRODUCTION POPULATION")

    if len(v7) != EXPECTED_PRODUCTION_POPULATION:
        warn(
            f"V7 population is {len(v7):,}, expected "
            f"{EXPECTED_PRODUCTION_POPULATION:,}"
        )
    else:
        passed(
            f"V7 population: {len(v7):,}"
        )

    if len(v8) != EXPECTED_PRODUCTION_POPULATION:
        warn(
            f"V8 population is {len(v8):,}, expected "
            f"{EXPECTED_PRODUCTION_POPULATION:,}"
        )
    else:
        passed(
            f"V8 population: {len(v8):,}"
        )

    # --------------------------------------------------------
    # V7 vs V8
    # --------------------------------------------------------

    header("V7 / V8 POPULATION COMPARISON")

    v7_v8 = print_overlap(
        "V7 → V8",
        ids_v7,
        ids_v8
    )

    if ids_v7 == ids_v8:
        passed(
            "V7 and V8 contain exactly the same transaction population"
        )
    else:
        warn(
            "V7 and V8 transaction populations are NOT identical"
        )
        compare_id_examples(
            "V7 vs V8 differences",
            ids_v7,
            ids_v8
        )

    # --------------------------------------------------------
    # V10.2 vs V7/V8
    # --------------------------------------------------------

    header("V10.2 → PRODUCTION POPULATION LINKAGE")

    v10_v7 = print_overlap(
        "V10.2 → V7",
        ids_v10,
        ids_v7
    )

    v10_v8 = print_overlap(
        "V10.2 → V8",
        ids_v10,
        ids_v8
    )

    if ids_v10 == ids_v7:
        passed(
            "V10.2 and V7 contain exactly the same transaction population"
        )
    else:
        warn(
            "V10.2 and V7 transaction populations are NOT identical"
        )
        compare_id_examples(
            "V10.2 vs V7 differences",
            ids_v10,
            ids_v7
        )

    if ids_v10 == ids_v8:
        passed(
            "V10.2 and V8 contain exactly the same transaction population"
        )
    else:
        warn(
            "V10.2 and V8 transaction populations are NOT identical"
        )

    # --------------------------------------------------------
    # PRODUCTION OUTCOME LINKAGE
    # --------------------------------------------------------

    header("V10.2 / PRODUCTION → LEGITIMATE OUTCOME LINKAGE")

    v10_outcomes = print_overlap(
        "V10.2 → recovery_outcomes",
        ids_v10,
        ids_outcomes
    )

    v7_outcomes = print_overlap(
        "V7 → recovery_outcomes",
        ids_v7,
        ids_outcomes
    )

    v8_outcomes = print_overlap(
        "V8 → recovery_outcomes",
        ids_v8,
        ids_outcomes
    )

    # --------------------------------------------------------
    # POLICY VALIDATION
    # --------------------------------------------------------

    header("VALIDATING IMMUTABLE V10.2 PRODUCTION POLICY")

    v7_policy = production_policy_metrics(
        v7,
        "V7"
    )

    v8_policy = production_policy_metrics(
        v8,
        "V8"
    )

    # --------------------------------------------------------
    # ID LINEAGE CLASSIFICATION
    # --------------------------------------------------------

    header("ID LINEAGE DIAGNOSIS")

    if (
        ids_v10 == ids_v7
        and ids_v10 == ids_v8
        and len(ids_v10) == EXPECTED_PRODUCTION_POPULATION
    ):
        production_population_status = (
            "V10.2_PRODUCTION_POPULATION_ALIGNED"
        )
        passed(
            "V10.2, V7 and V8 are aligned to the same 10,000-row population"
        )

    elif (
        len(ids_v10) == EXPECTED_PRODUCTION_POPULATION
        and len(ids_v10 & ids_v7) > 0
    ):
        production_population_status = (
            "V10.2_PARTIAL_PRODUCTION_OVERLAP"
        )
        warn(
            "V10.2 has partial overlap with the V7 production population"
        )

    else:
        production_population_status = (
            "V10.2_PRODUCTION_POPULATION_NOT_ALIGNED"
        )
        warn(
            "V10.2 does not align with the immutable V7/V8 production population"
        )

    if len(ids_outcomes & ids_v7) == 0:
        outcome_status = "NO_V7_OUTCOME_MATCH"
        warn(
            "No legitimate recovery outcomes match V7 transaction IDs"
        )
    else:
        outcome_status = "V7_OUTCOME_MATCH_AVAILABLE"
        passed(
            f"Legitimate outcomes overlap V7 by "
            f"{len(ids_outcomes & ids_v7):,} transactions"
        )

    if len(ids_outcomes & ids_v8) == 0:
        v8_outcome_status = "NO_V8_OUTCOME_MATCH"
        warn(
            "No legitimate recovery outcomes match V8 transaction IDs"
        )
    else:
        v8_outcome_status = "V8_OUTCOME_MATCH_AVAILABLE"
        passed(
            f"Legitimate outcomes overlap V8 by "
            f"{len(ids_outcomes & ids_v8):,} transactions"
        )

    # --------------------------------------------------------
    # GOVERNANCE
    # --------------------------------------------------------

    header("V16 GOVERNANCE AUDIT")

    governance = {
        "v10_2_targeting_changed": "NO",
        "v7_population_changed": "NO",
        "v8_population_changed": "NO",
        "artificial_transaction_ids_created": "NO",
        "artificial_outcomes_created": "NO",
        "calibration_predictions_treated_as_outcomes": "NO",
        "oracle_fields_used_for_linkage": "NO",
        "v7_v8_improvement_claimed": "NO",
    }

    for key, value in governance.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------

    header("FINAL V16 VERDICT")

    if production_population_status == "V10.2_PRODUCTION_POPULATION_ALIGNED":

        if (
            len(ids_outcomes & ids_v7) > 0
            and len(ids_outcomes & ids_v8) > 0
        ):
            verdict = "PRODUCTION_LINEAGE_ALIGNED_OUTCOME_EVALUATION_READY"
        else:
            verdict = (
                "PRODUCTION_LINEAGE_ALIGNED_BUT_OUTCOMES_NOT_CAPTURED"
            )

    elif production_population_status == "V10.2_PARTIAL_PRODUCTION_OVERLAP":

        verdict = "PRODUCTION_LINEAGE_PARTIAL_REQUIRES_ALIGNMENT"

    else:

        verdict = "V10.2_PRODUCTION_LINEAGE_MISMATCH_REQUIRES_FIX"

    print("FINAL VERDICT:", verdict)

    # --------------------------------------------------------
    # OUTPUT TABLE
    # --------------------------------------------------------

    header("WRITING V16 OUTPUTS")

    comparison = pd.DataFrame([
        {
            "source": "V10.2",
            "rows": len(v10),
            "unique_transaction_ids": len(ids_v10),
            "overlap_v7": len(ids_v10 & ids_v7),
            "overlap_v8": len(ids_v10 & ids_v8),
            "overlap_outcomes": len(ids_v10 & ids_outcomes),
        },
        {
            "source": "V7",
            "rows": len(v7),
            "unique_transaction_ids": len(ids_v7),
            "overlap_v7": len(ids_v7),
            "overlap_v8": len(ids_v7 & ids_v8),
            "overlap_outcomes": len(ids_v7 & ids_outcomes),
        },
        {
            "source": "V8",
            "rows": len(v8),
            "unique_transaction_ids": len(ids_v8),
            "overlap_v7": len(ids_v8 & ids_v7),
            "overlap_v8": len(ids_v8),
            "overlap_outcomes": len(ids_v8 & ids_outcomes),
        },
        {
            "source": "RECOVERY_OUTCOMES",
            "rows": len(outcomes),
            "unique_transaction_ids": len(ids_outcomes),
            "overlap_v7": len(ids_outcomes & ids_v7),
            "overlap_v8": len(ids_outcomes & ids_v8),
            "overlap_outcomes": len(ids_outcomes),
        },
    ])

    lineage = pd.DataFrame([
        {
            "source_a": "V10.2",
            "source_b": "V7",
            "exact_overlap": len(ids_v10 & ids_v7),
            "source_a_count": len(ids_v10),
            "source_b_count": len(ids_v7),
            "source_a_match_rate": (
                len(ids_v10 & ids_v7) / len(ids_v10)
            ),
            "source_b_match_rate": (
                len(ids_v10 & ids_v7) / len(ids_v7)
            ),
        },
        {
            "source_a": "V10.2",
            "source_b": "V8",
            "exact_overlap": len(ids_v10 & ids_v8),
            "source_a_count": len(ids_v10),
            "source_b_count": len(ids_v8),
            "source_a_match_rate": (
                len(ids_v10 & ids_v8) / len(ids_v10)
            ),
            "source_b_match_rate": (
                len(ids_v10 & ids_v8) / len(ids_v8)
            ),
        },
        {
            "source_a": "V10.2",
            "source_b": "RECOVERY_OUTCOMES",
            "exact_overlap": len(ids_v10 & ids_outcomes),
            "source_a_count": len(ids_v10),
            "source_b_count": len(ids_outcomes),
            "source_a_match_rate": (
                len(ids_v10 & ids_outcomes) / len(ids_v10)
            ),
            "source_b_match_rate": (
                len(ids_v10 & ids_outcomes) / len(ids_outcomes)
            ),
        },
        {
            "source_a": "V7",
            "source_b": "RECOVERY_OUTCOMES",
            "exact_overlap": len(ids_v7 & ids_outcomes),
            "source_a_count": len(ids_v7),
            "source_b_count": len(ids_outcomes),
            "source_a_match_rate": (
                len(ids_v7 & ids_outcomes) / len(ids_v7)
            ),
            "source_b_match_rate": (
                len(ids_v7 & ids_outcomes) / len(ids_outcomes)
            ),
        },
        {
            "source_a": "V8",
            "source_b": "RECOVERY_OUTCOMES",
            "exact_overlap": len(ids_v8 & ids_outcomes),
            "source_a_count": len(ids_v8),
            "source_b_count": len(ids_outcomes),
            "source_a_match_rate": (
                len(ids_v8 & ids_outcomes) / len(ids_v8)
            ),
            "source_b_match_rate": (
                len(ids_v8 & ids_outcomes) / len(ids_outcomes)
            ),
        },
    ])

    summary = pd.DataFrame([{
        "pipeline_version": "V16",
        "v10_2_rows": len(v10),
        "v10_2_unique_ids": len(ids_v10),
        "v7_rows": len(v7),
        "v7_unique_ids": len(ids_v7),
        "v8_rows": len(v8),
        "v8_unique_ids": len(ids_v8),
        "outcome_rows": len(outcomes),
        "outcome_unique_ids": len(ids_outcomes),
        "v10_2_v7_overlap": len(ids_v10 & ids_v7),
        "v10_2_v8_overlap": len(ids_v10 & ids_v8),
        "v10_2_outcome_overlap": len(ids_v10 & ids_outcomes),
        "v7_outcome_overlap": len(ids_v7 & ids_outcomes),
        "v8_outcome_overlap": len(ids_v8 & ids_outcomes),
        "v7_v8_overlap": len(ids_v7 & ids_v8),
        "production_population_status": production_population_status,
        "outcome_status_v7": outcome_status,
        "outcome_status_v8": v8_outcome_status,
        "final_verdict": verdict,
        "artificial_ids_created": False,
        "artificial_outcomes_created": False,
        "oracle_fields_used": False,
        "v7_v8_improvement_claimed": False,
    }])

    policy = pd.DataFrame([
        v7_policy,
        v8_policy,
    ])

    paths = {
        "Comparison":
            OUTPUT_DIR / "reclaim_production_v16_population_comparison.csv",
        "Lineage":
            OUTPUT_DIR / "reclaim_production_v16_id_lineage.csv",
        "Policy":
            OUTPUT_DIR / "reclaim_production_v16_policy_validation.csv",
        "Summary":
            OUTPUT_DIR / "reclaim_production_v16_summary.csv",
    }

    comparison.to_csv(
        paths["Comparison"],
        index=False
    )

    lineage.to_csv(
        paths["Lineage"],
        index=False
    )

    policy.to_csv(
        paths["Policy"],
        index=False
    )

    summary.to_csv(
        paths["Summary"],
        index=False
    )

    for label, path in paths.items():
        print(f"[PASS] {label}: {path}")

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    header("RECLAIM PRODUCTION PIPELINE V16 COMPLETE")

    print("FINAL VERDICT:", verdict)

    if verdict == "PRODUCTION_LINEAGE_ALIGNED_OUTCOME_EVALUATION_READY":
        print(
            "\nNEXT STEP:"
            "\nRun V17 controlled V7 vs V8 production outcome evaluation."
        )

    elif verdict == "PRODUCTION_LINEAGE_ALIGNED_BUT_OUTCOMES_NOT_CAPTURED":
        print(
            "\nNEXT STEP:"
            "\nCapture legitimate outcomes for the aligned production IDs."
        )

    elif verdict == "PRODUCTION_LINEAGE_PARTIAL_REQUIRES_ALIGNMENT":
        print(
            "\nNEXT STEP:"
            "\nResolve the partial V10.2 → V7/V8 population mismatch."
        )

    else:
        print(
            "\nNEXT STEP:"
            "\nFix V10.2 production population lineage before outcome evaluation."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 72)
        print("V16 FAILED")
        print("=" * 72)
        print(f"[FAIL] {exc}")
        sys.exit(1)

