"""
========================================================================
RECLAIM PRODUCTION PIPELINE V18
========================================================================
Deterministic Cross-Dataset Lineage Bridge Diagnostic

Purpose:
    Find legitimate existing-field relationships between:

        V10.2 / V7 / V8 production population
                    |
                    v
             legitimate outcomes

Important governance rules:
    - V10.2 TOP_RATE 5% targeting remains IMMUTABLE
    - No production outcome evaluation
    - No artificial transaction IDs
    - No artificial outcome generation
    - No random matching
    - No fuzzy matching
    - No oracle/evaluation fields
    - No outcome values used to manufacture linkage

This script only discovers whether existing datasets contain
legitimate deterministic lineage keys.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ======================================================================
# CONFIG
# ======================================================================

ROOT = Path(r"D:\RECLAIM")
GENERATED = ROOT / "data" / "generated"

V102_FILE = GENERATED / "causal_uplift_v10_2_predictions.csv"
V7_FILE = GENERATED / "reclaim_production_decisions_v7.csv"
V8_FILE = GENERATED / "reclaim_production_decisions_v8.csv"
OUTCOME_FILE = GENERATED / "recovery_outcomes.csv"

OUTPUT = GENERATED / "reclaim_production_v18_lineage_bridge.csv"
SUMMARY = GENERATED / "reclaim_production_v18_summary.csv"


# ======================================================================
# HELPERS
# ======================================================================

def normalize_column_name(col):
    return str(col).strip().lower()


def normalize_value(value):
    """
    Conservative normalization.

    We do NOT alter semantic values.
    Only remove surrounding whitespace and normalize strings.
    """
    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return None

        return value

    return value


def load_csv(path):
    print(f"[LOAD] {path}")

    if not path.exists():
        print(f"[WARN] File does not exist: {path}")
        return None

    df = pd.read_csv(path)

    df.columns = [normalize_column_name(c) for c in df.columns]

    print(f"       Rows: {len(df):,}")
    print(f"       Columns: {len(df.columns)}")

    return df


def find_common_columns(df_a, df_b):
    return sorted(set(df_a.columns).intersection(df_b.columns))


def exact_overlap(df_a, df_b, column):
    """
    Calculate deterministic exact overlap on one existing column.
    """

    if column not in df_a.columns or column not in df_b.columns:
        return None

    a = df_a[column].map(normalize_value).dropna()
    b = df_b[column].map(normalize_value).dropna()

    a_set = set(a)
    b_set = set(b)

    overlap = a_set.intersection(b_set)

    return {
        "column": column,
        "source_unique": len(a_set),
        "target_unique": len(b_set),
        "overlap": len(overlap),
        "source_match_rate": (
            len(overlap) / len(a_set) if len(a_set) else 0
        ),
        "target_match_rate": (
            len(overlap) / len(b_set) if len(b_set) else 0
        ),
    }


def assess_candidate(result):
    """
    Conservative ranking.

    We want fields that:
      1. exist in both datasets
      2. have substantial overlap
      3. are not obviously outcome-only fields
    """

    if result is None:
        return "INVALID"

    overlap = result["overlap"]
    source_rate = result["source_match_rate"]
    target_rate = result["target_match_rate"]

    if overlap == 0:
        return "NO_OVERLAP"

    if source_rate == 1.0 and target_rate == 1.0:
        return "EXACT_BIJECTIVE"

    if source_rate == 1.0:
        return "SOURCE_COMPLETE"

    if source_rate >= 0.9:
        return "HIGH_SOURCE_COVERAGE"

    if source_rate >= 0.5:
        return "PARTIAL"

    return "LOW_OVERLAP"


# ======================================================================
# LOAD DATA
# ======================================================================

print("=" * 72)
print("RECLAIM PRODUCTION PIPELINE V18")
print("=" * 72)

v102 = load_csv(V102_FILE)
v7 = load_csv(V7_FILE)
v8 = load_csv(V8_FILE)
outcomes = load_csv(OUTCOME_FILE)


if any(x is None for x in [v102, v7, v8, outcomes]):
    raise SystemExit(
        "[FAIL] One or more required datasets could not be loaded."
    )


# ======================================================================
# COMMON COLUMN INVENTORY
# ======================================================================

print()
print("=" * 72)
print("COMMON COLUMN INVENTORY")
print("=" * 72)

datasets = {
    "V10.2": v102,
    "V7": v7,
    "V8": v8,
    "OUTCOMES": outcomes,
}

for name, df in datasets.items():

    print()
    print(name)
    print("-" * 72)

    print(", ".join(df.columns))


# ======================================================================
# DIRECT PRODUCTION → OUTCOME FIELD ANALYSIS
# ======================================================================

print()
print("=" * 72)
print("DIRECT EXISTING-FIELD OVERLAP")
print("=" * 72)

records = []


production_datasets = {
    "V10.2": v102,
    "V7": v7,
    "V8": v8,
}


for prod_name, prod_df in production_datasets.items():

    common = find_common_columns(prod_df, outcomes)

    print()
    print(f"{prod_name} → OUTCOMES")
    print("-" * 72)

    for column in common:

        # Skip transaction ID because V17 already proved
        # it has zero overlap and the formats are different.
        if column == "transaction_id":
            continue

        result = exact_overlap(
            prod_df,
            outcomes,
            column
        )

        if result is None:
            continue

        result["source_dataset"] = prod_name
        result["target_dataset"] = "OUTCOMES"
        result["assessment"] = assess_candidate(result)

        records.append(result)

        print(
            f"{column:35s} "
            f"overlap={result['overlap']:6,d} "
            f"source={result['source_match_rate']:.2%} "
            f"target={result['target_match_rate']:.2%} "
            f"[{result['assessment']}]"
        )


# ======================================================================
# TWO-FIELD COMPOSITE KEY ANALYSIS
# ======================================================================

print()
print("=" * 72)
print("TWO-FIELD COMPOSITE KEY DIAGNOSTIC")
print("=" * 72)

composite_records = []


for prod_name, prod_df in production_datasets.items():

    common = find_common_columns(prod_df, outcomes)

    # Exclude obvious IDs and outcome fields.
    candidate_columns = [
        c for c in common
        if c != "transaction_id"
        and not c.endswith("_outcome")
        and "recovered" not in c
        and "recovery" not in c
    ]

    print()
    print(f"{prod_name} → OUTCOMES")
    print("-" * 72)

    # Only evaluate combinations of the first reasonable columns.
    # This prevents an uncontrolled combinatorial search.
    candidate_columns = candidate_columns[:20]

    for i in range(len(candidate_columns)):
        for j in range(i + 1, len(candidate_columns)):

            c1 = candidate_columns[i]
            c2 = candidate_columns[j]

            left = prod_df[[c1, c2]].copy()
            right = outcomes[[c1, c2]].copy()

            left[c1] = left[c1].map(normalize_value)
            left[c2] = left[c2].map(normalize_value)

            right[c1] = right[c1].map(normalize_value)
            right[c2] = right[c2].map(normalize_value)

            left = left.dropna()
            right = right.dropna()

            if len(left) == 0 or len(right) == 0:
                continue

            left_keys = set(
                zip(left[c1], left[c2])
            )

            right_keys = set(
                zip(right[c1], right[c2])
            )

            overlap = left_keys.intersection(right_keys)

            if len(overlap) == 0:
                continue

            source_rate = len(overlap) / len(left_keys)
            target_rate = len(overlap) / len(right_keys)

            record = {
                "source_dataset": prod_name,
                "target_dataset": "OUTCOMES",
                "field_1": c1,
                "field_2": c2,
                "source_unique_pairs": len(left_keys),
                "target_unique_pairs": len(right_keys),
                "overlap_pairs": len(overlap),
                "source_match_rate": source_rate,
                "target_match_rate": target_rate,
            }

            composite_records.append(record)

            print(
                f"{c1} + {c2} -> "
                f"overlap={len(overlap):,} "
                f"source={source_rate:.2%} "
                f"target={target_rate:.2%}"
            )


# ======================================================================
# CUSTOMER-LEVEL DIAGNOSTIC
# ======================================================================

print()
print("=" * 72)
print("CUSTOMER-LEVEL LINEAGE")
print("=" * 72)

if "customer_id" in v102.columns and "customer_id" in outcomes.columns:

    prod_customers = set(
        v102["customer_id"]
        .map(normalize_value)
        .dropna()
    )

    outcome_customers = set(
        outcomes["customer_id"]
        .map(normalize_value)
        .dropna()
    )

    customer_overlap = prod_customers.intersection(
        outcome_customers
    )

    print(f"V10.2 customers:        {len(prod_customers):,}")
    print(f"Outcome customers:      {len(outcome_customers):,}")
    print(f"Customer overlap:       {len(customer_overlap):,}")

    print(
        f"V10.2 customer coverage: "
        f"{len(customer_overlap) / len(prod_customers):.2%}"
    )

    print(
        f"Outcome customer coverage: "
        f"{len(customer_overlap) / len(outcome_customers):.2%}"
    )

else:

    print("[WARN] customer_id not available in both datasets.")


# ======================================================================
# SAVE SINGLE-FIELD RESULTS
# ======================================================================

single_df = pd.DataFrame(records)

if len(single_df):

    single_df = single_df.sort_values(
        [
            "source_match_rate",
            "target_match_rate",
            "overlap",
        ],
        ascending=False
    )


# ======================================================================
# SAVE COMPOSITE RESULTS
# ======================================================================

composite_df = pd.DataFrame(composite_records)

if len(composite_df):

    composite_df = composite_df.sort_values(
        [
            "source_match_rate",
            "target_match_rate",
            "overlap_pairs",
        ],
        ascending=False
    )


# ======================================================================
# WRITE OUTPUT
# ======================================================================

output_frames = []

if len(single_df):

    temp = single_df.copy()
    temp["analysis_type"] = "single_field"
    temp["field"] = temp["column"]

    output_frames.append(
        temp[
            [
                "analysis_type",
                "source_dataset",
                "target_dataset",
                "field",
                "source_unique",
                "target_unique",
                "overlap",
                "source_match_rate",
                "target_match_rate",
                "assessment",
            ]
        ]
    )


if len(composite_df):

    temp = composite_df.copy()
    temp["analysis_type"] = "two_field"
    temp["field"] = (
        temp["field_1"] + " + " + temp["field_2"]
    )

    temp["source_unique"] = temp["source_unique_pairs"]
    temp["target_unique"] = temp["target_unique_pairs"]
    temp["overlap"] = temp["overlap_pairs"]
    temp["assessment"] = temp.apply(
        lambda r: assess_candidate({
            "overlap": r["overlap"],
            "source_match_rate": r["source_match_rate"],
            "target_match_rate": r["target_match_rate"],
        }),
        axis=1,
    )

    output_frames.append(
        temp[
            [
                "analysis_type",
                "source_dataset",
                "target_dataset",
                "field",
                "source_unique",
                "target_unique",
                "overlap",
                "source_match_rate",
                "target_match_rate",
                "assessment",
            ]
        ]
    )


if output_frames:

    final_df = pd.concat(
        output_frames,
        ignore_index=True
    )

else:

    final_df = pd.DataFrame()


final_df.to_csv(
    OUTPUT,
    index=False
)


# ======================================================================
# SUMMARY
# ======================================================================

summary = {
    "v102_rows": len(v102),
    "v7_rows": len(v7),
    "v8_rows": len(v8),
    "outcome_rows": len(outcomes),
    "single_field_candidates": len(single_df),
    "two_field_candidates": len(composite_df),
    "exact_bijective_candidates": 0,
    "high_coverage_candidates": 0,
}


if len(final_df):

    summary["exact_bijective_candidates"] = int(
        (final_df["assessment"] == "EXACT_BIJECTIVE").sum()
    )

    summary["high_coverage_candidates"] = int(
        final_df["assessment"].isin(
            [
                "EXACT_BIJECTIVE",
                "SOURCE_COMPLETE",
                "HIGH_SOURCE_COVERAGE",
            ]
        ).sum()
    )


summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    SUMMARY,
    index=False
)


# ======================================================================
# FINAL VERDICT
# ======================================================================

print()
print("=" * 72)
print("V18 LINEAGE BRIDGE VERDICT")
print("=" * 72)

if summary["exact_bijective_candidates"] > 0:

    verdict = "DETERMINISTIC_LINEAGE_BRIDGE_FOUND"

elif summary["high_coverage_candidates"] > 0:

    verdict = "POTENTIAL_HIGH_COVERAGE_LINEAGE_BRIDGE"

elif len(final_df) > 0:

    verdict = "PARTIAL_LINEAGE_CANDIDATES_FOUND"

else:

    verdict = "NO_EXISTING_FIELD_LINEAGE_BRIDGE_FOUND"


print(f"FINAL VERDICT: {verdict}")

print()
print("Outputs:")
print(f"[PASS] Bridge analysis: {OUTPUT}")
print(f"[PASS] Summary:         {SUMMARY}")

print()
print("=" * 72)
print("RECLAIM PRODUCTION PIPELINE V18 COMPLETE")
print("=" * 72)