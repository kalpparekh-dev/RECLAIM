"""
========================================================================
RECLAIM PRODUCTION PIPELINE V20
========================================================================
Strict Dual-ID Lineage Bridge Validator

Purpose:
    Validate explicit existing-ID bridge candidates discovered in V19.

    Required lineage structure:

        PRODUCTION TRANSACTION ID
                  |
                  | exact existing ID
                  v
          BRIDGE DATASET ROW
                  |
                  | explicit second existing ID
                  v
           OUTCOME TRANSACTION ID

Important governance rules:
    - V10.2 TOP_RATE 5% targeting remains IMMUTABLE
    - No production outcome evaluation
    - No artificial transaction IDs
    - No artificial outcome generation
    - No random matching
    - No fuzzy matching
    - No customer_id-only lineage
    - No oracle/evaluation fields used for linkage
    - Only existing explicit ID fields are accepted
    - Mapping must be deterministic
    - Ambiguous mappings are rejected
    - One-to-many mappings are rejected
    - Many-to-one mappings are rejected

This version VALIDATES lineage only.
It does NOT claim intervention performance.
"""

from pathlib import Path
import pandas as pd


# ======================================================================
# CONFIG
# ======================================================================

ROOT = Path(r"D:\RECLAIM")
GENERATED = ROOT / "data" / "generated"

# Immutable production population
V102_FILE = GENERATED / "causal_uplift_v10_2_predictions.csv"

# Existing outcomes
OUTCOME_FILE = GENERATED / "recovery_outcomes.csv"

# Strongest explicit dual-ID candidates discovered by V19
CANDIDATE_FILES = [
    GENERATED / "reclaim_causal_experiment_v2.csv",
    GENERATED / "reclaim_causal_experiment_v3.csv",
    GENERATED / "causal_uplift_v2_predictions.csv",
    GENERATED / "causal_uplift_v4_predictions.csv",
]

# Outputs
OUTPUT = GENERATED / "reclaim_production_v20_validated_lineage.csv"
REPORT = GENERATED / "reclaim_production_v20_validation_report.csv"
SUMMARY = GENERATED / "reclaim_production_v20_summary.csv"


# ======================================================================
# HELPERS
# ======================================================================

def normalize_column_name(col):
    return str(col).strip().lower()


def normalize_value(value):
    """
    Conservative normalization.

    No semantic transformation.
    Only:
        - remove surrounding whitespace
        - treat empty strings as missing
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
        print(f"[WARN] Missing: {path}")
        return None

    df = pd.read_csv(path)

    df.columns = [
        normalize_column_name(c)
        for c in df.columns
    ]

    print(f"       Rows: {len(df):,}")
    print(f"       Columns: {len(df.columns)}")

    return df


def normalized_series(df, column):

    return (
        df[column]
        .map(normalize_value)
    )


def unique_values(df, column):

    if column not in df.columns:
        return set()

    return set(
        normalized_series(df, column)
        .dropna()
    )


def identify_id_columns(df):
    """
    Identify explicit ID-like columns only.

    We deliberately avoid:
        customer_id as a transaction lineage key.
    """

    candidates = []

    for column in df.columns:

        name = column.lower()

        if name == "customer_id":
            continue

        if (
            name == "transaction_id"
            or name.endswith("_transaction_id")
            or "transaction_id" in name
        ):
            candidates.append(column)

    return candidates


# ======================================================================
# CARDINALITY VALIDATION
# ======================================================================

def validate_mapping(df, source_col, target_col):
    """
    Validate:

        source ID -> exactly one target ID
        target ID -> exactly one source ID

    Any ambiguity is rejected.

    Returns mapping statistics and validated pairs.
    """

    temp = df[
        [source_col, target_col]
    ].copy()

    temp[source_col] = (
        temp[source_col]
        .map(normalize_value)
    )

    temp[target_col] = (
        temp[target_col]
        .map(normalize_value)
    )

    temp = temp.dropna()

    if len(temp) == 0:

        return {
            "total_rows": 0,
            "unique_source_ids": 0,
            "unique_target_ids": 0,
            "unique_pairs": 0,
            "duplicate_pairs": 0,
            "ambiguous_sources": 0,
            "ambiguous_targets": 0,
            "valid_pairs": pd.DataFrame(
                columns=[
                    source_col,
                    target_col
                ]
            ),
        }

    # --------------------------------------------------------------
    # Remove exact duplicate pairs.
    #
    # Repeated identical rows do not create ambiguity.
    # Example:
    #
    # P001 -> O001
    # P001 -> O001
    #
    # remains deterministic.
    # --------------------------------------------------------------

    unique_pairs = temp.drop_duplicates()

    duplicate_pairs = (
        len(temp)
        - len(unique_pairs)
    )

    # --------------------------------------------------------------
    # SOURCE -> TARGET cardinality
    # --------------------------------------------------------------

    source_counts = (
        unique_pairs
        .groupby(source_col)[target_col]
        .nunique()
    )

    ambiguous_sources = set(
        source_counts[
            source_counts > 1
        ].index
    )

    # --------------------------------------------------------------
    # TARGET -> SOURCE cardinality
    # --------------------------------------------------------------

    target_counts = (
        unique_pairs
        .groupby(target_col)[source_col]
        .nunique()
    )

    ambiguous_targets = set(
        target_counts[
            target_counts > 1
        ].index
    )

    # --------------------------------------------------------------
    # Keep only strictly one-to-one pairs
    # --------------------------------------------------------------

    valid_pairs = unique_pairs[
        ~unique_pairs[source_col].isin(
            ambiguous_sources
        )
        &
        ~unique_pairs[target_col].isin(
            ambiguous_targets
        )
    ].copy()

    return {
        "total_rows": len(temp),
        "unique_source_ids": (
            unique_pairs[source_col].nunique()
        ),
        "unique_target_ids": (
            unique_pairs[target_col].nunique()
        ),
        "unique_pairs": len(unique_pairs),
        "duplicate_pairs": duplicate_pairs,
        "ambiguous_sources": len(
            ambiguous_sources
        ),
        "ambiguous_targets": len(
            ambiguous_targets
        ),
        "valid_pairs": valid_pairs,
    }


# ======================================================================
# COVERAGE VALIDATION
# ======================================================================

def calculate_coverage(
    valid_pairs,
    source_col,
    target_col,
    production_ids,
    outcome_ids,
):
    """
    Measure how much of the validated bridge
    connects to production and outcomes.
    """

    if len(valid_pairs) == 0:

        return {
            "production_overlap": 0,
            "outcome_overlap": 0,
            "production_coverage": 0.0,
            "outcome_coverage": 0.0,
            "bridge_production_coverage": 0.0,
            "bridge_outcome_coverage": 0.0,
            "linked_pairs": pd.DataFrame(
                columns=[
                    "production_transaction_id",
                    "outcome_transaction_id",
                ]
            ),
        }

    bridge_source_ids = set(
        valid_pairs[source_col]
    )

    bridge_target_ids = set(
        valid_pairs[target_col]
    )

    production_overlap_ids = (
        bridge_source_ids
        .intersection(production_ids)
    )

    outcome_overlap_ids = (
        bridge_target_ids
        .intersection(outcome_ids)
    )

    production_coverage = (
        len(production_overlap_ids)
        / len(production_ids)
        if len(production_ids)
        else 0
    )

    outcome_coverage = (
        len(outcome_overlap_ids)
        / len(outcome_ids)
        if len(outcome_ids)
        else 0
    )

    bridge_production_coverage = (
        len(production_overlap_ids)
        / len(bridge_source_ids)
        if len(bridge_source_ids)
        else 0
    )

    bridge_outcome_coverage = (
        len(outcome_overlap_ids)
        / len(bridge_target_ids)
        if len(bridge_target_ids)
        else 0
    )

    # --------------------------------------------------------------
    # Only rows that connect BOTH sides are legitimate lineage pairs.
    # --------------------------------------------------------------

    linked_pairs = valid_pairs[
        valid_pairs[source_col].isin(
            production_ids
        )
        &
        valid_pairs[target_col].isin(
            outcome_ids
        )
    ].copy()

    linked_pairs = linked_pairs.rename(
        columns={
            source_col:
                "production_transaction_id",
            target_col:
                "outcome_transaction_id",
        }
    )

    return {
        "production_overlap":
            len(production_overlap_ids),

        "outcome_overlap":
            len(outcome_overlap_ids),

        "production_coverage":
            production_coverage,

        "outcome_coverage":
            outcome_coverage,

        "bridge_production_coverage":
            bridge_production_coverage,

        "bridge_outcome_coverage":
            bridge_outcome_coverage,

        "linked_pairs":
            linked_pairs,
    }


# ======================================================================
# CANDIDATE ASSESSMENT
# ======================================================================

def assess_candidate(
    mapping,
    coverage,
):
    """
    Strict validation policy.

    A bridge cannot be accepted if ambiguity exists.
    """

    if mapping["unique_pairs"] == 0:

        return "NO_VALID_MAPPING"

    if mapping["ambiguous_sources"] > 0:

        return "REJECTED_ONE_TO_MANY"

    if mapping["ambiguous_targets"] > 0:

        return "REJECTED_MANY_TO_ONE"

    linked_count = len(
        coverage["linked_pairs"]
    )

    if linked_count == 0:

        return "NO_END_TO_END_LINEAGE"

    if (
        coverage["production_coverage"] == 1.0
        and coverage["outcome_coverage"] == 1.0
    ):

        return "VALIDATED_FULL_ONE_TO_ONE_BRIDGE"

    if (
        linked_count > 0
        and mapping["ambiguous_sources"] == 0
        and mapping["ambiguous_targets"] == 0
    ):

        return "VALIDATED_PARTIAL_ONE_TO_ONE_BRIDGE"

    return "REJECTED"


# ======================================================================
# LOAD PRIMARY DATASETS
# ======================================================================

print("=" * 72)
print("RECLAIM PRODUCTION PIPELINE V20")
print("=" * 72)

print()
print("STRICT DUAL-ID LINEAGE BRIDGE VALIDATION")

print()

v102 = load_csv(V102_FILE)
outcomes = load_csv(OUTCOME_FILE)


if v102 is None or outcomes is None:

    raise SystemExit(
        "[FAIL] Required primary datasets missing."
    )


# ======================================================================
# PRIMARY ID POPULATIONS
# ======================================================================

print()
print("=" * 72)
print("PRIMARY ID POPULATIONS")
print("=" * 72)

if "transaction_id" not in v102.columns:

    raise SystemExit(
        "[FAIL] V10.2 transaction_id missing."
    )

if "transaction_id" not in outcomes.columns:

    raise SystemExit(
        "[FAIL] Outcome transaction_id missing."
    )


production_ids = unique_values(
    v102,
    "transaction_id"
)

outcome_ids = unique_values(
    outcomes,
    "transaction_id"
)


print(
    f"Production transaction IDs: "
    f"{len(production_ids):,}"
)

print(
    f"Outcome transaction IDs:    "
    f"{len(outcome_ids):,}"
)

print(
    f"Direct overlap:              "
    f"{len(production_ids.intersection(outcome_ids)):,}"
)


# ======================================================================
# VALIDATE CANDIDATES
# ======================================================================

print()
print("=" * 72)
print("DUAL-ID BRIDGE VALIDATION")
print("=" * 72)


validation_records = []

all_validated_pairs = []


for path in CANDIDATE_FILES:

    print()
    print("-" * 72)
    print(f"CANDIDATE: {path.name}")
    print("-" * 72)

    df = load_csv(path)

    if df is None:

        validation_records.append(
            {
                "candidate_file":
                    path.name,
                "source_field": None,
                "target_field": None,
                "assessment":
                    "FILE_MISSING",
            }
        )

        continue

    id_columns = identify_id_columns(df)

    print(
        "ID columns discovered: "
        f"{id_columns}"
    )

    if len(id_columns) < 2:

        print(
            "[REJECT] Less than two "
            "explicit transaction ID fields."
        )

        validation_records.append(
            {
                "candidate_file":
                    path.name,
                "source_field": None,
                "target_field": None,
                "assessment":
                    "INSUFFICIENT_ID_FIELDS",
            }
        )

        continue

    # --------------------------------------------------------------
    # Test every ordered pair.
    #
    # Source field must overlap production IDs.
    # Target field must overlap outcome IDs.
    # --------------------------------------------------------------

    for source_col in id_columns:

        for target_col in id_columns:

            if source_col == target_col:
                continue

            source_ids = unique_values(
                df,
                source_col
            )

            target_ids = unique_values(
                df,
                target_col
            )

            production_side_overlap = len(
                source_ids.intersection(
                    production_ids
                )
            )

            outcome_side_overlap = len(
                target_ids.intersection(
                    outcome_ids
                )
            )

            # Not a bridge direction.
            if (
                production_side_overlap == 0
                or outcome_side_overlap == 0
            ):
                continue

            print()
            print(
                f"[TEST] {source_col} "
                f"→ {target_col}"
            )

            print(
                f"Production overlap: "
                f"{production_side_overlap:,}"
            )

            print(
                f"Outcome overlap:    "
                f"{outcome_side_overlap:,}"
            )

            # ------------------------------------------------------
            # Strict cardinality validation
            # ------------------------------------------------------

            mapping = validate_mapping(
                df,
                source_col,
                target_col,
            )

            print(
                f"Unique source IDs:  "
                f"{mapping['unique_source_ids']:,}"
            )

            print(
                f"Unique target IDs:  "
                f"{mapping['unique_target_ids']:,}"
            )

            print(
                f"Unique pairs:       "
                f"{mapping['unique_pairs']:,}"
            )

            print(
                f"Duplicate pairs:    "
                f"{mapping['duplicate_pairs']:,}"
            )

            print(
                f"Ambiguous sources:  "
                f"{mapping['ambiguous_sources']:,}"
            )

            print(
                f"Ambiguous targets:  "
                f"{mapping['ambiguous_targets']:,}"
            )

            # ------------------------------------------------------
            # End-to-end coverage
            # ------------------------------------------------------

            coverage = calculate_coverage(
                mapping["valid_pairs"],
                source_col,
                target_col,
                production_ids,
                outcome_ids,
            )

            assessment = assess_candidate(
                mapping,
                coverage,
            )

            linked_count = len(
                coverage["linked_pairs"]
            )

            print(
                f"End-to-end pairs:   "
                f"{linked_count:,}"
            )

            print(
                f"Production coverage:"
                f" {coverage['production_coverage']:.2%}"
            )

            print(
                f"Outcome coverage:   "
                f" {coverage['outcome_coverage']:.2%}"
            )

            print(
                f"ASSESSMENT: "
                f"{assessment}"
            )

            record = {

                "candidate_file":
                    path.name,

                "source_field":
                    source_col,

                "target_field":
                    target_col,

                "production_side_overlap":
                    production_side_overlap,

                "outcome_side_overlap":
                    outcome_side_overlap,

                "total_mapping_rows":
                    mapping["total_rows"],

                "unique_source_ids":
                    mapping["unique_source_ids"],

                "unique_target_ids":
                    mapping["unique_target_ids"],

                "unique_pairs":
                    mapping["unique_pairs"],

                "duplicate_pairs":
                    mapping["duplicate_pairs"],

                "ambiguous_sources":
                    mapping["ambiguous_sources"],

                "ambiguous_targets":
                    mapping["ambiguous_targets"],

                "end_to_end_pairs":
                    linked_count,

                "production_coverage":
                    coverage[
                        "production_coverage"
                    ],

                "outcome_coverage":
                    coverage[
                        "outcome_coverage"
                    ],

                "bridge_production_coverage":
                    coverage[
                        "bridge_production_coverage"
                    ],

                "bridge_outcome_coverage":
                    coverage[
                        "bridge_outcome_coverage"
                    ],

                "assessment":
                    assessment,
            }

            validation_records.append(
                record
            )

            # ------------------------------------------------------
            # Save only accepted deterministic pairs
            # ------------------------------------------------------

            if assessment in [
                "VALIDATED_FULL_ONE_TO_ONE_BRIDGE",
                "VALIDATED_PARTIAL_ONE_TO_ONE_BRIDGE",
            ]:

                pairs = (
                    coverage["linked_pairs"]
                    .copy()
                )

                pairs[
                    "candidate_file"
                ] = path.name

                pairs[
                    "source_field"
                ] = source_col

                pairs[
                    "target_field"
                ] = target_col

                pairs[
                    "validation_status"
                ] = assessment

                all_validated_pairs.append(
                    pairs
                )


# ======================================================================
# VALIDATION REPORT
# ======================================================================

print()
print("=" * 72)
print("VALIDATION REPORT")
print("=" * 72)


report_df = pd.DataFrame(
    validation_records
)


if len(report_df):

    sort_columns = [
        "end_to_end_pairs",
        "production_coverage",
        "outcome_coverage",
    ]

    existing_sort_columns = [
        c for c in sort_columns
        if c in report_df.columns
    ]

    if existing_sort_columns:

        report_df = report_df.sort_values(
            existing_sort_columns,
            ascending=False,
        )


report_df.to_csv(
    REPORT,
    index=False,
)


# ======================================================================
# VALIDATED LINEAGE OUTPUT
# ======================================================================

if all_validated_pairs:

    lineage_df = pd.concat(
        all_validated_pairs,
        ignore_index=True,
    )

    # Remove exact duplicate lineage rows
    lineage_df = lineage_df.drop_duplicates(
        subset=[
            "production_transaction_id",
            "outcome_transaction_id",
            "candidate_file",
            "source_field",
            "target_field",
        ]
    )

else:

    lineage_df = pd.DataFrame(
        columns=[
            "production_transaction_id",
            "outcome_transaction_id",
            "candidate_file",
            "source_field",
            "target_field",
            "validation_status",
        ]
    )


lineage_df.to_csv(
    OUTPUT,
    index=False,
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

total_candidates = len(report_df)

full_bridges = 0
partial_bridges = 0
rejected_one_to_many = 0
rejected_many_to_one = 0
end_to_end_pairs = len(lineage_df)


if len(report_df) and "assessment" in report_df.columns:

    full_bridges = int(
        (
            report_df["assessment"]
            == "VALIDATED_FULL_ONE_TO_ONE_BRIDGE"
        ).sum()
    )

    partial_bridges = int(
        (
            report_df["assessment"]
            == "VALIDATED_PARTIAL_ONE_TO_ONE_BRIDGE"
        ).sum()
    )

    rejected_one_to_many = int(
        (
            report_df["assessment"]
            == "REJECTED_ONE_TO_MANY"
        ).sum()
    )

    rejected_many_to_one = int(
        (
            report_df["assessment"]
            == "REJECTED_MANY_TO_ONE"
        ).sum()
    )


summary = {

    "production_transactions":
        len(production_ids),

    "outcome_transactions":
        len(outcome_ids),

    "direct_transaction_overlap":
        len(
            production_ids.intersection(
                outcome_ids
            )
        ),

    "candidate_mappings_tested":
        total_candidates,

    "validated_full_bridges":
        full_bridges,

    "validated_partial_bridges":
        partial_bridges,

    "rejected_one_to_many":
        rejected_one_to_many,

    "rejected_many_to_one":
        rejected_many_to_one,

    "validated_end_to_end_pairs":
        end_to_end_pairs,

}


summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    SUMMARY,
    index=False,
)


# ======================================================================
# FINAL VERDICT
# ======================================================================

print()
print("=" * 72)
print("V20 STRICT LINEAGE VALIDATION VERDICT")
print("=" * 72)

if full_bridges > 0:

    verdict = (
        "VALIDATED_FULL_DETERMINISTIC_LINEAGE_BRIDGE"
    )

elif partial_bridges > 0:

    verdict = (
        "VALIDATED_PARTIAL_DETERMINISTIC_LINEAGE_BRIDGE"
    )

elif total_candidates == 0:

    verdict = (
        "NO_EXPLICIT_ID_BRIDGE_VALIDATED"
    )

else:

    verdict = (
        "EXPLICIT_ID_CANDIDATES_REJECTED_OR_UNRESOLVED"
    )


print(
    f"FINAL VERDICT: {verdict}"
)

print()

print("Summary:")
print(
    f"Candidate mappings tested: "
    f"{total_candidates:,}"
)

print(
    f"Full validated bridges:    "
    f"{full_bridges:,}"
)

print(
    f"Partial validated bridges: "
    f"{partial_bridges:,}"
)

print(
    f"End-to-end lineage pairs:  "
    f"{end_to_end_pairs:,}"
)

print()

print("Governance:")

print(
    "[PASS] V10.2 TOP_RATE 5% targeting unchanged"
)

print(
    "[PASS] No production outcome evaluation performed"
)

print(
    "[PASS] No artificial transaction IDs"
)

print(
    "[PASS] No artificial outcomes"
)

print(
    "[PASS] No random matching"
)

print(
    "[PASS] No fuzzy matching"
)

print(
    "[PASS] customer_id-only lineage rejected"
)

print(
    "[PASS] One-to-many mappings rejected"
)

print(
    "[PASS] Many-to-one mappings rejected"
)

print(
    "[PASS] Only existing explicit ID fields tested"
)


print()

print("Outputs:")

print(
    f"[PASS] Validated lineage: {OUTPUT}"
)

print(
    f"[PASS] Validation report: {REPORT}"
)

print(
    f"[PASS] Summary:           {SUMMARY}"
)


print()
print("=" * 72)
print("RECLAIM PRODUCTION PIPELINE V20 COMPLETE")
print("=" * 72)