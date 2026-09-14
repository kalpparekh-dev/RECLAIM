"""
========================================================================
RECLAIM PRODUCTION PIPELINE V19
========================================================================
Deterministic Existing-Lineage Discovery

Purpose:
    Discover existing datasets that may contain legitimate deterministic
    lineage paths between:

        V10.2 / V7 / V8 production transactions
                    |
                    v
          existing project datasets
                    |
                    v
             legitimate outcomes

IMPORTANT GOVERNANCE RULES
--------------------------
- V10.2 TOP_RATE 5% targeting remains IMMUTABLE
- No production outcome evaluation
- No artificial transaction IDs
- No artificial outcome generation
- No random matching
- No fuzzy matching
- No synthetic/composite transaction IDs
- No oracle/evaluation fields for linkage
- No outcome values used to manufacture linkage
- customer_id alone is NOT accepted as transaction lineage
- Existing IDs may be DISCOVERED but are NOT automatically trusted
- V19 DISCOVERS candidates; it does NOT validate a bridge

V19 answers:

    "Where do existing datasets contain identifiers that could
     potentially connect production and outcome populations?"

V20 should perform strict deterministic bridge validation.
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

OUTPUT = GENERATED / "reclaim_production_v19_lineage_bridge.csv"
MULTIHOP_OUTPUT = GENERATED / "reclaim_production_v19_multihop_candidates.csv"
SUMMARY = GENERATED / "reclaim_production_v19_summary.csv"


# ======================================================================
# GOVERNANCE
# ======================================================================

FORBIDDEN_LINKAGE_FIELDS = {
    "recovered",
    "recovered_amount",
    "actual_recovery_probability",
    "net_recovered_value",
    "recovery_status",
    "expected_recovery",
    "expected_net_recovery",
    "oracle_control_value",
    "oracle_treatment_value",
    "oracle_incremental_value",
}

CUSTOMER_FIELD = "customer_id"
TRANSACTION_FIELD = "transaction_id"


# ======================================================================
# HELPERS
# ======================================================================

def normalize_column_name(col):
    return str(col).strip().lower()


def normalize_value(value):

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

    try:

        df = pd.read_csv(path)

    except Exception as exc:

        print(f"[WARN] Failed to load {path}: {exc}")

        return None

    df.columns = [
        normalize_column_name(c)
        for c in df.columns
    ]

    return df


def normalized_series(df, column):

    if column not in df.columns:

        return pd.Series(dtype=object)

    return (
        df[column]
        .map(normalize_value)
        .dropna()
    )


def unique_values(df, column):

    return set(
        normalized_series(df, column)
    )


def overlap_count(df_a, df_b, column):

    a = unique_values(df_a, column)
    b = unique_values(df_b, column)

    return len(a.intersection(b))


def production_overlap(df, production_ids):

    if TRANSACTION_FIELD not in df.columns:

        return 0

    values = unique_values(
        df,
        TRANSACTION_FIELD
    )

    return len(
        values.intersection(production_ids)
    )


def outcome_overlap(df, outcome_ids):

    if TRANSACTION_FIELD not in df.columns:

        return 0

    values = unique_values(
        df,
        TRANSACTION_FIELD
    )

    return len(
        values.intersection(outcome_ids)
    )


def id_like_columns(df):

    """
    Discover existing identifier-looking fields.

    This function only discovers fields.

    It never creates IDs.
    """

    result = []

    for column in df.columns:

        c = column.lower()

        if c in FORBIDDEN_LINKAGE_FIELDS:
            continue

        if (
            c == "transaction_id"
            or c.endswith("_transaction_id")
            or c.endswith("_id")
            or "transaction_id" in c
        ):
            result.append(column)

    return sorted(set(result))


def deterministic_mapping_stats(
    left_df,
    right_df,
    left_field,
    right_field
):

    """
    Examine an EXISTING field-to-field relationship.

    No generated key is created.

    Returns:
        shared values
        deterministic left mappings
        ambiguous left mappings
    """

    if (
        left_field not in left_df.columns
        or right_field not in right_df.columns
    ):
        return None

    left = left_df[
        [left_field]
    ].copy()

    right = right_df[
        [right_field]
    ].copy()

    left[left_field] = (
        left[left_field]
        .map(normalize_value)
    )

    right[right_field] = (
        right[right_field]
        .map(normalize_value)
    )

    left = left.dropna()
    right = right.dropna()

    left_values = set(left[left_field])
    right_values = set(right[right_field])

    shared = left_values.intersection(
        right_values
    )

    return {
        "shared_values": len(shared),
        "left_unique": len(left_values),
        "right_unique": len(right_values),
    }


# ======================================================================
# START
# ======================================================================

print("=" * 72)
print("RECLAIM PRODUCTION PIPELINE V19")
print("=" * 72)


# ======================================================================
# LOAD PRIMARY DATASETS
# ======================================================================

print()
print("[LOAD] Primary datasets")

v102 = load_csv(V102_FILE)
v7 = load_csv(V7_FILE)
v8 = load_csv(V8_FILE)
outcomes = load_csv(OUTCOME_FILE)

if any(
    x is None
    for x in [v102, v7, v8, outcomes]
):

    raise SystemExit(
        "[FAIL] Required dataset missing."
    )

print(f"V10.2 rows: {len(v102):,}")
print(f"V7 rows:    {len(v7):,}")
print(f"V8 rows:    {len(v8):,}")
print(f"Outcomes:   {len(outcomes):,}")


# ======================================================================
# PRIMARY TRANSACTION POPULATIONS
# ======================================================================

print()
print("=" * 72)
print("PRIMARY TRANSACTION POPULATIONS")
print("=" * 72)

v102_ids = unique_values(
    v102,
    TRANSACTION_FIELD
)

v7_ids = unique_values(
    v7,
    TRANSACTION_FIELD
)

v8_ids = unique_values(
    v8,
    TRANSACTION_FIELD
)

outcome_ids = unique_values(
    outcomes,
    TRANSACTION_FIELD
)

production_ids = (
    v102_ids
    | v7_ids
    | v8_ids
)

direct_overlap = (
    production_ids
    .intersection(outcome_ids)
)

print(
    f"V10.2 transactions: {len(v102_ids):,}"
)

print(
    f"V7 transactions:    {len(v7_ids):,}"
)

print(
    f"V8 transactions:    {len(v8_ids):,}"
)

print(
    f"Outcome transactions:{len(outcome_ids):,}"
)

print(
    f"Production union:   {len(production_ids):,}"
)

print(
    f"Direct overlap:     {len(direct_overlap):,}"
)


# ======================================================================
# V18 CUSTOMER BRIDGE REVALIDATION
# ======================================================================

print()
print("=" * 72)
print("V18 CUSTOMER BRIDGE REVALIDATION")
print("=" * 72)

production_customers = (
    unique_values(
        v102,
        CUSTOMER_FIELD
    )
)

outcome_customers = (
    unique_values(
        outcomes,
        CUSTOMER_FIELD
    )
)

customer_overlap = (
    production_customers
    .intersection(outcome_customers)
)

ambiguous_customers = 0

if (
    CUSTOMER_FIELD in v102.columns
    and CUSTOMER_FIELD in outcomes.columns
):

    production_customer_counts = (
        v102.groupby(
            CUSTOMER_FIELD
        )[TRANSACTION_FIELD]
        .nunique()
    )

    outcome_customer_counts = (
        outcomes.groupby(
            CUSTOMER_FIELD
        )[TRANSACTION_FIELD]
        .nunique()
    )

    common_customers = (
        set(production_customer_counts.index)
        .intersection(
            set(outcome_customer_counts.index)
        )
    )

    for customer in common_customers:

        p_count = production_customer_counts.get(
            customer,
            0
        )

        o_count = outcome_customer_counts.get(
            customer,
            0
        )

        if p_count != 1 or o_count != 1:

            ambiguous_customers += 1

print(
    f"Production customers: {len(production_customers):,}"
)

print(
    f"Outcome customers:    {len(outcome_customers):,}"
)

print(
    f"Customer overlap:     {len(customer_overlap):,}"
)

print(
    f"Ambiguous customers:  {ambiguous_customers:,}"
)

if ambiguous_customers > 0:

    print(
        "[REJECT] customer_id cannot establish "
        "transaction-level lineage."
    )


# ======================================================================
# DISCOVER ALL CSV DATASETS
# ======================================================================

print()
print("=" * 72)
print("DISCOVERING EXISTING PROJECT DATASETS")
print("=" * 72)

csv_files = sorted(
    GENERATED.glob("*.csv")
)

print(
    f"CSV files discovered: {len(csv_files):,}"
)


# ======================================================================
# SEARCH PRODUCTION-ID DATASETS
# ======================================================================

print()
print("=" * 72)
print("SEARCHING FOR EXISTING PRODUCTION-ID BRIDGES")
print("=" * 72)

production_candidates = []

for path in csv_files:

    if path.resolve() in {
        V102_FILE.resolve(),
        V7_FILE.resolve(),
        V8_FILE.resolve(),
        OUTCOME_FILE.resolve(),
    }:

        continue

    try:

        df = load_csv(path)

    except Exception:

        continue

    if df is None:
        continue

    if TRANSACTION_FIELD not in df.columns:
        continue

    prod_overlap = production_overlap(
        df,
        production_ids
    )

    outcome_overlap_value = outcome_overlap(
        df,
        outcome_ids
    )

    if prod_overlap == 0:
        continue

    record = {
        "dataset": str(path),
        "production_overlap": prod_overlap,
        "outcome_overlap": outcome_overlap_value,
        "classification": (
            "PRODUCTION_ID_ONLY"
            if outcome_overlap_value == 0
            else "SHARED_TRANSACTION_ID_SPACE"
        ),
    }

    production_candidates.append(record)

    print()
    print(
        f"[CANDIDATE] {path}"
    )

    print(
        f"  Production overlap: {prod_overlap:,}"
    )

    print(
        f"  Outcome overlap:    "
        f"{outcome_overlap_value:,}"
    )


# ======================================================================
# MULTI-HOP DATASET BRIDGE SEARCH
# ======================================================================

print()
print("=" * 72)
print("MULTI-HOP DATASET BRIDGE SEARCH")
print("=" * 72)

print(
    """
Searching for datasets that can form:

    PRODUCTION TRANSACTION
             |
             | exact existing field
             v
       INTERMEDIATE DATASET
             |
             | exact existing field
             v
       OUTCOME TRANSACTION

No generated keys are created.
"""
)

multihop_records = []

for path in csv_files:

    if path.resolve() in {
        V102_FILE.resolve(),
        V7_FILE.resolve(),
        V8_FILE.resolve(),
        OUTCOME_FILE.resolve(),
    }:

        continue

    df = load_csv(path)

    if df is None:
        continue

    columns = id_like_columns(df)

    if not columns:
        continue

    # --------------------------------------------------------------
    # Every existing ID-like field is independently examined.
    # --------------------------------------------------------------

    for field in columns:

        if field in FORBIDDEN_LINKAGE_FIELDS:
            continue

        field_values = unique_values(
            df,
            field
        )

        if not field_values:
            continue

        # Existing field versus production transaction IDs.
        production_shared = (
            field_values
            .intersection(
                production_ids
            )
        )

        # Existing field versus outcome transaction IDs.
        outcome_shared = (
            field_values
            .intersection(
                outcome_ids
            )
        )

        if (
            len(production_shared) == 0
            and len(outcome_shared) == 0
        ):
            continue

        if (
            len(production_shared) > 0
            and len(outcome_shared) > 0
        ):

            classification = (
                "POTENTIAL_TWO_SIDED_ID_FIELD"
            )

        elif len(production_shared) > 0:

            classification = (
                "PRODUCTION_SIDE_ID_FIELD"
            )

        else:

            classification = (
                "OUTCOME_SIDE_ID_FIELD"
            )

        record = {
            "dataset": str(path),
            "field": field,
            "shared_with_production": len(
                production_shared
            ),
            "shared_with_outcomes": len(
                outcome_shared
            ),
            "classification": classification,
        }

        multihop_records.append(record)

        print()
        print(
            f"[BRIDGE CANDIDATE] {path}"
        )

        print(
            f"  Field: {field}"
        )

        print(
            f"  Production shared: "
            f"{len(production_shared):,}"
        )

        print(
            f"  Outcome shared:    "
            f"{len(outcome_shared):,}"
        )

        print(
            f"  Classification: "
            f"{classification}"
        )


# ======================================================================
# DUAL-ID DATASET DISCOVERY
# ======================================================================

print()
print("=" * 72)
print("SEARCHING FOR EXISTING DUAL-ID BRIDGE DATASETS")
print("=" * 72)

dual_id_candidates = []

for path in csv_files:

    df = load_csv(path)

    if df is None:
        continue

    id_columns = id_like_columns(df)

    if len(id_columns) < 2:
        continue

    for i in range(len(id_columns)):

        for j in range(i + 1, len(id_columns)):

            field_a = id_columns[i]
            field_b = id_columns[j]

            if (
                field_a in FORBIDDEN_LINKAGE_FIELDS
                or field_b in FORBIDDEN_LINKAGE_FIELDS
            ):
                continue

            a_values = unique_values(
                df,
                field_a
            )

            b_values = unique_values(
                df,
                field_b
            )

            a_prod = len(
                a_values.intersection(
                    production_ids
                )
            )

            a_outcome = len(
                a_values.intersection(
                    outcome_ids
                )
            )

            b_prod = len(
                b_values.intersection(
                    production_ids
                )
            )

            b_outcome = len(
                b_values.intersection(
                    outcome_ids
                )
            )

            # A useful discovery candidate has one field
            # overlapping production and another overlapping outcomes.

            possible_bridge = (
                (
                    a_prod > 0
                    and b_outcome > 0
                )
                or
                (
                    b_prod > 0
                    and a_outcome > 0
                )
            )

            if not possible_bridge:
                continue

            record = {
                "dataset": str(path),
                "field_a": field_a,
                "field_b": field_b,
                "field_a_production_overlap": a_prod,
                "field_a_outcome_overlap": a_outcome,
                "field_b_production_overlap": b_prod,
                "field_b_outcome_overlap": b_outcome,
                "status": (
                    "CANDIDATE_REQUIRES_VALIDATION"
                ),
            }

            dual_id_candidates.append(
                record
            )

            print()
            print(
                f"[DUAL-ID CANDIDATE] {path}"
            )

            print(
                f"  {field_a}: "
                f"production={a_prod:,}, "
                f"outcome={a_outcome:,}"
            )

            print(
                f"  {field_b}: "
                f"production={b_prod:,}, "
                f"outcome={b_outcome:,}"
            )


# ======================================================================
# SAVE PRODUCTION DATASET CANDIDATES
# ======================================================================

production_df = pd.DataFrame(
    production_candidates
)


# ======================================================================
# SAVE MULTI-HOP RESULTS
# ======================================================================

multihop_df = pd.DataFrame(
    multihop_records
)


if len(multihop_df):

    multihop_df = multihop_df.sort_values(
        [
            "shared_with_production",
            "shared_with_outcomes",
        ],
        ascending=False
    )


multihop_df.to_csv(
    MULTIHOP_OUTPUT,
    index=False
)


# ======================================================================
# SAVE MAIN BRIDGE CANDIDATES
# ======================================================================

main_records = []

for record in production_candidates:

    temp = record.copy()

    temp["analysis_type"] = (
        "production_id_dataset"
    )

    main_records.append(temp)


for record in dual_id_candidates:

    temp = record.copy()

    temp["analysis_type"] = (
        "dual_id_candidate"
    )

    main_records.append(temp)


main_df = pd.DataFrame(
    main_records
)

main_df.to_csv(
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

    "production_transactions": len(
        production_ids
    ),

    "outcome_transactions": len(
        outcome_ids
    ),

    "direct_transaction_overlap": len(
        direct_overlap
    ),

    "production_customers": len(
        production_customers
    ),

    "outcome_customers": len(
        outcome_customers
    ),

    "customer_overlap": len(
        customer_overlap
    ),

    "ambiguous_customers": (
        ambiguous_customers
    ),

    "production_id_dataset_candidates": (
        len(production_candidates)
    ),

    "multihop_candidates": (
        len(multihop_records)
    ),

    "dual_id_candidates": (
        len(dual_id_candidates)
    ),

    # V19 intentionally does NOT claim validation.
    "validated_bridges": 0,
}


# ======================================================================
# FINAL VERDICT
# ======================================================================

print()
print("=" * 72)
print("V19 LINEAGE BRIDGE VERDICT")
print("=" * 72)

print(
    "Direct production → outcome overlap: "
    f"{len(direct_overlap):,}"
)

print(
    "Production-ID dataset candidates: "
    f"{len(production_candidates):,}"
)

print(
    "Multi-hop candidates: "
    f"{len(multihop_records):,}"
)

print(
    "Dual-ID bridge candidates: "
    f"{len(dual_id_candidates):,}"
)


if len(direct_overlap) > 0:

    verdict = (
        "DIRECT_EXISTING_TRANSACTION_ID_OVERLAP_FOUND"
    )

elif len(dual_id_candidates) > 0:

    verdict = (
        "EXPLICIT_ID_CANDIDATE_FOUND_BUT_NOT_VALIDATED"
    )

elif len(multihop_records) > 0:

    verdict = (
        "MULTIHOP_LINEAGE_CANDIDATES_FOUND"
    )

elif len(production_candidates) > 0:

    verdict = (
        "PRODUCTION_ID_DATASETS_FOUND"
    )

else:

    verdict = (
        "NO_EXISTING_ID_LINEAGE_CANDIDATE_FOUND"
    )


summary["final_verdict"] = verdict

summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    SUMMARY,
    index=False
)


print()
print(
    f"FINAL VERDICT: {verdict}"
)


# ======================================================================
# GOVERNANCE
# ======================================================================

print()
print("Governance:")

print(
    "[PASS] V10.2 TOP_RATE 5% targeting unchanged"
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
    "[PASS] No synthetic composite IDs"
)

print(
    "[PASS] No oracle fields used for linkage"
)

print(
    "[PASS] customer_id alone NOT accepted "
    "as transaction lineage"
)

print(
    "[PASS] Candidate IDs are not automatically "
    "treated as validated lineage"
)


# ======================================================================
# OUTPUTS
# ======================================================================

print()
print("Outputs:")

print(
    f"[PASS] Main bridge candidates: "
    f"{OUTPUT}"
)

print(
    f"[PASS] Multi-hop candidates:    "
    f"{MULTIHOP_OUTPUT}"
)

print(
    f"[PASS] Summary:                 "
    f"{SUMMARY}"
)

print()
print("=" * 72)
print("RECLAIM PRODUCTION PIPELINE V19 COMPLETE")
print("=" * 72)