
from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\RECLAIM")
DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = DATA_DIR / "generated"

V10_2_FILE = GENERATED_DIR / "causal_uplift_v10_2_predictions.csv"
V7_FILE = GENERATED_DIR / "reclaim_production_decisions_v7.csv"
V8_FILE = GENERATED_DIR / "reclaim_production_decisions_v8.csv"
OUTCOME_FILE = GENERATED_DIR / "recovery_outcomes.csv"

OUTPUT_DIR = GENERATED_DIR


# ============================================================
# HELPERS
# ============================================================

def header(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def pass_msg(text):
    print(f"[PASS] {text}")


def warn_msg(text):
    print(f"[WARN] {text}")


def fail_msg(text):
    print(f"[FAIL] {text}")


def find_id_column(df):
    candidates = [
        "transaction_id",
        "experiment_transaction_id",
        "production_transaction_id",
        "id",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def clean_ids(series):
    return (
        series
        .astype(str)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        .dropna()
    )


def id_set(df, column):
    return set(clean_ids(df[column]))


def compare_ids(name_a, ids_a, name_b, ids_b):
    overlap = ids_a & ids_b

    only_a = ids_a - ids_b
    only_b = ids_b - ids_a

    source_rate = len(overlap) / len(ids_a) if ids_a else 0
    target_rate = len(overlap) / len(ids_b) if ids_b else 0

    print(f"\n{name_a} → {name_b}")
    print(f"  Source IDs:        {len(ids_a):,}")
    print(f"  Target IDs:        {len(ids_b):,}")
    print(f"  Exact overlap:     {len(overlap):,}")
    print(f"  Source match rate: {source_rate:.2%}")
    print(f"  Target match rate: {target_rate:.2%}")

    return {
        "source": name_a,
        "target": name_b,
        "source_ids": len(ids_a),
        "target_ids": len(ids_b),
        "overlap": len(overlap),
        "source_match_rate": source_rate,
        "target_match_rate": target_rate,
        "only_source": len(only_a),
        "only_target": len(only_b),
    }


def normalize_columns(df):
    df = df.copy()

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    return df


# ============================================================
# LOAD FILE
# ============================================================

def load_file(path, label):
    if not path.exists():
        warn_msg(f"{label} not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
        pass_msg(
            f"{label} loaded: {len(df):,} rows, {len(df.columns)} columns"
        )
        return df

    except Exception as exc:
        fail_msg(f"Could not load {label}: {exc}")
        return None


# ============================================================
# DATASET SUMMARY
# ============================================================

def dataset_summary(name, df, id_col):
    ids = id_set(df, id_col)

    duplicate_count = len(df) - len(ids)

    print(f"\n{name}")
    print("-" * 72)
    print(f"Rows:                 {len(df):,}")
    print(f"Columns:              {len(df.columns):,}")
    print(f"Transaction ID col:   {id_col}")
    print(f"Unique transaction IDs:{len(ids):,}")
    print(f"Duplicate ID rows:    {duplicate_count:,}")

    return ids


# ============================================================
# SEARCH FOR POSSIBLE SOURCE FILES
# ============================================================

def discover_csv_files():
    header("DISCOVERING CSV DATA SOURCES")

    files = []

    search_roots = [
        DATA_DIR,
        GENERATED_DIR,
    ]

    seen = set()

    for root in search_roots:
        if not root.exists():
            continue

        for path in root.rglob("*.csv"):
            resolved = str(path.resolve())

            if resolved in seen:
                continue

            seen.add(resolved)
            files.append(path)

    files = sorted(files)

    print(f"CSV files discovered: {len(files):,}")

    for path in files:
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {path} | {size_mb:.2f} MB")
        except Exception:
            print(f"  {path}")

    return files


# ============================================================
# INSPECT POSSIBLE TRANSACTION SOURCES
# ============================================================

def inspect_candidate_files(files, production_ids, outcome_ids):
    header("SEARCHING FOR TRANSACTION-ID LINEAGE")

    results = []

    protected = {
        V10_2_FILE.resolve(),
        V7_FILE.resolve(),
        V8_FILE.resolve(),
        OUTCOME_FILE.resolve(),
    }

    for path in files:

        try:
            if path.resolve() in protected:
                continue
        except Exception:
            pass

        try:
            sample = pd.read_csv(path, nrows=1000)

        except Exception:
            continue

        id_col = find_id_column(sample)

        if id_col is None:
            continue

        try:
            full = pd.read_csv(path, usecols=[id_col])
        except Exception:
            continue

        ids = id_set(full, id_col)

        prod_overlap = len(ids & production_ids)
        outcome_overlap = len(ids & outcome_ids)

        if prod_overlap == 0 and outcome_overlap == 0:
            continue

        prod_rate = (
            prod_overlap / len(production_ids)
            if production_ids else 0
        )

        outcome_rate = (
            outcome_overlap / len(outcome_ids)
            if outcome_ids else 0
        )

        results.append({
            "file": str(path),
            "id_column": id_col,
            "rows": len(full),
            "unique_ids": len(ids),
            "production_overlap": prod_overlap,
            "production_match_rate": prod_rate,
            "outcome_overlap": outcome_overlap,
            "outcome_match_rate": outcome_rate,
        })

        print("\nCandidate:")
        print(f"  File:              {path}")
        print(f"  ID column:         {id_col}")
        print(f"  Rows:              {len(full):,}")
        print(f"  Unique IDs:        {len(ids):,}")
        print(f"  Production overlap:{prod_overlap:,} ({prod_rate:.2%})")
        print(f"  Outcome overlap:   {outcome_overlap:,} ({outcome_rate:.2%})")

    return results


# ============================================================
# CUSTOMER / TIMESTAMP / AMOUNT DIAGNOSTICS
# ============================================================

def secondary_linkage(v10, outcomes):
    header("SECONDARY LINEAGE DIAGNOSTICS")

    checks = []

    # Customer ID
    if "customer_id" in v10.columns and "customer_id" in outcomes.columns:

        a = set(clean_ids(v10["customer_id"]))
        b = set(clean_ids(outcomes["customer_id"]))

        overlap = a & b

        rate_a = len(overlap) / len(a) if a else 0
        rate_b = len(overlap) / len(b) if b else 0

        print("\nCUSTOMER ID")
        print(f"  V10.2 unique customers: {len(a):,}")
        print(f"  Outcome unique customers:{len(b):,}")
        print(f"  Overlap:                {len(overlap):,}")
        print(f"  V10.2 match rate:       {rate_a:.2%}")
        print(f"  Outcome match rate:     {rate_b:.2%}")

        checks.append({
            "field": "customer_id",
            "source_unique": len(a),
            "target_unique": len(b),
            "overlap": len(overlap),
            "source_match_rate": rate_a,
            "target_match_rate": rate_b,
        })

    # Amount
    if "amount" in v10.columns and "amount" in outcomes.columns:

        a = pd.to_numeric(v10["amount"], errors="coerce")
        b = pd.to_numeric(outcomes["amount"], errors="coerce")

        print("\nAMOUNT")

        print(
            f"  V10.2 amount range: "
            f"{a.min():.2f} → {a.max():.2f}"
        )

        print(
            f"  Outcome amount range: "
            f"{b.min():.2f} → {b.max():.2f}"
        )

        print(
            f"  V10.2 median amount: "
            f"{a.median():.2f}"
        )

        print(
            f"  Outcome median amount: "
            f"{b.median():.2f}"
        )

        checks.append({
            "field": "amount",
            "source_median": a.median(),
            "target_median": b.median(),
        })

    # Payment method
    if (
        "payment_method" in v10.columns
        and "payment_method" in outcomes.columns
    ):

        print("\nPAYMENT METHOD")

        print("V10.2:")
        print(
            v10["payment_method"]
            .value_counts(dropna=False)
            .to_string()
        )

        print("\nOutcomes:")
        print(
            outcomes["payment_method"]
            .value_counts(dropna=False)
            .to_string()
        )

    # Timestamp
    if "timestamp" in v10.columns and "timestamp" in outcomes.columns:

        va = pd.to_datetime(v10["timestamp"], errors="coerce")
        ob = pd.to_datetime(outcomes["timestamp"], errors="coerce")

        print("\nTIMESTAMP")

        print(
            f"  V10.2 timestamp range: "
            f"{va.min()} → {va.max()}"
        )

        print(
            f"  Outcome timestamp range: "
            f"{ob.min()} → {ob.max()}"
        )

        checks.append({
            "field": "timestamp",
            "source_min": str(va.min()),
            "source_max": str(va.max()),
            "target_min": str(ob.min()),
            "target_max": str(ob.max()),
        })

    return checks


# ============================================================
# ID FORMAT DIAGNOSTIC
# ============================================================

def id_format_diagnostic(name, ids):
    if not ids:
        return

    sample = list(ids)[:10]

    lengths = pd.Series([len(str(x)) for x in ids])

    print(f"\n{name} ID FORMAT")
    print("-" * 72)
    print("Sample IDs:")

    for x in sample:
        print(f"  {x}")

    print(f"Minimum length: {lengths.min()}")
    print(f"Maximum length: {lengths.max()}")
    print(f"Median length:  {lengths.median()}")

    prefixes = {}

    for x in list(ids)[:10000]:
        s = str(x)

        if len(s) >= 3:
            p = s[:3]
            prefixes[p] = prefixes.get(p, 0) + 1

    top = sorted(
        prefixes.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print("Common prefixes:")

    for prefix, count in top:
        print(f"  {prefix}: {count:,}")


# ============================================================
# GENERATION-SOURCE SEARCH
# ============================================================

def search_text_for_terms():
    header("SEARCHING PROJECT FILES FOR ID GENERATION LOGIC")

    terms = [
        "transaction_id",
        "recovery_outcomes",
        "experiment_transaction_id",
        "causal_experiment",
        "transaction",
        "uuid",
        "random_seed",
        "np.random",
        "recovered_amount",
    ]

    extensions = {
        ".py",
        ".txt",
        ".md",
        ".json",
        ".yaml",
        ".yml",
    }

    matches = []

    roots = [
        BASE_DIR,
    ]

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in extensions:
                continue

            # Skip very large files.
            try:
                if path.stat().st_size > 5 * 1024 * 1024:
                    continue
            except Exception:
                continue

            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )
            except Exception:
                continue

            found = [
                term
                for term in terms
                if term.lower() in text.lower()
            ]

            if found:

                matches.append({
                    "file": str(path),
                    "terms": ", ".join(found),
                })

                print(
                    f"[MATCH] {path} "
                    f"-> {', '.join(found)}"
                )

    print(
        f"\nFiles containing relevant lineage terms: "
        f"{len(matches):,}"
    )

    return matches


# ============================================================
# MAIN
# ============================================================

def main():

    header("RECLAIM PRODUCTION PIPELINE V17")
    print("Transaction-ID Lineage Diagnostic")
    print("V10.2 TOP_RATE 5% targeting is IMMUTABLE")
    print("NO production outcome evaluation")
    print("NO artificial outcome generation")
    print("NO artificial ID matching")

    # --------------------------------------------------------
    # LOAD MAIN DATASETS
    # --------------------------------------------------------

    header("LOADING PRIMARY DATASETS")

    v10 = load_file(
        V10_2_FILE,
        "V10.2 predictions"
    )

    v7 = load_file(
        V7_FILE,
        "V7 production decisions"
    )

    v8 = load_file(
        V8_FILE,
        "V8 production decisions"
    )

    outcomes = load_file(
        OUTCOME_FILE,
        "Legitimate recovery outcomes"
    )

    if any(
        x is None
        for x in [v10, v7, v8, outcomes]
    ):
        raise RuntimeError(
            "One or more required datasets could not be loaded."
        )

    # --------------------------------------------------------
    # ID COLUMNS
    # --------------------------------------------------------

    header("IDENTIFYING TRANSACTION-ID COLUMNS")

    id_v10 = find_id_column(v10)
    id_v7 = find_id_column(v7)
    id_v8 = find_id_column(v8)
    id_out = find_id_column(outcomes)

    if not all([id_v10, id_v7, id_v8, id_out]):
        raise ValueError(
            "Could not identify transaction ID column in all datasets."
        )

    pass_msg(f"V10.2 ID column:      {id_v10}")
    pass_msg(f"V7 ID column:         {id_v7}")
    pass_msg(f"V8 ID column:         {id_v8}")
    pass_msg(f"Outcome ID column:    {id_out}")

    # --------------------------------------------------------
    # DATASET SUMMARIES
    # --------------------------------------------------------

    header("DATASET ID SUMMARIES")

    ids_v10 = dataset_summary(
        "V10.2",
        v10,
        id_v10
    )

    ids_v7 = dataset_summary(
        "V7",
        v7,
        id_v7
    )

    ids_v8 = dataset_summary(
        "V8",
        v8,
        id_v8
    )

    ids_out = dataset_summary(
        "Recovery Outcomes",
        outcomes,
        id_out
    )

    # --------------------------------------------------------
    # ID FORMAT
    # --------------------------------------------------------

    header("TRANSACTION-ID FORMAT DIAGNOSTIC")

    id_format_diagnostic(
        "V10.2",
        ids_v10
    )

    id_format_diagnostic(
        "Recovery Outcomes",
        ids_out
    )

    # --------------------------------------------------------
    # EXACT ID COMPARISONS
    # --------------------------------------------------------

    header("EXACT TRANSACTION-ID COMPARISON")

    comparisons = []

    comparisons.append(
        compare_ids(
            "V7",
            ids_v7,
            "V8",
            ids_v8
        )
    )

    comparisons.append(
        compare_ids(
            "V10.2",
            ids_v10,
            "V7",
            ids_v7
        )
    )

    comparisons.append(
        compare_ids(
            "V10.2",
            ids_v10,
            "V8",
            ids_v8
        )
    )

    comparisons.append(
        compare_ids(
            "V10.2",
            ids_v10,
            "Recovery Outcomes",
            ids_out
        )
    )

    comparisons.append(
        compare_ids(
            "V7",
            ids_v7,
            "Recovery Outcomes",
            ids_out
        )
    )

    comparisons.append(
        compare_ids(
            "V8",
            ids_v8,
            "Recovery Outcomes",
            ids_out
        )
    )

    # --------------------------------------------------------
    # SECONDARY FIELDS
    # --------------------------------------------------------

    secondary = secondary_linkage(
        v10,
        outcomes
    )

    # --------------------------------------------------------
    # DISCOVER OTHER CSV SOURCES
    # --------------------------------------------------------

    csv_files = discover_csv_files()

    candidates = inspect_candidate_files(
        csv_files,
        ids_v10,
        ids_out
    )

    # --------------------------------------------------------
    # SEARCH PROJECT CODE
    # --------------------------------------------------------

    text_matches = search_text_for_terms()

    # --------------------------------------------------------
    # BUILD DIAGNOSTIC OUTPUTS
    # --------------------------------------------------------

    header("WRITING V17 DIAGNOSTIC OUTPUTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison_df = pd.DataFrame(comparisons)

    secondary_df = pd.DataFrame(secondary)

    candidate_df = pd.DataFrame(candidates)

    text_match_df = pd.DataFrame(text_matches)

    # --------------------------------------------------------
    # DIAGNOSTIC VERDICT
    # --------------------------------------------------------

    production_outcome_overlap = len(
        ids_v10 & ids_out
    )

    if production_outcome_overlap > 0:

        verdict = (
            "PRODUCTION_OUTCOME_LINKAGE_EXISTS"
        )

    else:

        if len(candidates) > 0:

            verdict = (
                "POSSIBLE_ALTERNATE_OUTCOME_SOURCE_FOUND"
            )

        else:

            verdict = (
                "PRODUCTION_LINEAGE_ALIGNED_BUT_OUTCOME_SOURCE_MISMATCHED"
            )

    summary = pd.DataFrame([
        {
            "v10_2_rows": len(v10),
            "v7_rows": len(v7),
            "v8_rows": len(v8),
            "outcome_rows": len(outcomes),

            "v10_2_unique_ids": len(ids_v10),
            "v7_unique_ids": len(ids_v7),
            "v8_unique_ids": len(ids_v8),
            "outcome_unique_ids": len(ids_out),

            "v10_2_v7_overlap": len(ids_v10 & ids_v7),
            "v10_2_v8_overlap": len(ids_v10 & ids_v8),
            "v7_v8_overlap": len(ids_v7 & ids_v8),

            "v10_2_outcome_overlap":
                len(ids_v10 & ids_out),

            "v7_outcome_overlap":
                len(ids_v7 & ids_out),

            "v8_outcome_overlap":
                len(ids_v8 & ids_out),

            "candidate_source_files":
                len(candidates),

            "project_files_with_lineage_terms":
                len(text_matches),

            "artificial_ids_created":
                "NO",

            "artificial_outcomes_created":
                "NO",

            "calibration_predictions_used_as_outcomes":
                "NO",

            "v10_2_targeting_changed":
                "NO",

            "v7_v8_improvement_claimed":
                "NO",

            "verdict":
                verdict,
        }
    ])

    paths = {
        "Summary":
            OUTPUT_DIR /
            "reclaim_production_v17_summary.csv",

        "ID Comparisons":
            OUTPUT_DIR /
            "reclaim_production_v17_id_comparisons.csv",

        "Secondary Linkage":
            OUTPUT_DIR /
            "reclaim_production_v17_secondary_linkage.csv",

        "Candidate Sources":
            OUTPUT_DIR /
            "reclaim_production_v17_candidate_sources.csv",

        "Code Lineage":
            OUTPUT_DIR /
            "reclaim_production_v17_code_lineage.csv",
    }

    summary.to_csv(
        paths["Summary"],
        index=False
    )

    comparison_df.to_csv(
        paths["ID Comparisons"],
        index=False
    )

    secondary_df.to_csv(
        paths["Secondary Linkage"],
        index=False
    )

    candidate_df.to_csv(
        paths["Candidate Sources"],
        index=False
    )

    text_match_df.to_csv(
        paths["Code Lineage"],
        index=False
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    header("V17 LINEAGE DIAGNOSTIC SUMMARY")

    print(
        f"V10.2 transactions:        {len(v10):,}"
    )

    print(
        f"V7 transactions:           {len(v7):,}"
    )

    print(
        f"V8 transactions:           {len(v8):,}"
    )

    print(
        f"Recovery outcomes:         {len(outcomes):,}"
    )

    print(
        f"V10.2 → V7 overlap:        "
        f"{len(ids_v10 & ids_v7):,}"
    )

    print(
        f"V10.2 → V8 overlap:        "
        f"{len(ids_v10 & ids_v8):,}"
    )

    print(
        f"V7 → V8 overlap:           "
        f"{len(ids_v7 & ids_v8):,}"
    )

    print(
        f"V10.2 → outcomes overlap:  "
        f"{len(ids_v10 & ids_out):,}"
    )

    print(
        f"V7 → outcomes overlap:     "
        f"{len(ids_v7 & ids_out):,}"
    )

    print(
        f"V8 → outcomes overlap:     "
        f"{len(ids_v8 & ids_out):,}"
    )

    print(
        f"Possible source files:      "
        f"{len(candidates):,}"
    )

    print(
        f"Relevant code files:        "
        f"{len(text_matches):,}"
    )

    header("V17 GOVERNANCE")

    print(
        "V10.2 targeting changed:                         NO"
    )

    print(
        "V7/V8 population changed:                       NO"
    )

    print(
        "Artificial transaction IDs created:             NO"
    )

    print(
        "Artificial outcomes created:                    NO"
    )

    print(
        "Calibration predictions treated as outcomes:   NO"
    )

    print(
        "Oracle/evaluation fields used for linkage:      NO"
    )

    print(
        "V7 vs V8 improvement claimed:                   NO"
    )

    header("FINAL V17 VERDICT")

    print(f"FINAL VERDICT: {verdict}")

    print("\nOutput files:")

    for name, path in paths.items():
        print(f"[PASS] {name}: {path}")

    if production_outcome_overlap == 0:

        print("\nNEXT STEP:")

        if candidates:

            print(
                "Review the candidate source files listed in "
                "reclaim_production_v17_candidate_sources.csv."
            )

            print(
                "At least one discovered dataset shares transaction "
                "IDs with the production or outcome population."
            )

        else:

            print(
                "No alternate CSV source with matching transaction IDs "
                "was discovered automatically."
            )

            print(
                "Inspect the code-lineage output to identify where "
                "transaction IDs for recovery_outcomes.csv are generated."
            )

            print(
                "Do NOT fabricate or transform IDs to create a match."
            )

    else:

        print(
            "\nLegitimate production outcome linkage exists."
        )

        print(
            "V7 vs V8 evaluation can proceed only after validating "
            "the matched outcome cohort."
        )

    header("RECLAIM PRODUCTION PIPELINE V17 COMPLETE")


if __name__ == "__main__":
    main()

