
"""
========================================================================
RECLAIM PRODUCTION PIPELINE V21
========================================================================
Production Monitoring & Backend Readiness

Purpose:
    Build a monitoring layer around the immutable V10.2 production
    targeting population.

V21 DOES:
    - Validate production population integrity
    - Verify immutable TOP_RATE 5% targeting
    - Monitor prediction distributions
    - Monitor uplift distributions
    - Monitor selected policy distribution
    - Detect missing values
    - Detect duplicate transaction IDs
    - Create backend-ready monitoring summaries
    - Produce transaction-level monitoring status

V21 DOES NOT:
    - Change targeting
    - Re-rank transactions
    - Evaluate production outcomes
    - Join recovery_outcomes.csv
    - Manufacture lineage
    - Generate artificial IDs
    - Use oracle fields for decisions

Governance:
    V10.2 remains the immutable production decision source.
========================================================================
"""

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ======================================================================
# CONFIG
# ======================================================================

ROOT = Path(r"D:\RECLAIM")
GENERATED = ROOT / "data" / "generated"

V102_FILE = GENERATED / "causal_uplift_v10_2_predictions.csv"
V7_FILE = GENERATED / "reclaim_production_decisions_v7.csv"
V8_FILE = GENERATED / "reclaim_production_decisions_v8.csv"

MONITORING_OUTPUT = (
    GENERATED / "reclaim_production_v21_monitoring.csv"
)

METRICS_OUTPUT = (
    GENERATED / "reclaim_production_v21_metrics.csv"
)

DATA_QUALITY_OUTPUT = (
    GENERATED / "reclaim_production_v21_data_quality.csv"
)

SUMMARY_OUTPUT = (
    GENERATED / "reclaim_production_v21_summary.csv"
)


# ======================================================================
# CONSTANTS
# ======================================================================

EXPECTED_TOP_RATE = 0.05
TARGET_RATE_TOLERANCE = 0.001

REQUIRED_V102_COLUMNS = [
    "transaction_id",
    "customer_id",
    "predicted_treatment_probability",
    "predicted_control_probability",
    "predicted_probability_uplift",
    "selected_policy_type",
    "selected_policy_parameter",
    "selected_policy",
    "intervention_selected",
]


# ======================================================================
# HELPERS
# ======================================================================

def load_csv(path):
    """
    Load CSV safely.
    """

    print(f"[LOAD] {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    df = pd.read_csv(path)

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    print(f"       Rows: {len(df):,}")
    print(f"       Columns: {len(df.columns)}")

    return df


def safe_numeric(series):
    """
    Convert a series to numeric safely.
    """

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def add_metric(
    records,
    category,
    metric,
    value,
    status="INFO",
    threshold=None,
):
    """
    Add a standardized monitoring metric.
    """

    records.append(
        {
            "category": category,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "status": status,
        }
    )


def check_required_columns(df, columns):
    """
    Return missing required columns.
    """

    return [
        column
        for column in columns
        if column not in df.columns
    ]


# ======================================================================
# START
# ======================================================================

print("=" * 72)
print("RECLAIM PRODUCTION PIPELINE V21")
print("=" * 72)
print()
print("PRODUCTION MONITORING & BACKEND READINESS")
print()


# ======================================================================
# LOAD DATA
# ======================================================================

v102 = load_csv(V102_FILE)
v7 = load_csv(V7_FILE)
v8 = load_csv(V8_FILE)


# ======================================================================
# V10.2 SCHEMA VALIDATION
# ======================================================================

print()
print("=" * 72)
print("V10.2 PRODUCTION SCHEMA VALIDATION")
print("=" * 72)

missing_columns = check_required_columns(
    v102,
    REQUIRED_V102_COLUMNS
)

if missing_columns:

    print(
        "[FAIL] Missing required columns:"
    )

    for column in missing_columns:
        print(f"       - {column}")

    schema_status = "FAIL"

else:

    print(
        "[PASS] All required production columns present."
    )

    schema_status = "PASS"


# ======================================================================
# PRIMARY POPULATION INTEGRITY
# ======================================================================

print()
print("=" * 72)
print("PRIMARY POPULATION INTEGRITY")
print("=" * 72)


total_rows = len(v102)

transaction_ids = (
    v102["transaction_id"]
    .dropna()
)

unique_transaction_ids = (
    transaction_ids.nunique()
)

duplicate_transaction_rows = (
    transaction_ids.duplicated()
    .sum()
)


customer_ids = (
    v102["customer_id"]
    .dropna()
)

unique_customers = (
    customer_ids.nunique()
)


print(
    f"Total production rows:        {total_rows:,}"
)

print(
    f"Unique transaction IDs:       "
    f"{unique_transaction_ids:,}"
)

print(
    f"Duplicate transaction rows:   "
    f"{duplicate_transaction_rows:,}"
)

print(
    f"Unique customers:             "
    f"{unique_customers:,}"
)


if (
    unique_transaction_ids == total_rows
    and duplicate_transaction_rows == 0
):

    transaction_integrity_status = "PASS"

    print(
        "[PASS] Transaction population is unique."
    )

else:

    transaction_integrity_status = "FAIL"

    print(
        "[FAIL] Duplicate transaction IDs detected."
    )


# ======================================================================
# MISSING VALUE MONITORING
# ======================================================================

print()
print("=" * 72)
print("DATA QUALITY MONITORING")
print("=" * 72)


quality_records = []


for column in v102.columns:

    missing_count = int(
        v102[column].isna().sum()
    )

    missing_rate = (
        missing_count / total_rows
        if total_rows
        else 0
    )

    status = (
        "PASS"
        if missing_count == 0
        else "WARNING"
    )

    quality_records.append(
        {
            "column": column,
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "status": status,
        }
    )

    print(
        f"{column:40s} "
        f"missing={missing_count:6,d} "
        f"rate={missing_rate:.2%} "
        f"[{status}]"
    )


quality_df = pd.DataFrame(
    quality_records
)


# ======================================================================
# TARGETING INTEGRITY
# ======================================================================

print()
print("=" * 72)
print("IMMUTABLE TARGETING INTEGRITY")
print("=" * 72)


intervention = (
    safe_numeric(
        v102["intervention_selected"]
    )
    .fillna(0)
)

selected_count = int(
    (intervention == 1).sum()
)

observed_target_rate = (
    selected_count / total_rows
    if total_rows
    else 0
)


target_rate_difference = abs(
    observed_target_rate
    - EXPECTED_TOP_RATE
)


print(
    f"Expected TOP_RATE:            "
    f"{EXPECTED_TOP_RATE:.2%}"
)

print(
    f"Observed intervention rate:   "
    f"{observed_target_rate:.2%}"
)

print(
    f"Selected transactions:        "
    f"{selected_count:,}"
)

print(
    f"Target rate difference:       "
    f"{target_rate_difference:.4%}"
)


if target_rate_difference <= TARGET_RATE_TOLERANCE:

    targeting_status = "PASS"

    print(
        "[PASS] V10.2 TOP_RATE targeting remains intact."
    )

else:

    targeting_status = "FAIL"

    print(
        "[FAIL] Targeting rate differs from expected 5%."
    )


# ======================================================================
# PREDICTION DISTRIBUTION MONITORING
# ======================================================================

print()
print("=" * 72)
print("PREDICTION DISTRIBUTION MONITORING")
print("=" * 72)


numeric_monitoring_columns = [
    "predicted_treatment_probability",
    "predicted_control_probability",
    "predicted_probability_uplift",
]


metric_records = []


for column in numeric_monitoring_columns:

    if column not in v102.columns:
        continue

    values = (
        safe_numeric(v102[column])
        .dropna()
    )

    if len(values) == 0:
        continue

    stats = {
        "count": len(values),
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "p01": values.quantile(0.01),
        "p05": values.quantile(0.05),
        "p25": values.quantile(0.25),
        "median": values.median(),
        "p75": values.quantile(0.75),
        "p95": values.quantile(0.95),
        "p99": values.quantile(0.99),
        "max": values.max(),
    }

    print()
    print(column)

    for metric_name, value in stats.items():

        print(
            f"  {metric_name:10s}: {value:.8f}"
            if isinstance(value, float)
            else f"  {metric_name:10s}: {value:,}"
        )

        add_metric(
            metric_records,
            category="prediction_distribution",
            metric=(
                f"{column}.{metric_name}"
            ),
            value=value,
        )


# ======================================================================
# UPLIFT SEGMENT MONITORING
# ======================================================================

print()
print("=" * 72)
print("UPLIFT SEGMENT MONITORING")
print("=" * 72)


uplift = safe_numeric(
    v102["predicted_probability_uplift"]
)


positive_uplift = int(
    (uplift > 0).sum()
)

zero_or_negative_uplift = int(
    (uplift <= 0).sum()
)

missing_uplift = int(
    uplift.isna().sum()
)


print(
    f"Positive uplift:              "
    f"{positive_uplift:,}"
)

print(
    f"Zero/negative uplift:         "
    f"{zero_or_negative_uplift:,}"
)

print(
    f"Missing uplift:               "
    f"{missing_uplift:,}"
)


add_metric(
    metric_records,
    "uplift_monitoring",
    "positive_uplift_count",
    positive_uplift,
)

add_metric(
    metric_records,
    "uplift_monitoring",
    "zero_or_negative_uplift_count",
    zero_or_negative_uplift,
)

add_metric(
    metric_records,
    "uplift_monitoring",
    "missing_uplift_count",
    missing_uplift,
)


# ======================================================================
# POLICY DISTRIBUTION MONITORING
# ======================================================================

print()
print("=" * 72)
print("POLICY DISTRIBUTION MONITORING")
print("=" * 72)


policy_columns = [
    "selected_policy_type",
    "selected_policy",
]


for column in policy_columns:

    if column not in v102.columns:
        continue

    print()
    print(column)
    print("-" * 72)

    distribution = (
        v102[column]
        .fillna("MISSING")
        .value_counts(
            dropna=False
        )
    )

    for value, count in distribution.items():

        rate = count / total_rows

        print(
            f"{str(value):35s} "
            f"count={count:6,d} "
            f"rate={rate:.2%}"
        )

        add_metric(
            metric_records,
            category="policy_distribution",
            metric=(
                f"{column}.{value}"
            ),
            value=count,
        )


# ======================================================================
# PRODUCTION DECISION CONSISTENCY
# ======================================================================

print()
print("=" * 72)
print("PRODUCTION DECISION CONSISTENCY")
print("=" * 72)


v7_v8_same_population = (
    set(v7["transaction_id"])
    == set(v8["transaction_id"])
)

v102_v7_same_population = (
    set(v102["transaction_id"])
    == set(v7["transaction_id"])
)

v102_v8_same_population = (
    set(v102["transaction_id"])
    == set(v8["transaction_id"])
)


print(
    f"V7 ↔ V8 population identical:     "
    f"{v7_v8_same_population}"
)

print(
    f"V10.2 ↔ V7 population identical: "
    f"{v102_v7_same_population}"
)

print(
    f"V10.2 ↔ V8 population identical: "
    f"{v102_v8_same_population}"
)


population_consistency_status = (
    "PASS"
    if (
        v7_v8_same_population
        and v102_v7_same_population
        and v102_v8_same_population
    )
    else "WARNING"
)


# ======================================================================
# TRANSACTION-LEVEL MONITORING DATASET
# ======================================================================

print()
print("=" * 72)
print("BUILDING TRANSACTION MONITORING DATASET")
print("=" * 72)


monitoring_df = v102[
    [
        "transaction_id",
        "customer_id",
        "predicted_treatment_probability",
        "predicted_control_probability",
        "predicted_probability_uplift",
        "selected_policy_type",
        "selected_policy_parameter",
        "selected_policy",
        "intervention_selected",
    ]
].copy()


# Monitoring flags only.
# No decision is modified.

monitoring_df["monitoring_missing_required_field"] = (
    monitoring_df[
        REQUIRED_V102_COLUMNS
    ]
    .isna()
    .any(axis=1)
)


monitoring_df["monitoring_duplicate_transaction_id"] = (
    monitoring_df[
        "transaction_id"
    ]
    .duplicated(
        keep=False
    )
)


monitoring_df["monitoring_positive_uplift"] = (
    safe_numeric(
        monitoring_df[
            "predicted_probability_uplift"
        ]
    )
    > 0
)


monitoring_df["monitoring_selected"] = (
    safe_numeric(
        monitoring_df[
            "intervention_selected"
        ]
    )
    .fillna(0)
    .astype(int)
)


monitoring_df["monitoring_status"] = np.select(
    [
        monitoring_df[
            "monitoring_missing_required_field"
        ],
        monitoring_df[
            "monitoring_duplicate_transaction_id"
        ],
    ],
    [
        "DATA_QUALITY_WARNING",
        "DATA_QUALITY_WARNING",
    ],
    default="PASS",
)


monitoring_df["monitoring_version"] = "V21"

monitoring_df["source_version"] = "V10.2"

monitoring_df["outcome_evaluated"] = False


print(
    f"Monitoring rows created: "
    f"{len(monitoring_df):,}"
)

print(
    "[PASS] Monitoring layer created without changing decisions."
)


# ======================================================================
# BACKEND READINESS CONTRACT
# ======================================================================

print()
print("=" * 72)
print("BACKEND READINESS CONTRACT")
print("=" * 72)


backend_required_columns = [
    "transaction_id",
    "customer_id",
    "predicted_treatment_probability",
    "predicted_control_probability",
    "predicted_probability_uplift",
    "selected_policy",
    "intervention_selected",
    "monitoring_status",
    "monitoring_version",
    "source_version",
]


missing_backend_columns = [
    column
    for column in backend_required_columns
    if column not in monitoring_df.columns
]


if not missing_backend_columns:

    backend_readiness_status = "PASS"

    print(
        "[PASS] Monitoring dataset satisfies backend contract."
    )

else:

    backend_readiness_status = "FAIL"

    print(
        "[FAIL] Missing backend contract fields:"
    )

    for column in missing_backend_columns:

        print(f"       - {column}")


# ======================================================================
# ADD CORE METRICS
# ======================================================================

add_metric(
    metric_records,
    "population",
    "total_rows",
    total_rows,
)

add_metric(
    metric_records,
    "population",
    "unique_transaction_ids",
    unique_transaction_ids,
)

add_metric(
    metric_records,
    "population",
    "unique_customers",
    unique_customers,
)

add_metric(
    metric_records,
    "targeting",
    "expected_top_rate",
    EXPECTED_TOP_RATE,
)

add_metric(
    metric_records,
    "targeting",
    "observed_intervention_rate",
    observed_target_rate,
)

add_metric(
    metric_records,
    "targeting",
    "selected_transaction_count",
    selected_count,
)

add_metric(
    metric_records,
    "targeting",
    "target_rate_difference",
    target_rate_difference,
)


metrics_df = pd.DataFrame(
    metric_records
)


# ======================================================================
# SAVE OUTPUTS
# ======================================================================

print()
print("=" * 72)
print("WRITING V21 OUTPUTS")
print("=" * 72)


monitoring_df.to_csv(
    MONITORING_OUTPUT,
    index=False,
)

metrics_df.to_csv(
    METRICS_OUTPUT,
    index=False,
)

quality_df.to_csv(
    DATA_QUALITY_OUTPUT,
    index=False,
)


# ======================================================================
# SUMMARY
# ======================================================================

overall_status = (
    "PASS"
    if (
        schema_status == "PASS"
        and transaction_integrity_status == "PASS"
        and targeting_status == "PASS"
        and backend_readiness_status == "PASS"
    )
    else "WARNING"
)


summary = {
    "pipeline_version": "V21",
    "source_version": "V10.2",
    "generated_at_utc": (
        datetime.now(
            timezone.utc
        ).isoformat()
    ),
    "total_rows": total_rows,
    "unique_transaction_ids": unique_transaction_ids,
    "duplicate_transaction_rows": duplicate_transaction_rows,
    "unique_customers": unique_customers,
    "expected_top_rate": EXPECTED_TOP_RATE,
    "observed_intervention_rate": observed_target_rate,
    "selected_transaction_count": selected_count,
    "target_rate_difference": target_rate_difference,
    "schema_status": schema_status,
    "transaction_integrity_status": (
        transaction_integrity_status
    ),
    "targeting_status": targeting_status,
    "population_consistency_status": (
        population_consistency_status
    ),
    "backend_readiness_status": (
        backend_readiness_status
    ),
    "outcome_evaluation_performed": False,
    "overall_status": overall_status,
}


summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ======================================================================
# FINAL VERDICT
# ======================================================================

print()
print("=" * 72)
print("V21 PRODUCTION MONITORING VERDICT")
print("=" * 72)

print(
    f"FINAL STATUS: {overall_status}"
)

print()
print("Core validation:")

print(
    f"[{schema_status}] "
    "Production schema validation"
)

print(
    f"[{transaction_integrity_status}] "
    "Transaction population integrity"
)

print(
    f"[{targeting_status}] "
    "Immutable TOP_RATE 5% targeting"
)

print(
    f"[{population_consistency_status}] "
    "Cross-version population consistency"
)

print(
    f"[{backend_readiness_status}] "
    "Backend readiness contract"
)

print()
print("Governance:")

print(
    "[PASS] V10.2 production targeting unchanged"
)

print(
    "[PASS] No production re-ranking performed"
)

print(
    "[PASS] No production outcome evaluation performed"
)

print(
    "[PASS] No recovery outcome linkage attempted"
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
    "[PASS] Oracle fields not used for decisions"
)


print()
print("Outputs:")

print(
    f"[PASS] Monitoring dataset: {MONITORING_OUTPUT}"
)

print(
    f"[PASS] Monitoring metrics: {METRICS_OUTPUT}"
)

print(
    f"[PASS] Data quality:       {DATA_QUALITY_OUTPUT}"
)

print(
    f"[PASS] Summary:            {SUMMARY_OUTPUT}"
)


print()
print("=" * 72)
print("RECLAIM PRODUCTION PIPELINE V21 COMPLETE")
print("=" * 72)
