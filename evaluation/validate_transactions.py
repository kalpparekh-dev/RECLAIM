from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

CUSTOMER_PATH = Path(
    "data/generated/customers.csv"
)

TRANSACTION_PATH = Path(
    "data/generated/transactions.csv"
)

SIMULATION_DAYS = 30

HIDDEN_COLUMNS = {
    "reliability",
    "financial_stability",
    "responsiveness",
    "retry_sensitivity",
    "reminder_sensitivity",
    "activity_level",
}


# ============================================================
# Utility
# ============================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


# ============================================================
# File validation
# ============================================================

def validate_files() -> bool:

    customers_exist = CUSTOMER_PATH.exists()
    transactions_exist = TRANSACTION_PATH.exists()

    print(
        f"[{status(customers_exist)}] "
        f"Customer dataset exists"
    )

    print(
        f"[{status(transactions_exist)}] "
        f"Transaction dataset exists"
    )

    return customers_exist and transactions_exist


# ============================================================
# Structural validation
# ============================================================

def validate_transaction_structure(
    transactions: pd.DataFrame,
) -> list[bool]:

    print_section("TRANSACTION STRUCTURAL VALIDATION")

    results = []

    required_columns = {
        "transaction_id",
        "customer_id",
        "timestamp",
        "amount",
        "payment_method",
    }

    missing_columns = (
        required_columns
        - set(transactions.columns)
    )

    passed = len(missing_columns) == 0

    results.append(passed)

    print(
        f"[{status(passed)}] "
        "Required transaction columns"
    )

    if missing_columns:
        print("Missing columns:")

        for column in sorted(missing_columns):
            print(f"  - {column}")

        return results

    # --------------------------------------------------------
    # Transaction ID uniqueness
    # --------------------------------------------------------

    unique_ids = (
        transactions["transaction_id"]
        .is_unique
    )

    results.append(unique_ids)

    print(
        f"[{status(unique_ids)}] "
        "Transaction IDs are unique"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = (
        transactions.isna().sum().sum()
    )

    no_missing = missing_values == 0

    results.append(no_missing)

    print(
        f"[{status(no_missing)}] "
        "No missing transaction values"
    )

    if not no_missing:
        print(
            f"Missing values: {missing_values}"
        )

    # --------------------------------------------------------
    # Amount validity
    # --------------------------------------------------------

    positive_amounts = (
        transactions["amount"] > 0
    ).all()

    results.append(positive_amounts)

    print(
        f"[{status(positive_amounts)}] "
        "All amounts are positive"
    )

    # --------------------------------------------------------
    # Timestamp validity
    # --------------------------------------------------------

    timestamps_valid = pd.api.types.is_datetime64_any_dtype(
        transactions["timestamp"]
    )

    results.append(timestamps_valid)

    print(
        f"[{status(timestamps_valid)}] "
        "Timestamp column has datetime type"
    )

    return results


# ============================================================
# Customer coverage
# ============================================================

def analyze_customer_coverage(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
) -> list[bool]:

    print_section("CUSTOMER COVERAGE")

    transaction_counts = (
        transactions
        .groupby("customer_id")
        .size()
    )

    customers_with_transactions = (
        transaction_counts.index
    )

    total_customers = len(customers)

    active_customers = len(
        customers_with_transactions
    )

    inactive_customers = (
        total_customers
        - active_customers
    )

    print(
        f"Total customers:             "
        f"{total_customers:,}"
    )

    print(
        f"Customers with transactions: "
        f"{active_customers:,}"
    )

    print(
        f"Customers without transactions:"
        f" {inactive_customers:,}"
    )

    coverage = (
        active_customers / total_customers
    )

    print(
        f"Transaction coverage: "
        f"{coverage * 100:.2f}%"
    )

    # For our current simulator, having every customer
    # participate at least once is intentional.
    passed = active_customers == total_customers

    print(
        f"[{status(passed)}] "
        "All customers have transaction activity"
    )

    return [passed]


# ============================================================
# Activity → transaction relationship
# ============================================================

def analyze_activity_relationship(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
) -> list[bool]:

    print_section(
        "ACTIVITY LEVEL → TRANSACTION FREQUENCY"
    )

    transaction_counts = (
        transactions
        .groupby("customer_id")
        .size()
        .rename("transaction_count")
    )

    customer_activity = customers[
        [
            "customer_id",
            "activity_level",
        ]
    ].copy()

    customer_activity = customer_activity.merge(
        transaction_counts,
        on="customer_id",
        how="left",
    )

    customer_activity["transaction_count"] = (
        customer_activity["transaction_count"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Create activity quartiles
    # --------------------------------------------------------

    customer_activity["activity_quartile"] = (
        pd.qcut(
            customer_activity["activity_level"],
            q=4,
            labels=[
                "Q1 Low",
                "Q2",
                "Q3",
                "Q4 High",
            ],
            duplicates="drop",
        )
    )

    summary = (
        customer_activity
        .groupby(
            "activity_quartile",
            observed=True,
        )["transaction_count"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
        .round(2)
    )

    print(summary.to_string())

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    correlation = (
        customer_activity[
            [
                "activity_level",
                "transaction_count",
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    print(
        f"\nCorrelation between activity level "
        f"and transaction count: "
        f"{correlation:.3f}"
    )

    # --------------------------------------------------------
    # Monotonicity check
    # --------------------------------------------------------

    means = (
        summary["mean"]
        .tolist()
    )

    monotonic = all(
        means[i] < means[i + 1]
        for i in range(len(means) - 1)
    )

    print(
        f"[{status(monotonic)}] "
        "Transaction frequency increases "
        "monotonically across activity quartiles"
    )

    # Correlation should be meaningfully positive.
    correlation_valid = correlation > 0.50

    print(
        f"[{status(correlation_valid)}] "
        "Activity has a strong positive relationship "
        "with transaction frequency"
    )

    return [
        monotonic,
        correlation_valid,
    ]


# ============================================================
# Transaction concentration
# ============================================================

def analyze_transaction_concentration(
    transactions: pd.DataFrame,
) -> None:

    print_section(
        "TRANSACTION CONCENTRATION"
    )

    counts = (
        transactions
        .groupby("customer_id")
        .size()
    )

    buckets = pd.cut(
        counts,
        bins=[
            -1,
            10,
            25,
            50,
            100,
            np.inf,
        ],
        labels=[
            "0–10",
            "11–25",
            "26–50",
            "51–100",
            "100+",
        ],
    )

    distribution = (
        buckets
        .value_counts()
        .sort_index()
    )

    for bucket, count in distribution.items():

        percentage = (
            count / len(counts)
        ) * 100

        print(
            f"{str(bucket):<10} "
            f"{count:>5} customers "
            f"({percentage:>6.2f}%)"
        )

    print("\nTransaction count statistics:")

    print(
        counts.describe()
        .round(2)
        .to_string()
    )


# ============================================================
# Amount distribution
# ============================================================

def analyze_amount_distribution(
    transactions: pd.DataFrame,
) -> None:

    print_section(
        "TRANSACTION AMOUNT DISTRIBUTION"
    )

    amounts = transactions["amount"]

    statistics = {
        "Mean": amounts.mean(),
        "Median": amounts.median(),
        "P90": amounts.quantile(0.90),
        "P95": amounts.quantile(0.95),
        "P99": amounts.quantile(0.99),
        "Maximum": amounts.max(),
    }

    for name, value in statistics.items():

        print(
            f"{name:<10}: ₹{value:,.2f}"
        )


# ============================================================
# Daily volume analysis
# ============================================================

def analyze_daily_volume(
    transactions: pd.DataFrame,
) -> list[bool]:

    print_section(
        "DAILY TRANSACTION VOLUME"
    )

    daily = (
        transactions
        .assign(
            date=transactions[
                "timestamp"
            ].dt.date
        )
        .groupby("date")
        .size()
    )

    print(
        daily.describe()
        .round(2)
        .to_string()
    )

    mean = daily.mean()
    std = daily.std()

    coefficient_of_variation = (
        std / mean
        if mean != 0
        else 0
    )

    print(
        f"\nMean daily volume: "
        f"{mean:.2f}"
    )

    print(
        f"Daily standard deviation: "
        f"{std:.2f}"
    )

    print(
        f"Coefficient of variation: "
        f"{coefficient_of_variation:.4f}"
    )

    # This is intentionally informational rather than a hard
    # pass/fail gate. We want to inspect the result rather than
    # artificially force randomness into the simulator.

    reasonable_days = (
        len(daily) == SIMULATION_DAYS
    )

    print(
        f"[{status(reasonable_days)}] "
        f"Expected {SIMULATION_DAYS}-day simulation"
    )

    return [reasonable_days]


# ============================================================
# Payment method consistency
# ============================================================

def analyze_payment_methods(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
) -> list[bool]:

    print_section(
        "PAYMENT METHOD CONSISTENCY"
    )

    merged = transactions.merge(
        customers[
            [
                "customer_id",
                "preferred_payment_method",
            ]
        ],
        on="customer_id",
        how="left",
    )

    matches = (
        merged["payment_method"]
        ==
        merged["preferred_payment_method"]
    )

    match_rate = matches.mean()

    print(
        f"Transaction/payment preference match: "
        f"{match_rate * 100:.2f}%"
    )

    passed = match_rate == 1.0

    print(
        f"[{status(passed)}] "
        "Transactions use customer's preferred method"
    )

    return [passed]


# ============================================================
# Hidden-feature leakage check
# ============================================================

def check_hidden_feature_leakage(
    transactions: pd.DataFrame,
) -> list[bool]:

    print_section(
        "HIDDEN FEATURE LEAKAGE CHECK"
    )

    leaked = (
        HIDDEN_COLUMNS
        & set(transactions.columns)
    )

    passed = len(leaked) == 0

    print(
        f"[{status(passed)}] "
        "Hidden simulator traits are not present "
        "in transaction data"
    )

    if leaked:
        print("\nLeaked columns:")

        for column in sorted(leaked):
            print(f"  - {column}")

    return [passed]


# ============================================================
# Final report
# ============================================================

def main() -> None:

    print("=" * 70)
    print("RECLAIM TRANSACTION WORLD VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    if not validate_files():
        return

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    customers = pd.read_csv(
        CUSTOMER_PATH
    )

    transactions = pd.read_csv(
        TRANSACTION_PATH,
        parse_dates=["timestamp"],
    )

    print(
        f"\nCustomers:    {len(customers):,}"
    )

    print(
        f"Transactions: {len(transactions):,}"
    )

    results = []

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    results.extend(
        validate_transaction_structure(
            transactions
        )
    )

    # --------------------------------------------------------
    # Customer coverage
    # --------------------------------------------------------

    results.extend(
        analyze_customer_coverage(
            customers,
            transactions,
        )
    )

    # --------------------------------------------------------
    # Activity relationship
    # --------------------------------------------------------

    results.extend(
        analyze_activity_relationship(
            customers,
            transactions,
        )
    )

    # --------------------------------------------------------
    # Concentration
    # --------------------------------------------------------

    analyze_transaction_concentration(
        transactions
    )

    # --------------------------------------------------------
    # Amount distribution
    # --------------------------------------------------------

    analyze_amount_distribution(
        transactions
    )

    # --------------------------------------------------------
    # Daily volume
    # --------------------------------------------------------

    results.extend(
        analyze_daily_volume(
            transactions
        )
    )

    # --------------------------------------------------------
    # Payment method
    # --------------------------------------------------------

    results.extend(
        analyze_payment_methods(
            customers,
            transactions,
        )
    )

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

    results.extend(
        check_hidden_feature_leakage(
            transactions
        )
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print_section(
        "FINAL TRANSACTION WORLD RESULT"
    )

    passed = sum(results)
    total = len(results)

    print(
        f"Checks passed: "
        f"{passed}/{total}"
    )

    if all(results):

        print(
            "\nTRANSACTION WORLD STATUS: VALID"
        )

        print(
            "The transaction population passed "
            "the current validation gates."
        )

    else:

        print(
            "\nTRANSACTION WORLD STATUS: "
            "REVIEW REQUIRED"
        )

        print(
            "One or more validation gates failed."
        )


if __name__ == "__main__":
    main()