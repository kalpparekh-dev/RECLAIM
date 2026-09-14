from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

TRANSACTION_PATH = Path(
    "data/generated/transaction_outcomes.csv"
)

OUTPUT_PATH = Path(
    "data/generated/recovery_context.csv"
)

LOOKBACK_DAYS = 7


# ============================================================
# Columns that must NEVER reach the agent
# ============================================================

FORBIDDEN_COLUMNS = {
    "reliability",
    "financial_stability",
    "responsiveness",
    "retry_sensitivity",
    "reminder_sensitivity",
    "activity_level",
    "_simulated_failure_probability",
}


# ============================================================
# Utility
# ============================================================

def print_section(title: str) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Main feature construction
# ============================================================

def build_recovery_context(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    print_section(
        "BUILDING OBSERVABLE RECOVERY CONTEXT"
    )

    df = transactions.copy()

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "customer_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Basic customer history
    # --------------------------------------------------------

    customer_groups = df.groupby(
        "customer_id",
        sort=False,
    )

    # Total historical transactions before current row.
    df["historical_transaction_count"] = (
        customer_groups.cumcount()
    )

    # --------------------------------------------------------
    # Previous successes / failures
    # --------------------------------------------------------

    success_flag = (
        df["status"] == "SUCCESS"
    ).astype(int)

    failure_flag = (
        df["status"] == "FAILED"
    ).astype(int)

    df["_success_flag"] = success_flag
    df["_failure_flag"] = failure_flag

    df["historical_success_count"] = (
        df.groupby("customer_id")[
            "_success_flag"
        ]
        .cumsum()
        .shift(1)
        .fillna(0)
    )

    df["historical_failure_count"] = (
        df.groupby("customer_id")[
            "_failure_flag"
        ]
        .cumsum()
        .shift(1)
        .fillna(0)
    )

    # --------------------------------------------------------
    # Historical success rate
    # --------------------------------------------------------

    history_total = (
        df["historical_success_count"]
        +
        df["historical_failure_count"]
    )

    df["historical_success_rate"] = np.where(
        history_total > 0,
        (
            df["historical_success_count"]
            / history_total
        ),
        0.0,
    )

    # --------------------------------------------------------
    # Previous transaction timestamp
    # --------------------------------------------------------

    previous_timestamp = (
        df.groupby("customer_id")[
            "timestamp"
        ]
        .shift(1)
    )

    df["hours_since_previous_transaction"] = (
        (
            df["timestamp"]
            -
            previous_timestamp
        )
        .dt.total_seconds()
        / 3600
    )

    df[
        "hours_since_previous_transaction"
    ] = (
        df[
            "hours_since_previous_transaction"
        ]
        .fillna(-1)
    )

    # --------------------------------------------------------
    # Previous transaction status
    # --------------------------------------------------------

    previous_status = (
        df.groupby("customer_id")[
            "status"
        ]
        .shift(1)
    )

    df["previous_transaction_success"] = (
        previous_status == "SUCCESS"
    ).astype(int)

    # For the first transaction, there is no previous event.
    first_transaction = (
        previous_status.isna()
    )

    df.loc[
        first_transaction,
        "previous_transaction_success",
    ] = -1

    # --------------------------------------------------------
    # Previous failure reason
    # --------------------------------------------------------

    previous_failure_reason = (
        df.groupby("customer_id")[
            "failure_reason"
        ]
        .shift(1)
    )

    df["previous_failure_reason"] = (
        previous_failure_reason
        .fillna("NONE")
    )

    # --------------------------------------------------------
    # Rolling 7-day behavior
    # --------------------------------------------------------

    print(
        "Calculating 7-day behavioral history..."
    )

    # We calculate these per customer using time-aware
    # rolling windows.
    #
    # Current transaction is excluded from the window.
    # This is extremely important because otherwise the model
    # would see information about the event it is trying to
    # explain.

    rolling_success_counts = []
    rolling_failure_counts = []
    rolling_transaction_counts = []
    rolling_amount_means = []

    for customer_id, group in df.groupby(
        "customer_id",
        sort=False,
    ):

        group = group.sort_values(
            "timestamp"
        ).copy()

        time_index = (
            group.set_index("timestamp")
        )

        success_series = (
            time_index["_success_flag"]
        )

        failure_series = (
            time_index["_failure_flag"]
        )

        amount_series = (
            time_index["amount"]
        )

        # Shift by one event before rolling to prevent
        # current-event leakage.
        previous_success = (
            success_series.shift(1)
        )

        previous_failure = (
            failure_series.shift(1)
        )

        previous_amount = (
            amount_series.shift(1)
        )

        rolling_success = (
            previous_success
            .rolling(
                f"{LOOKBACK_DAYS}D",
                min_periods=1,
            )
            .sum()
        )

        rolling_failure = (
            previous_failure
            .rolling(
                f"{LOOKBACK_DAYS}D",
                min_periods=1,
            )
            .sum()
        )

        rolling_transactions = (
            (
                previous_success
                +
                previous_failure
            )
            .rolling(
                f"{LOOKBACK_DAYS}D",
                min_periods=1,
            )
            .sum()
        )

        rolling_amount = (
            previous_amount
            .rolling(
                f"{LOOKBACK_DAYS}D",
                min_periods=1,
            )
            .mean()
        )

        rolling_success_counts.extend(
            zip(
                group.index,
                rolling_success.values,
            )
        )

        rolling_failure_counts.extend(
            zip(
                group.index,
                rolling_failure.values,
            )
        )

        rolling_transaction_counts.extend(
            zip(
                group.index,
                rolling_transactions.values,
            )
        )

        rolling_amount_means.extend(
            zip(
                group.index,
                rolling_amount.values,
            )
        )

    # --------------------------------------------------------
    # Put rolling values back into dataframe
    # --------------------------------------------------------

    rolling_success_dict = dict(
        rolling_success_counts
    )

    rolling_failure_dict = dict(
        rolling_failure_counts
    )

    rolling_transaction_dict = dict(
        rolling_transaction_counts
    )

    rolling_amount_dict = dict(
        rolling_amount_means
    )

    df["transactions_last_7d"] = (
        df.index.map(
            rolling_transaction_dict
        )
        .fillna(0)
    )

    df["successful_payments_last_7d"] = (
        df.index.map(
            rolling_success_dict
        )
        .fillna(0)
    )

    df["failed_payments_last_7d"] = (
        df.index.map(
            rolling_failure_dict
        )
        .fillna(0)
    )

    df["avg_amount_last_7d"] = (
        df.index.map(
            rolling_amount_dict
        )
    )

    df["avg_amount_last_7d"] = (
        df["avg_amount_last_7d"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # 7-day failure rate
    # --------------------------------------------------------

    df["failure_rate_last_7d"] = np.where(
        df["transactions_last_7d"] > 0,
        (
            df["failed_payments_last_7d"]
            /
            df["transactions_last_7d"]
        ),
        0.0,
    )

    # --------------------------------------------------------
    # Amount deviation
    # --------------------------------------------------------

    df["amount_vs_recent_average"] = (
        np.where(
            df["avg_amount_last_7d"] > 0,
            (
                df["amount"]
                /
                df["avg_amount_last_7d"]
            ),
            1.0,
        )
    )

    # --------------------------------------------------------
    # Customer payment momentum
    # --------------------------------------------------------

    df["payment_momentum"] = (
        df["transactions_last_7d"]
        /
        (
            df["historical_transaction_count"]
            .clip(lower=1)
        )
    )

    # --------------------------------------------------------
    # Failure streak
    # --------------------------------------------------------

    failure_streak = []

    current_customer = None
    current_streak = 0

    for _, row in df.iterrows():

        customer_id = row[
            "customer_id"
        ]

        if customer_id != current_customer:

            current_customer = customer_id
            current_streak = 0

        if row["status"] == "FAILED":

            current_streak += 1

        else:

            current_streak = 0

        failure_streak.append(
            current_streak
        )

    df["failure_streak"] = (
        failure_streak
    )

    # --------------------------------------------------------
    # Time since last successful payment
    # --------------------------------------------------------

    last_success_timestamp = []

    last_success_by_customer = {}

    for _, row in df.iterrows():

        customer_id = row[
            "customer_id"
        ]

        current_timestamp = row[
            "timestamp"
        ]

        last_success = (
            last_success_by_customer
            .get(customer_id)
        )

        if last_success is None:

            last_success_timestamp.append(
                -1
            )

        else:

            hours = (
                current_timestamp
                -
                last_success
            ).total_seconds() / 3600

            last_success_timestamp.append(
                hours
            )

        if row["status"] == "SUCCESS":

            last_success_by_customer[
                customer_id
            ] = current_timestamp

    df["hours_since_last_success"] = (
        last_success_timestamp
    )

    # --------------------------------------------------------
    # Time since last failure
    # --------------------------------------------------------

    last_failure_timestamp = []

    last_failure_by_customer = {}

    for _, row in df.iterrows():

        customer_id = row[
            "customer_id"
        ]

        current_timestamp = row[
            "timestamp"
        ]

        last_failure = (
            last_failure_by_customer
            .get(customer_id)
        )

        if last_failure is None:

            last_failure_timestamp.append(
                -1
            )

        else:

            hours = (
                current_timestamp
                -
                last_failure
            ).total_seconds() / 3600

            last_failure_timestamp.append(
                hours
            )

        if row["status"] == "FAILED":

            last_failure_by_customer[
                customer_id
            ] = current_timestamp

    df["hours_since_last_failure"] = (
        last_failure_timestamp
    )

    # --------------------------------------------------------
    # Payment method consistency
    # --------------------------------------------------------

    # Whether this customer historically used the same method.
    method_counts = (
        df.groupby(
            [
                "customer_id",
                "payment_method",
            ]
        )
        .size()
        .rename("method_count")
        .reset_index()
    )

    method_totals = (
        method_counts
        .groupby("customer_id")[
            "method_count"
        ]
        .sum()
        .rename("customer_method_total")
        .reset_index()
    )

    method_counts = method_counts.merge(
        method_totals,
        on="customer_id",
        how="left",
    )

    method_counts[
        "method_share"
    ] = (
        method_counts["method_count"]
        /
        method_counts[
            "customer_method_total"
        ]
    )

    df = df.merge(
        method_counts[
            [
                "customer_id",
                "payment_method",
                "method_share",
            ]
        ],
        on=[
            "customer_id",
            "payment_method",
        ],
        how="left",
    )

    df["method_share"] = (
        df["method_share"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Extract only failed payments
    # --------------------------------------------------------

    recovery_context = df[
        df["status"] == "FAILED"
    ].copy()

    print(
        f"Failed payments available for "
        f"recovery analysis: "
        f"{len(recovery_context):,}"
    )

    # --------------------------------------------------------
    # Remove simulator-only information
    # --------------------------------------------------------

    columns_to_remove = (
        FORBIDDEN_COLUMNS
        |
        {
            "_success_flag",
            "_failure_flag",
        }
    )

    recovery_context = (
        recovery_context.drop(
            columns=[
                column
                for column in columns_to_remove
                if column in recovery_context.columns
            ],
            errors="ignore",
        )
    )

    # --------------------------------------------------------
    # Sort final dataset
    # --------------------------------------------------------

    recovery_context = (
        recovery_context
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return recovery_context


# ============================================================
# Validation
# ============================================================

def validate_recovery_context(
    context: pd.DataFrame,
) -> None:

    print_section(
        "RECOVERY CONTEXT VALIDATION"
    )

    # --------------------------------------------------------
    # Required observable columns
    # --------------------------------------------------------

    required_columns = {
        "transaction_id",
        "customer_id",
        "timestamp",
        "amount",
        "payment_method",
        "status",
        "failure_reason",
        "historical_transaction_count",
        "historical_success_count",
        "historical_failure_count",
        "historical_success_rate",
        "transactions_last_7d",
        "successful_payments_last_7d",
        "failed_payments_last_7d",
        "failure_rate_last_7d",
        "avg_amount_last_7d",
        "amount_vs_recent_average",
        "failure_streak",
        "hours_since_last_success",
        "hours_since_last_failure",
        "previous_transaction_success",
        "previous_failure_reason",
    }

    missing = (
        required_columns
        -
        set(context.columns)
    )

    passed = len(missing) == 0

    print(
        f"[{'PASS' if passed else 'FAIL'}] "
        "Required observable recovery features"
    )

    if missing:

        print("\nMissing columns:")

        for column in sorted(missing):

            print(f"  - {column}")

    # --------------------------------------------------------
    # No hidden feature leakage
    # --------------------------------------------------------

    leaked = (
        FORBIDDEN_COLUMNS
        &
        set(context.columns)
    )

    leakage_pass = len(leaked) == 0

    print(
        f"[{'PASS' if leakage_pass else 'FAIL'}] "
        "No hidden simulator traits exposed"
    )

    if leaked:

        print("\nLeaked columns:")

        for column in sorted(leaked):

            print(f"  - {column}")

    # --------------------------------------------------------
    # Only failed transactions
    # --------------------------------------------------------

    only_failed = (
        context["status"]
        == "FAILED"
    ).all()

    print(
        f"[{'PASS' if only_failed else 'FAIL'}] "
        "Recovery context contains only failed payments"
    )

    # --------------------------------------------------------
    # No missing values in important features
    # --------------------------------------------------------

    important_features = [
        "amount",
        "transactions_last_7d",
        "successful_payments_last_7d",
        "failed_payments_last_7d",
        "failure_rate_last_7d",
        "avg_amount_last_7d",
        "amount_vs_recent_average",
        "failure_streak",
    ]

    missing_features = (
        context[
            important_features
        ]
        .isna()
        .sum()
        .sum()
    )

    missing_pass = (
        missing_features == 0
    )

    print(
        f"[{'PASS' if missing_pass else 'FAIL'}] "
        "Important behavioral features contain no missing values"
    )

    # --------------------------------------------------------
    # Range checks
    # --------------------------------------------------------

    success_rate_valid = (
        context[
            "historical_success_rate"
        ]
        .between(0, 1)
        .all()
    )

    failure_rate_valid = (
        context[
            "failure_rate_last_7d"
        ]
        .between(0, 1)
        .all()
    )

    print(
        f"[{'PASS' if success_rate_valid else 'FAIL'}] "
        "Historical success rate is between 0 and 1"
    )

    print(
        f"[{'PASS' if failure_rate_valid else 'FAIL'}] "
        "7-day failure rate is between 0 and 1"
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print(
        "\nObservable feature summary:"
    )

    summary_columns = [
        "amount",
        "historical_transaction_count",
        "historical_success_rate",
        "transactions_last_7d",
        "failed_payments_last_7d",
        "failure_rate_last_7d",
        "avg_amount_last_7d",
        "failure_streak",
    ]

    print(
        context[
            summary_columns
        ]
        .describe()
        .round(3)
        .to_string()
    )


# ============================================================
# Save
# ============================================================

def save_context(
    context: pd.DataFrame,
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    context.to_csv(
        OUTPUT_PATH,
        index=False,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("RECLAIM RECOVERY CONTEXT BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # Input check
    # --------------------------------------------------------

    if not TRANSACTION_PATH.exists():

        raise FileNotFoundError(
            f"Transaction outcome dataset not found: "
            f"{TRANSACTION_PATH}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    transactions = pd.read_csv(
        TRANSACTION_PATH,
        parse_dates=["timestamp"],
    )

    print(
        f"Transactions loaded: "
        f"{len(transactions):,}"
    )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_recovery_context(
        transactions
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_recovery_context(
        context
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_context(
        context
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        "\nRECOVERY CONTEXT STATUS: READY"
    )


if __name__ == "__main__":
    main()