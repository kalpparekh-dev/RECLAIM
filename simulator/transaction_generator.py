from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

CUSTOMER_DATA_PATH = Path(
    "data/generated/customers.csv"
)

OUTPUT_PATH = Path(
    "data/generated/transactions.csv"
)

DEFAULT_SEED = 20260822

SIMULATION_START = datetime(
    2026,
    1,
    1,
    0,
    0,
    0,
)

SIMULATION_DAYS = 30


# ============================================================
# Utility functions
# ============================================================

def generate_transaction_amount(
    rng: np.random.Generator,
) -> float:
    """
    Generate a realistic right-skewed transaction amount.

    Most transactions are relatively small while a smaller
    number of transactions are significantly larger.
    """

    amount = rng.lognormal(
        mean=np.log(1500),
        sigma=0.85,
    )

    # Keep the pilot economically bounded.
    amount = np.clip(
        amount,
        100,
        50000,
    )

    return round(float(amount), 2)


def generate_transaction_count(
    activity_level: float,
    rng: np.random.Generator,
) -> int:
    """
    Determine how many payment attempts a customer makes
    during the 30-day simulation.

    Higher activity_level means more transactions.

    The minimum is zero because not every customer must
    transact during a particular simulation window.
    """

    # Base transaction intensity.
    #
    # activity_level = 0
    #     → approximately 0.5 expected transactions/day
    #
    # activity_level = 1
    #     → approximately 2.5 expected transactions/day
    #
    # This produces meaningful variation without making
    # transaction volume explode.
    daily_rate = (
        0.5
        + 2.0 * activity_level
    )

    expected_transactions = (
        daily_rate * SIMULATION_DAYS
    )

    count = rng.poisson(
        expected_transactions
    )

    return int(count)


def generate_timestamp(
    rng: np.random.Generator,
) -> datetime:
    """
    Generate a random timestamp within the simulation window.
    """

    total_seconds = (
        SIMULATION_DAYS * 24 * 60 * 60
    )

    random_seconds = rng.integers(
        0,
        total_seconds,
    )

    return (
        SIMULATION_START
        + timedelta(
            seconds=int(random_seconds)
        )
    )


# ============================================================
# Main transaction generator
# ============================================================

def generate_transactions(
    customers: pd.DataFrame,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic payment attempts for all customers.
    """

    rng = np.random.default_rng(seed)

    transactions = []

    transaction_counter = 1

    for _, customer in customers.iterrows():

        activity_level = float(
            customer["activity_level"]
        )

        payment_method = (
            customer["preferred_payment_method"]
        )

        transaction_count = generate_transaction_count(
            activity_level=activity_level,
            rng=rng,
        )

        for _ in range(transaction_count):

            transaction = {
                "transaction_id": (
                    f"T{transaction_counter:07d}"
                ),

                "customer_id": (
                    customer["customer_id"]
                ),

                "timestamp": generate_timestamp(
                    rng
                ),

                "amount": generate_transaction_amount(
                    rng
                ),

                "payment_method": payment_method,
            }

            transactions.append(
                transaction
            )

            transaction_counter += 1

    if not transactions:
        return pd.DataFrame(
            columns=[
                "transaction_id",
                "customer_id",
                "timestamp",
                "amount",
                "payment_method",
            ]
        )

    transactions_df = pd.DataFrame(
        transactions
    )

    # Ensure timestamps are proper datetime values.
    transactions_df["timestamp"] = pd.to_datetime(
        transactions_df["timestamp"]
    )

    # Sort chronologically.
    transactions_df = transactions_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return transactions_df


# ============================================================
# Validation helpers
# ============================================================

def validate_transactions(
    transactions: pd.DataFrame,
) -> None:

    print("\n" + "=" * 60)
    print("TRANSACTION SANITY CHECK")
    print("=" * 60)

    print(
        f"Transactions: "
        f"{len(transactions):,}"
    )

    if transactions.empty:
        print("WARNING: No transactions generated.")
        return

    # --------------------------------------------------------
    # Unique transaction IDs
    # --------------------------------------------------------

    unique_ids = (
        transactions["transaction_id"]
        .is_unique
    )

    print(
        f"[{'PASS' if unique_ids else 'FAIL'}] "
        "Transaction IDs are unique"
    )

    # --------------------------------------------------------
    # Amount validation
    # --------------------------------------------------------

    positive_amounts = (
        transactions["amount"] > 0
    ).all()

    print(
        f"[{'PASS' if positive_amounts else 'FAIL'}] "
        "All transaction amounts are positive"
    )

    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    timestamps_valid = (
        transactions["timestamp"].min()
        >= SIMULATION_START
        and
        transactions["timestamp"].max()
        <
        SIMULATION_START
        + timedelta(days=SIMULATION_DAYS)
    )

    print(
        f"[{'PASS' if timestamps_valid else 'FAIL'}] "
        "Timestamps are inside simulation window"
    )

    # --------------------------------------------------------
    # Customer references
    # --------------------------------------------------------

    customer_ids = set(
        transactions["customer_id"]
    )

    print(
        f"Unique customers with transactions: "
        f"{len(customer_ids):,}"
    )

    # --------------------------------------------------------
    # Amount statistics
    # --------------------------------------------------------

    print("\nAmount statistics:")

    print(
        transactions["amount"]
        .describe()
        .round(2)
        .to_string()
    )

    # --------------------------------------------------------
    # Daily transaction volume
    # --------------------------------------------------------

    daily_counts = (
        transactions
        .assign(
            date=transactions["timestamp"].dt.date
        )
        .groupby("date")
        .size()
    )

    print("\nDaily transaction volume:")

    print(
        daily_counts.describe()
        .round(2)
        .to_string()
    )

    # --------------------------------------------------------
    # Payment method distribution
    # --------------------------------------------------------

    print(
        "\nPayment method distribution:"
    )

    distribution = (
        transactions[
            "payment_method"
        ]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    for method, percentage in distribution.items():
        print(
            f"{method:<12}: "
            f"{percentage:>6.2f}%"
        )


# ============================================================
# Save function
# ============================================================

def save_transactions(
    transactions: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    transactions.to_csv(
        output_path,
        index=False,
    )


# ============================================================
# Script entry point
# ============================================================

def main() -> None:

    print("=" * 60)
    print("RECLAIM TRANSACTION GENERATOR")
    print("=" * 60)

    # --------------------------------------------------------
    # Load customers
    # --------------------------------------------------------

    if not CUSTOMER_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Customer dataset not found: "
            f"{CUSTOMER_DATA_PATH}"
        )

    customers = pd.read_csv(
        CUSTOMER_DATA_PATH
    )

    print(
        f"Customers loaded: "
        f"{len(customers):,}"
    )

    # --------------------------------------------------------
    # Generate transactions
    # --------------------------------------------------------

    print(
        "\nGenerating 30-day transaction world..."
    )

    transactions = generate_transactions(
        customers=customers,
        seed=DEFAULT_SEED,
    )

    print(
        "Transaction generation complete."
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_transactions(
        transactions
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_transactions(
        transactions
    )

    print(
        "\nSaved to:"
        f" {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()