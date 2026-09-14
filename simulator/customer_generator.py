from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd


PAYMENT_METHODS = ["UPI", "CARD", "NETBANKING", "WALLET"]


@dataclass
class Customer:
    customer_id: str
    signup_date: date
    preferred_payment_method: str

    # Hidden simulator traits.
    # These must NEVER be exposed to the RECLAIM agent.
    reliability: float
    financial_stability: float
    responsiveness: float
    retry_sensitivity: float
    reminder_sensitivity: float
    activity_level: float


def _clip(values: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Keep generated latent traits inside the [0, 1] range."""
    return np.clip(values, low, high)


def generate_customers(
    n_customers: int = 1000,
    seed: int = 20260822,
    simulation_start: date = date(2026, 1, 1),
) -> pd.DataFrame:
    """
    Generate the synthetic customer population for the RECLAIM simulator.

    The generated latent traits represent hidden customer behavior.
    They are used by the simulator to generate realistic outcomes but
    must not be used directly as features by the future RECLAIM agent.
    """

    if n_customers <= 0:
        raise ValueError("n_customers must be greater than 0.")

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 1. Generate correlated latent variables
    # ---------------------------------------------------------
    #
    # We create a latent "general reliability" factor first.
    # Other traits are partially influenced by it, rather than
    # being completely independent random numbers.
    #
    general_reliability = rng.beta(5.0, 2.0, n_customers)

    financial_stability = _clip(
        0.55 * general_reliability
        + 0.45 * rng.beta(4.0, 3.0, n_customers)
        + rng.normal(0, 0.06, n_customers)
    )

    responsiveness = _clip(
        0.25 * general_reliability
        + 0.75 * rng.beta(4.0, 2.5, n_customers)
        + rng.normal(0, 0.07, n_customers)
    )

    reliability = _clip(
        0.75 * general_reliability
        + 0.25 * rng.beta(5.0, 2.0, n_customers)
        + rng.normal(0, 0.04, n_customers)
    )

    # Intervention-specific sensitivities are deliberately not
    # perfectly correlated with reliability.
    retry_sensitivity = _clip(
        0.30 * reliability
        + 0.70 * rng.beta(3.0, 2.5, n_customers)
        + rng.normal(0, 0.08, n_customers)
    )

    reminder_sensitivity = _clip(
        0.25 * responsiveness
        + 0.75 * rng.beta(3.5, 2.5, n_customers)
        + rng.normal(0, 0.08, n_customers)
    )

    # Activity is deliberately more independent.
    activity_level = _clip(
        rng.beta(2.5, 2.5, n_customers)
        + rng.normal(0, 0.05, n_customers)
    )

    # ---------------------------------------------------------
    # 2. Generate customer identity information
    # ---------------------------------------------------------

    customer_ids = [
        f"C{i:05d}"
        for i in range(1, n_customers + 1)
    ]

    # Customers can have different account ages.
    signup_offsets = rng.integers(
        low=0,
        high=365,
        size=n_customers,
    )

    signup_dates = [
        simulation_start - timedelta(days=int(offset))
        for offset in signup_offsets
    ]

    # Payment method distribution is configurable later.
    payment_methods = rng.choice(
        PAYMENT_METHODS,
        size=n_customers,
        p=[0.50, 0.30, 0.12, 0.08],
    )

    # ---------------------------------------------------------
    # 3. Construct dataframe
    # ---------------------------------------------------------

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "signup_date": signup_dates,
            "preferred_payment_method": payment_methods,

            # Hidden simulator state
            "reliability": reliability,
            "financial_stability": financial_stability,
            "responsiveness": responsiveness,
            "retry_sensitivity": retry_sensitivity,
            "reminder_sensitivity": reminder_sensitivity,
            "activity_level": activity_level,
        }
    )

    return customers


def save_customers(
    customers: pd.DataFrame,
    output_path: str = "data/generated/customers.csv",
) -> None:
    """Save generated customers to CSV."""

    customers_to_save = customers.copy()

    customers_to_save.to_csv(
        output_path,
        index=False,
    )


if __name__ == "__main__":
    customers = generate_customers()

    print("=" * 50)
    print("RECLAIM CUSTOMER GENERATOR")
    print("=" * 50)

    print(f"Customers generated: {len(customers):,}")

    print("\nPayment method distribution:")
    print(
        customers["preferred_payment_method"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    print("\nLatent trait summary:")
    trait_columns = [
        "reliability",
        "financial_stability",
        "responsiveness",
        "retry_sensitivity",
        "reminder_sensitivity",
        "activity_level",
    ]

    print(
        customers[trait_columns]
        .describe()
        .loc[["mean", "std", "min", "max"]]
        .round(3)
    )

    print("\nFirst 5 customers:")
    print(customers.head())

    save_customers(customers)

    print("\nSaved to: data/generated/customers.csv")