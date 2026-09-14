from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("data/generated/customers.csv")

EXPECTED_CUSTOMERS = 1000

PAYMENT_METHODS = {
    "UPI",
    "CARD",
    "NETBANKING",
    "WALLET",
}

TRAIT_COLUMNS = [
    "reliability",
    "financial_stability",
    "responsiveness",
    "retry_sensitivity",
    "reminder_sensitivity",
    "activity_level",
]


# ============================================================
# Utility
# ============================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


# ============================================================
# Validation functions
# ============================================================

def validate_file_exists() -> bool:
    exists = DATA_PATH.exists()

    print(f"[{pass_fail(exists)}] Dataset exists")

    if not exists:
        print(f"Missing file: {DATA_PATH}")

    return exists


def validate_customer_count(df: pd.DataFrame) -> bool:
    passed = len(df) == EXPECTED_CUSTOMERS

    print(
        f"[{pass_fail(passed)}] Customer count: "
        f"{len(df):,} / {EXPECTED_CUSTOMERS:,}"
    )

    return passed


def validate_required_columns(df: pd.DataFrame) -> bool:
    required_columns = {
        "customer_id",
        "signup_date",
        "preferred_payment_method",
        *TRAIT_COLUMNS,
    }

    missing = required_columns - set(df.columns)

    passed = len(missing) == 0

    print(f"[{pass_fail(passed)}] Required columns")

    if missing:
        print("Missing columns:")
        for column in sorted(missing):
            print(f"  - {column}")

    return passed


def validate_customer_ids(df: pd.DataFrame) -> bool:
    unique_ids = df["customer_id"].is_unique
    no_missing = df["customer_id"].notna().all()

    passed = unique_ids and no_missing

    print(f"[{pass_fail(passed)}] Customer IDs")

    if not unique_ids:
        print("Duplicate customer IDs detected.")

    if not no_missing:
        print("Missing customer IDs detected.")

    return passed


def validate_missing_values(df: pd.DataFrame) -> bool:
    missing_counts = df.isna().sum()

    total_missing = missing_counts.sum()

    passed = total_missing == 0

    print(f"[{pass_fail(passed)}] Missing values")

    if not passed:
        print("\nMissing values by column:")

        for column, count in missing_counts.items():
            if count > 0:
                print(f"  {column}: {count}")

    return passed


def validate_trait_ranges(df: pd.DataFrame) -> bool:
    passed = True

    print("\nTrait ranges:")

    for column in TRAIT_COLUMNS:
        minimum = df[column].min()
        maximum = df[column].max()

        valid = (
            df[column].between(0.0, 1.0).all()
        )

        if not valid:
            passed = False

        print(
            f"  [{pass_fail(valid)}] "
            f"{column:<24} "
            f"min={minimum:.3f} "
            f"max={maximum:.3f}"
        )

    return passed


def validate_payment_methods(df: pd.DataFrame) -> bool:
    actual_methods = set(
        df["preferred_payment_method"].dropna().unique()
    )

    passed = actual_methods.issubset(PAYMENT_METHODS)

    print(f"[{pass_fail(passed)}] Payment methods")

    unexpected = actual_methods - PAYMENT_METHODS

    if unexpected:
        print("Unexpected payment methods:")
        for method in sorted(unexpected):
            print(f"  - {method}")

    return passed


def validate_distributions(df: pd.DataFrame) -> None:
    print_section("PAYMENT METHOD DISTRIBUTION")

    distribution = (
        df["preferred_payment_method"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    for method, percentage in distribution.items():
        print(f"{method:<12}: {percentage:>6.2f}%")


def print_trait_statistics(df: pd.DataFrame) -> None:
    print_section("LATENT TRAIT STATISTICS")

    statistics = (
        df[TRAIT_COLUMNS]
        .describe()
        .loc[
            [
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        ]
        .round(3)
    )

    print(statistics.to_string())


def print_correlations(df: pd.DataFrame) -> None:
    print_section("LATENT TRAIT CORRELATIONS")

    correlations = (
        df[TRAIT_COLUMNS]
        .corr()
        .round(3)
    )

    print(correlations.to_string())


def validate_extreme_concentration(df: pd.DataFrame) -> bool:
    """
    Check whether any trait is suspiciously concentrated
    near 0 or 1.

    This is a sanity check, not a statistical proof.
    """

    passed = True

    print_section("EXTREME VALUE CHECK")

    for column in TRAIT_COLUMNS:
        values = df[column]

        near_zero = (values < 0.05).mean()
        near_one = (values > 0.95).mean()

        print(
            f"{column:<24} "
            f"<0.05: {near_zero * 100:5.2f}% | "
            f">0.95: {near_one * 100:5.2f}%"
        )

        # We flag extreme concentration if more than 10%
        # of the population is near either boundary.
        if near_zero > 0.10 or near_one > 0.10:
            passed = False

    print(
        f"\n[{pass_fail(passed)}] "
        "No suspicious extreme concentration"
    )

    return passed


# ============================================================
# Main validation pipeline
# ============================================================

def main() -> None:

    print("=" * 70)
    print("RECLAIM CUSTOMER POPULATION VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # File check
    # --------------------------------------------------------

    if not validate_file_exists():
        return

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset: {DATA_PATH}")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    print_section("STRUCTURAL VALIDATION")

    results = []

    results.append(
        validate_customer_count(df)
    )

    results.append(
        validate_required_columns(df)
    )

    results.append(
        validate_customer_ids(df)
    )

    results.append(
        validate_missing_values(df)
    )

    results.append(
        validate_trait_ranges(df)
    )

    results.append(
        validate_payment_methods(df)
    )

    # --------------------------------------------------------
    # Distribution analysis
    # --------------------------------------------------------

    validate_distributions(df)

    print_trait_statistics(df)

    # --------------------------------------------------------
    # Correlation analysis
    # --------------------------------------------------------

    print_correlations(df)

    # --------------------------------------------------------
    # Extreme-value analysis
    # --------------------------------------------------------

    results.append(
        validate_extreme_concentration(df)
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print_section("FINAL VALIDATION RESULT")

    passed_checks = sum(results)
    total_checks = len(results)

    print(
        f"Checks passed: {passed_checks}/{total_checks}"
    )

    if all(results):
        print("\nWORLD STATUS: VALID")
        print(
            "Customer population passed all structural "
            "and distribution sanity checks."
        )
    else:
        print("\nWORLD STATUS: REVIEW REQUIRED")
        print(
            "At least one validation check failed. "
            "Do not freeze the customer population yet."
        )


if __name__ == "__main__":
    main()