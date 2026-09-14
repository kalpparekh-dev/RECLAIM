"""
RECLAIM - INCREMENTAL RECOVERY OPTIMIZER

Purpose:
    Optimize recovery interventions using incremental value rather than
    simply maximizing total recovered value.

Inputs:
    data/generated/recovery_model_predictions.csv
    data/generated/recovery_outcomes.csv

Important:
    recovery_model_predictions.csv contains the held-out prediction population
    (currently 581 rows), while recovery_outcomes.csv contains the full
    population (currently 2,904 rows).

    Therefore, outcomes are filtered to prediction transaction IDs before
    evaluation.

Outputs:
    data/generated/incremental_recovery_comparison.csv
    data/generated/incremental_recovery_policy.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "generated"

PREDICTIONS_FILE = DATA_DIR / "recovery_model_predictions.csv"
OUTCOMES_FILE = DATA_DIR / "recovery_outcomes.csv"

COMPARISON_FILE = DATA_DIR / "incremental_recovery_comparison.csv"
POLICY_FILE = DATA_DIR / "incremental_recovery_policy.csv"


# Economic assumptions
INTERVENTION_COST = 1.50
FRICTION_COST = 0.25

# Minimum probability for intervention
MIN_PROBABILITY = 0.05

# Threshold grid
THRESHOLDS = np.round(
    np.arange(0.05, 0.951, 0.025),
    3
)

# Top-k policies
TOP_PERCENTAGES = [
    5,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
]


# ============================================================
# DISPLAY HELPERS
# ============================================================

def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def money(value):
    return f"₹{float(value):,.2f}"


# ============================================================
# COLUMN HELPERS
# ============================================================

def find_column(df, candidates):
    """
    Return the first available column from candidates.
    """
    for column in candidates:
        if column in df.columns:
            return column
    return None


def normalize_binary(value):
    """
    Normalize recovery flag values to 0/1.
    """
    if pd.isna(value):
        return 0

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(float(value) > 0)

    text = str(value).strip().lower()

    if text in {
        "1",
        "true",
        "yes",
        "y",
        "recovered",
        "success",
        "successful",
    }:
        return 1

    return 0


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required = [
        "transaction_id",
        "amount",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Prediction dataset missing columns: {missing}"
        )

    print(
        f"Prediction rows: {len(df):,}"
    )

    print("[PASS] Prediction dataset structure")

    if df["transaction_id"].duplicated().any():
        raise ValueError(
            "Prediction transaction IDs are not unique."
        )

    print("[PASS] Prediction transaction IDs are unique")

    # --------------------------------------------------------
    # Normalize probability column
    # --------------------------------------------------------

    probability_column = find_column(
        df,
        [
            "recovery_probability",
            "calibrated_recovery_probability",
            "base_recovery_probability",
            "action_recovery_probability",
        ],
    )

    if probability_column is None:
        raise ValueError(
            "No recovery probability column found."
        )

    if probability_column != "recovery_probability":
        df["recovery_probability"] = pd.to_numeric(
            df[probability_column],
            errors="coerce"
        )
    else:
        df["recovery_probability"] = pd.to_numeric(
            df["recovery_probability"],
            errors="coerce"
        )

    df["recovery_probability"] = (
        df["recovery_probability"]
        .clip(0, 1)
    )

    if df["recovery_probability"].isna().any():
        raise ValueError(
            "Recovery probability contains missing values."
        )

    print("[PASS] Recovery probabilities normalized")

    # --------------------------------------------------------
    # Normalize amount
    # --------------------------------------------------------

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    if df["amount"].isna().any():
        raise ValueError(
            "Prediction amount contains missing values."
        )

    if (df["amount"] <= 0).any():
        raise ValueError(
            "Prediction amount contains non-positive values."
        )

    return df


# ============================================================
# LOAD AND NORMALIZE OUTCOMES
# ============================================================

def load_outcomes():
    if not OUTCOMES_FILE.exists():
        raise FileNotFoundError(
            f"Outcome file not found:\n{OUTCOMES_FILE}"
        )

    df = pd.read_csv(OUTCOMES_FILE)

    print()
    print("Normalizing recovery outcome schema...")

    print("Outcome columns detected:")
    print(list(df.columns))

    if "transaction_id" not in df.columns:
        raise ValueError(
            "Outcome dataset must contain transaction_id."
        )

    # --------------------------------------------------------
    # Recovery flag
    # --------------------------------------------------------

    if "recovery_flag" in df.columns:

        df["recovery_flag"] = df[
            "recovery_flag"
        ].apply(normalize_binary)

        print(
            "[PASS] Existing `recovery_flag` normalized"
        )

    elif "recovered" in df.columns:

        df["recovery_flag"] = df[
            "recovered"
        ].apply(normalize_binary)

        print(
            "[PASS] Derived `recovery_flag` from `recovered`"
        )

    elif "recovery_status" in df.columns:

        df["recovery_flag"] = (
            df["recovery_status"]
            .astype(str)
            .str.upper()
            .eq("RECOVERED")
            .astype(int)
        )

        print(
            "[PASS] Derived `recovery_flag` from `recovery_status`"
        )

    else:

        raise ValueError(
            "Could not derive recovery_flag."
        )

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    if "amount" not in df.columns:

        raise ValueError(
            "Outcome dataset does not contain amount."
        )

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Recovered amount
    # --------------------------------------------------------

    if "recovered_amount" in df.columns:

        df["recovered_amount"] = pd.to_numeric(
            df["recovered_amount"],
            errors="coerce"
        )

        # Missing recovered amount becomes zero for
        # non-recovered transactions.
        df["recovered_amount"] = (
            df["recovered_amount"]
            .fillna(0)
        )

        print(
            "[PASS] Existing `recovered_amount` normalized"
        )

    elif "recovered" in df.columns:

        recovered_numeric = pd.to_numeric(
            df["recovered"],
            errors="coerce"
        )

        recovered_numeric = recovered_numeric.fillna(
            df["recovery_flag"]
        )

        df["recovered_amount"] = np.where(
            df["recovery_flag"] == 1,
            df["amount"],
            0.0
        )

        print(
            "[PASS] Rebuilt `recovered_amount` from `recovered`"
        )

    else:

        df["recovered_amount"] = np.where(
            df["recovery_flag"] == 1,
            df["amount"],
            0.0
        )

        print(
            "[PASS] Rebuilt `recovered_amount` from recovery_flag"
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if df["transaction_id"].duplicated().any():

        raise ValueError(
            "Outcome transaction IDs are not unique."
        )

    print(
        "[PASS] Outcome transaction IDs are unique"
    )

    df["recovered_amount"] = (
        df["recovered_amount"]
        .clip(lower=0)
    )

    return df


# ============================================================
# MERGE DATASETS
# ============================================================

def load_data():
    print("=" * 70)
    print("RECLAIM INCREMENTAL RECOVERY OPTIMIZER")
    print("=" * 70)

    # ------------------------------------------------------------
    # FILE PATHS
    # ------------------------------------------------------------
    prediction_path = "data/generated/recovery_model_predictions.csv"
    outcome_path = "data/generated/recovery_outcomes.csv"

    # ------------------------------------------------------------
    # LOAD FILES
    # ------------------------------------------------------------
    predictions = pd.read_csv(prediction_path)
    outcomes = pd.read_csv(outcome_path)

    print(f"Prediction rows: {len(predictions):,}")

    # ------------------------------------------------------------
    # VALIDATE PREDICTION DATASET
    # ------------------------------------------------------------
    required_prediction_columns = [
        "transaction_id",
        "amount",
    ]

    for column in required_prediction_columns:
        if column not in predictions.columns:
            raise ValueError(
                f"Missing prediction column: {column}"
            )

    print("[PASS] Prediction dataset structure")

    if predictions["transaction_id"].duplicated().any():
        raise ValueError(
            "Prediction transaction IDs are not unique."
        )

    print("[PASS] Prediction transaction IDs are unique")

    # ------------------------------------------------------------
    # NORMALIZE PROBABILITY COLUMN
    # ------------------------------------------------------------
    probability_candidates = [
        "recovery_probability",
        "calibrated_recovery_probability",
        "base_recovery_probability",
        "action_recovery_probability",
    ]

    probability_column = None

    for column in probability_candidates:
        if column in predictions.columns:
            probability_column = column
            break

    if probability_column is None:
        raise ValueError(
            "No recovery probability column found in prediction dataset."
        )

    predictions["recovery_probability"] = pd.to_numeric(
        predictions[probability_column],
        errors="coerce"
    ).clip(0, 1)

    if predictions["recovery_probability"].isna().any():
        raise ValueError(
            "Invalid recovery probabilities detected."
        )

    print("[PASS] Recovery probabilities normalized")

    # ------------------------------------------------------------
    # NORMALIZE OUTCOME SCHEMA
    # ------------------------------------------------------------
    print()
    print("Normalizing recovery outcome schema...")

    print("Outcome columns detected:")
    print(list(outcomes.columns))

    # ------------------------------------------------------------
    # RECOVERY FLAG
    # ------------------------------------------------------------
    if "recovery_flag" not in outcomes.columns:

        if "recovered" in outcomes.columns:
            recovered_numeric = pd.to_numeric(
                outcomes["recovered"],
                errors="coerce"
            ).fillna(0)

            outcomes["recovery_flag"] = (
                recovered_numeric > 0
            ).astype(int)

            print(
                "[PASS] Derived `recovery_flag` from `recovered`"
            )

        elif "recovery_status" in outcomes.columns:

            outcomes["recovery_flag"] = (
                outcomes["recovery_status"]
                .astype(str)
                .str.upper()
                .eq("RECOVERED")
                .astype(int)
            )

            print(
                "[PASS] Derived `recovery_flag` from "
                "`recovery_status`"
            )

        else:
            raise ValueError(
                "Cannot derive recovery_flag. "
                "Need `recovered` or `recovery_status`."
            )

    else:
        outcomes["recovery_flag"] = pd.to_numeric(
            outcomes["recovery_flag"],
            errors="coerce"
        ).fillna(0).astype(int)

    # ------------------------------------------------------------
    # RECOVERED AMOUNT
    # ------------------------------------------------------------
    if "recovered_amount" in outcomes.columns:

        outcomes["recovered_amount"] = pd.to_numeric(
            outcomes["recovered_amount"],
            errors="coerce"
        )

        # If missing, rebuild from recovery flag and amount.
        missing_mask = outcomes["recovered_amount"].isna()

        if missing_mask.any():

            outcomes.loc[
                missing_mask,
                "recovered_amount"
            ] = (
                pd.to_numeric(
                    outcomes.loc[missing_mask, "amount"],
                    errors="coerce"
                ).fillna(0)
                *
                outcomes.loc[
                    missing_mask,
                    "recovery_flag"
                ]
            )

            print(
                "[PASS] Filled missing `recovered_amount`"
            )

        else:
            print(
                "[PASS] Existing `recovered_amount` normalized"
            )

    else:

        # Build it directly from amount + recovery flag.
        outcomes["recovered_amount"] = (
            pd.to_numeric(
                outcomes["amount"],
                errors="coerce"
            ).fillna(0)
            *
            outcomes["recovery_flag"]
        )

        print(
            "[PASS] Rebuilt `recovered_amount`"
        )

    # ------------------------------------------------------------
    # CLEAN NUMERIC COLUMNS
    # ------------------------------------------------------------
    outcomes["amount"] = pd.to_numeric(
        outcomes["amount"],
        errors="coerce"
    )

    outcomes["recovered_amount"] = pd.to_numeric(
        outcomes["recovered_amount"],
        errors="coerce"
    ).fillna(0)

    outcomes["recovery_flag"] = (
        pd.to_numeric(
            outcomes["recovery_flag"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    # Never allow recovered amount > transaction amount.
    outcomes["recovered_amount"] = (
        outcomes["recovered_amount"]
        .clip(lower=0)
    )

    outcomes.loc[
        outcomes["recovered_amount"] > outcomes["amount"],
        "recovered_amount"
    ] = outcomes.loc[
        outcomes["recovered_amount"] > outcomes["amount"],
        "amount"
    ]

    print("[PASS] Recovery outcome schema normalized")

    # ------------------------------------------------------------
    # VALIDATE OUTCOME IDS
    # ------------------------------------------------------------
    if outcomes["transaction_id"].duplicated().any():
        raise ValueError(
            "Outcome transaction IDs are not unique."
        )

    print("[PASS] Outcome transaction IDs are unique")
    print(f"Outcome rows: {len(outcomes):,}")

    # ------------------------------------------------------------
    # ALIGN POPULATIONS
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("ALIGNING PREDICTION AND OUTCOME POPULATIONS")
    print("=" * 70)

    prediction_ids = set(
        predictions["transaction_id"].astype(str)
    )

    outcome_ids = set(
        outcomes["transaction_id"].astype(str)
    )

    missing_in_outcomes = prediction_ids - outcome_ids
    extra_in_outcomes = outcome_ids - prediction_ids

    print(
        f"Prediction population: {len(prediction_ids):,}"
    )
    print(
        f"Outcome population:    {len(outcome_ids):,}"
    )
    print(
        f"Missing in outcomes:   {len(missing_in_outcomes):,}"
    )
    print(
        f"Extra in outcomes:     {len(extra_in_outcomes):,}"
    )

    if missing_in_outcomes:
        raise ValueError(
            "Some prediction transactions are missing "
            "from recovery outcomes."
        )

    # IMPORTANT:
    # The model predictions are only for the held-out 581 rows.
    # Therefore filter the 2,904-row outcome dataset to those 581.
    outcomes = outcomes[
        outcomes["transaction_id"]
        .astype(str)
        .isin(prediction_ids)
    ].copy()

    print(
        f"[PASS] Filtered outcomes to prediction population: "
        f"{len(outcomes):,}"
    )

    # ------------------------------------------------------------
    # MERGE
    # ------------------------------------------------------------
    df = predictions.merge(
        outcomes[
            [
                "transaction_id",
                "recovery_flag",
                "recovered_amount",
                "recovered",
                "recovery_status",
            ]
        ],
        on="transaction_id",
        how="inner",
        suffixes=("_prediction", "_outcome"),
        validate="one_to_one",
    )

    print(
        f"[PASS] Merged dataset: {len(df):,} transactions"
    )

    # ------------------------------------------------------------
    # REPAIR MERGE SUFFIXES
    # ------------------------------------------------------------
    # If prediction dataset also contains amount/recovered_amount,
    # pandas may create suffixed columns.
    #
    # We ALWAYS want the outcome amount/recovery fields here.

    if "amount_outcome" in df.columns:
        df["amount"] = pd.to_numeric(
            df["amount_outcome"],
            errors="coerce"
        )
    elif "amount" not in df.columns:
        raise ValueError(
            "Canonical `amount` column could not be created."
        )

    # recovered_amount should come from outcomes.
    if "recovered_amount_outcome" in df.columns:

        df["recovered_amount"] = pd.to_numeric(
            df["recovered_amount_outcome"],
            errors="coerce"
        ).fillna(0)

    elif "recovered_amount" in df.columns:

        df["recovered_amount"] = pd.to_numeric(
            df["recovered_amount"],
            errors="coerce"
        ).fillna(0)

    else:

        # Final fallback: reconstruct it.
        df["recovered_amount"] = (
            pd.to_numeric(
                df["amount"],
                errors="coerce"
            ).fillna(0)
            *
            pd.to_numeric(
                df["recovery_flag"],
                errors="coerce"
            ).fillna(0)
        )

        print(
            "[PASS] Rebuilt merged `recovered_amount`"
        )

    # ------------------------------------------------------------
    # NORMALIZE RECOVERY FLAG AFTER MERGE
    # ------------------------------------------------------------
    if "recovery_flag" not in df.columns:

        if "recovery_flag_outcome" in df.columns:

            df["recovery_flag"] = pd.to_numeric(
                df["recovery_flag_outcome"],
                errors="coerce"
            ).fillna(0).astype(int)

        else:
            raise ValueError(
                "Canonical `recovery_flag` column could not "
                "be created after merge."
            )

    else:
        df["recovery_flag"] = pd.to_numeric(
            df["recovery_flag"],
            errors="coerce"
        ).fillna(0).astype(int)

    # ------------------------------------------------------------
    # RECOVERED COLUMN
    # ------------------------------------------------------------
    if "recovered" in df.columns:

        df["recovered"] = pd.to_numeric(
            df["recovered"],
            errors="coerce"
        ).fillna(0)

    elif "recovered_outcome" in df.columns:

        df["recovered"] = pd.to_numeric(
            df["recovered_outcome"],
            errors="coerce"
        ).fillna(0)

    else:

        df["recovered"] = df["recovery_flag"]

    # ------------------------------------------------------------
    # FINAL NUMERIC CLEANUP
    # ------------------------------------------------------------
    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    df["recovered_amount"] = pd.to_numeric(
        df["recovered_amount"],
        errors="coerce"
    ).fillna(0)

    df["recovered_amount"] = (
        df["recovered_amount"]
        .clip(lower=0)
    )

    # Never recover more than the transaction amount.
    df.loc[
        df["recovered_amount"] > df["amount"],
        "recovered_amount"
    ] = df.loc[
        df["recovered_amount"] > df["amount"],
        "amount"
    ]

    # ------------------------------------------------------------
    # FINAL VALIDATION
    # ------------------------------------------------------------
    if df["amount"].isna().any():
        raise ValueError(
            "Missing transaction amounts after merge."
        )

    if df["recovered_amount"].isna().any():
        raise ValueError(
            "Missing recovered_amount after normalization."
        )

    if df["recovery_flag"].isna().any():
        raise ValueError(
            "Missing recovery_flag after normalization."
        )

    print(
        "[PASS] Canonical amount column: `amount`"
    )

    print(
        "[PASS] Canonical recovered amount column: "
        "`recovered_amount`"
    )

    print(
        "[PASS] Canonical recovery flag column: "
        "`recovery_flag`"
    )

    print(
        f"[PASS] Final modeling population: "
        f"{len(df):,}"
    )

    return df

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    df = predictions.merge(
        outcomes_subset[
            [
                "transaction_id",
                "recovery_flag",
                "recovered_amount",
            ]
        ],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    if len(df) != len(predictions):

        raise ValueError(
            "Merged dataset size changed unexpectedly."
        )

    print(
        f"[PASS] Merged dataset: {len(df):,} transactions"
    )

    # --------------------------------------------------------
    # Validate merged data
    # --------------------------------------------------------

    if df["recovery_flag"].isna().any():

        raise ValueError(
            "Merged recovery_flag contains missing values."
        )

    if df["recovered_amount"].isna().any():

        raise ValueError(
            "Merged recovered_amount contains missing values."
        )

    df["recovery_flag"] = (
        df["recovery_flag"]
        .astype(int)
    )

    df["recovered_amount"] = (
        pd.to_numeric(
            df["recovered_amount"],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    df["amount"] = (
        pd.to_numeric(
            df["amount"],
            errors="coerce"
        )
    )

    if df["amount"].isna().any():
        raise ValueError(
            "Merged amount contains missing values."
        )

    print("[PASS] Recovery flags valid")
    print("[PASS] Recovered amounts valid")

    print(
        f"Observed recovery rate: "
        f"{df['recovery_flag'].mean():.2%}"
    )

    return df


# ============================================================
# ECONOMIC CALCULATIONS
# ============================================================

def calculate_economic_features(df):

    section("CALCULATING INCREMENTAL ECONOMIC VALUE")

    df = df.copy()

    # --------------------------------------------------------
    # Expected gross recovery
    # --------------------------------------------------------

    df["expected_recovery"] = (
        df["amount"]
        * df["recovery_probability"]
    )

    # --------------------------------------------------------
    # Expected intervention cost
    # --------------------------------------------------------

    df["expected_intervention_cost"] = (
        INTERVENTION_COST
    )

    df["expected_friction_cost"] = (
        FRICTION_COST
    )

    # --------------------------------------------------------
    # Expected net recovery
    # --------------------------------------------------------

    df["expected_net_recovery"] = (
        df["expected_recovery"]
        - df["expected_intervention_cost"]
        - df["expected_friction_cost"]
    )

    print(
        "[PASS] Expected recovery calculated"
    )

    print(
        "[PASS] Expected net recovery calculated"
    )

    # --------------------------------------------------------
    # Incremental value
    #
    # If we do nothing, expected recovery is zero.
    # Therefore positive expected net recovery is the
    # incremental economic opportunity.
    # --------------------------------------------------------

    df["incremental_value"] = (
        df["expected_net_recovery"]
    )

    print(
        "[PASS] Incremental economic value calculated"
    )

    return df


# ============================================================
# POLICY MASKS
# ============================================================

def top_k_mask(df, percentage):

    n = len(df)

    if percentage >= 100:
        return pd.Series(
            True,
            index=df.index
        )

    k = max(
        1,
        int(np.ceil(
            n * percentage / 100
        ))
    )

    ranked = (
        df.sort_values(
            "incremental_value",
            ascending=False
        )
    )

    selected_ids = set(
        ranked.head(k)["transaction_id"]
    )

    return df[
        "transaction_id"
    ].isin(selected_ids)


def probability_mask(df, threshold):

    return (
        df["recovery_probability"]
        >= threshold
    )


# ============================================================
# EVALUATE POLICY
# ============================================================

def evaluate_policy(
    df,
    policy_name,
    mask,
):

    selected = df.loc[mask].copy()

    interventions = len(selected)

    recovered_payments = int(
        selected["recovery_flag"].sum()
    )

    recovered_value = float(
        selected["recovered_amount"].sum()
    )

    total_failed_value = float(
        df["amount"].sum()
    )

    recovery_rate_value = (
        recovered_value / total_failed_value
        if total_failed_value > 0
        else 0.0
    )

    recovery_rate_count = (
        recovered_payments / interventions
        if interventions > 0
        else 0.0
    )

    intervention_cost = (
        interventions
        * INTERVENTION_COST
    )

    friction_cost = (
        interventions
        * FRICTION_COST
    )

    net_recovered_value = (
        recovered_value
        - intervention_cost
        - friction_cost
    )

    net_per_intervention = (
        net_recovered_value / interventions
        if interventions > 0
        else 0.0
    )

    expected_recovery = float(
        selected["expected_recovery"].sum()
    )

    expected_net_recovery = float(
        selected["expected_net_recovery"].sum()
    )

    intervention_rate = (
        interventions / len(df)
        if len(df) > 0
        else 0.0
    )

    return {
        "policy": policy_name,
        "transactions": len(df),
        "interventions": interventions,
        "intervention_rate": intervention_rate,
        "recovered_payments": recovered_payments,
        "recovered_value": recovered_value,
        "recovery_rate_value": recovery_rate_value,
        "recovery_rate_count": recovery_rate_count,
        "expected_recovery": expected_recovery,
        "expected_net_recovery": expected_net_recovery,
        "intervention_cost": intervention_cost,
        "friction_cost": friction_cost,
        "net_recovered_value": net_recovered_value,
        "net_per_intervention": net_per_intervention,
    }


# ============================================================
# TEST POLICIES
# ============================================================

def evaluate_all_policies(df):

    section("TESTING INCREMENTAL RECOVERY POLICIES")

    results = []

    # --------------------------------------------------------
    # RETRY ALL
    # --------------------------------------------------------

    results.append(
        evaluate_policy(
            df,
            "RETRY_ALL",
            pd.Series(
                True,
                index=df.index
            ),
        )
    )

    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    for percentage in TOP_PERCENTAGES:

        mask = top_k_mask(
            df,
            percentage
        )

        results.append(
            evaluate_policy(
                df,
                f"TOP_{percentage}",
                mask,
            )
        )

    # --------------------------------------------------------
    # PROBABILITY THRESHOLDS
    # --------------------------------------------------------

    for threshold in [
        0.40,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
    ]:

        mask = probability_mask(
            df,
            threshold
        )

        results.append(
            evaluate_policy(
                df,
                f"PROBABILITY_{threshold:.2f}",
                mask,
            )
        )

    # --------------------------------------------------------
    # POSITIVE ECONOMIC VALUE
    # --------------------------------------------------------

    positive_mask = (
        df["incremental_value"]
        > 0
    )

    results.append(
        evaluate_policy(
            df,
            "POSITIVE_INCREMENTAL_VALUE",
            positive_mask,
        )
    )

    # --------------------------------------------------------
    # HIGH-VALUE + PROBABILITY
    # --------------------------------------------------------

    hybrid_mask = (
        (df["recovery_probability"] >= 0.60)
        &
        (df["incremental_value"] > 0)
    )

    results.append(
        evaluate_policy(
            df,
            "HIGH_PROBABILITY_POSITIVE_VALUE",
            hybrid_mask,
        )
    )

    results_df = pd.DataFrame(results)

    return results_df


# ============================================================
# BUILD RECLAIM POLICY
# ============================================================

def build_reclaim_policy(df):

    section("BUILDING INCREMENTAL RECLAIM POLICY")

    policy = df.copy()

    # --------------------------------------------------------
    # Core principle:
    #
    # Intervene only when expected incremental value
    # is positive.
    #
    # Then rank opportunities by expected net value.
    # --------------------------------------------------------

    policy["reclaim_intervene"] = (
        policy["incremental_value"] > 0
    )

    # Confidence-aware adjustment if confidence exists.
    if "confidence" in policy.columns:

        policy["reclaim_intervene"] &= (
            policy["confidence"] >= 0.20
        )

    elif "model_confidence" in policy.columns:

        confidence = pd.to_numeric(
            policy["model_confidence"],
            errors="coerce"
        ).fillna(0)

        policy["reclaim_intervene"] &= (
            confidence >= 0.20
        )

    policy["reclaim_action"] = np.where(
        policy["reclaim_intervene"],
        np.where(
            policy["recovery_probability"] >= 0.65,
            "RETRY_NOW",
            "DELAYED_RETRY",
        ),
        "NO_ACTION",
    )

    # Rank only intervention opportunities
    policy["incremental_rank"] = np.nan

    active = policy[
        policy["reclaim_intervene"]
    ].copy()

    if len(active) > 0:

        active = active.sort_values(
            [
                "incremental_value",
                "recovery_probability",
                "amount",
            ],
            ascending=False,
        )

        ranks = pd.Series(
            np.arange(
                1,
                len(active) + 1
            ),
            index=active.index,
        )

        policy.loc[
            active.index,
            "incremental_rank"
        ] = ranks

    interventions = int(
        policy["reclaim_intervene"].sum()
    )

    print(
        f"RECLAIM interventions: {interventions:,}"
    )

    print(
        f"RECLAIM intervention rate: "
        f"{interventions / len(policy):.2%}"
    )

    return policy


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison(results):

    section("POLICY COMPARISON")

    display_df = results.copy()

    for column in [
        "intervention_rate",
        "recovery_rate_value",
        "recovery_rate_count",
    ]:
        display_df[column] = display_df[
            column
        ].map(
            lambda x: f"{x:.2%}"
        )

    for column in [
        "recovered_value",
        "expected_recovery",
        "expected_net_recovery",
        "intervention_cost",
        "friction_cost",
        "net_recovered_value",
        "net_per_intervention",
    ]:
        display_df[column] = display_df[
            column
        ].map(
            money
        )

    print(
        display_df[
            [
                "policy",
                "interventions",
                "intervention_rate",
                "recovered_payments",
                "recovered_value",
                "recovery_rate_value",
                "intervention_cost",
                "friction_cost",
                "net_recovered_value",
                "net_per_intervention",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# OPTIMAL POLICY
# ============================================================

def find_optimal_policy(results):

    # IMPORTANT:
    #
    # Choose by actual measured net recovery first.
    #
    # This prevents a policy with very few interventions
    # and high ₹/action from incorrectly being considered
    # globally optimal when it recovers much less total value.
    #

    eligible = results[
        results["interventions"] > 0
    ].copy()

    best = eligible.sort_values(
        [
            "net_recovered_value",
            "recovered_value",
            "net_per_intervention",
        ],
        ascending=False,
    ).iloc[0]

    section("OPTIMAL INCREMENTAL POLICY")

    print(
        f"Best policy:             {best['policy']}"
    )

    print(
        f"Interventions:           "
        f"{int(best['interventions']):,}"
    )

    print(
        f"Intervention rate:      "
        f"{best['intervention_rate']:.2%}"
    )

    print(
        f"Recovered value:         "
        f"{money(best['recovered_value'])}"
    )

    print(
        f"Net recovered value:     "
        f"{money(best['net_recovered_value'])}"
    )

    print(
        f"Net ₹ / intervention:    "
        f"{money(best['net_per_intervention'])}"
    )

    return best


# ============================================================
# LIFT ANALYSIS
# ============================================================

def print_lift_analysis(
    results,
    reclaim_result,
):

    section("INCREMENTAL RECLAIM LIFT")

    retry_all = results[
        results["policy"] == "RETRY_ALL"
    ].iloc[0]

    reclaim = reclaim_result

    net_difference = (
        reclaim["net_recovered_value"]
        - retry_all["net_recovered_value"]
    )

    interventions_avoided = (
        retry_all["interventions"]
        - reclaim["interventions"]
    )

    intervention_reduction = (
        interventions_avoided
        / retry_all["interventions"]
        if retry_all["interventions"] > 0
        else 0
    )

    print(
        "INCREMENTAL RECLAIM vs RETRY ALL"
    )

    print("-" * 55)

    print(
        f"Net recovery difference: "
        f"{money(net_difference)}"
    )

    print(
        f"Interventions avoided:   "
        f"{int(interventions_avoided):,}"
    )

    print(
        f"Intervention reduction:  "
        f"{intervention_reduction:.2%}"
    )

    if net_difference >= 0:

        print(
            "[PASS] INCREMENTAL RECLAIM beats "
            "RETRY_ALL on net recovered value."
        )

    else:

        print(
            "[WARN] INCREMENTAL RECLAIM does NOT "
            "yet beat RETRY_ALL on net recovered value."
        )


# ============================================================
# VALIDATE POLICY
# ============================================================

def validate_policy(policy):

    section("INCREMENTAL POLICY VALIDATION")

    required = [
        "transaction_id",
        "amount",
        "recovery_probability",
        "recovery_flag",
        "recovered_amount",
        "expected_recovery",
        "expected_net_recovery",
        "incremental_value",
        "reclaim_intervene",
        "reclaim_action",
        "incremental_rank",
    ]

    missing = [
        c
        for c in required
        if c not in policy.columns
    ]

    if missing:
        raise ValueError(
            f"Missing policy columns: {missing}"
        )

    print(
        "[PASS] Required policy columns"
    )

    if not (
        policy["recovery_probability"]
        .between(0, 1)
        .all()
    ):
        raise ValueError(
            "Invalid recovery probabilities."
        )

    print(
        "[PASS] Probability range"
    )

    if (
        policy["expected_recovery"] < 0
    ).any():

        raise ValueError(
            "Expected recovery cannot be negative."
        )

    print(
        "[PASS] Expected recovery values"
    )

    if (
        policy["recovered_amount"] < 0
    ).any():

        raise ValueError(
            "Recovered amount cannot be negative."
        )

    print(
        "[PASS] Recovered amounts"
    )

    valid_actions = {
        "RETRY_NOW",
        "DELAYED_RETRY",
        "NO_ACTION",
    }

    if not set(
        policy["reclaim_action"]
    ).issubset(valid_actions):

        raise ValueError(
            "Invalid reclaim actions."
        )

    print(
        "[PASS] Actions bounded"
    )

    # Ensure NO_ACTION never has intervention flag
    invalid_abstentions = policy[
        (policy["reclaim_action"] == "NO_ACTION")
        &
        (policy["reclaim_intervene"])
    ]

    if len(invalid_abstentions) > 0:

        raise ValueError(
            "NO_ACTION transactions cannot be marked "
            "as intervention."
        )

    print(
        "[PASS] Abstention rule enforced"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "RECLAIM INCREMENTAL RECOVERY OPTIMIZER"
    )

    # --------------------------------------------------------
    # Load and align datasets
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Economic calculations
    # --------------------------------------------------------

    df = calculate_economic_features(df)

    # --------------------------------------------------------
    # Test policies
    # --------------------------------------------------------

    results = evaluate_all_policies(
        df
    )

    print(
        f"Policies evaluated: {len(results)}"
    )

    # --------------------------------------------------------
    # Build RECLAIM
    # --------------------------------------------------------

    reclaim_policy = build_reclaim_policy(
        df
    )

    reclaim_mask = (
        reclaim_policy[
            "reclaim_intervene"
        ]
    )

    reclaim_result = evaluate_policy(
        reclaim_policy,
        "INCREMENTAL_RECLAIM",
        reclaim_mask,
    )

    # Add RECLAIM to policy comparison
    results = pd.concat(
        [
            results,
            pd.DataFrame(
                [reclaim_result]
            ),
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print_comparison(
        results
    )

    # --------------------------------------------------------
    # Optimal policy
    # --------------------------------------------------------

    best = find_optimal_policy(
        results
    )

    # --------------------------------------------------------
    # Lift
    # --------------------------------------------------------

    print_lift_analysis(
        results,
        reclaim_result,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_policy(
        reclaim_policy
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    results.to_csv(
        COMPARISON_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Prepare final policy output
    # --------------------------------------------------------

    final_columns = [
        "transaction_id",
        "amount",
        "recovery_probability",
        "recovery_flag",
        "recovered_amount",
        "expected_recovery",
        "expected_intervention_cost",
        "expected_friction_cost",
        "expected_net_recovery",
        "incremental_value",
        "reclaim_intervene",
        "reclaim_action",
        "incremental_rank",
    ]

    # Add optional descriptive columns
    optional_columns = [
        "customer_id",
        "failure_reason",
        "model_confidence",
        "confidence",
        "recovery_rank",
        "actual_recovered",
    ]

    output_columns = []

    for column in optional_columns:
        if column in reclaim_policy.columns:
            output_columns.append(
                column
            )

    output_columns.extend(
        final_columns
    )

    output_columns = list(
        dict.fromkeys(output_columns)
    )

    final_policy = reclaim_policy[
        output_columns
    ].copy()

    final_policy.to_csv(
        POLICY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    section(
        "INCREMENTAL OPTIMIZATION COMPLETE"
    )

    print(
        f"Best measured policy: "
        f"{best['policy']}"
    )

    print(
        f"Best net recovered value: "
        f"{money(best['net_recovered_value'])}"
    )

    print()
    print(
        "Saved comparison to:"
    )

    print(
        COMPARISON_FILE
    )

    print()
    print(
        "Saved final policy to:"
    )

    print(
        POLICY_FILE
    )

    print()
    print(
        "INCREMENTAL RECOVERY STATUS: READY"
    )


if __name__ == "__main__":
    main()