"""
RECLAIM Recovery Model
----------------------

Purpose:
    Learn which failed payments are more likely to be recovered.

Important design rule:
    Only observable merchant-side features are used.

The model produces:
    - recovery probability
    - confidence
    - expected recovery
    - expected net recovery
    - recovery rank

This module does NOT execute payments.
It only creates recovery intelligence.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "generated"

CONTEXT_PATH = DATA_DIR / "recovery_context.csv"
OUTCOME_PATH = DATA_DIR / "recovery_outcomes.csv"

OUTPUT_PATH = DATA_DIR / "recovery_model_predictions.csv"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Observable numerical features.
NUMERIC_FEATURES = [
    "amount",
    "historical_transaction_count",
    "historical_success_rate",
    "transactions_last_7d",
    "failed_payments_last_7d",
    "failure_rate_last_7d",
    "avg_amount_last_7d",
    "failure_streak",
]

# Observable categorical feature.
CATEGORICAL_FEATURES = [
    "failure_reason",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("RECLAIM RECOVERY MODEL")
    print("=" * 70)

    if not CONTEXT_PATH.exists():
        raise FileNotFoundError(
            f"Missing recovery context:\n{CONTEXT_PATH}"
        )

    if not OUTCOME_PATH.exists():
        raise FileNotFoundError(
            f"Missing recovery outcomes:\n{OUTCOME_PATH}"
        )

    context = pd.read_csv(CONTEXT_PATH)

    outcomes = pd.read_csv(OUTCOME_PATH)

    print(
        f"Recovery context rows: {len(context):,}"
    )

    print(
        f"Recovery outcome rows: {len(outcomes):,}"
    )

    return context, outcomes


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(context, outcomes):

    print("\n" + "=" * 70)
    print("PREPARING RECOVERY MODEL DATA")
    print("=" * 70)

    required_context = [
        "transaction_id",
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    ]

    missing_context = [
        col
        for col in required_context
        if col not in context.columns
    ]

    if missing_context:
        raise ValueError(
            f"Missing context columns: {missing_context}"
        )

    if "transaction_id" not in outcomes.columns:
        raise ValueError(
            "Recovery outcomes missing transaction_id."
        )

    if "recovered_amount" not in outcomes.columns:
        raise ValueError(
            "Recovery outcomes missing recovered_amount."
        )

    # --------------------------------------------------------
    # Construct actual binary recovery label.
    #
    # recovered_amount > 0 means the payment was recovered.
    # --------------------------------------------------------

    outcomes = outcomes.copy()

    outcomes["actual_recovered"] = (
        pd.to_numeric(
            outcomes["recovered_amount"],
            errors="coerce",
        ).fillna(0)
        > 0
    ).astype(int)

    # --------------------------------------------------------
    # Keep only information required for modeling.
    # --------------------------------------------------------

    context_subset = context[
        required_context
    ].copy()

    outcome_subset = outcomes[
        [
            "transaction_id",
            "recovered_amount",
            "actual_recovered",
        ]
    ].copy()

    # --------------------------------------------------------
    # One outcome per transaction.
    # --------------------------------------------------------

    if context_subset["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs in recovery context."
        )

    if outcome_subset["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs in recovery outcomes."
        )

    # --------------------------------------------------------
    # Merge.
    # --------------------------------------------------------

    df = context_subset.merge(
        outcome_subset,
        on="transaction_id",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(context_subset):
        raise ValueError(
            "Some recovery-context transactions "
            "were lost during outcome merge."
        )

    print(
        f"[PASS] Modeling dataset: {len(df):,} rows"
    )

    print(
        f"Recovered payments: "
        f"{df['actual_recovered'].sum():,}"
    )

    recovery_rate = df["actual_recovered"].mean()

    print(
        f"Observed recovery rate: "
        f"{recovery_rate:.2%}"
    )

    return df


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(df):

    print("\n" + "=" * 70)
    print("FEATURE VALIDATION")
    print("=" * 70)

    forbidden_features = [
        "reliability",
        "financial_stability",
        "responsiveness",
        "retry_sensitivity",
        "reminder_sensitivity",
        "activity_level",
    ]

    leaked = [
        feature
        for feature in forbidden_features
        if feature in df.columns
    ]

    if leaked:
        raise ValueError(
            "HIDDEN FEATURE LEAKAGE DETECTED: "
            f"{leaked}"
        )

    print(
        "[PASS] Hidden simulator traits are not used"
    )

    for feature in NUMERIC_FEATURES:

        if feature not in df.columns:
            raise ValueError(
                f"Missing numeric feature: {feature}"
            )

    for feature in CATEGORICAL_FEATURES:

        if feature not in df.columns:
            raise ValueError(
                f"Missing categorical feature: {feature}"
            )

    print(
        "[PASS] Required observable features present"
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(df):

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    # Keep the split deterministic.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    indices = np.arange(len(df))

    rng.shuffle(indices)

    split_index = int(
        len(indices) * 0.80
    )

    train_indices = indices[:split_index]

    test_indices = indices[split_index:]

    train = df.iloc[
        train_indices
    ].copy()

    test = df.iloc[
        test_indices
    ].copy()

    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"Test rows:     {len(test):,}"
    )

    print(
        f"Train recovery rate: "
        f"{train['actual_recovered'].mean():.2%}"
    )

    print(
        f"Test recovery rate:  "
        f"{test['actual_recovered'].mean():.2%}"
    )

    return train, test


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(train):

    print("\n" + "=" * 70)
    print("TRAINING RECOVERY MODEL")
    print("=" * 70)

    X_train = train[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]

    y_train = train[
        "actual_recovered"
    ]

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    print(
        "[PASS] Recovery model trained"
    )

    return model


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def predict(model, test):

    print("\n" + "=" * 70)
    print("GENERATING RECOVERY PREDICTIONS")
    print("=" * 70)

    X_test = test[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    predictions = test.copy()

    predictions[
        "recovery_probability"
    ] = probabilities

    # --------------------------------------------------------
    # Confidence:
    #
    # Distance from uncertainty (0.5).
    #
    # 0.50 probability -> low confidence
    # 0.95 probability -> high confidence
    # --------------------------------------------------------

    predictions["confidence"] = (
        np.abs(
            probabilities - 0.5
        ) * 2
    )

    # --------------------------------------------------------
    # Expected recovery.
    # --------------------------------------------------------

    predictions[
        "expected_recovery"
    ] = (
        predictions["amount"]
        * predictions[
            "recovery_probability"
        ]
    )

    # --------------------------------------------------------
    # Sort by expected recovery.
    # --------------------------------------------------------

    predictions = predictions.sort_values(
        "expected_recovery",
        ascending=False,
    ).reset_index(
        drop=True
    )

    predictions[
        "recovery_rank"
    ] = (
        np.arange(
            len(predictions)
        ) + 1
    )

    return predictions


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_predictions(predictions):

    print("\n" + "=" * 70)
    print("RECOVERY MODEL VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Probability range.
    # --------------------------------------------------------

    if not predictions[
        "recovery_probability"
    ].between(0, 1).all():

        raise ValueError(
            "Recovery probabilities outside [0,1]."
        )

    print(
        "[PASS] Recovery probability range"
    )

    # --------------------------------------------------------
    # Confidence range.
    # --------------------------------------------------------

    if not predictions[
        "confidence"
    ].between(0, 1).all():

        raise ValueError(
            "Confidence outside [0,1]."
        )

    print(
        "[PASS] Confidence range"
    )

    # --------------------------------------------------------
    # Expected recovery.
    # --------------------------------------------------------

    if (
        predictions[
            "expected_recovery"
        ] < 0
    ).any():

        raise ValueError(
            "Negative expected recovery detected."
        )

    print(
        "[PASS] Expected recovery values"
    )

    # --------------------------------------------------------
    # Ranking.
    # --------------------------------------------------------

    if predictions[
        "recovery_rank"
    ].duplicated().any():

        raise ValueError(
            "Duplicate recovery ranks."
        )

    print(
        "[PASS] Recovery ranking"
    )

    # --------------------------------------------------------
    # Actual outcome exists only for evaluation.
    # --------------------------------------------------------

    if "actual_recovered" not in predictions.columns:
        raise ValueError(
            "Actual recovery labels missing."
        )

    print(
        "[PASS] Held-out actual outcomes available"
    )


# ============================================================
# REPORT
# ============================================================

def print_report(predictions):

    print("\n" + "=" * 70)
    print("RECLAIM RECOVERY MODEL REPORT")
    print("=" * 70)

    total = len(predictions)

    recovered = int(
        predictions[
            "actual_recovered"
        ].sum()
    )

    print(
        f"Test transactions:    {total:,}"
    )

    print(
        f"Actually recovered:   {recovered:,}"
    )

    print(
        f"Observed recovery:    "
        f"{recovered / total:.2%}"
    )

    print(
        "\nTop 10 predicted recovery opportunities:"
    )

    columns = [
        "transaction_id",
        "amount",
        "failure_reason",
        "recovery_probability",
        "confidence",
        "expected_recovery",
        "recovery_rank",
        "actual_recovered",
        "recovered_amount",
    ]

    print(
        predictions[
            columns
        ].head(10).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Top-K evaluation.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RECOVERY FRONTIER")
    print("=" * 70)

    for percentage in [
        5,
        10,
        20,
        30,
        40,
        50,
        75,
        100,
    ]:

        k = max(
            1,
            int(
                total
                * percentage
                / 100
            ),
        )

        top_k = predictions.head(k)

        recovered_value = (
            top_k[
                "recovered_amount"
            ].sum()
        )

        attempted_value = (
            top_k[
                "amount"
            ].sum()
        )

        actual_recovered = (
            top_k[
                "actual_recovered"
            ].sum()
        )

        recovery_rate = (
            actual_recovered / k
        )

        print(
            f"Top {percentage:>3}% | "
            f"Actions={k:4d} | "
            f"Recovered ₹={recovered_value:,.2f} | "
            f"Recovery rate={recovery_rate:.2%} | "
            f"₹/action="
            f"₹{recovered_value / k:,.2f}"
        )


# ============================================================
# SAVE
# ============================================================

def save_predictions(predictions):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save only useful fields plus evaluation labels.
    # --------------------------------------------------------

    output_columns = [
        "transaction_id",
        "amount",
        "failure_reason",
        *NUMERIC_FEATURES[
            1:
        ],
        "recovery_probability",
        "confidence",
        "expected_recovery",
        "recovery_rank",
        "actual_recovered",
        "recovered_amount",
    ]

    # Remove duplicates while preserving order.
    output_columns = list(
        dict.fromkeys(
            output_columns
        )
    )

    output_columns = [
        col
        for col in output_columns
        if col in predictions.columns
    ]

    predictions[
        output_columns
    ].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    context, outcomes = load_data()

    df = prepare_data(
        context,
        outcomes,
    )

    validate_features(df)

    train, test = split_data(df)

    model = train_model(train)

    predictions = predict(
        model,
        test,
    )

    validate_predictions(
        predictions
    )

    print_report(
        predictions
    )

    save_predictions(
        predictions
    )

    print("\n" + "=" * 70)
    print(
        "RECOVERY MODEL STATUS: READY"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()