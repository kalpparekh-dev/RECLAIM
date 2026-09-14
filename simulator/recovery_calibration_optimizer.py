import os
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "generated"
)

PREDICTIONS_FILE = os.path.join(
    DATA_DIR,
    "recovery_model_predictions.csv"
)

OUTCOMES_FILE = os.path.join(
    DATA_DIR,
    "recovery_outcomes.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "calibrated_recovery_predictions.csv"
)

POLICY_FILE = os.path.join(
    DATA_DIR,
    "calibrated_recovery_policy.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

INTERVENTION_COST = 1.50
FRICTION_COST = 0.25

MIN_PROBABILITY = 0.05
MAX_PROBABILITY = 0.95


# ============================================================
# DISPLAY
# ============================================================

def banner(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# VALIDATION
# ============================================================

def validate_probability(series, name):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    if values.isna().any():
        raise ValueError(
            f"{name} contains missing values"
        )

    if ((values < 0) | (values > 1)).any():
        raise ValueError(
            f"{name} must be between 0 and 1"
        )

    print(
        f"[PASS] {name} range"
    )


# ============================================================
# NORMALIZE RECOVERY OUTCOMES
# ============================================================

def normalize_outcomes(outcomes):

    print()
    print("Normalizing recovery outcome schema...")

    print(
        "Outcome columns detected:"
    )

    print(
        list(outcomes.columns)
    )

    # --------------------------------------------------------
    # recovery_flag
    # --------------------------------------------------------

    if "recovery_flag" not in outcomes.columns:

        if "recovered" in outcomes.columns:

            outcomes["recovery_flag"] = (
                pd.to_numeric(
                    outcomes["recovered"],
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )

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
                "[PASS] Derived `recovery_flag` "
                "from `recovery_status`"
            )

        else:

            raise ValueError(
                "Unable to derive recovery_flag"
            )

    # --------------------------------------------------------
    # recovered_amount
    # --------------------------------------------------------

    if "recovered_amount" not in outcomes.columns:

        if "amount" not in outcomes.columns:

            raise ValueError(
                "Cannot derive recovered_amount: "
                "`amount` is missing from recovery outcomes"
            )

        outcomes["recovered_amount"] = np.where(
            outcomes["recovery_flag"] == 1,
            pd.to_numeric(
                outcomes["amount"],
                errors="coerce"
            ),
            0.0
        )

        print(
            "[PASS] Derived `recovered_amount` "
            "from amount + recovery_flag"
        )

    else:

        outcomes["recovered_amount"] = pd.to_numeric(
            outcomes["recovered_amount"],
            errors="coerce"
        )

        missing = (
            outcomes["recovered_amount"]
            .isna()
            .sum()
        )

        if missing > 0:

            print(
                f"[WARN] {missing} missing recovered_amount values"
            )

            outcomes["recovered_amount"] = (
                outcomes["recovered_amount"]
                .fillna(
                    np.where(
                        outcomes["recovery_flag"] == 1,
                        pd.to_numeric(
                            outcomes["amount"],
                            errors="coerce"
                        ),
                        0.0
                    )
                )
            )

    # --------------------------------------------------------
    # Final normalization
    # --------------------------------------------------------

    outcomes["recovery_flag"] = (
        pd.to_numeric(
            outcomes["recovery_flag"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    outcomes["recovered_amount"] = (
        pd.to_numeric(
            outcomes["recovered_amount"],
            errors="coerce"
        )
        .fillna(0.0)
        .clip(lower=0)
    )

    print(
        "[PASS] Recovery outcome schema normalized"
    )

    return outcomes


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(PREDICTIONS_FILE):

        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS_FILE}"
        )

    if not os.path.exists(OUTCOMES_FILE):

        raise FileNotFoundError(
            f"Outcome file not found:\n{OUTCOMES_FILE}"
        )

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    outcomes = pd.read_csv(
        OUTCOMES_FILE
    )

    outcomes = normalize_outcomes(
        outcomes
    )

    print()
    print(
        f"Prediction rows: {len(predictions):,}"
    )

    print(
        f"Outcome rows:    {len(outcomes):,}"
    )

    # --------------------------------------------------------
    # Prediction validation
    # --------------------------------------------------------

    required_prediction_columns = [
        "transaction_id",
        "amount",
        "recovery_probability",
        "confidence",
    ]

    for column in required_prediction_columns:

        if column not in predictions.columns:

            raise ValueError(
                f"Missing prediction column: {column}"
            )

    print(
        "[PASS] Prediction dataset structure"
    )

    if predictions["transaction_id"].duplicated().any():

        raise ValueError(
            "Prediction transaction IDs are not unique"
        )

    if outcomes["transaction_id"].duplicated().any():

        raise ValueError(
            "Outcome transaction IDs are not unique"
        )

    print(
        "[PASS] Transaction IDs are unique"
    )

    # --------------------------------------------------------
    # Validate prediction probabilities
    # --------------------------------------------------------

    validate_probability(
        predictions["recovery_probability"],
        "Recovery probability"
    )

    validate_probability(
        predictions["confidence"],
        "Confidence"
    )

    # --------------------------------------------------------
    # Merge ONLY required outcome columns
    # --------------------------------------------------------

    outcome_subset = outcomes[
        [
            "transaction_id",
            "recovery_flag",
            "recovered_amount",
        ]
    ].copy()

    df = predictions.merge(
        outcome_subset,
        on="transaction_id",
        how="inner",
        validate="one_to_one"
    )

    if len(df) != len(predictions):

        raise ValueError(
            "Prediction population changed after merge"
        )

    print(
        "[PASS] Same transaction population"
    )

    print(
        f"[PASS] Modeling population: {len(df):,} rows"
    )

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    if "recovered_amount" not in df.columns:

        print(
            "[WARN] recovered_amount missing after merge"
        )

        df["recovered_amount"] = np.where(
            df["recovery_flag"] == 1,
            df["amount"],
            0.0
        )

        print(
            "[PASS] Rebuilt recovered_amount"
        )

    # --------------------------------------------------------
    # Final amount normalization
    # --------------------------------------------------------

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    df["recovered_amount"] = pd.to_numeric(
        df["recovered_amount"],
        errors="coerce"
    ).fillna(0.0)

    if df["amount"].isna().any():

        raise ValueError(
            "Amount contains invalid values"
        )

    if (df["amount"] <= 0).any():

        raise ValueError(
            "Transaction amounts must be positive"
        )

    df["recovered_amount"] = (
        df["recovered_amount"]
        .clip(lower=0)
    )

    print(
        "[PASS] Canonical amount column: `amount`"
    )

    print(
        "[PASS] Canonical recovered amount column: "
        "`recovered_amount`"
    )

    return df


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_predictions(df):

    banner(
        "RECOVERY PROBABILITY CALIBRATION"
    )

    y = (
        df["recovery_flag"]
        .astype(int)
        .values
    )

    raw = (
        df["recovery_probability"]
        .astype(float)
        .values
    )

    print(
        f"Observed recovery rate: "
        f"{y.mean() * 100:.2f}%"
    )

    raw_clipped = np.clip(
        raw,
        1e-6,
        1 - 1e-6
    )

    raw_brier = brier_score_loss(
        y,
        raw
    )

    raw_logloss = log_loss(
        y,
        raw_clipped
    )

    raw_auc = roc_auc_score(
        y,
        raw
    )

    print()
    print("Before calibration")
    print("------------------")

    print(
        f"Brier score : {raw_brier:.6f}"
    )

    print(
        f"Log loss    : {raw_logloss:.6f}"
    )

    print(
        f"ROC-AUC     : {raw_auc:.6f}"
    )

    # --------------------------------------------------------
    # Deterministic calibration split
    # --------------------------------------------------------

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    indices = np.arange(
        len(df)
    )

    rng.shuffle(
        indices
    )

    split = int(
        len(indices) * 0.70
    )

    train_idx = indices[:split]
    validation_idx = indices[split:]

    x_train = raw[train_idx]
    y_train = y[train_idx]

    x_validation = raw[validation_idx]
    y_validation = y[validation_idx]

    # --------------------------------------------------------
    # Logistic calibration
    # --------------------------------------------------------

    logistic = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000
    )

    logistic.fit(
        x_train.reshape(-1, 1),
        y_train
    )

    logistic_predictions = (
        logistic.predict_proba(
            x_validation.reshape(-1, 1)
        )[:, 1]
    )

    logistic_brier = brier_score_loss(
        y_validation,
        logistic_predictions
    )

    # --------------------------------------------------------
    # Isotonic calibration
    # --------------------------------------------------------

    isotonic = IsotonicRegression(
        y_min=MIN_PROBABILITY,
        y_max=MAX_PROBABILITY,
        out_of_bounds="clip"
    )

    isotonic.fit(
        x_train,
        y_train
    )

    isotonic_predictions = (
        isotonic.predict(
            x_validation
        )
    )

    isotonic_brier = brier_score_loss(
        y_validation,
        isotonic_predictions
    )

    print()
    print("Calibration comparison")
    print("----------------------")

    print(
        f"Raw Brier score       : {raw_brier:.6f}"
    )

    print(
        f"Logistic Brier score  : {logistic_brier:.6f}"
    )

    print(
        f"Isotonic Brier score  : {isotonic_brier:.6f}"
    )

    # --------------------------------------------------------
    # Select calibration model
    # --------------------------------------------------------

    if logistic_brier <= isotonic_brier:

        method = "LOGISTIC"

        final_calibrator = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        )

        final_calibrator.fit(
            raw.reshape(-1, 1),
            y
        )

        calibrated = (
            final_calibrator
            .predict_proba(
                raw.reshape(-1, 1)
            )[:, 1]
        )

    else:

        method = "ISOTONIC"

        final_calibrator = IsotonicRegression(
            y_min=MIN_PROBABILITY,
            y_max=MAX_PROBABILITY,
            out_of_bounds="clip"
        )

        final_calibrator.fit(
            raw,
            y
        )

        calibrated = (
            final_calibrator.predict(
                raw
            )
        )

    calibrated = np.clip(
        calibrated,
        MIN_PROBABILITY,
        MAX_PROBABILITY
    )

    df["calibrated_probability"] = (
        calibrated
    )

    print()
    print(
        f"[PASS] Selected calibration method: "
        f"{method}"
    )

    return df, method


# ============================================================
# POLICY EVALUATION
# ============================================================

def evaluate_threshold(
    df,
    threshold
):

    selected = df[
        df["calibrated_probability"]
        >= threshold
    ].copy()

    interventions = len(
        selected
    )

    if interventions == 0:

        return {
            "threshold": threshold,
            "interventions": 0,
            "intervention_rate": 0.0,
            "recovered_payments": 0,
            "recovered_value": 0.0,
            "recovery_rate": 0.0,
            "intervention_cost": 0.0,
            "friction_cost": 0.0,
            "net_recovered_value": 0.0,
            "net_per_intervention": 0.0,
        }

    recovered = selected[
        selected["recovery_flag"] == 1
    ]

    recovered_payments = len(
        recovered
    )

    recovered_value = (
        recovered["recovered_amount"]
        .sum()
    )

    intervention_cost = (
        interventions
        * INTERVENTION_COST
    )

    friction_cost = (
        interventions
        * FRICTION_COST
    )

    net_value = (
        recovered_value
        - intervention_cost
        - friction_cost
    )

    recovery_rate = (
        recovered_payments
        / interventions
    )

    return {
        "threshold": threshold,
        "interventions": interventions,
        "intervention_rate": (
            interventions / len(df)
        ),
        "recovered_payments": recovered_payments,
        "recovered_value": recovered_value,
        "recovery_rate": recovery_rate,
        "intervention_cost": intervention_cost,
        "friction_cost": friction_cost,
        "net_recovered_value": net_value,
        "net_per_intervention": (
            net_value / interventions
        ),
    }


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_threshold(df):

    banner(
        "OPTIMIZING CALIBRATED RECOVERY POLICY"
    )

    thresholds = np.round(
        np.arange(
            0.05,
            0.951,
            0.025
        ),
        3
    )

    results = []

    for threshold in thresholds:

        result = evaluate_threshold(
            df,
            threshold
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Print search
    # --------------------------------------------------------

    for _, row in results_df.iterrows():

        print(
            f"Threshold {row['threshold']:.3f} | "
            f"Actions={int(row['interventions']):4d} | "
            f"Recovered=₹{row['recovered_value']:,.2f} | "
            f"Net=₹{row['net_recovered_value']:,.2f} | "
            f"Recovery={row['recovery_rate'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Best total net recovery
    # --------------------------------------------------------

    best = results_df.sort_values(
        "net_recovered_value",
        ascending=False
    ).iloc[0]

    print()
    print("=" * 70)
    print(
        "OPTIMAL CALIBRATED POLICY"
    )
    print("=" * 70)

    print(
        f"Best threshold:       "
        f"{best['threshold']:.3f}"
    )

    print(
        f"Interventions:        "
        f"{int(best['interventions']):,}"
    )

    print(
        f"Intervention rate:    "
        f"{best['intervention_rate'] * 100:.2f}%"
    )

    print(
        f"Recovered value:      "
        f"₹{best['recovered_value']:,.2f}"
    )

    print(
        f"Net recovered value:  "
        f"₹{best['net_recovered_value']:,.2f}"
    )

    print(
        f"Net ₹ / intervention: "
        f"₹{best['net_per_intervention']:,.2f}"
    )

    return results_df, best


# ============================================================
# BUILD FINAL POLICY
# ============================================================

def build_final_policy(
    df,
    best
):

    threshold = float(
        best["threshold"]
    )

    df["recommended_policy"] = np.where(
        df["calibrated_probability"]
        >= threshold,
        "INTERVENE",
        "ABSTAIN"
    )

    df["expected_recovery"] = (
        df["amount"]
        * df["calibrated_probability"]
    )

    df["expected_intervention_cost"] = (
        np.where(
            df["recommended_policy"]
            == "INTERVENE",
            INTERVENTION_COST,
            0.0
        )
    )

    df["expected_friction_cost"] = (
        np.where(
            df["recommended_policy"]
            == "INTERVENE",
            FRICTION_COST,
            0.0
        )
    )

    df["expected_net_recovery"] = (
        np.where(
            df["recommended_policy"]
            == "INTERVENE",
            df["expected_recovery"]
            - INTERVENTION_COST
            - FRICTION_COST,
            0.0
        )
    )

    df["policy_threshold"] = (
        threshold
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "RECLAIM RECOVERY CALIBRATION OPTIMIZER"
    )

    df = load_data()

    df, calibration_method = (
        calibrate_predictions(df)
    )

    policy_results, best = (
        optimize_threshold(df)
    )

    final_df = build_final_policy(
        df,
        best
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    banner(
        "CALIBRATION VALIDATION"
    )

    validate_probability(
        final_df[
            "calibrated_probability"
        ],
        "Calibrated probability"
    )

    if final_df[
        "expected_recovery"
    ].isna().any():

        raise ValueError(
            "Expected recovery contains missing values"
        )

    if final_df[
        "expected_net_recovery"
    ].isna().any():

        raise ValueError(
            "Expected net recovery contains missing values"
        )

    if (
        final_df["recovered_amount"] < 0
    ).any():

        raise ValueError(
            "Recovered amount cannot be negative"
        )

    print(
        "[PASS] Calibrated probabilities valid"
    )

    print(
        "[PASS] Expected recovery values valid"
    )

    print(
        "[PASS] Expected net recovery values valid"
    )

    print(
        "[PASS] Recovered amounts valid"
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    banner(
        "RECLAIM CALIBRATED RECOVERY REPORT"
    )

    interventions = (
        final_df[
            "recommended_policy"
        ]
        == "INTERVENE"
    )

    intervention_count = (
        interventions.sum()
    )

    abstention_count = (
        len(final_df)
        - intervention_count
    )

    recovered = final_df[
        interventions
        & (
            final_df["recovery_flag"]
            == 1
        )
    ]

    recovered_value = (
        recovered[
            "recovered_amount"
        ].sum()
    )

    intervention_cost = (
        intervention_count
        * INTERVENTION_COST
    )

    friction_cost = (
        intervention_count
        * FRICTION_COST
    )

    net_value = (
        recovered_value
        - intervention_cost
        - friction_cost
    )

    print(
        f"Calibration method:   "
        f"{calibration_method}"
    )

    print(
        f"Transactions:         "
        f"{len(final_df):,}"
    )

    print(
        f"Interventions:        "
        f"{intervention_count:,}"
    )

    print(
        f"Abstentions:          "
        f"{abstention_count:,}"
    )

    print(
        f"Intervention rate:    "
        f"{intervention_count / len(final_df) * 100:.2f}%"
    )

    print(
        f"Recovered payments:   "
        f"{len(recovered):,}"
    )

    print(
        f"Recovered value:      "
        f"₹{recovered_value:,.2f}"
    )

    print(
        f"Intervention cost:    "
        f"₹{intervention_cost:,.2f}"
    )

    print(
        f"Friction cost:        "
        f"₹{friction_cost:,.2f}"
    )

    print(
        f"Net recovered value:  "
        f"₹{net_value:,.2f}"
    )

    # --------------------------------------------------------
    # Save calibrated predictions
    # --------------------------------------------------------

    final_columns = [
        "transaction_id",
        "amount",
        "failure_reason",
        "recovery_probability",
        "calibrated_probability",
        "confidence",
        "recovery_flag",
        "recovered_amount",
        "recommended_policy",
        "policy_threshold",
        "expected_recovery",
        "expected_intervention_cost",
        "expected_friction_cost",
        "expected_net_recovery",
    ]

    final_columns = [
        column
        for column in final_columns
        if column in final_df.columns
    ]

    final_df[
        final_columns
    ].to_csv(
        OUTPUT_FILE,
        index=False
    )

    policy_results.to_csv(
        POLICY_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print(
        "CALIBRATION OPTIMIZATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Saved calibrated predictions to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Saved threshold analysis to:"
    )

    print(
        POLICY_FILE
    )

    print()
    print(
        "RECOVERY CALIBRATION STATUS: READY"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()