from __future__ import annotations

from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# FILES
# ============================================================

FAILED_CONTEXT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "recovery_context.csv"
)

ECONOMIC_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "economic_opportunities.csv"
)

OUTCOME_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "recovery_outcomes.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "economic_baseline_comparison.csv"
)


# ============================================================
# ECONOMIC PARAMETERS
# ============================================================

ACTION_COST = {
    "RETRY_NOW": 1.50,
    "DELAYED_RETRY": 1.00,
}

FRICTION_COST = {
    "RETRY_NOW": 0.25,
    "DELAYED_RETRY": 0.50,
}


# ============================================================
# DISPLAY
# ============================================================

def section(title: str) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("RECLAIM ECONOMIC BASELINE COMPARISON")
    print("=" * 70)

    if not ECONOMIC_PATH.exists():
        raise FileNotFoundError(
            f"Economic opportunities not found:\n"
            f"{ECONOMIC_PATH}"
        )

    if not OUTCOME_PATH.exists():
        raise FileNotFoundError(
            f"Recovery outcomes not found:\n"
            f"{OUTCOME_PATH}"
        )

    economic = pd.read_csv(
        ECONOMIC_PATH
    )

    outcomes = pd.read_csv(
        OUTCOME_PATH
    )

    print(
        f"Economic opportunities: "
        f"{len(economic):,}"
    )

    print(
        f"Recovery outcomes:      "
        f"{len(outcomes):,}"
    )

    return economic, outcomes


# ============================================================
# VALIDATION
# ============================================================

def validate_data(
    economic: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:

    section(
        "INPUT VALIDATION"
    )

    economic_required = {
        "transaction_id",
        "amount",
        "failure_reason",
        "recommended_action",
        "retry_now_probability",
        "delayed_retry_probability",
    }

    outcome_required = {
    "transaction_id",
    "amount",
    "recovered_amount",
}

    missing_economic = (
        economic_required
        -
        set(economic.columns)
    )

    missing_outcome = (
        outcome_required
        -
        set(outcomes.columns)
    )

    if missing_economic:

        raise ValueError(
            "Missing economic columns: "
            f"{sorted(missing_economic)}"
        )

    if missing_outcome:

        raise ValueError(
            "Missing outcome columns: "
            f"{sorted(missing_outcome)}"
        )

    print(
        "[PASS] Economic dataset structure"
    )

    print(
        "[PASS] Recovery outcome structure"
    )


    economic_ids = set(
        economic[
            "transaction_id"
        ]
    )

    outcome_ids = set(
        outcomes[
            "transaction_id"
        ]
    )

    if economic_ids != outcome_ids:

        missing_from_outcomes = (
            economic_ids
            -
            outcome_ids
        )

        missing_from_economic = (
            outcome_ids
            -
            economic_ids
        )

        raise ValueError(
            "Transaction populations do not match.\n"
            f"Missing outcomes: "
            f"{len(missing_from_outcomes)}\n"
            f"Missing economic records: "
            f"{len(missing_from_economic)}"
        )

    print(
        "[PASS] Same transaction population"
    )


    if economic[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate economic transaction IDs."
        )

    if outcomes[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate outcome transaction IDs."
        )

    print(
        "[PASS] Transaction IDs are unique"
    )


# ============================================================
# MERGE
# ============================================================
def prepare_dataset(
    economic: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:

    print("\n======================================================================")
    print("PREPARING ECONOMIC BASELINE DATASET")
    print("======================================================================")

    # ------------------------------------------------------------
    # Keep only outcome fields that are actually needed.
    # IMPORTANT:
    # Do NOT bring `amount` from outcomes because economic already
    # contains the canonical transaction amount.
    # ------------------------------------------------------------

    outcome_columns = [
        "transaction_id",
        "recovered_amount",
    ]

    if "recovery_status" in outcomes.columns:
        outcome_columns.append("recovery_status")

    if "actual_recovery_probability" in outcomes.columns:
        outcome_columns.append("actual_recovery_probability")

    outcome_subset = outcomes[outcome_columns].copy()

    # ------------------------------------------------------------
    # Validate uniqueness before merging
    # ------------------------------------------------------------

    if outcome_subset["transaction_id"].duplicated().any():
        raise ValueError(
            "Recovery outcomes contain duplicate transaction IDs."
        )

    if economic["transaction_id"].duplicated().any():
        raise ValueError(
            "Economic opportunities contain duplicate transaction IDs."
        )

    # ------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------

    merged = economic.merge(
        outcome_subset,
        on="transaction_id",
        how="inner",
        validate="one_to_one",
    )

    # ------------------------------------------------------------
    # Validate merge
    # ------------------------------------------------------------

    if len(merged) != len(economic):
        raise ValueError(
            f"Merge lost transactions. "
            f"Economic={len(economic)}, merged={len(merged)}"
        )

    # ------------------------------------------------------------
    # Ensure canonical amount exists
    # ------------------------------------------------------------

    if "amount" not in merged.columns:
        raise ValueError(
            "Economic dataset must contain canonical `amount` column."
        )

    # ------------------------------------------------------------
    # Final schema validation
    # ------------------------------------------------------------

    required_final_columns = [
        "transaction_id",
        "amount",
        "recommended_action",
        "expected_recovery",
        "expected_net_recovery",
        "recovered_amount",
    ]

    missing = [
        col
        for col in required_final_columns
        if col not in merged.columns
    ]

    if missing:
        raise ValueError(
            f"Prepared dataset missing columns: {missing}"
        )

    # ------------------------------------------------------------
    # Numeric cleanup
    # ------------------------------------------------------------

    merged["amount"] = pd.to_numeric(
        merged["amount"],
        errors="raise",
    )

    merged["recovered_amount"] = pd.to_numeric(
        merged["recovered_amount"],
        errors="raise",
    )

    # ------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------

    print(
        f"[PASS] Merged dataset: {len(merged):,} transactions"
    )

    print(
        f"[PASS] Canonical amount column: `amount`"
    )

    print(
        f"[PASS] Recovered amount column: `recovered_amount`"
    )

    print(
        f"[PASS] Final columns: {len(merged.columns)}"
    )

    return merged


# ============================================================
# DETERMINISTIC RANDOM NUMBER
# ============================================================

def stable_random(
    transaction_id: str,
    strategy: str,
) -> float:

    """
    Deterministic pseudo-random value.

    This ensures every strategy uses the same random draw
    for a given transaction, preventing random-number noise
    from deciding the winner.
    """

    key = (
        f"{transaction_id}|{strategy}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        key
    ).hexdigest()

    integer = int(
        digest[:16],
        16,
    )

    return (
        integer
        /
        float(16**16 - 1)
    )


# ============================================================
# SIMULATE STRATEGY
# ============================================================

def simulate_strategy(
    df: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:

    result = df.copy()

    actions = []
    probabilities = []

    for _, row in result.iterrows():

        if strategy == "RETRY_ALL":

            action = "RETRY_NOW"

            probability = float(
                row[
                    "retry_now_probability"
                ]
            )

        elif strategy == "RANDOM_SELECTIVE":

            # Same intervention rate as economic RECLAIM.
            #
            # This gives us a selective baseline rather than
            # comparing against a random 50/50 strategy.

            threshold = (
                df[
                    "recommended_action"
                ]
                !=
                "NO_ACTION"
            ).mean()

            random_value = stable_random(
                row["transaction_id"],
                strategy,
            )

            if random_value < threshold:

                action = "RETRY_NOW"

                probability = float(
                    row[
                        "retry_now_probability"
                    ]
                )

            else:

                action = "NO_ACTION"

                probability = 0.0

        elif strategy == "ECONOMIC_RECLAIM":

            action = str(
                row[
                    "recommended_action"
                ]
            )

            if action == "RETRY_NOW":

                probability = float(
                    row[
                        "retry_now_probability"
                    ]
                )

            elif action == "DELAYED_RETRY":

                probability = float(
                    row[
                        "delayed_retry_probability"
                    ]
                )

            else:

                probability = 0.0

        else:

            raise ValueError(
                f"Unknown strategy: {strategy}"
            )

        actions.append(
            action
        )

        probabilities.append(
            probability
        )

    result[
        "simulated_action"
    ] = actions

    result[
        "simulated_probability"
    ] = probabilities

    # --------------------------------------------------------
    # Counterfactual recovery simulation
    # --------------------------------------------------------
    #
    # We use a deterministic draw per transaction.
    #
    # This is a simulation benchmark, not a claim that these
    # are live Razorpay outcomes.
    # --------------------------------------------------------

    simulated_recovered = []
    simulated_recovered_amount = []

    for _, row in result.iterrows():

        action = row[
            "simulated_action"
        ]

        probability = row[
            "simulated_probability"
        ]

        if action == "NO_ACTION":

            recovered = 0
            recovered_amount = 0.0

        else:

            random_value = stable_random(
                row["transaction_id"],
                strategy,
            )

            recovered = (
                1
                if random_value
                <
                probability
                else 0
            )

            recovered_amount = (
                float(row["amount"])
                if recovered
                else 0.0
            )

        simulated_recovered.append(
            recovered
        )

        simulated_recovered_amount.append(
            recovered_amount
        )

    result[
        "simulated_recovered"
    ] = simulated_recovered

    result[
        "simulated_recovered_amount"
    ] = simulated_recovered_amount

    return result


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    result: pd.DataFrame,
    strategy: str,
) -> dict:

    total = len(
        result
    )

    failed_value = float(
        result[
            "amount"
        ].sum()
    )

    intervention_mask = (
        result[
            "simulated_action"
        ]
        !=
        "NO_ACTION"
    )

    interventions = int(
        intervention_mask.sum()
    )

    intervention_rate = (
        interventions
        /
        total
        *
        100
    )


    recovered_payments = int(
        result[
            "simulated_recovered"
        ].sum()
    )

    recovered_value = float(
        result[
            "simulated_recovered_amount"
        ].sum()
    )


    recovery_rate_value = (
        recovered_value
        /
        failed_value
        *
        100
        if failed_value > 0
        else 0
    )

    recovery_rate_count = (
        recovered_payments
        /
        interventions
        *
        100
        if interventions > 0
        else 0
    )


    intervention_cost = 0.0

    friction_cost = 0.0

    for action, count in (
        result[
            "simulated_action"
        ]
        .value_counts()
        .items()
    ):

        if action in ACTION_COST:

            intervention_cost += (
                count
                *
                ACTION_COST[action]
            )

            friction_cost += (
                count
                *
                FRICTION_COST[action]
            )


    net_recovered_value = (
        recovered_value
        -
        intervention_cost
        -
        friction_cost
    )


    recovery_per_intervention = (
        recovered_value
        /
        interventions
        if interventions > 0
        else 0
    )

    net_per_intervention = (
        net_recovered_value
        /
        interventions
        if interventions > 0
        else 0
    )


    return {
        "strategy":
            strategy,

        "transactions":
            total,

        "interventions":
            interventions,

        "intervention_rate":
            intervention_rate,

        "recovered_payments":
            recovered_payments,

        "recovered_value":
            recovered_value,

        "recovery_rate_value":
            recovery_rate_value,

        "recovery_rate_count":
            recovery_rate_count,

        "intervention_cost":
            intervention_cost,

        "friction_cost":
            friction_cost,

        "net_recovered_value":
            net_recovered_value,

        "recovery_per_intervention":
            recovery_per_intervention,

        "net_per_intervention":
            net_per_intervention,
    }


# ============================================================
# PRINT STRATEGY
# ============================================================

def print_strategy(
    metrics: dict,
) -> None:

    print(
        f"\n{metrics['strategy']}"
    )

    print("-" * 55)

    print(
        f"Interventions:          "
        f"{metrics['interventions']:,}"
    )

    print(
        f"Intervention rate:      "
        f"{metrics['intervention_rate']:.2f}%"
    )

    print(
        f"Recovered payments:     "
        f"{metrics['recovered_payments']:,}"
    )

    print(
        f"Recovered value:        "
        f"₹{metrics['recovered_value']:,.2f}"
    )

    print(
        f"Recovery by value:      "
        f"{metrics['recovery_rate_value']:.2f}%"
    )

    print(
        f"Recovery by transaction:"
        f" {metrics['recovery_rate_count']:.2f}%"
    )

    print(
        f"Intervention cost:      "
        f"₹{metrics['intervention_cost']:,.2f}"
    )

    print(
        f"Friction cost:          "
        f"₹{metrics['friction_cost']:,.2f}"
    )

    print(
        f"Net recovered value:    "
        f"₹{metrics['net_recovered_value']:,.2f}"
    )

    print(
        f"₹ recovered / action:   "
        f"₹{metrics['recovery_per_intervention']:,.2f}"
    )

    print(
        f"Net ₹ / action:         "
        f"₹{metrics['net_per_intervention']:,.2f}"
    )


# ============================================================
# LIFT ANALYSIS
# ============================================================

def calculate_lift(
    baseline: dict,
    reclaim: dict,
) -> dict:

    net_difference = (
        reclaim[
            "net_recovered_value"
        ]
        -
        baseline[
            "net_recovered_value"
        ]
    )

    recovery_difference = (
        reclaim[
            "recovered_value"
        ]
        -
        baseline[
            "recovered_value"
        ]
    )

    intervention_difference = (
        baseline[
            "interventions"
        ]
        -
        reclaim[
            "interventions"
        ]
    )

    intervention_reduction = (
        intervention_difference
        /
        baseline[
            "interventions"
        ]
        *
        100
        if baseline[
            "interventions"
        ] > 0
        else 0
    )

    return {
        "net_recovery_difference":
            net_difference,

        "recovered_value_difference":
            recovery_difference,

        "interventions_avoided":
            intervention_difference,

        "intervention_reduction":
            intervention_reduction,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    economic, outcomes = (
        load_data()
    )

    validate_data(
        economic,
        outcomes,
    )

    df = prepare_dataset(
        economic,
        outcomes,
    )


    # ========================================================
    # STRATEGY 1 — RETRY ALL
    # ========================================================

    section(
        "SIMULATING RETRY ALL"
    )

    retry_all = simulate_strategy(
        df,
        "RETRY_ALL",
    )

    retry_all_metrics = (
        calculate_metrics(
            retry_all,
            "RETRY_ALL",
        )
    )

    print(
        "[PASS] Retry-all simulation complete"
    )


    # ========================================================
    # STRATEGY 2 — RANDOM SELECTIVE
    # ========================================================

    section(
        "SIMULATING RANDOM SELECTIVE"
    )

    random_selective = (
        simulate_strategy(
            df,
            "RANDOM_SELECTIVE",
        )
    )

    random_metrics = (
        calculate_metrics(
            random_selective,
            "RANDOM_SELECTIVE",
        )
    )

    print(
        "[PASS] Random-selective simulation complete"
    )


    # ========================================================
    # STRATEGY 3 — ECONOMIC RECLAIM
    # ========================================================

    section(
        "SIMULATING ECONOMIC RECLAIM"
    )

    economic_reclaim = (
        simulate_strategy(
            df,
            "ECONOMIC_RECLAIM",
        )
    )

    economic_metrics = (
        calculate_metrics(
            economic_reclaim,
            "ECONOMIC_RECLAIM",
        )
    )

    print(
        "[PASS] Economic RECLAIM simulation complete"
    )


    # ========================================================
    # COMPARISON
    # ========================================================

    section(
        "STRATEGY COMPARISON"
    )

    metrics = [
        retry_all_metrics,
        random_metrics,
        economic_metrics,
    ]

    for item in metrics:

        print_strategy(
            item
        )


    comparison = pd.DataFrame(
        metrics
    )


    # ========================================================
    # HEAD TO HEAD
    # ========================================================

    section(
        "HEAD-TO-HEAD COMPARISON"
    )

    print(
        comparison.to_string(
            index=False
        )
    )


    # ========================================================
    # LIFT
    # ========================================================

    section(
        "ECONOMIC RECLAIM LIFT ANALYSIS"
    )

    lift_vs_retry = (
        calculate_lift(
            retry_all_metrics,
            economic_metrics,
        )
    )

    lift_vs_random = (
        calculate_lift(
            random_metrics,
            economic_metrics,
        )
    )


    print(
        "ECONOMIC RECLAIM vs RETRY ALL"
    )

    print("-" * 55)

    print(
        "Net recovery difference: "
        f"₹{lift_vs_retry['net_recovery_difference']:,.2f}"
    )

    print(
        "Recovered value difference: "
        f"₹{lift_vs_retry['recovered_value_difference']:,.2f}"
    )

    print(
        "Interventions avoided: "
        f"{lift_vs_retry['interventions_avoided']:,}"
    )

    print(
        "Intervention reduction: "
        f"{lift_vs_retry['intervention_reduction']:.2f}%"
    )


    print(
        "\nECONOMIC RECLAIM vs RANDOM SELECTIVE"
    )

    print("-" * 55)

    print(
        "Net recovery difference: "
        f"₹{lift_vs_random['net_recovery_difference']:,.2f}"
    )

    print(
        "Recovered value difference: "
        f"₹{lift_vs_random['recovered_value_difference']:,.2f}"
    )

    print(
        "Interventions avoided: "
        f"{lift_vs_random['interventions_avoided']:,}"
    )

    print(
        "Intervention reduction: "
        f"{lift_vs_random['intervention_reduction']:.2f}%"
    )


    # ========================================================
    # VERDICT
    # ========================================================

    section(
        "ECONOMIC VERDICT"
    )

    reclaim_net = (
        economic_metrics[
            "net_recovered_value"
        ]
    )

    retry_net = (
        retry_all_metrics[
            "net_recovered_value"
        ]
    )

    random_net = (
        random_metrics[
            "net_recovered_value"
        ]
    )


    if reclaim_net > retry_net:

        print(
            "[PASS] ECONOMIC RECLAIM beats RETRY_ALL "
            "on net recovered value."
        )

    else:

        print(
            "[WARN] ECONOMIC RECLAIM does NOT beat "
            "RETRY_ALL on net recovered value."
        )


    if reclaim_net > random_net:

        print(
            "[PASS] ECONOMIC RECLAIM beats "
            "RANDOM_SELECTIVE."
        )

    else:

        print(
            "[WARN] ECONOMIC RECLAIM does NOT beat "
            "RANDOM_SELECTIVE."
        )


    if (
        economic_metrics[
            "intervention_rate"
        ]
        <
        retry_all_metrics[
            "intervention_rate"
        ]
    ):

        print(
            "[PASS] Economic RECLAIM reduces "
            "intervention volume."
        )

    else:

        print(
            "[WARN] Economic RECLAIM does not "
            "reduce intervention volume."
        )


    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved comparison to:"
        f"\n{OUTPUT_PATH}"
    )

    print(
        "\nECONOMIC BASELINE COMPARISON STATUS: READY"
    )


if __name__ == "__main__":
    main()