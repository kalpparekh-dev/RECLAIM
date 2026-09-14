from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# MAKE SIMULATOR IMPORTABLE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from simulator.recovery_outcome_model import (
    calculate_actual_recovery_probability,
    ACTION_COST,
    ACTION_FRICTION,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "recovery_opportunities.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "baseline_comparison.csv"
)

RANDOM_SEED = 42


# ============================================================
# BASELINE CONFIGURATION
# ============================================================

# These baselines answer different questions:
#
# 1. RETRY_ALL
#    What happens if we simply retry every failed payment?
#
# 2. RANDOM_SELECTIVE
#    What happens if we intervene on the same number of
#    payments as RECLAIM, but choose them randomly?
#
# 3. RECLAIM
#    What happens when our intelligence layer chooses the
#    opportunities?
#
# This makes the comparison much more meaningful.
# ============================================================


# ============================================================
# DISPLAY
# ============================================================

def print_section(title: str) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(df: pd.DataFrame) -> None:

    required = {
        "transaction_id",
        "customer_id",
        "amount",
        "failure_reason",
        "recommended_action",
        "historical_success_rate",
        "failure_rate_last_7d",
        "failure_streak",
        "previous_transaction_success",
        "model_confidence",
    }

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:

        print(
            "[FAIL] Missing required columns:"
        )

        for column in sorted(missing):

            print(
                f"  - {column}"
            )

        raise ValueError(
            "Input validation failed."
        )

    print(
        "[PASS] Required comparison columns"
    )

    if df.empty:

        raise ValueError(
            "Input dataset is empty."
        )

    print(
        f"[PASS] Dataset contains "
        f"{len(df):,} failed payments"
    )


# ============================================================
# SIMULATE ONE STRATEGY
# ============================================================

def simulate_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    actions: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:

    result = df[
        [
            "transaction_id",
            "customer_id",
            "amount",
            "failure_reason",
        ]
    ].copy()

    result[
        "strategy"
    ] = strategy_name

    result[
        "action"
    ] = actions.values

    # --------------------------------------------------------
    # Calculate counterfactual recovery probability
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # We do NOT use the probability predicted by RECLAIM.
    #
    # We ask the same outcome model:
    #
    # "What would have happened if this action had been used?"
    #
    # This prevents the comparison from simply rewarding
    # RECLAIM because RECLAIM created the probability.
    # --------------------------------------------------------

    temp = df.copy()

    temp[
        "recommended_action"
    ] = actions.values

    probabilities = temp.apply(
        calculate_actual_recovery_probability,
        axis=1,
    )

    result[
        "recovery_probability"
    ] = probabilities.values

    # --------------------------------------------------------
    # Simulate actual response
    # --------------------------------------------------------

    random_values = rng.random(
        len(result)
    )

    result[
        "recovered"
    ] = (
        random_values
        <
        result[
            "recovery_probability"
        ]
    ).astype(int)

    # --------------------------------------------------------
    # No-action cannot recover
    # --------------------------------------------------------

    no_action = (
        result["action"]
        == "NO_ACTION"
    )

    result.loc[
        no_action,
        "recovered"
    ] = 0

    # --------------------------------------------------------
    # Recovered amount
    # --------------------------------------------------------

    result[
        "recovered_amount"
    ] = np.where(
        result["recovered"] == 1,
        result["amount"],
        0.0,
    )

    # --------------------------------------------------------
    # Intervention cost
    # --------------------------------------------------------

    result[
        "intervention_cost"
    ] = result[
        "action"
    ].map(
        ACTION_COST
    )

    # --------------------------------------------------------
    # Customer friction
    # --------------------------------------------------------

    result[
        "customer_friction_cost"
    ] = result[
        "action"
    ].map(
        ACTION_FRICTION
    )

    # --------------------------------------------------------
    # Net recovered value
    # --------------------------------------------------------

    result[
        "net_recovered_value"
    ] = (
        result[
            "recovered_amount"
        ]
        -
        result[
            "intervention_cost"
        ]
        -
        result[
            "customer_friction_cost"
        ]
    ).clip(
        lower=0
    )

    return result


# ============================================================
# BUILD RETRY-ALL BASELINE
# ============================================================

def build_retry_all(
    df: pd.DataFrame,
) -> pd.Series:

    return pd.Series(
        "RETRY_NOW",
        index=df.index,
        dtype="object",
    )


# ============================================================
# BUILD RANDOM SELECTIVE BASELINE
# ============================================================

def build_random_selective(
    df: pd.DataFrame,
    target_count: int,
    seed: int,
) -> pd.Series:

    actions = pd.Series(
        "NO_ACTION",
        index=df.index,
        dtype="object",
    )

    rng = np.random.default_rng(
        seed
    )

    if target_count <= 0:

        return actions

    target_count = min(
        target_count,
        len(df),
    )

    selected_indices = rng.choice(
        df.index.to_numpy(),
        size=target_count,
        replace=False,
    )

    actions.loc[
        selected_indices
    ] = "RETRY_NOW"

    return actions


# ============================================================
# BUILD RECLAIM STRATEGY
# ============================================================

def build_reclaim_strategy(
    df: pd.DataFrame,
) -> pd.Series:

    return (
        df[
            "recommended_action"
        ]
        .astype(str)
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    result: pd.DataFrame,
) -> dict:

    total_transactions = len(
        result
    )

    total_failed_value = (
        result["amount"]
        .sum()
    )

    intervention_mask = (
        result["action"]
        != "NO_ACTION"
    )

    intervention_count = (
        intervention_mask.sum()
    )

    recovered_count = (
        result["recovered"]
        .sum()
    )

    recovered_value = (
        result["recovered_amount"]
        .sum()
    )

    intervention_cost = (
        result["intervention_cost"]
        .sum()
    )

    friction_cost = (
        result["customer_friction_cost"]
        .sum()
    )

    net_recovered_value = (
        result["net_recovered_value"]
        .sum()
    )

    intervention_rate = (
        intervention_count
        /
        total_transactions
        *
        100
    )

    recovery_rate_value = (
        recovered_value
        /
        total_failed_value
        *
        100
    )

    recovery_rate_count = (
        recovered_count
        /
        total_transactions
        *
        100
    )

    recovery_per_intervention = (
        recovered_value
        /
        intervention_count
        if intervention_count > 0
        else 0
    )

    net_per_intervention = (
        net_recovered_value
        /
        intervention_count
        if intervention_count > 0
        else 0
    )

    return {
        "transactions":
            total_transactions,

        "interventions":
            int(intervention_count),

        "intervention_rate":
            intervention_rate,

        "recovered_payments":
            int(recovered_count),

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
# PRINT METRICS
# ============================================================

def print_strategy_metrics(
    strategy_name: str,
    metrics: dict,
) -> None:

    print(
        f"\n{strategy_name}"
    )

    print(
        "-" * 55
    )

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
        f"Recovery by count:      "
        f"{metrics['recovery_rate_count']:.2f}%"
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
# COMPARE STRATEGIES
# ============================================================

def compare_strategies(
    results: dict[str, pd.DataFrame],
) -> pd.DataFrame:

    rows = []

    for strategy_name, result in (
        results.items()
    ):

        metrics = calculate_metrics(
            result
        )

        row = {
            "strategy":
                strategy_name,

            **metrics,
        }

        rows.append(
            row
        )

    comparison = pd.DataFrame(
        rows
    )

    return comparison


# ============================================================
# RECLAIM LIFT ANALYSIS
# ============================================================

def calculate_lift(
    comparison: pd.DataFrame,
) -> None:

    print_section(
        "RECLAIM LIFT ANALYSIS"
    )

    reclaim = comparison.loc[
        comparison["strategy"]
        == "RECLAIM"
    ].iloc[0]

    retry_all = comparison.loc[
        comparison["strategy"]
        == "RETRY_ALL"
    ].iloc[0]

    random_selective = comparison.loc[
        comparison["strategy"]
        == "RANDOM_SELECTIVE"
    ].iloc[0]

    # --------------------------------------------------------
    # Against retry all
    # --------------------------------------------------------

    print(
        "RECLAIM vs RETRY ALL"
    )

    print(
        "-" * 55
    )

    net_difference = (
        reclaim[
            "net_recovered_value"
        ]
        -
        retry_all[
            "net_recovered_value"
        ]
    )

    intervention_reduction = (
        retry_all[
            "interventions"
        ]
        -
        reclaim[
            "interventions"
        ]
    )

    intervention_reduction_pct = (
        intervention_reduction
        /
        retry_all[
            "interventions"
        ]
        *
        100
    )

    print(
        f"Net recovery difference: "
        f"₹{net_difference:,.2f}"
    )

    print(
        f"Interventions avoided:    "
        f"{intervention_reduction:,}"
    )

    print(
        f"Intervention reduction:   "
        f"{intervention_reduction_pct:.2f}%"
    )

    # --------------------------------------------------------
    # Against random selective
    # --------------------------------------------------------

    print(
        "\nRECLAIM vs RANDOM SELECTIVE"
    )

    print(
        "-" * 55
    )

    random_net_difference = (
        reclaim[
            "net_recovered_value"
        ]
        -
        random_selective[
            "net_recovered_value"
        ]
    )

    random_recovery_difference = (
        reclaim[
            "recovered_value"
        ]
        -
        random_selective[
            "recovered_value"
        ]
    )

    print(
        f"Net recovery difference: "
        f"₹{random_net_difference:,.2f}"
    )

    print(
        f"Additional recovered value: "
        f"₹{random_recovery_difference:,.2f}"
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    print(
        "\nPRELIMINARY VERDICT"
    )

    if (
        reclaim[
            "net_recovered_value"
        ]
        >
        retry_all[
            "net_recovered_value"
        ]
    ):

        print(
            "[PASS] RECLAIM beats RETRY_ALL "
            "on net recovered value."
        )

    else:

        print(
            "[WARN] RECLAIM does NOT beat "
            "RETRY_ALL on net recovered value."
        )

    if (
        reclaim[
            "net_recovered_value"
        ]
        >
        random_selective[
            "net_recovered_value"
        ]
    ):

        print(
            "[PASS] RECLAIM beats RANDOM_SELECTIVE "
            "on net recovered value."
        )

    else:

        print(
            "[WARN] RECLAIM does NOT beat "
            "RANDOM_SELECTIVE."
        )


# ============================================================
# SAVE
# ============================================================

def save_comparison(
    comparison: pd.DataFrame,
) -> None:

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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("RECLAIM BASELINE COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Recovery opportunities not found:\n"
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Failed payments loaded: "
        f"{len(df):,}"
    )

    validate_input(
        df
    )

    # --------------------------------------------------------
    # Determine RECLAIM intervention count
    # --------------------------------------------------------

    reclaim_actions = (
        build_reclaim_strategy(
            df
        )
    )

    reclaim_intervention_count = (
        reclaim_actions
        .ne("NO_ACTION")
        .sum()
    )

    print(
        "\nRECLAIM intervention count: "
        f"{reclaim_intervention_count:,}"
    )

    # --------------------------------------------------------
    # Build strategies
    # --------------------------------------------------------

    retry_all_actions = (
        build_retry_all(
            df
        )
    )

    random_actions = (
        build_random_selective(
            df,
            target_count=int(
                reclaim_intervention_count
            ),
            seed=RANDOM_SEED,
        )
    )

    # --------------------------------------------------------
    # Simulate
    # --------------------------------------------------------
    #
    # Separate RNG streams are used so one strategy's random
    # outcome does not affect another strategy.
    # --------------------------------------------------------

    print_section(
        "SIMULATING RETRY ALL"
    )

    retry_all_result = (
        simulate_strategy(
            df,
            "RETRY_ALL",
            retry_all_actions,
            np.random.default_rng(
                RANDOM_SEED
            ),
        )
    )

    print(
        "[PASS] Retry-all simulation complete"
    )

    print_section(
        "SIMULATING RANDOM SELECTIVE"
    )

    random_result = (
        simulate_strategy(
            df,
            "RANDOM_SELECTIVE",
            random_actions,
            np.random.default_rng(
                RANDOM_SEED + 1
            ),
        )
    )

    print(
        "[PASS] Random-selective simulation complete"
    )

    print_section(
        "SIMULATING RECLAIM"
    )

    reclaim_result = (
        simulate_strategy(
            df,
            "RECLAIM",
            reclaim_actions,
            np.random.default_rng(
                RANDOM_SEED + 2
            ),
        )
    )

    print(
        "[PASS] RECLAIM simulation complete"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    results = {
        "RETRY_ALL":
            retry_all_result,

        "RANDOM_SELECTIVE":
            random_result,

        "RECLAIM":
            reclaim_result,
    }

    comparison = compare_strategies(
        results
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_section(
        "STRATEGY COMPARISON"
    )

    for strategy_name, result in (
        results.items()
    ):

        metrics = calculate_metrics(
            result
        )

        print_strategy_metrics(
            strategy_name,
            metrics,
        )

    # --------------------------------------------------------
    # Comparison table
    # --------------------------------------------------------

    print_section(
        "HEAD-TO-HEAD COMPARISON"
    )

    display_columns = [
        "strategy",
        "interventions",
        "intervention_rate",
        "recovered_payments",
        "recovered_value",
        "recovery_rate_value",
        "intervention_cost",
        "friction_cost",
        "net_recovered_value",
        "recovery_per_intervention",
        "net_per_intervention",
    ]

    print(
        comparison[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Lift
    # --------------------------------------------------------

    calculate_lift(
        comparison
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_comparison(
        comparison
    )

    print(
        "\nBASELINE COMPARISON STATUS: READY"
    )


if __name__ == "__main__":
    main()