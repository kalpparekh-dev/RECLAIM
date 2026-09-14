from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "generated" / "reclaim_production_decisions_v5.csv"
OUT_DECISIONS = ROOT / "data" / "generated" / "reclaim_production_decisions_v6.csv"
OUT_RECON = ROOT / "data" / "generated" / "reclaim_production_financial_reconciliation_v6.csv"
OUT_MONITOR = ROOT / "data" / "generated" / "reclaim_production_monitoring_v6.csv"

TARGET_RATE = 0.05
ECONOMIC_BASE_VALUE = 1500.0
COSTS = {"HIGH_INTENSITY":60.0, "STANDARD_TARGET":50.0,
         "LOW_INTENSITY":45.0, "MONITOR":40.0, "RETRY_ALL":25.0}

def fail(msg):
    raise ValueError("[FAIL] " + msg)

def require_columns(df):
    required = [
        "transaction_id","predicted_probability_uplift","production_target_rate",
        "production_intervention_selected","production_decision",
        "expected_incremental_recovery_value","economic_opportunity_score",
        "recommended_action","estimated_action_cost","estimated_net_action_value"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing: fail(f"Missing required V5 columns: {missing}")

def money(x):
    return f"₹{float(x):,.2f}"

def main():
    print("="*70)
    print("RECLAIM PRODUCTION PIPELINE V6")
    print("Financial Reconciliation + Production Monitoring")
    print("="*70)
    print(f"Input: {INPUT}")
    print("Validated policy: TOP_RATE")
    print("Validated target rate: 5.00%")
    print(f"Economic base value: {money(ECONOMIC_BASE_VALUE)}")

    if not INPUT.exists(): fail(f"Input file not found: {INPUT}")
    df = pd.read_csv(INPUT)
    print(f"[PASS] Rows loaded: {len(df)}")
    print(f"[PASS] Columns loaded: {len(df.columns)}")
    require_columns(df)
    print("[PASS] V5 decision schema validated")

    numeric = ["predicted_probability_uplift","production_target_rate",
               "production_intervention_selected","expected_incremental_recovery_value",
               "economic_opportunity_score","estimated_action_cost",
               "estimated_net_action_value"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any(): fail(f"Non-numeric/NaN values found in {c}")
    print("[PASS] Numeric fields validated")

    print("\n"+"="*70)
    print("RUNNING V6 RECONCILIATION AUDITS")
    print("="*70)

    target = df["production_intervention_selected"].astype(int).eq(1)
    rate = float(df["production_target_rate"].iloc[0])
    if not np.isclose(rate, TARGET_RATE, atol=1e-12):
        fail(f"Validated target rate changed: {rate}")
    if int(target.sum()) != round(len(df)*TARGET_RATE):
        fail(f"Target population mismatch: {int(target.sum())}")
    if not (df.loc[target,"production_decision"]=="TARGET").all():
        fail("Intervention-selected rows are not all TARGET")
    if not (df.loc[~target,"production_decision"]=="RETRY_ALL").all():
        fail("Non-selected rows are not all RETRY_ALL")
    print("[PASS] Validated 5% targeting population preserved")

    positive = df["predicted_probability_uplift"].clip(lower=0)
    value_error = (df["expected_incremental_recovery_value"] -
                   positive*ECONOMIC_BASE_VALUE).abs().max()
    score_error = (df["economic_opportunity_score"] -
                   positive*100).abs().max()
    if value_error > 1e-6:
        fail(f"V3 expected value mismatch. Max error={value_error:.10f}")
    if score_error > 1e-6:
        fail(f"V3 opportunity score mismatch. Max error={score_error:.10f}")
    print("[PASS] V3 economic definitions reconciled")

    for action, cost in COSTS.items():
        mask = df["recommended_action"].eq(action)
        if mask.any() and not np.allclose(df.loc[mask,"estimated_action_cost"], cost):
            fail(f"Action-cost proxy mismatch for {action}")
    net_error = (df["estimated_net_action_value"] -
                 (df["expected_incremental_recovery_value"] -
                  df["estimated_action_cost"])).abs().max()
    if net_error > 1e-6:
        fail(f"V4 net-value arithmetic mismatch. Max error={net_error:.10f}")
    print("[PASS] V4 action-cost and net-value arithmetic reconciled")

    oracle = sorted({"oracle_control_value","oracle_treatment_value",
                     "oracle_incremental_value"}.intersection(df.columns))
    print("[PASS] Oracle/evaluation columns audit-only: " +
          (", ".join(oracle) if oracle else "NONE"))

    # Reconciliation is explicitly split by scope.
    retry = ~target
    rows = []
    for scope, mask, interpretation in [
        ("TARGET", target, "Validated intervention population; V3/V4 economic scope."),
        ("RETRY_ALL", retry, "Outside validated target capacity; planning proxy only.")
    ]:
        ev = float(df.loc[mask,"expected_incremental_recovery_value"].sum())
        cost = float(df.loc[mask,"estimated_action_cost"].sum())
        net = float(df.loc[mask,"estimated_net_action_value"].sum())
        rows.append([scope, int(mask.sum()), ev, cost, net, interpretation])

    portfolio = pd.DataFrame(rows, columns=[
        "scope","transactions","expected_incremental_value",
        "estimated_action_cost","estimated_net_action_value","interpretation"
    ])
    total = pd.DataFrame([[
        "PORTFOLIO", len(df),
        portfolio["expected_incremental_value"].sum(),
        portfolio["estimated_action_cost"].sum(),
        portfolio["estimated_net_action_value"].sum(),
        "Planning aggregate only; not a causal estimate."
    ]], columns=portfolio.columns)
    recon = pd.concat([portfolio,total], ignore_index=True)

    out = df.copy()
    out["v6_scope"] = np.where(target,"TARGET","RETRY_ALL")
    out["v6_target_expected_value"] = np.where(
        target,out["expected_incremental_recovery_value"],0.0)
    out["v6_target_action_cost"] = np.where(
        target,out["estimated_action_cost"],0.0)
    out["v6_target_net_action_value"] = (
        out["v6_target_expected_value"] - out["v6_target_action_cost"])
    out["v6_production_decision_preserved"] = True

    monitoring = pd.DataFrame([
        ["target_rate",TARGET_RATE,float(target.mean()),"PASS"],
        ["target_count",round(len(df)*TARGET_RATE),int(target.sum()),"PASS"],
        ["mean_predicted_uplift",np.nan,float(df["predicted_probability_uplift"].mean()),"INFO"],
        ["target_mean_predicted_uplift",np.nan,float(df.loc[target,"predicted_probability_uplift"].mean()),"INFO"],
        ["target_expected_value",np.nan,float(df.loc[target,"expected_incremental_recovery_value"].sum()),"INFO"],
        ["target_action_cost",np.nan,float(df.loc[target,"estimated_action_cost"].sum()),"INFO"],
        ["target_net_action_value",np.nan,float(df.loc[target,"estimated_net_action_value"].sum()),"INFO"],
    ], columns=["metric","expected","observed","status"])

    OUT_DECISIONS.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUT_DECISIONS,index=False)
    recon.to_csv(OUT_RECON,index=False)
    monitoring.to_csv(OUT_MONITOR,index=False)

    t = recon[recon.scope=="TARGET"].iloc[0]
    r = recon[recon.scope=="RETRY_ALL"].iloc[0]
    p = recon[recon.scope=="PORTFOLIO"].iloc[0]

    print("\n"+"="*70)
    print("V6 FINANCIAL RECONCILIATION SUMMARY")
    print("="*70)
    print(f"Total transactions: {len(df)}")
    print(f"Validated TARGET transactions: {int(target.sum())}")
    print(f"RETRY_ALL transactions: {int(retry.sum())}")
    print(f"Actual intervention rate: {target.mean():.2%}")
    print("\nTARGET — validated intervention scope")
    print(f"  Expected incremental value: {money(t.expected_incremental_value)}")
    print(f"  Estimated action cost:      {money(t.estimated_action_cost)}")
    print(f"  Estimated net value:        {money(t.estimated_net_action_value)}")
    print("\nRETRY_ALL — planning scope")
    print(f"  Expected incremental value: {money(r.expected_incremental_value)}")
    print(f"  Estimated action cost:      {money(r.estimated_action_cost)}")
    print(f"  Estimated net value:        {money(r.estimated_net_action_value)}")
    print("\nPORTFOLIO — planning aggregate only")
    print(f"  Expected incremental value: {money(p.expected_incremental_value)}")
    print(f"  Estimated action cost:      {money(p.estimated_action_cost)}")
    print(f"  Estimated net value:        {money(p.estimated_net_action_value)}")
    print("\nGOVERNANCE")
    print("Target population changed by V6: NO")
    print("Oracle/evaluation fields used for decisions: NO")
    print("Action costs: transparent V4 planning proxies")
    print("Portfolio aggregate is causal estimate: NO")
    print("\n"+"="*70)
    print("FINAL V6 VERDICT")
    print("="*70)
    print("FINAL VERDICT: V6_FINANCIAL_RECONCILIATION_PASSED")
    print("="*70)
    print(f"[PASS] Decisions: {OUT_DECISIONS}")
    print(f"[PASS] Reconciliation: {OUT_RECON}")
    print(f"[PASS] Monitoring: {OUT_MONITOR}")

if __name__ == "__main__":
    main()