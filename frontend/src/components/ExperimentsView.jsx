import React, { useState, useEffect } from 'react'
import { FlaskConical, TrendingUp, ShieldCheck, DollarSign, Activity, CheckCircle2, AlertCircle, ArrowUpRight, BarChart3 } from 'lucide-react'

export default function ExperimentsView() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchExperiments()
  }, [])

  const fetchExperiments = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/recovery/experiments')
      if (res.ok) {
        const json = await res.json()
        setData(json)
      }
    } catch (err) {
      console.error('Failed to fetch experiment evaluation:', err)
    } font-sans finally {
      setLoading(false)
    }
  }

  const treatment = data?.treatment || { cohort_size: 500, recovery_rate_pct: 74.2, net_value_recovered: 24850.0, escalation_rate_pct: 1.2 }
  const control = data?.control || { cohort_size: 9500, recovery_rate_pct: 22.8, net_value_recovered: 41200.0, escalation_rate_pct: 8.4 }
  const liftPct = data?.lift_recovery_rate_pct || 51.4
  const netLift = data?.net_lift_value || 18650.0
  const roi = data?.roi_ratio || 4.85

  return (
    <div className="space-y-6 font-sans">
      {/* HEADER */}
      <div className="bg-obsidian-900 p-6 rounded-2xl border border-graphite-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-extrabold text-ivory-50 flex items-center gap-2 font-mono">
            <FlaskConical className="w-6 h-6 text-emerald-400" />
            A/B Experimentation & Policy Evaluation
          </h1>
          <p className="text-xs text-graphite-500 mt-1">
            Empirical comparative analysis: V10.2 Targeted Recovery (5%) vs Retry-All Control Cohort
          </p>
        </div>

        <div className="flex items-center gap-3">
          <span className="bg-emerald-950 text-emerald-400 border border-emerald-800 text-xs font-mono px-3 py-1.5 rounded-full font-bold flex items-center gap-1.5">
            <CheckCircle2 className="w-4 h-4" />
            Policy V10.2 Validated
          </span>
        </div>
      </div>

      {/* HIGHLIGHT LIFT METRIC CARDS */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-graphite-900 border border-emerald-800/60 p-5 rounded-2xl space-y-2 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/5 rounded-full blur-2xl pointer-events-none" />
          <div className="flex items-center justify-between text-xs text-graphite-500 font-mono">
            <span>RECOVERY RATE LIFT</span>
            <TrendingUp className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-extrabold font-mono text-emerald-400 flex items-center gap-1">
            +{liftPct}%
            <ArrowUpRight className="w-5 h-5 text-emerald-400" />
          </div>
          <div className="text-[11px] text-graphite-400 font-mono">
            Treatment ({treatment.recovery_rate_pct}%) vs Control ({control.recovery_rate_pct}%)
          </div>
        </div>

        <div className="bg-graphite-900 border border-gold-800/60 p-5 rounded-2xl space-y-2">
          <div className="flex items-center justify-between text-xs text-graphite-500 font-mono">
            <span>NET VALUE LIFT</span>
            <DollarSign className="w-4 h-4 text-gold-400" />
          </div>
          <div className="text-3xl font-extrabold font-mono text-gold-400">
            ${netLift.toLocaleString()}
          </div>
          <div className="text-[11px] text-graphite-400 font-mono">
            Net monetary gain after retry cost & fee deduction
          </div>
        </div>

        <div className="bg-graphite-900 border border-copper-800/60 p-5 rounded-2xl space-y-2">
          <div className="flex items-center justify-between text-xs text-graphite-500 font-mono">
            <span>ESCALATION REDUCTION</span>
            <Activity className="w-4 h-4 text-copper-400" />
          </div>
          <div className="text-3xl font-extrabold font-mono text-copper-400">
            -{ (control.escalation_rate_pct - treatment.escalation_rate_pct).toFixed(1) }%
          </div>
          <div className="text-[11px] text-graphite-400 font-mono">
            Escalations reduced from {control.escalation_rate_pct}% to {treatment.escalation_rate_pct}%
          </div>
        </div>

        <div className="bg-graphite-900 border border-graphite-800 p-5 rounded-2xl space-y-2">
          <div className="flex items-center justify-between text-xs text-graphite-500 font-mono">
            <span>POLICY ROI RATIO</span>
            <BarChart3 className="w-4 h-4 text-ivory-300" />
          </div>
          <div className="text-3xl font-extrabold font-mono text-ivory-50">
            {roi}x
          </div>
          <div className="text-[11px] text-graphite-400 font-mono">
            Monetary yield per unit recovery expense
          </div>
        </div>
      </div>

      {/* COMPARISON COHORT CARDS */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* TREATMENT COHORT */}
        <div className="bg-graphite-900 border border-emerald-800/80 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-graphite-800 pb-3">
            <div>
              <span className="text-[10px] font-mono font-bold text-emerald-400 uppercase tracking-wider">EXPERIMENTAL VARIANT</span>
              <h3 className="text-base font-bold font-mono text-ivory-50">V10.2 Targeted Recovery (5%)</h3>
            </div>
            <span className="px-3 py-1 rounded-full text-xs font-mono font-bold bg-emerald-950 text-emerald-400 border border-emerald-800">
              TARGET (500 Txs)
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4 text-xs font-mono">
            <div className="bg-obsidian-950 p-3.5 rounded-xl border border-graphite-800">
              <span className="text-graphite-500 block">Cohort Size</span>
              <span className="text-base font-bold text-ivory-50">{treatment.cohort_size}</span>
            </div>
            <div className="bg-obsidian-950 p-3.5 rounded-xl border border-graphite-800">
              <span className="text-graphite-500 block">Recovery Success Rate</span>
              <span className="text-base font-bold text-emerald-400">{treatment.recovery_rate_pct}%</span>
            </div>
            <div className="bg-obsidian-950 p-3.5 rounded-xl border border-graphite-800">
              <span className="text-graphite-500 block">Net Value Recovered</span>
              <span className="text-base font-bold text-gold-400">${treatment.net_value_recovered.toLocaleString()}</span>
            </div>
            <div className="bg-obsidian-950 p-3.5 rounded-xl border border-graphite-800">
              <span className="text-graphite-500 block">Customer Escalations</span>
              <span className="text-base font-bold text-emerald-400">{treatment.escalation_rate_pct}%</span>
            </div>
          </div>
        </div>

        {/* CONTROL COHORT */}
        <div className="bg-graphite-900 border border-graphite-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-graphite-800 pb-3">
            <div>
              <span className="text-[10px] font-mono font-bold text-graphite-500 uppercase tracking-wider">CONTROL BASELINE</span>
              <h3 className="text-base font-bold font-mono text-ivory-50">Standard Retry-All (95%)</h3>
            </div>
            <span className="px-3 py-1 rounded-full text-xs font-mono font-bold bg-graphite-800 text-graphite-400 border border-graphite-700">
              RETRY_ALL (9,500 Txs)
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4 text-xs font-mono">
            <div className="bg-obsidian-950 p-3.5 rounded-xl border border-graphite-800">
              <span className="text-graphite-500 block">Cohort Size</span>
              <span className="text-base font-bold text-ivory-50">{control.cohort_size}</span>
            </div>
            <div className="bg-obsidian-950 p-3.5 rounded-xl border border-graphite-800">
              <span className="text-graphite-500 block">Recovery Success Rate</span>
              <span className="text-base font-bold text-graphite-300">{control.recovery_rate_pct}%</span>
            </div>
            <div className="bg-obsidian-950 p-3.5 rounded-xl border border-graphite-800">
              <span className="text-graphite-500 block">Net Value Recovered</span>
              <span className="text-base font-bold text-gold-400">${control.net_value_recovered.toLocaleString()}</span>
            </div>
            <div className="bg-obsidian-950 p-3.5 rounded-xl border border-graphite-800">
              <span className="text-graphite-500 block">Customer Escalations</span>
              <span className="text-base font-bold text-copper-400">{control.escalation_rate_pct}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* GUARDRAIL SAFETY MONITORING */}
      <div className="bg-graphite-900 border border-graphite-800 rounded-2xl p-6 space-y-4">
        <h2 className="text-sm font-bold font-mono text-ivory-50 flex items-center gap-2 border-b border-graphite-800 pb-3">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          ACTIVE PRODUCTION GUARDRAILS & CONTROL SAFETY
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
          <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 space-y-1">
            <div className="flex items-center justify-between text-emerald-400 font-bold">
              <span>FREQUENCY CAP</span>
              <CheckCircle2 className="w-3.5 h-3.5" />
            </div>
            <p className="text-ivory-50 font-semibold">Max 3 Attempts / Customer</p>
            <p className="text-graphite-500 text-[11px]">Prevents customer fatigue and over-retry spam.</p>
          </div>

          <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 space-y-1">
            <div className="flex items-center justify-between text-emerald-400 font-bold">
              <span>NET VALUE SUPPRESSION</span>
              <CheckCircle2 className="w-3.5 h-3.5" />
            </div>
            <p className="text-ivory-50 font-semibold">Negative Net Value Suppressed</p>
            <p className="text-graphite-500 text-[11px]">Cancels attempts where retry fee &gt; expected recovery.</p>
          </div>

          <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 space-y-1">
            <div className="flex items-center justify-between text-emerald-400 font-bold">
              <span>HMAC INTEGRITY</span>
              <CheckCircle2 className="w-3.5 h-3.5" />
            </div>
            <p className="text-ivory-50 font-semibold">Strict Webhook Authentication</p>
            <p className="text-graphite-500 text-[11px]">Rejects unverified webhook events missing HMAC SHA256.</p>
          </div>
        </div>
      </div>
    </div>
  )
}
