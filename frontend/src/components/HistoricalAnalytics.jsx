import React, { useState, useEffect } from 'react'
import { 
  History, 
  AlertCircle, 
  CreditCard, 
  Info,
  TrendingUp
} from 'lucide-react'

export default function HistoricalAnalytics() {
  const [outcomes, setOutcomes] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchOutcomes()
  }, [])

  const fetchOutcomes = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/outcomes/summary')
      if (res.ok) {
        setOutcomes(await res.json())
      }
    } catch (e) {
      console.error('Failed to fetch historical outcomes', e)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 font-sans">
      {/* Explicit Warning & Research Disclaimer Banner */}
      <div className="bg-copper-950/70 border border-copper-700/80 rounded-3xl p-6 space-y-2 shadow-2xl">
        <div className="flex items-center space-x-2 text-copper-400 font-bold text-sm font-mono">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span className="uppercase tracking-widest">HISTORICAL / RESEARCH ANALYTICS</span>
        </div>
        <p className="text-xs text-copper-200/90 leading-relaxed font-mono">
          These metrics represent historical or synthetic research results and are not validated production recovery outcomes. 
          {outcomes?.disclaimer}
        </p>
      </div>

      {/* KPI Overview Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 font-mono">
        <div className="bg-graphite-900/90 border border-graphite-800 rounded-2xl p-5 space-y-1 shadow-xl">
          <span className="text-xs font-medium text-graphite-500">Historical Records</span>
          <div className="text-3xl font-black text-ivory-50">{outcomes?.total_records || 2905}</div>
          <p className="text-[11px] text-graphite-500">Unique Customers: {outcomes?.unique_customers || 280}</p>
        </div>

        <div className="bg-graphite-900/90 border border-graphite-800 rounded-2xl p-5 space-y-1 shadow-xl">
          <span className="text-xs font-medium text-graphite-500">Recovery Rate</span>
          <div className="text-3xl font-black text-emerald-400">
            {outcomes ? (outcomes.historical_recovery_rate * 100).toFixed(1) : '0'}%
          </div>
          <p className="text-[11px] text-graphite-500">{outcomes?.recovered_count} Recovered</p>
        </div>

        <div className="bg-graphite-900/90 border border-graphite-800 rounded-2xl p-5 space-y-1 shadow-xl">
          <span className="text-xs font-medium text-graphite-500">Total Recovered</span>
          <div className="text-3xl font-black text-gold-400">
            ${outcomes ? outcomes.total_recovered_amount.toLocaleString() : '0'}
          </div>
          <p className="text-[11px] text-graphite-500">From ${outcomes ? outcomes.total_failed_volume.toLocaleString() : '0'} failed</p>
        </div>

        <div className="bg-graphite-900/90 border border-graphite-800 rounded-2xl p-5 space-y-1 shadow-xl">
          <span className="text-xs font-medium text-graphite-500">Avg Score</span>
          <div className="text-3xl font-black text-ivory-200">{outcomes?.avg_recoverability_score || '0'} / 100</div>
          <p className="text-[11px] text-graphite-500">Historical score metric</p>
        </div>
      </div>

      {/* Visual Breakdowns */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 font-mono text-xs">
        {/* Payment Method Breakdown (Gold & Emerald Bars) */}
        <div className="bg-graphite-900/90 border border-graphite-800 rounded-3xl p-6 space-y-4 shadow-xl">
          <h3 className="text-sm font-bold text-ivory-50 flex items-center gap-2">
            <CreditCard className="w-4 h-4 text-gold-400" />
            Payment Method Breakdown
          </h3>

          <div className="space-y-3">
            {outcomes?.payment_method_breakdown && Object.entries(outcomes.payment_method_breakdown).map(([method, count]) => {
              const pct = ((count / outcomes.total_records) * 100).toFixed(1)
              return (
                <div key={method} className="space-y-1">
                  <div className="flex justify-between">
                    <span className="text-ivory-200 font-semibold">{method}</span>
                    <span className="text-graphite-500">{count} txns ({pct}%)</span>
                  </div>
                  <div className="w-full h-2.5 bg-obsidian-950 rounded-full overflow-hidden border border-graphite-800">
                    <div className="bg-gold-500 h-full rounded-full" style={{ width: `${pct}%` }}></div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Failure Reason Breakdown (Copper Bars) */}
        <div className="bg-graphite-900/90 border border-graphite-800 rounded-3xl p-6 space-y-4 shadow-xl">
          <h3 className="text-sm font-bold text-ivory-50 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-copper-400" />
            Failure Reason Breakdown
          </h3>

          <div className="space-y-3">
            {outcomes?.failure_reason_breakdown && Object.entries(outcomes.failure_reason_breakdown).map(([reason, count]) => {
              const pct = ((count / outcomes.total_records) * 100).toFixed(1)
              return (
                <div key={reason} className="space-y-1">
                  <div className="flex justify-between">
                    <span className="text-ivory-200 font-semibold">{reason}</span>
                    <span className="text-graphite-500">{count} txns ({pct}%)</span>
                  </div>
                  <div className="w-full h-2.5 bg-obsidian-950 rounded-full overflow-hidden border border-graphite-800">
                    <div className="bg-copper-600 h-full rounded-full" style={{ width: `${pct}%` }}></div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
