import React, { useState, useEffect } from 'react'
import { 
  Cpu, 
  Search, 
  ShieldCheck, 
  CheckCircle2, 
  ArrowRight, 
  PlayCircle, 
  Lock,
  Layers,
  Activity,
  Sparkles
} from 'lucide-react'

export default function DecisionConsole({ initialTxId, onNavigateDemo }) {
  const [txIdInput, setTxIdInput] = useState(initialTxId || 'RECLAIM-V2-000004')
  const [decision, setDecision] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Animated processing steps state
  const [processingStep, setProcessingStep] = useState(0)

  useEffect(() => {
    if (initialTxId) {
      setTxIdInput(initialTxId)
      fetchDecision(initialTxId)
    } else {
      fetchDecision('RECLAIM-V2-000004')
    }
  }, [initialTxId])

  const fetchDecision = async (idToFetch) => {
    const targetId = (idToFetch || txIdInput).trim()
    if (!targetId) return

    setLoading(true)
    setError(null)
    setProcessingStep(1) // "Reading transaction data"

    setTimeout(() => setProcessingStep(2), 300) // "Applying V10.2 policy"
    setTimeout(() => setProcessingStep(3), 600) // "Checking targeting threshold"

    try {
      const res = await fetch('/api/decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transaction_id: targetId })
      })

      setTimeout(async () => {
        if (res.ok) {
          const data = await res.json()
          setDecision(data)
          setProcessingStep(4) // Final Decision
        } else {
          const errData = await res.json()
          setError(errData.detail || 'Transaction ID not found in production policy V10.2')
          setDecision(null)
          setProcessingStep(0)
        }
        setLoading(false)
      }, 900)
    } catch (e) {
      setError('Network error connecting to decision API')
      setDecision(null)
      setLoading(false)
      setProcessingStep(0)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    fetchDecision()
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6 font-sans">
      {/* Console Input & Query Bar */}
      <div className="bg-graphite-900/90 border border-graphite-800 rounded-3xl p-6 shadow-2xl space-y-4">
        <div className="flex items-center space-x-3 border-b border-graphite-800 pb-4">
          <div className="p-3 bg-obsidian-950 text-emerald-400 rounded-2xl border border-emerald-900">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-ivory-50 font-mono">Causal Decision Engine Console</h2>
            <p className="text-xs text-graphite-500">Query read-only V10.2 production decision logic for any transaction ID</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="flex gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-graphite-500 absolute left-4 top-4" />
            <input
              type="text"
              placeholder="Enter Transaction ID (e.g. RECLAIM-V2-000004)..."
              value={txIdInput}
              onChange={(e) => setTxIdInput(e.target.value)}
              className="w-full bg-obsidian-950 border border-graphite-800 rounded-2xl pl-11 pr-4 py-3.5 text-sm font-mono text-emerald-400 placeholder-graphite-600 focus:outline-none focus:border-emerald-500 font-bold"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 text-obsidian-950 font-extrabold px-6 py-3.5 rounded-2xl transition-all shadow-lg shadow-emerald-950 text-xs flex items-center gap-2 font-mono uppercase tracking-wider shrink-0"
          >
            {loading ? (
              <div className="w-4 h-4 border-2 border-obsidian-950 border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <>
                <span>Evaluate Policy</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>

        {error && (
          <div className="bg-copper-950/40 border border-copper-800 rounded-xl p-4 text-copper-300 text-xs font-mono">
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* STEP-BY-STEP DECISION PROCESSING ANIMATION */}
      {loading && (
        <div className="bg-obsidian-950 border border-graphite-800 rounded-3xl p-8 shadow-2xl space-y-6 text-center font-mono">
          <div className="relative w-20 h-20 mx-auto flex items-center justify-center">
            <div className="absolute inset-0 rounded-full border-2 border-dashed border-emerald-500/40 animate-spin-slow"></div>
            <Cpu className="w-8 h-8 text-emerald-400 animate-pulse" />
          </div>

          <div className="space-y-2">
            <div className="text-xs font-bold text-gold-400 uppercase tracking-widest">
              Processing Causal Decision
            </div>
            <div className="space-y-1.5 text-xs text-graphite-500 max-w-sm mx-auto">
              <div className={`flex items-center justify-between p-2 rounded-lg ${processingStep >= 1 ? 'text-emerald-400 font-bold' : ''}`}>
                <span>1. Reading Transaction Data</span>
                {processingStep >= 1 && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
              </div>
              <div className={`flex items-center justify-between p-2 rounded-lg ${processingStep >= 2 ? 'text-emerald-400 font-bold' : ''}`}>
                <span>2. Applying Policy V10.2</span>
                {processingStep >= 2 && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
              </div>
              <div className={`flex items-center justify-between p-2 rounded-lg ${processingStep >= 3 ? 'text-emerald-400 font-bold' : ''}`}>
                <span>3. Checking TOP_RATE 5% Threshold</span>
                {processingStep >= 3 && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DECISION RESULT DISPLAY CARD */}
      {!loading && decision && (
        <div className="bg-graphite-900/90 border border-graphite-800 rounded-3xl p-6 sm:p-8 shadow-2xl space-y-6">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-graphite-800 pb-5">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xs font-mono text-graphite-500">Transaction ID:</span>
                <span className="text-lg font-extrabold font-mono text-emerald-400">{decision.transaction_id}</span>
              </div>
              <div className="text-xs text-graphite-500 font-mono mt-0.5">
                Customer ID: <strong className="text-ivory-200">{decision.customer_id}</strong>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <span className="bg-obsidian-950 text-emerald-400 border border-emerald-800 text-[11px] font-mono px-3 py-1 rounded-full font-bold flex items-center gap-1.5">
                <Lock className="w-3.5 h-3.5" />
                READ-ONLY IMMUTABLE
              </span>
            </div>
          </div>

          {/* Result State Banner */}
          <div className={`p-6 rounded-2xl border ${
            decision.intervention_selected === 1
              ? 'bg-emerald-950/40 border-emerald-700/80 shadow-xl shadow-emerald-950/50'
              : 'bg-obsidian-950 border-graphite-800'
          }`}>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="space-y-1">
                <span className="text-xs font-mono text-graphite-500 uppercase tracking-wider">Assigned Policy Action</span>
                <div className="text-3xl font-black font-mono tracking-tight text-ivory-50 flex items-center gap-3">
                  <span>{decision.selected_policy}</span>
                  {decision.intervention_selected === 1 ? (
                    <span className="bg-emerald-500 text-obsidian-950 text-xs px-3 py-1 rounded-full font-sans font-black uppercase">
                      Targeted Recovery (Top 5%)
                    </span>
                  ) : (
                    <span className="bg-graphite-800 text-graphite-400 text-xs px-3 py-1 rounded-full font-sans font-medium">
                      Normal Retry Path
                    </span>
                  )}
                </div>
              </div>

              <div className="text-right font-mono">
                <span className="text-[11px] text-graphite-500">Targeting Rate</span>
                <div className="text-xl font-bold text-emerald-400">
                  {(decision.selected_policy_parameter * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-graphite-800/80">
              <span className="text-xs font-bold text-ivory-200 font-mono">Decision Rationale:</span>
              <p className="text-xs text-graphite-500 mt-1 leading-relaxed">{decision.decision_rationale}</p>
            </div>
          </div>

          {/* Causal Uplift Probabilities Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono">
            <div className="bg-obsidian-950 border border-graphite-800 rounded-2xl p-5 space-y-1">
              <span className="text-xs text-graphite-500 font-medium">P(Treatment Success)</span>
              <div className="text-2xl font-black text-emerald-400">
                {(decision.predicted_treatment_probability * 100).toFixed(2)}%
              </div>
              <p className="text-[10px] text-graphite-500">Predicted probability under intervention</p>
            </div>

            <div className="bg-obsidian-950 border border-graphite-800 rounded-2xl p-5 space-y-1">
              <span className="text-xs text-graphite-500 font-medium">P(Control Success)</span>
              <div className="text-2xl font-black text-ivory-200">
                {(decision.predicted_control_probability * 100).toFixed(2)}%
              </div>
              <p className="text-[10px] text-graphite-500">Predicted probability under standard retry</p>
            </div>

            <div className="bg-obsidian-950 border border-graphite-800 rounded-2xl p-5 space-y-1">
              <span className="text-xs text-graphite-500 font-medium">Predicted Uplift (\(\Delta P\))</span>
              <div className={`text-2xl font-black ${
                decision.predicted_probability_uplift > 0 ? 'text-gold-400' : 'text-graphite-500'
              }`}>
                {decision.predicted_probability_uplift > 0 ? '+' : ''}
                {(decision.predicted_probability_uplift * 100).toFixed(4)}%
              </div>
              <p className="text-[10px] text-graphite-500">Net causal improvement</p>
            </div>
          </div>

          {/* Action Footer */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-4 border-t border-graphite-800 font-mono">
            <div className="flex items-center space-x-2 text-xs text-graphite-500">
              <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>{decision.governance_note}</span>
            </div>

            <button
              onClick={() => onNavigateDemo(decision.transaction_id)}
              className="bg-copper-600 hover:bg-copper-500 text-ivory-50 text-xs font-bold px-6 py-3 rounded-xl transition-all shadow-lg shadow-copper-950 flex items-center gap-2 uppercase tracking-wider"
            >
              <PlayCircle className="w-4 h-4" />
              <span>Simulate Operator Action</span>
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
