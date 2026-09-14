import React, { useState, useEffect } from 'react'
import { 
  PlayCircle, 
  AlertTriangle, 
  CheckCircle2, 
  History, 
  ArrowRight,
  ShieldAlert,
  CreditCard,
  Send,
  RefreshCw,
  MessageSquare
} from 'lucide-react'

export default function DemoWorkflow({ initialTxId }) {
  const [txIdInput, setTxIdInput] = useState(initialTxId || 'RECLAIM-V2-000004')
  const [operatorAction, setOperatorAction] = useState('SEND_PAYMENT_REMINDER')
  const [operatorNotes, setOperatorNotes] = useState('')
  
  const [decision, setDecision] = useState(null)
  const [simResult, setSimResult] = useState(null)
  const [demoEvents, setDemoEvents] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchDemoEvents()
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
    try {
      const res = await fetch('/api/decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transaction_id: targetId })
      })

      if (res.ok) {
        setDecision(await res.json())
      } else {
        setError(`Transaction ID '${targetId}' not found in production policy.`)
        setDecision(null)
      }
    } catch (e) {
      setError('Failed to fetch transaction decision.')
      setDecision(null)
    } finally {
      setLoading(false)
    }
  }

  const fetchDemoEvents = async () => {
    try {
      const res = await fetch('/api/demo/events')
      if (res.ok) {
        const data = await res.json()
        setDemoEvents(data.events || [])
      }
    } catch (e) {
      console.error('Failed to load demo events', e)
    }
  }

  const handleExecuteSimulation = async (e) => {
    e.preventDefault()
    if (!decision) return

    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/demo/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transaction_id: decision.transaction_id,
          operator_action: operatorAction,
          operator_notes: operatorNotes || 'Executed via RECLAIM Interactive Demo Simulator'
        })
      })

      if (res.ok) {
        const data = await res.json()
        setSimResult(data.demo_event)
        fetchDemoEvents()
      } else {
        const errData = await res.json()
        setError(errData.detail || 'Simulation execution failed.')
      }
    } catch (e) {
      setError('Network error during simulation execution.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 font-sans">
      {/* Prominent Demo Mode Disclaimer Banner */}
      <div className="bg-copper-950/70 border border-copper-700/80 rounded-3xl p-6 text-copper-200 space-y-2 shadow-2xl">
        <div className="flex items-center space-x-2 font-bold text-sm text-copper-400 font-mono">
          <AlertTriangle className="w-5 h-5 shrink-0" />
          <span className="uppercase tracking-widest">DEMO MODE — No real payment or financial action is executed.</span>
        </div>
        <p className="text-xs text-copper-200/90 leading-relaxed font-mono">
          This workspace simulates operator recovery interventions. 
          All results are purely simulated demo events stored in isolated JSON storage. No real payment attempt is triggered.
        </p>
      </div>

      {/* Main Simulation Workflow Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Workflow Configuration */}
        <div className="lg:col-span-7 space-y-6">
          {/* Step 1: Select Transaction */}
          <div className="bg-graphite-900/90 border border-graphite-800 rounded-3xl p-6 space-y-4 shadow-xl">
            <div className="flex items-center justify-between border-b border-graphite-800 pb-3">
              <h3 className="text-sm font-bold text-ivory-50 flex items-center gap-2 font-mono">
                <span className="bg-copper-500 text-obsidian-950 text-xs w-5 h-5 rounded-full flex items-center justify-center font-bold">1</span>
                Select Production Transaction
              </h3>
              <span className="text-xs font-mono text-emerald-400">POLICY V10.2</span>
            </div>

            <div className="flex gap-2">
              <input
                type="text"
                value={txIdInput}
                onChange={(e) => setTxIdInput(e.target.value)}
                placeholder="Transaction ID (e.g. RECLAIM-V2-000004)"
                className="flex-1 bg-obsidian-950 border border-graphite-800 rounded-xl px-4 py-2.5 text-xs font-mono text-emerald-400 focus:outline-none focus:border-copper-500 font-bold"
              />
              <button
                onClick={() => fetchDecision(txIdInput)}
                className="bg-graphite-800 hover:bg-graphite-700 text-ivory-50 text-xs font-bold px-4 py-2.5 rounded-xl transition-colors font-mono uppercase"
              >
                Fetch Decision
              </button>
            </div>

            {decision && (
              <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 space-y-3 text-xs font-mono">
                <div className="flex justify-between items-center">
                  <div>
                    <span className="text-graphite-500">Txn:</span> <strong className="text-emerald-400">{decision.transaction_id}</strong>
                  </div>
                  <span className={`px-2.5 py-0.5 rounded text-[11px] font-bold ${
                    decision.intervention_selected === 1 
                      ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' 
                      : 'bg-graphite-800 text-graphite-400'
                  }`}>
                    {decision.selected_policy}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2 text-[11px] text-ivory-200 pt-2 border-t border-graphite-800">
                  <div>P(Trt): <strong className="text-emerald-400">{(decision.predicted_treatment_probability * 100).toFixed(1)}%</strong></div>
                  <div>P(Ctrl): <strong className="text-graphite-500">{(decision.predicted_control_probability * 100).toFixed(1)}%</strong></div>
                  <div>Uplift: <strong className="text-gold-400">{(decision.predicted_probability_uplift * 100).toFixed(2)}%</strong></div>
                </div>
              </div>
            )}
          </div>

          {/* Step 2: Choose & Execute Simulated Recovery Action */}
          {decision && (
            <div className="bg-graphite-900/90 border border-graphite-800 rounded-3xl p-6 space-y-4 shadow-xl">
              <div className="flex items-center justify-between border-b border-graphite-800 pb-3">
                <h3 className="text-sm font-bold text-ivory-50 flex items-center gap-2 font-mono">
                  <span className="bg-copper-500 text-obsidian-950 text-xs w-5 h-5 rounded-full flex items-center justify-center font-bold">2</span>
                  Execute Simulated Recovery Action
                </h3>
                <span className="text-xs font-mono text-gold-400">DEMO SIMULATOR</span>
              </div>

              <form onSubmit={handleExecuteSimulation} className="space-y-4 text-xs font-mono">
                <div>
                  <label className="block text-ivory-200 font-medium mb-1">Select Recovery Action</label>
                  <select
                    value={operatorAction}
                    onChange={(e) => setOperatorAction(e.target.value)}
                    className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl px-3.5 py-3 text-xs text-ivory-50 focus:outline-none focus:border-copper-500"
                  >
                    <option value="SEND_PAYMENT_REMINDER">SEND_PAYMENT_REMINDER — Send Frictionless Payment Recovery Ping</option>
                    <option value="OFFER_ALTERNATE_METHOD">OFFER_ALTERNATE_METHOD — Prompt Alternate Card/UPI Channel</option>
                    <option value="SCHEDULE_INTELLIGENT_RETRY">SCHEDULE_INTELLIGENT_RETRY — Schedule Retry in Off-Peak Window</option>
                    <option value="DISPLAY_RECOVERY_MESSAGE">DISPLAY_RECOVERY_MESSAGE — Show In-App Gateway Alert</option>
                    <option value="ROUTE_ASSISTED_RECOVERY">ROUTE_ASSISTED_RECOVERY — Route to Assisted Merchant Desk</option>
                  </select>
                </div>

                <div>
                  <label className="block text-ivory-200 font-medium mb-1">Operator Notes</label>
                  <input
                    type="text"
                    placeholder="e.g. Executing demo simulation step for Razorpay hackathon..."
                    value={operatorNotes}
                    onChange={(e) => setOperatorNotes(e.target.value)}
                    className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl px-4 py-2.5 text-xs text-ivory-50 focus:outline-none focus:border-copper-500"
                  />
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-gradient-to-r from-copper-600 via-copper-500 to-copper-600 hover:from-copper-500 hover:to-copper-400 text-ivory-50 font-bold py-3.5 rounded-xl transition-all shadow-lg shadow-copper-950 flex items-center justify-center gap-2 uppercase tracking-wider"
                >
                  <PlayCircle className="w-4 h-4" />
                  <span>Execute Demo Action</span>
                </button>
              </form>
            </div>
          )}
        </div>

        {/* Right Column: 3D Illustration & Simulation Event Timeline */}
        <div className="lg:col-span-5 space-y-6">
          {/* Latest Simulation Result */}
          {simResult && (
            <div className="bg-graphite-900/90 border border-copper-500/50 rounded-3xl p-6 space-y-4 shadow-2xl relative overflow-hidden">
              <div className="bg-copper-600 text-ivory-50 text-[10px] font-black tracking-widest uppercase px-3 py-1 text-center font-mono rounded-lg">
                SIMULATED — NOT AN OBSERVED RECOVERY
              </div>

              <div className="space-y-3 pt-2 font-mono">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-graphite-500">Event ID:</span>
                  <span className="font-bold text-gold-400">{simResult.event_id}</span>
                </div>

                <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 space-y-1.5">
                  <div className="text-xs text-emerald-400 font-bold flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    <span>Demo action completed successfully.</span>
                  </div>
                  <div className="text-sm font-black text-ivory-50">
                    {simResult.simulated_result}
                  </div>
                  <p className="text-[11px] text-graphite-500">Action: {simResult.operator_action}</p>
                </div>

                <div className="text-[11px] text-copper-300 bg-copper-950/40 border border-copper-800 p-3 rounded-xl">
                  {simResult.disclaimer}
                </div>
              </div>
            </div>
          )}

          {/* 3D Payment Card Illustration */}
          <div className="bg-obsidian-950 border border-graphite-800 rounded-3xl p-6 text-center space-y-4 relative overflow-hidden">
            <div className="w-24 h-16 mx-auto bg-gradient-to-tr from-gold-600 to-copper-500 rounded-xl p-2 shadow-xl animate-float flex flex-col justify-between text-left font-mono">
              <CreditCard className="w-5 h-5 text-obsidian-950" />
              <span className="text-[9px] text-obsidian-950 font-bold">RECLAIM PAY</span>
            </div>
            <div className="text-xs font-mono text-graphite-500">
              Isolated Event Store: <strong className="text-gold-400">data/demo_execution_store.json</strong>
            </div>
          </div>

          {/* Recorded Demo Events Audit Trail */}
          <div className="bg-graphite-900/90 border border-graphite-800 rounded-3xl p-5 space-y-4 shadow-xl">
            <h3 className="text-sm font-bold text-ivory-50 flex items-center gap-2 font-mono">
              <History className="w-4 h-4 text-gold-400" />
              Recorded Demo Events Store ({demoEvents.length})
            </h3>

            <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
              {demoEvents.length === 0 ? (
                <div className="text-xs text-graphite-500 text-center py-6 font-mono">No demo events recorded yet.</div>
              ) : (
                demoEvents.map((evt) => (
                  <div key={evt.event_id} className="bg-obsidian-950 border border-graphite-800 rounded-xl p-3 text-xs space-y-1 font-mono">
                    <div className="flex justify-between items-center">
                      <span className="text-gold-400 font-bold">{evt.event_id}</span>
                      <span className="text-graphite-500 text-[10px]">{evt.timestamp.slice(11, 19)}</span>
                    </div>
                    <div className="text-ivory-200">Txn: <strong className="text-emerald-400">{evt.transaction_id}</strong> | {evt.operator_action}</div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
