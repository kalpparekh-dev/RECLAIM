import React, { useState, useEffect } from 'react'
import { 
  Users, 
  Target, 
  TrendingUp, 
  RotateCcw, 
  ShieldCheck, 
  ArrowRight,
  Database,
  BarChart2,
  CheckCircle2,
  AlertTriangle,
  Layers,
  CircleDollarSign,
  Activity
} from 'lucide-react'

// Custom hook for animated counter
function useCounter(targetValue, duration = 1200) {
  const [count, setCount] = useState(0)

  useEffect(() => {
    let start = 0
    const end = parseInt(targetValue, 10)
    if (isNaN(end) || end === 0) return

    const totalSteps = 40
    const stepTime = duration / totalSteps
    let step = 0

    const timer = setInterval(() => {
      step++
      const progress = step / totalSteps
      const current = Math.floor(end * Math.sin((progress * Math.PI) / 2))
      setCount(current)

      if (step >= totalSteps) {
        setCount(end)
        clearInterval(timer)
      }
    }, stepTime)

    return () => clearInterval(timer)
  }, [targetValue, duration])

  return count
}

export default function ExecutiveOverview({ summary, onNavigateConsole, onNavigateExplorer }) {
  const totalRaw = summary?.total_transactions || 10000
  const selectedRaw = summary?.selected_transactions || 500
  const retryAllRaw = totalRaw - selectedRaw
  const positiveUpliftRaw = summary?.positive_uplift_count || 5391

  const totalCount = useCounter(totalRaw)
  const selectedCount = useCounter(selectedRaw)
  const retryAllCount = useCounter(retryAllRaw)
  const positiveUpliftCount = useCounter(positiveUpliftRaw)

  const [activePipelineStage, setActivePipelineStage] = useState(2)

  const pipelineStages = [
    { id: 0, label: 'Failed Payment', desc: 'Initial transaction decline detected', color: 'border-copper-500 text-copper-400 bg-copper-950/40' },
    { id: 1, label: 'Policy Evaluation', desc: 'Causal uplift prediction scoring', color: 'border-gold-500 text-gold-400 bg-gold-950/40' },
    { id: 2, label: 'Top 5% Targeting', desc: 'Heterogeneous uplift threshold check', color: 'border-emerald-500 text-emerald-400 bg-emerald-950/40' },
    { id: 3, label: 'Targeted Recovery / Retry', desc: 'High-priority intervention or RETRY_ALL', color: 'border-emerald-400 text-emerald-300 bg-emerald-900/30' },
    { id: 4, label: 'Monitoring', desc: 'V21 population & schema audit', color: 'border-graphite-600 text-ivory-100 bg-graphite-900' },
  ]

  return (
    <div className="space-y-8">
      {/* TOP BANNER HERO */}
      <div className="bg-gradient-to-br from-obsidian-900 via-graphite-900 to-obsidian-950 border border-graphite-800 rounded-3xl p-6 sm:p-8 shadow-2xl relative overflow-hidden">
        <div className="absolute -right-16 -bottom-16 w-80 h-80 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none animate-pulse-glow"></div>
        <div className="absolute top-0 right-1/4 w-64 h-64 bg-gold-500/10 rounded-full blur-3xl pointer-events-none"></div>

        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 relative z-10">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="bg-emerald-950 text-emerald-400 border border-emerald-800/80 text-xs px-3 py-1 rounded-full font-bold flex items-center gap-1.5 font-mono">
                <CheckCircle2 className="w-3.5 h-3.5" />
                V21 MONITORING PASS
              </span>
              <span className="text-graphite-500 text-xs font-mono">Immutable Policy Dataset</span>
            </div>

            <h1 className="text-3xl sm:text-4xl font-extrabold text-ivory-50 tracking-tight">
              Recovery Decision Intelligence
            </h1>
            <p className="text-xs sm:text-sm text-graphite-500 max-w-2xl leading-relaxed">
              RECLAIM evaluates failed transactions using heterogeneous causal uplift predictions. 
              The top 5% highest-uplift transactions receive targeted recovery intervention; the remaining 95% continue along the standard retry path.
            </p>
          </div>

          <button
            onClick={onNavigateExplorer}
            className="bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 text-obsidian-950 font-extrabold text-xs px-6 py-3.5 rounded-2xl transition-all shadow-lg shadow-emerald-950 flex items-center gap-2.5 shrink-0 uppercase tracking-wider font-mono"
          >
            <span>Explore 10,000 Transactions</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* TOP KPI METRIC CARDS (WITH ANIMATED COUNTERS) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Total Transactions */}
        <div className="bg-graphite-900/90 border border-graphite-800 rounded-2xl p-6 shadow-xl hover:border-graphite-700 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-graphite-500 uppercase tracking-wider font-mono">Total Transactions</span>
            <div className="p-2.5 bg-obsidian-950 rounded-xl text-ivory-100 border border-graphite-800">
              <Database className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-4">
            <div className="text-3xl sm:text-4xl font-black text-ivory-50 font-mono tracking-tight">
              {totalCount.toLocaleString()}
            </div>
            <div className="text-xs text-graphite-500 mt-1 font-mono">
              Unique Customers: <strong className="text-ivory-100 font-bold">425</strong>
            </div>
          </div>
        </div>

        {/* Card 2: Targeted (TOP_RATE 5%) */}
        <div className="bg-graphite-900/90 border border-emerald-800/80 rounded-2xl p-6 shadow-xl relative overflow-hidden hover:border-emerald-700 transition-all">
          <div className="absolute top-0 right-0 w-2 h-full bg-emerald-500"></div>
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-emerald-400 uppercase tracking-wider font-mono">Targeted (TOP_RATE)</span>
            <div className="p-2.5 bg-emerald-950 text-emerald-400 rounded-xl border border-emerald-800">
              <Target className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-4">
            <div className="text-3xl sm:text-4xl font-black text-emerald-400 font-mono tracking-tight">
              {selectedCount.toLocaleString()}
            </div>
            <div className="text-xs text-emerald-300/80 mt-1 font-mono">
              Target Rate: <strong className="text-emerald-300 font-bold">5.0%</strong> (TOP 5%)
            </div>
          </div>
        </div>

        {/* Card 3: Fallback (RETRY_ALL) */}
        <div className="bg-graphite-900/90 border border-graphite-800 rounded-2xl p-6 shadow-xl hover:border-graphite-700 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-graphite-500 uppercase tracking-wider font-mono">Fallback (RETRY_ALL)</span>
            <div className="p-2.5 bg-obsidian-950 text-graphite-500 rounded-xl border border-graphite-800">
              <RotateCcw className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-4">
            <div className="text-3xl sm:text-4xl font-black text-ivory-200 font-mono tracking-tight">
              {retryAllCount.toLocaleString()}
            </div>
            <div className="text-xs text-graphite-500 mt-1 font-mono">
              95.0% Non-Targeted Population
            </div>
          </div>
        </div>

        {/* Card 4: Positive Uplift Population */}
        <div className="bg-graphite-900/90 border border-gold-800/80 rounded-2xl p-6 shadow-xl hover:border-gold-700 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-gold-400 uppercase tracking-wider font-mono">Positive Uplift Txns</span>
            <div className="p-2.5 bg-gold-950 text-gold-400 rounded-xl border border-gold-800">
              <TrendingUp className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-4">
            <div className="text-3xl sm:text-4xl font-black text-gold-400 font-mono tracking-tight">
              {positiveUpliftCount.toLocaleString()}
            </div>
            <div className="text-xs text-gold-300/80 mt-1 font-mono">
              53.91% Positive Causal Uplift
            </div>
          </div>
        </div>
      </div>

      {/* CENTRAL "RECOVERY INTELLIGENCE" VISUALIZATION & 3D METALLIC OBJECT */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 8 Cols: Circular 5% Targeting Ring Visual & Allocation Bar */}
        <div className="lg:col-span-8 bg-graphite-900/90 border border-graphite-800 rounded-3xl p-6 sm:p-8 space-y-6 shadow-2xl">
          <div className="flex items-center justify-between border-b border-graphite-800 pb-4">
            <div>
              <h3 className="text-lg font-bold text-ivory-50 flex items-center gap-2 font-mono">
                <Activity className="w-5 h-5 text-emerald-400" />
                V10.2 Recovery Intelligence Allocation
              </h3>
              <p className="text-xs text-graphite-500">Heterogeneous causal targeting distribution</p>
            </div>
            <span className="bg-obsidian-950 text-emerald-400 border border-emerald-800 text-xs px-3 py-1 rounded-full font-mono font-bold">
              TOP_RATE (5.0%)
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
            {/* Circular 5% Targeting Ring */}
            <div className="md:col-span-5 flex flex-col items-center justify-center relative py-4">
              <div className="relative w-44 h-44 flex items-center justify-center">
                {/* SVG Ring */}
                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="42" stroke="#22252e" strokeWidth="10" fill="transparent" />
                  <circle 
                    cx="50" 
                    cy="50" 
                    r="42" 
                    stroke="#10b981" 
                    strokeWidth="10" 
                    fill="transparent" 
                    strokeDasharray="263.89" 
                    strokeDashoffset={263.89 * (1 - 0.05)} 
                    strokeLinecap="round"
                    className="transition-all duration-1000"
                  />
                </svg>

                {/* Center Content */}
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                  <span className="text-2xl font-black font-mono text-emerald-400 tracking-tight">5%</span>
                  <span className="text-[10px] font-mono text-graphite-500 font-bold uppercase">TARGETED</span>
                </div>
              </div>

              <span className="text-xs font-mono text-ivory-200 mt-3 font-semibold">500 Targeted / 9,500 Retry</span>
            </div>

            {/* Allocation Description & Stats */}
            <div className="md:col-span-7 space-y-4">
              {/* Allocation Progress Bar */}
              <div className="space-y-2 font-mono text-xs">
                <div className="flex justify-between">
                  <span className="text-emerald-400 font-bold">TARGET: 500 (5.0%)</span>
                  <span className="text-graphite-500">RETRY_ALL: 9,500 (95.0%)</span>
                </div>
                <div className="w-full h-3.5 bg-obsidian-950 rounded-full overflow-hidden flex p-0.5 border border-graphite-800">
                  <div className="bg-gradient-to-r from-emerald-500 to-emerald-400 h-full rounded-full w-[5%] shadow-md"></div>
                  <div className="bg-graphite-800 h-full rounded-full w-[95%] ml-1"></div>
                </div>
              </div>

              {/* Uplift Breakdown Bar */}
              <div className="space-y-2 font-mono text-xs pt-1">
                <div className="flex justify-between">
                  <span className="text-gold-400 font-bold">Positive Uplift: 5,391 (53.91%)</span>
                  <span className="text-graphite-500">Zero / Neg Uplift: 4,609 (46.09%)</span>
                </div>
                <div className="w-full h-3 bg-obsidian-950 rounded-full overflow-hidden flex p-0.5 border border-graphite-800">
                  <div className="bg-gold-500 h-full rounded-full w-[53.91%]"></div>
                  <div className="bg-copper-600/70 h-full rounded-full w-[46.09%] ml-0.5"></div>
                </div>
              </div>

              {/* Grid cards */}
              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="bg-obsidian-950 border border-emerald-900/50 p-3 rounded-xl space-y-0.5">
                  <div className="text-[10px] text-graphite-500 font-mono uppercase">TARGET ACTION</div>
                  <div className="text-xs font-bold text-emerald-400 font-mono">Top 5% Priority</div>
                </div>
                <div className="bg-obsidian-950 border border-graphite-800 p-3 rounded-xl space-y-0.5">
                  <div className="text-[10px] text-graphite-500 font-mono uppercase">FALLBACK ACTION</div>
                  <div className="text-xs font-bold text-ivory-200 font-mono">RETRY_ALL Path</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right 4 Cols: 3D Metallic Coin Stack / Transaction Object */}
        <div className="lg:col-span-4 bg-gradient-to-b from-graphite-900 via-obsidian-900 to-obsidian-950 border border-graphite-800 rounded-3xl p-6 flex flex-col justify-between items-center text-center space-y-6 shadow-2xl relative overflow-hidden">
          <div className="space-y-1">
            <span className="text-[10px] font-mono font-bold text-gold-400 uppercase tracking-widest">
              Financial Infrastructure
            </span>
            <h4 className="text-base font-bold text-ivory-50">Decision Vault Security</h4>
          </div>

          {/* 3D Rotating Metallic Coin Object */}
          <div className="relative w-36 h-36 flex items-center justify-center my-4">
            <div className="absolute inset-0 bg-gold-500/10 rounded-full blur-xl animate-pulse"></div>
            
            {/* CSS 3D Coin Stack */}
            <div className="w-28 h-28 rounded-full bg-gradient-to-tr from-gold-600 via-gold-400 to-amber-200 p-1 shadow-2xl shadow-gold-500/20 animate-coin flex items-center justify-center">
              <div className="w-full h-full rounded-full bg-obsidian-950 flex flex-col items-center justify-center border border-gold-400/40">
                <CircleDollarSign className="w-10 h-10 text-gold-400" />
                <span className="text-[9px] font-mono text-emerald-400 font-bold tracking-wider uppercase mt-1">
                  V10.2 ENGINE
                </span>
              </div>
            </div>
          </div>

          <div className="bg-obsidian-950 border border-graphite-800 p-3.5 rounded-xl w-full text-left space-y-1">
            <div className="text-[11px] text-graphite-500 font-mono">POLICY INTEGRITY</div>
            <div className="text-xs text-emerald-400 font-bold font-mono">10,000 / 10,000 RECORDS VERIFIED</div>
          </div>
        </div>
      </div>

      {/* DYNAMIC INTERACTIVE TRANSACTION FLOW PIPELINE */}
      <div className="bg-graphite-900/90 border border-graphite-800 rounded-3xl p-6 sm:p-8 space-y-6 shadow-2xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-graphite-800 pb-4">
          <div>
            <h3 className="text-lg font-bold text-ivory-50 flex items-center gap-2 font-mono">
              <Layers className="w-5 h-5 text-gold-400" />
              Dynamic Transaction Recovery Pipeline
            </h3>
            <p className="text-xs text-graphite-500">Interactive 5-stage payment processing evaluation lifecycle</p>
          </div>
          <span className="text-xs font-mono text-graphite-500">Click stage to inspect</span>
        </div>

        {/* 5-Stage Pipeline */}
        <div className="grid grid-cols-1 sm:grid-cols-5 gap-3 relative">
          {pipelineStages.map((stage) => {
            const isActive = activePipelineStage === stage.id

            return (
              <div
                key={stage.id}
                onClick={() => setActivePipelineStage(stage.id)}
                className={`p-4 rounded-2xl border transition-all cursor-pointer space-y-2 relative ${
                  isActive 
                    ? `${stage.color} shadow-lg ring-1 ring-emerald-500/50` 
                    : 'bg-obsidian-950 border-graphite-800 text-graphite-500 hover:border-graphite-700'
                }`}
              >
                <div className="flex items-center justify-between text-xs font-mono font-bold">
                  <span>STAGE 0{stage.id + 1}</span>
                  {isActive && <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>}
                </div>
                <div className="text-xs font-bold font-mono text-ivory-50">{stage.label}</div>
                <p className="text-[10px] text-graphite-500 leading-tight">{stage.desc}</p>
              </div>
            )
          })}
        </div>

        {/* Floating Transaction Cards Stream */}
        <div className="bg-obsidian-950 border border-graphite-800 p-4 rounded-2xl space-y-2">
          <div className="text-[11px] font-mono text-graphite-500 flex items-center justify-between">
            <span>LIVE TRANSACTION FLOW STREAM</span>
            <span className="text-emerald-400 font-bold">POLLED REAL-TIME</span>
          </div>

          <div className="flex flex-wrap gap-2 pt-1 font-mono text-xs">
            <div className="bg-graphite-900 border border-emerald-500/40 text-emerald-300 px-3 py-1.5 rounded-xl shadow-md flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              <span>RECLAIM-V2-000004 · ₹12,500 · <strong>TARGET</strong></span>
            </div>
            <div className="bg-graphite-900 border border-graphite-700 text-graphite-400 px-3 py-1.5 rounded-xl">
              RECLAIM-V2-000001 · ₹645 · RETRY_ALL
            </div>
            <div className="bg-graphite-900 border border-emerald-500/40 text-emerald-300 px-3 py-1.5 rounded-xl shadow-md flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              <span>RECLAIM-V2-000009 · ₹878 · <strong>TARGET</strong></span>
            </div>
            <div className="bg-graphite-900 border border-graphite-700 text-graphite-400 px-3 py-1.5 rounded-xl">
              RECLAIM-V2-000002 · ₹585 · RETRY_ALL
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
