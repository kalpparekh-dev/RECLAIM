import React, { useState, useEffect } from 'react'
import { 
  LayoutDashboard, 
  Search, 
  Cpu, 
  Activity, 
  PlayCircle, 
  History, 
  ShieldCheck,
  Settings,
  CheckCircle2,
  Bell,
  User,
  ChevronLeft,
  ChevronRight,
  LogOut,
  Sliders,
  Clock,
  Circle,
  CreditCard,
  Radio,
  Layers,
  FlaskConical,
  Server
} from 'lucide-react'

import Login from './components/Login'
import ExecutiveOverview from './components/ExecutiveOverview'
import TransactionExplorer from './components/TransactionExplorer'
import DecisionConsole from './components/DecisionConsole'
import MonitoringView from './components/MonitoringView'
import DemoWorkflow from './components/DemoWorkflow'
import HistoricalAnalytics from './components/HistoricalAnalytics'
import GovernanceDrawer from './components/GovernanceDrawer'
import PaymentOperations from './components/PaymentOperations'
import GatewaySandbox from './components/GatewaySandbox'
import FeedbackOutcomesView from './components/FeedbackOutcomesView'
import ExperimentsView from './components/ExperimentsView'
import SystemHealthView from './components/SystemHealthView'

const API_BASE = ''

export default function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [policySummary, setPolicySummary] = useState(null)
  const [isGovernanceOpen, setIsGovernanceOpen] = useState(false)
  const [selectedTxId, setSelectedTxId] = useState(null)
  const [currentTime, setCurrentTime] = useState('')

  useEffect(() => {
    fetchSummary()
    updateClock()
    const timer = setInterval(updateClock, 1000)
    return () => clearInterval(timer)
  }, [])

  const updateClock = () => {
    const now = new Date()
    setCurrentTime(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }))
  }

  const fetchSummary = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/policy/summary`)
      if (res.ok) {
        setPolicySummary(await res.json())
      }
    } catch (e) {
      console.error('Summary fetch failed', e)
    }
  }

  const navigateToConsoleWithTx = (txId) => {
    setSelectedTxId(txId)
    setActiveTab('console')
  }

  const navigateToDemoWithTx = (txId) => {
    setSelectedTxId(txId)
    setActiveTab('demo')
  }

  // Render Login Page if not logged in
  if (!isLoggedIn) {
    return <Login onLogin={() => setIsLoggedIn(true)} />
  }

  const navItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'explorer', label: 'Transactions', icon: Search },
    { id: 'console', label: 'Decision Console', icon: Cpu },
    { id: 'payment-ops', label: 'Payment Operations', icon: CreditCard },
    { id: 'gateway-sandbox', label: 'Gateway Sandbox', icon: Radio },
    { id: 'feedback-outcomes', label: 'Reconciliation & CSAT', icon: Layers },
    { id: 'experiments', label: 'A/B Experiments', icon: FlaskConical },
    { id: 'system-health', label: 'System Telemetry', icon: Server },
    { id: 'monitoring', label: 'Monitoring', icon: Activity },
    { id: 'demo', label: 'Demo Recovery', icon: PlayCircle, highlight: true },
    { id: 'historical', label: 'Historical Analytics', icon: History },
    { id: 'governance', label: 'Governance', icon: ShieldCheck },
    { id: 'settings', label: 'Settings', icon: Settings },
  ]

  return (
    <div className="min-h-screen bg-obsidian-950 text-ivory-50 flex font-sans selection:bg-emerald-500 selection:text-obsidian-950">
      {/* SIDEBAR NAVIGATION */}
      <aside 
        className={`bg-obsidian-900 border-r border-graphite-800/80 transition-all duration-300 z-30 flex flex-col justify-between sticky top-0 h-screen ${
          isSidebarCollapsed ? 'w-20' : 'w-64'
        }`}
      >
        <div className="p-4 space-y-6 overflow-y-auto max-h-[calc(100vh-80px)]">
          {/* Logo & Toggle */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3 overflow-hidden">
              <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-gold-500 p-0.5 shadow-md shrink-0">
                <div className="w-full h-full bg-obsidian-950 rounded-[10px] flex items-center justify-center">
                  <svg className="w-5 h-5 text-emerald-400 animate-spin-slow" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                    <path d="M3 3v5h5" />
                    <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
                    <path d="M16 16h5v5" />
                  </svg>
                </div>
              </div>
              {!isSidebarCollapsed && (
                <div>
                  <span className="font-extrabold text-xl tracking-wider text-ivory-50 font-mono">RECLAIM</span>
                  <span className="block text-[10px] text-graphite-500 font-mono font-medium">RECOVER MORE. RETRY SMARTER.</span>
                </div>
              )}
            </div>

            <button
              onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
              className="p-1.5 rounded-lg bg-graphite-800 text-graphite-500 hover:text-ivory-50 transition-colors hidden sm:block"
            >
              {isSidebarCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
            </button>
          </div>

          {/* Navigation Items */}
          <nav className="space-y-1 pt-2">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = activeTab === item.id

              if (item.id === 'governance') {
                return (
                  <button
                    key={item.id}
                    onClick={() => setIsGovernanceOpen(true)}
                    className={`w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs font-semibold transition-all ${
                      isSidebarCollapsed ? 'justify-center' : ''
                    } text-graphite-500 hover:text-ivory-50 hover:bg-graphite-800/60`}
                  >
                    <Icon className="w-4 h-4 text-emerald-400 shrink-0" />
                    {!isSidebarCollapsed && <span>Governance Panel</span>}
                  </button>
                )
              }

              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs font-semibold transition-all ${
                    isSidebarCollapsed ? 'justify-center' : ''
                  } ${
                    isActive
                      ? item.highlight
                        ? 'bg-gradient-to-r from-copper-600 to-gold-600 text-ivory-50 shadow-md shadow-copper-950 font-bold'
                        : 'bg-gradient-to-r from-emerald-600 to-emerald-700 text-obsidian-950 font-bold shadow-md shadow-emerald-950'
                      : 'text-graphite-500 hover:text-ivory-50 hover:bg-graphite-800/60'
                  }`}
                >
                  <Icon className={`w-4 h-4 shrink-0 ${isActive ? (item.highlight ? 'text-ivory-50' : 'text-obsidian-950') : ''}`} />
                  {!isSidebarCollapsed && <span>{item.label}</span>}
                </button>
              )
            })}
          </nav>
        </div>

        {/* Sidebar Footer User Info */}
        <div className="p-4 border-t border-graphite-800/80 space-y-3">
          {!isSidebarCollapsed && (
            <div className="bg-obsidian-950 p-3 rounded-xl border border-graphite-800 space-y-1">
              <div className="text-[11px] font-mono text-graphite-500">POLICY ENGINE</div>
              <div className="text-xs font-bold text-emerald-400 font-mono">V10.2 · TOP_RATE (5.0%)</div>
            </div>
          )}

          <button
            onClick={() => setIsLoggedIn(false)}
            className={`w-full flex items-center space-x-3 px-3.5 py-2 rounded-xl text-xs font-semibold text-copper-400 hover:bg-copper-950/40 transition-colors ${
              isSidebarCollapsed ? 'justify-center' : ''
            }`}
          >
            <LogOut className="w-4 h-4 shrink-0" />
            {!isSidebarCollapsed && <span>Sign Out Console</span>}
          </button>
        </div>
      </aside>

      {/* MAIN CONTENT AREA */}
      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        {/* TOP NAVIGATION BAR */}
        <header className="border-b border-graphite-800/80 bg-obsidian-900/90 backdrop-blur sticky top-0 z-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            {/* Left: Policy & Environment Badges */}
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2">
                <span className="bg-emerald-950 text-emerald-400 border border-emerald-800/80 text-[11px] font-mono px-3 py-1 rounded-full font-bold flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  POLICY V10.2
                </span>
                <span className="bg-graphite-800 text-gold-400 border border-gold-500/30 text-[11px] font-mono px-3 py-1 rounded-full font-semibold">
                  PRODUCTION POLICY
                </span>
              </div>
            </div>

            {/* Right: Status, Time, Profile */}
            <div className="flex items-center space-x-5">
              {/* Live Time */}
              <div className="hidden md:flex items-center space-x-1.5 text-xs text-graphite-500 font-mono">
                <Clock className="w-3.5 h-3.5 text-gold-400" />
                <span>{currentTime} IST</span>
              </div>

              {/* Status Indicator */}
              <div className="flex items-center space-x-2 bg-obsidian-950 border border-graphite-800 px-3 py-1.5 rounded-lg text-xs">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <span className="text-ivory-50 font-mono text-[11px]">SYSTEM OPERATIONAL</span>
              </div>

              {/* User Avatar Menu */}
              <div className="flex items-center space-x-2 pl-2 border-l border-graphite-800">
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-emerald-600 to-gold-500 p-0.5">
                  <div className="w-full h-full bg-obsidian-950 rounded-full flex items-center justify-center text-ivory-50 font-bold text-xs font-mono">
                    OP
                  </div>
                </div>
                <span className="hidden sm:inline text-xs font-semibold text-ivory-50 font-mono">Operator</span>
              </div>
            </div>
          </div>
        </header>

        {/* MAIN VIEW CONTENT */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full">
          {activeTab === 'overview' && (
            <ExecutiveOverview
              summary={policySummary}
              onNavigateConsole={navigateToConsoleWithTx}
              onNavigateExplorer={() => setActiveTab('explorer')}
            />
          )}

          {activeTab === 'explorer' && (
            <TransactionExplorer
              onSelectConsole={navigateToConsoleWithTx}
              onSelectDemo={navigateToDemoWithTx}
            />
          )}

          {activeTab === 'console' && (
            <DecisionConsole
              initialTxId={selectedTxId}
              onNavigateDemo={navigateToDemoWithTx}
            />
          )}

          {activeTab === 'payment-ops' && <PaymentOperations />}

          {activeTab === 'gateway-sandbox' && <GatewaySandbox />}

          {activeTab === 'feedback-outcomes' && <FeedbackOutcomesView />}

          {activeTab === 'experiments' && <ExperimentsView />}

          {activeTab === 'system-health' && <SystemHealthView />}

          {activeTab === 'monitoring' && <MonitoringView />}

          {activeTab === 'demo' && <DemoWorkflow initialTxId={selectedTxId} />}

          {activeTab === 'historical' && <HistoricalAnalytics />}

          {activeTab === 'settings' && (
            <div className="bg-graphite-900 border border-graphite-800 rounded-2xl p-8 max-w-2xl mx-auto space-y-4 font-sans">
              <h2 className="text-lg font-bold text-ivory-50 flex items-center gap-2">
                <Settings className="w-5 h-5 text-gold-400" />
                System Settings & Policy Parameters
              </h2>
              <div className="space-y-3 text-xs font-mono text-graphite-500">
                <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 justify-between flex items-center">
                  <span>Targeting Policy:</span>
                  <strong className="text-emerald-400">TOP_RATE (0.05 / 5%)</strong>
                </div>
                <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 justify-between flex items-center">
                  <span>Policy Dataset:</span>
                  <strong className="text-ivory-50">causal_uplift_v10_2_predictions.csv</strong>
                </div>
                <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 justify-between flex items-center">
                  <span>Policy Immutability:</span>
                  <strong className="text-emerald-400">LOCKED & READ-ONLY</strong>
                </div>
                <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 justify-between flex items-center">
                  <span>Gateway Mode:</span>
                  <strong className="text-gold-400">SIMULATOR SANDBOX / RAZORPAY ADAPTER</strong>
                </div>
              </div>
            </div>
          )}
        </main>

        {/* FOOTER */}
        <footer className="border-t border-graphite-800/80 bg-obsidian-950 py-4 text-center text-xs text-graphite-500 font-mono">
          <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
            <span>RECLAIM — Intelligent Payment Recovery & Decisioning Platform</span>
            <span className="text-gold-400">Production Policy V10.2</span>
            <button onClick={() => setIsGovernanceOpen(true)} className="text-emerald-400 hover:underline">
              Governance Directives
            </button>
          </div>
        </footer>
      </div>

      {/* Governance Slide-Over Drawer */}
      <GovernanceDrawer
        isOpen={isGovernanceOpen}
        onClose={() => setIsGovernanceOpen(false)}
      />
    </div>
  )
}
