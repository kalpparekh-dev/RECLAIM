import React, { useState } from 'react'
import { 
  Shield, 
  Lock, 
  ArrowRight, 
  Eye, 
  EyeOff, 
  CheckCircle2, 
  Key, 
  TrendingUp, 
  CircleDollarSign,
  Activity,
  Layers
} from 'lucide-react'

export default function Login({ onLogin }) {
  const [email, setEmail] = useState('operator@razorpay.hackathon')
  const [password, setPassword] = useState('••••••••••••')
  const [showPassword, setShowPassword] = useState(false)
  const [rememberMe, setRememberMe] = useState(true)
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = (e) => {
    e.preventDefault()
    setIsLoading(true)
    setTimeout(() => {
      setIsLoading(false)
      onLogin()
    }, 600)
  }

  return (
    <div className="min-h-screen bg-obsidian-950 text-ivory-50 flex flex-col lg:flex-row relative overflow-hidden font-sans selection:bg-emerald-500 selection:text-obsidian-950">
      {/* Background Ambient Lights (Emerald & Gold Only) */}
      <div className="absolute top-1/4 left-1/6 w-96 h-96 bg-emerald-500/10 rounded-full blur-[120px] pointer-events-none animate-pulse-glow"></div>
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-gold-500/10 rounded-full blur-[120px] pointer-events-none animate-pulse-glow" style={{ animationDelay: '2s' }}></div>

      {/* LEFT SIDE: Animated 3D Financial Intelligence Visual */}
      <div className="lg:w-7/12 p-8 lg:p-16 flex flex-col justify-between relative z-10 border-b lg:border-b-0 lg:border-r border-graphite-800/60 bg-gradient-to-br from-obsidian-950 via-obsidian-900 to-obsidian-950">
        {/* Brand Header */}
        <div className="flex items-center space-x-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-gold-500 p-0.5 shadow-lg shadow-emerald-950">
            <div className="w-full h-full bg-obsidian-950 rounded-[10px] flex items-center justify-center">
              {/* Interlocking Recovery Arrows Symbol */}
              <svg className="w-5 h-5 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
                <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
                <path d="M16 16h5v5" />
              </svg>
            </div>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-black text-2xl tracking-wider text-ivory-50 font-mono">RECLAIM</span>
              <span className="bg-emerald-950 text-emerald-400 border border-emerald-800/80 text-[10px] font-mono px-2 py-0.5 rounded-full font-bold">
                POLICY V10.2
              </span>
            </div>
            <p className="text-xs text-graphite-500 font-medium">Payment-Recovery Causal Decision Engine</p>
          </div>
        </div>

        {/* 3D Visual Control Center Workspace */}
        <div className="my-12 relative flex items-center justify-center min-h-[380px]">
          {/* Rotating Financial Outer Ring */}
          <div className="absolute w-80 h-80 sm:w-96 sm:h-96 rounded-full border border-dashed border-emerald-500/20 animate-spin-slow flex items-center justify-center">
            <div className="w-64 h-64 sm:w-80 sm:h-80 rounded-full border border-gold-500/20"></div>
          </div>

          {/* Connected Financial Nodes & Currency Symbols */}
          <div className="relative z-10 space-y-6 text-center">
            {/* Center 3D Coin Stack Graphic */}
            <div className="relative mx-auto w-36 h-36 flex items-center justify-center">
              <div className="absolute inset-0 bg-gradient-to-tr from-emerald-500/20 to-gold-500/20 rounded-full blur-xl animate-pulse"></div>
              
              {/* Metallic 3D Floating Coin */}
              <div className="w-28 h-28 rounded-full bg-gradient-to-tr from-gold-600 via-gold-400 to-amber-200 p-1 shadow-2xl shadow-gold-500/30 transform rotate-12 animate-float flex items-center justify-center">
                <div className="w-full h-full rounded-full bg-gradient-to-br from-obsidian-900 via-graphite-900 to-obsidian-950 flex flex-col items-center justify-center border border-gold-400/40">
                  <span className="text-3xl font-black text-gold-400 font-mono">₹</span>
                  <span className="text-[9px] font-mono text-emerald-400 font-bold tracking-widest uppercase">RECOVERY</span>
                </div>
              </div>

              {/* Floating Orbiting Currency Nodes */}
              <div className="absolute -top-3 -right-3 bg-graphite-900 border border-emerald-500/40 text-emerald-400 w-9 h-9 rounded-xl flex items-center justify-center text-sm font-bold shadow-lg animate-float" style={{ animationDelay: '1s' }}>
                $
              </div>
              <div className="absolute -bottom-2 -left-3 bg-graphite-900 border border-gold-500/40 text-gold-400 w-9 h-9 rounded-xl flex items-center justify-center text-sm font-bold shadow-lg animate-float" style={{ animationDelay: '2s' }}>
                €
              </div>
            </div>

            {/* Headline and Value Proposition */}
            <div className="max-w-md mx-auto space-y-2 pt-4">
              <h2 className="text-2xl sm:text-3xl font-extrabold text-ivory-50 tracking-tight">
                Recover More. <span className="text-emerald-400">Retry Smarter.</span>
              </h2>
              <p className="text-xs sm:text-sm text-graphite-500 leading-relaxed">
                Autonomous decision intelligence identifying the top 5% highest-uplift payment recovery opportunities in real time.
              </p>
            </div>

            {/* Floating Live Transaction Badges */}
            <div className="flex flex-wrap justify-center gap-2 pt-2 text-[11px] font-mono">
              <div className="bg-obsidian-900/90 border border-emerald-500/30 text-emerald-300 px-3 py-1.5 rounded-lg flex items-center space-x-1.5 shadow-md">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
                <span>TXN-8842 · ₹14,500 · <strong>TARGET (TOP 5%)</strong></span>
              </div>
              <div className="bg-obsidian-900/90 border border-graphite-700 text-ivory-200/70 px-3 py-1.5 rounded-lg flex items-center space-x-1.5 shadow-md">
                <span>TXN-8843 · ₹2,100 · RETRY_ALL</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer Security Badges */}
        <div className="flex items-center justify-between text-xs text-graphite-500 border-t border-graphite-800/60 pt-6">
          <div className="flex items-center space-x-2">
            <Shield className="w-4 h-4 text-emerald-400" />
            <span>256-Bit Immutable Decision Vault</span>
          </div>
          <span className="font-mono text-[11px]">Razorpay Hackathon Edition</span>
        </div>
      </div>

      {/* RIGHT SIDE: Premium Login Form Card */}
      <div className="lg:w-5/12 p-8 lg:p-16 flex items-center justify-center relative z-10 bg-obsidian-950">
        <div className="w-full max-w-md space-y-8">
          <div className="space-y-2">
            <span className="text-xs font-mono font-semibold text-emerald-400 uppercase tracking-widest">
              Secure Operations Access
            </span>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-ivory-50 tracking-tight">
              Command Center Sign In
            </h1>
            <p className="text-xs sm:text-sm text-graphite-500">
              Enter your credentials to access the production policy decision engine.
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-ivory-200">Work Email / Username</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full bg-graphite-900 border border-graphite-700 rounded-xl px-4 py-3 text-xs text-ivory-50 placeholder-graphite-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-colors font-mono"
              />
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <label className="block text-xs font-medium text-ivory-200">Password</label>
                <a href="#forgot" onClick={(e) => e.preventDefault()} className="text-[11px] text-emerald-400 hover:underline">
                  Forgot password?
                </a>
              </div>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="w-full bg-graphite-900 border border-graphite-700 rounded-xl pl-4 pr-10 py-3 text-xs text-ivory-50 placeholder-graphite-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-colors font-mono"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-3.5 text-graphite-500 hover:text-ivory-200 transition-colors"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between text-xs">
              <label className="flex items-center space-x-2 cursor-pointer text-graphite-500">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="rounded bg-graphite-900 border-graphite-700 text-emerald-500 focus:ring-emerald-500 w-4 h-4"
                />
                <span className="text-ivory-200 text-xs">Remember this console device</span>
              </label>
            </div>

            {/* Primary Sign In Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-gradient-to-r from-emerald-600 via-emerald-500 to-emerald-600 hover:from-emerald-500 hover:to-emerald-400 text-obsidian-950 font-extrabold py-3.5 rounded-xl shadow-lg shadow-emerald-950 transition-all text-xs tracking-wider uppercase flex items-center justify-center space-x-2"
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-obsidian-950 border-t-transparent rounded-full animate-spin"></div>
              ) : (
                <>
                  <span>Sign In To Command Center</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>

            {/* Direct Demo Access Shortcut Button */}
            <div className="pt-2">
              <button
                type="button"
                onClick={onLogin}
                className="w-full bg-graphite-900 hover:bg-graphite-800 text-gold-400 border border-gold-500/30 font-semibold py-3 rounded-xl transition-all text-xs flex items-center justify-center space-x-2"
              >
                <Key className="w-4 h-4 text-gold-400" />
                <span>Launch Instant Demo Access (No Credentials Required)</span>
              </button>
            </div>
          </form>

          {/* Governance Notice */}
          <div className="bg-graphite-900/60 border border-graphite-800 p-4 rounded-xl space-y-1">
            <div className="flex items-center space-x-1.5 text-xs text-emerald-400 font-bold">
              <CheckCircle2 className="w-4 h-4" />
              <span>Immutable Production Policy V10.2</span>
            </div>
            <p className="text-[11px] text-graphite-500 leading-normal">
              Targeting rate locked at 5.0% (500/10,000 txns). All live decision data is read-only.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
