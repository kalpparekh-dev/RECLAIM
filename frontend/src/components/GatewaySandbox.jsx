import React, { useState } from 'react'
import { 
  Play, 
  Terminal, 
  Send, 
  AlertOctagon, 
  CheckCircle2, 
  RefreshCw, 
  ShieldAlert, 
  FileCode, 
  Cpu,
  HelpCircle,
  Copy
} from 'lucide-react'

const API_BASE = ''

export default function GatewaySandbox() {
  // Create Payment State
  const [customerId, setCustomerId] = useState('CUST_9901')
  const [amount, setAmount] = useState('499.00')
  const [currency, setCurrency] = useState('INR')
  const [method, setMethod] = useState('CARD')
  const [initialOutcome, setInitialOutcome] = useState('FAILURE')
  const [failureReason, setFailureReason] = useState('BANK_SERVER_ERROR')

  // Result & Output Console
  const [lastPaymentResult, setLastPaymentResult] = useState(null)
  const [webhookPackage, setWebhookPackage] = useState(null)
  const [webhookResult, setWebhookResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleCreateSandboxPayment = async (e) => {
    e.preventDefault()
    setIsLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/gateway/sandbox/create-payment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          customer_id: customerId,
          amount: parseFloat(amount),
          currency,
          payment_method: method,
          initial_outcome: initialOutcome,
          failure_reason: failureReason
        })
      })
      if (res.ok) {
        const data = await res.json()
        setLastPaymentResult(data)
      }
    } catch (err) {
      console.error('Sandbox payment creation failed', err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleGenerateWebhook = async (tamperSig = false, alterBody = false) => {
    if (!lastPaymentResult?.id) return
    setIsLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/gateway/sandbox/send-signed-webhook`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          payment_id: lastPaymentResult.id,
          event_type: 'payment.failed',
          tamper_signature: tamperSig,
          alter_body: alterBody
        })
      })
      if (res.ok) {
        const data = await res.json()
        setWebhookPackage(data.simulator_package)
      }
    } catch (err) {
      console.error('Webhook generation failed', err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleEmitWebhook = async () => {
    if (!webhookPackage) return
    setIsLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/webhooks/simulator`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Razorpay-Signature': webhookPackage.signature
        },
        body: webhookPackage.raw_body
      })
      const data = await res.json()
      setWebhookResult({ status: res.status, data })
    } catch (err) {
      setWebhookResult({ status: 500, data: { error: str(err) } })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* SANDBOX DISCLAIMER BANNER */}
      <div className="bg-copper-950/60 border border-copper-800/80 p-5 rounded-2xl flex items-start gap-4">
        <AlertOctagon className="w-6 h-6 text-copper-400 shrink-0 mt-0.5" />
        <div>
          <span className="text-xs font-mono font-bold text-copper-400 uppercase tracking-wider">Gateway Sandbox & Webhook Test Harness</span>
          <h2 className="text-sm font-extrabold text-ivory-50 font-mono mt-0.5">LOCAL GATEWAY SIMULATOR MODE ACTIVE</h2>
          <p className="text-xs text-graphite-400 font-mono mt-1">
            This Sandbox environment simulates Razorpay payment lifecycle events, failure injection, signed HMAC webhooks, and retry recovery pipelines locally.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* LEFT COLUMN: PAYMENT CREATOR & OUTCOME SIMULATOR */}
        <div className="bg-obsidian-900 border border-graphite-800 rounded-2xl p-6 space-y-6">
          <h3 className="text-sm font-bold text-ivory-50 font-mono flex items-center gap-2 border-b border-graphite-800 pb-3">
            <Play className="w-4 h-4 text-emerald-400" />
            1. Sandbox Payment & Outcome Ingestion
          </h3>

          <form onSubmit={handleCreateSandboxPayment} className="space-y-4 text-xs font-mono">
            <div>
              <label className="text-graphite-400 block mb-1">Customer Identifier</label>
              <input
                type="text"
                value={customerId}
                onChange={(e) => setCustomerId(e.target.value)}
                className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl px-3 py-2 text-ivory-50 focus:border-emerald-500 focus:outline-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-graphite-400 block mb-1">Amount (INR)</label>
                <input
                  type="number"
                  step="0.01"
                  value={amount}
                  onChange={(e) => setAmount(e.target.value)}
                  className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl px-3 py-2 text-ivory-50 focus:border-emerald-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="text-graphite-400 block mb-1">Payment Method</label>
                <select
                  value={method}
                  onChange={(e) => setMethod(e.target.value)}
                  className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl px-3 py-2 text-ivory-50 focus:border-emerald-500 focus:outline-none"
                >
                  <option value="CARD">CARD</option>
                  <option value="UPI">UPI</option>
                  <option value="NETBANKING">NETBANKING</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-graphite-400 block mb-1">Initial Gateway Outcome</label>
                <select
                  value={initialOutcome}
                  onChange={(e) => setInitialOutcome(e.target.value)}
                  className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl px-3 py-2 text-ivory-50 focus:border-emerald-500 focus:outline-none"
                >
                  <option value="FAILURE">FAILURE (Triggers Recovery)</option>
                  <option value="SUCCESS">SUCCESS (Captured)</option>
                  <option value="PENDING">PENDING</option>
                </select>
              </div>
              <div>
                <label className="text-graphite-400 block mb-1">Failure Reason Code</label>
                <select
                  value={failureReason}
                  onChange={(e) => setFailureReason(e.target.value)}
                  className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl px-3 py-2 text-ivory-50 focus:border-emerald-500 focus:outline-none"
                >
                  <option value="BANK_SERVER_ERROR">BANK_SERVER_ERROR</option>
                  <option value="INSUFFICIENT_FUNDS">INSUFFICIENT_FUNDS</option>
                  <option value="GATEWAY_TIMEOUT">GATEWAY_TIMEOUT</option>
                  <option value="UPI_LIMIT_EXCEEDED">UPI_LIMIT_EXCEEDED</option>
                  <option value="NETWORK_ERROR">NETWORK_ERROR</option>
                  <option value="EXPIRED_CARD">EXPIRED_CARD</option>
                </select>
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3 bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-500 hover:to-emerald-600 text-obsidian-950 font-bold rounded-xl text-xs font-mono transition-colors shadow-lg shadow-emerald-950 flex items-center justify-center gap-2"
            >
              <Send className="w-4 h-4" /> Create Sandbox Payment & Trigger Failure Workflow
            </button>
          </form>

          {/* CREATED PAYMENT JSON OUTPUT */}
          {lastPaymentResult && (
            <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 space-y-2 text-xs font-mono">
              <div className="flex items-center justify-between text-emerald-400 font-bold">
                <span>Created Payment Record:</span>
                <span className="text-[10px] bg-emerald-950 text-emerald-300 px-2 py-0.5 rounded border border-emerald-800">
                  ID: {lastPaymentResult.id}
                </span>
              </div>
              <pre className="text-[11px] text-graphite-300 overflow-x-auto p-2 bg-obsidian-900 rounded border border-graphite-800/60 max-h-48">
                {JSON.stringify(lastPaymentResult, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN: WEBHOOK EMITTER & SIGNATURE TESTER */}
        <div className="bg-obsidian-900 border border-graphite-800 rounded-2xl p-6 space-y-6">
          <h3 className="text-sm font-bold text-ivory-50 font-mono flex items-center gap-2 border-b border-graphite-800 pb-3">
            <Terminal className="w-4 h-4 text-gold-400" />
            2. Webhook Ingestion & HMAC Verification Tester
          </h3>

          <div className="space-y-4 text-xs font-mono">
            <p className="text-graphite-400">
              Generate signed Razorpay webhook payloads and test signature validation security against the backend.
            </p>

            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => handleGenerateWebhook(false, false)}
                disabled={!lastPaymentResult || isLoading}
                className="py-2.5 bg-graphite-800 hover:bg-graphite-700 disabled:opacity-40 text-ivory-50 rounded-xl font-semibold transition-colors text-[11px]"
              >
                Generate Valid Webhook
              </button>
              <button
                onClick={() => handleGenerateWebhook(true, false)}
                disabled={!lastPaymentResult || isLoading}
                className="py-2.5 bg-copper-950 hover:bg-copper-900 text-copper-400 border border-copper-800/80 disabled:opacity-40 rounded-xl font-semibold transition-colors text-[11px]"
              >
                Generate Invalid Signature
              </button>
            </div>

            {webhookPackage && (
              <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 space-y-3">
                <div className="flex items-center justify-between text-gold-400 font-bold">
                  <span>Generated Webhook Package:</span>
                  <span className="text-[10px] bg-gold-950 text-gold-300 px-2 py-0.5 rounded border border-gold-800">
                    HMAC SHA256 Ready
                  </span>
                </div>
                <div className="text-[11px]">
                  <span className="text-graphite-500 block">X-Razorpay-Signature:</span>
                  <code className="text-emerald-400 block bg-obsidian-900 p-1.5 rounded border border-graphite-800 break-all text-[10px]">
                    {webhookPackage.signature}
                  </code>
                </div>

                <button
                  onClick={handleEmitWebhook}
                  disabled={isLoading}
                  className="w-full py-2.5 bg-gold-600 hover:bg-gold-500 text-obsidian-950 font-bold rounded-xl text-xs transition-colors flex items-center justify-center gap-2"
                >
                  <Send className="w-3.5 h-3.5" /> Emit Signed Webhook to /api/webhooks/simulator
                </button>
              </div>
            )}

            {/* WEBHOOK RESPONSE OUTPUT */}
            {webhookResult && (
              <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 space-y-2">
                <div className="flex items-center justify-between text-ivory-50 font-bold">
                  <span>Webhook Receiver Response:</span>
                  <span className={`text-[10px] px-2 py-0.5 rounded font-mono ${webhookResult.status === 200 ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-copper-950 text-copper-400 border border-copper-800'}`}>
                    HTTP {webhookResult.status}
                  </span>
                </div>
                <pre className="text-[11px] text-graphite-300 overflow-x-auto p-2 bg-obsidian-900 rounded border border-graphite-800/60">
                  {JSON.stringify(webhookResult.data, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
