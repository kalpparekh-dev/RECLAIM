import React, { useState, useEffect } from 'react'
import { 
  X, 
  CreditCard, 
  Clock, 
  CheckCircle2, 
  AlertTriangle, 
  ShieldCheck, 
  Cpu, 
  RefreshCw, 
  User, 
  FileText, 
  Zap,
  Star
} from 'lucide-react'

const API_BASE = ''

export default function PaymentDetailModal({ paymentId, onClose }) {
  const [timelineData, setTimelineData] = useState(null)
  const [paymentDetail, setPaymentDetail] = useState(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    fetchPaymentData()
  }, [paymentId])

  const fetchPaymentData = async () => {
    setIsLoading(true)
    try {
      const [resPay, resTimeline] = await Promise.all([
        fetch(`${API_BASE}/api/payments/${paymentId}`),
        fetch(`${API_BASE}/api/payments/${paymentId}/timeline`)
      ])
      if (resPay.ok) setPaymentDetail(await resPay.json())
      if (resTimeline.ok) setTimelineData(await resTimeline.json())
    } catch (e) {
      console.error('Timeline fetch failed', e)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 bg-obsidian-950/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-obsidian-900 border border-graphite-800 rounded-2xl max-w-4xl w-full max-h-[90vh] flex flex-col overflow-hidden shadow-2xl">
        {/* MODAL HEADER */}
        <div className="p-6 border-b border-graphite-800 flex items-center justify-between bg-obsidian-950">
          <div>
            <span className="text-xs font-mono text-emerald-400 font-bold uppercase tracking-wider">Payment Lifecycle & Event Audit</span>
            <h2 className="text-lg font-extrabold text-ivory-50 font-mono mt-0.5">{paymentId}</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl bg-graphite-800 text-graphite-400 hover:text-ivory-50 hover:bg-graphite-700 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* MODAL BODY */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1">
          {isLoading ? (
            <div className="py-12 text-center text-graphite-500 font-mono text-xs flex flex-col items-center gap-2">
              <RefreshCw className="w-6 h-6 animate-spin text-emerald-400" />
              Loading payment lifecycle timeline...
            </div>
          ) : (
            <>
              {/* PAYMENT OVERVIEW CARD */}
              {paymentDetail && (
                <div className="bg-obsidian-950 p-5 rounded-xl border border-graphite-800 grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
                  <div>
                    <span className="text-graphite-500 block text-[10px]">CUSTOMER ID</span>
                    <strong className="text-emerald-400 font-semibold">{paymentDetail.masked_customer_id || paymentDetail.customer_id}</strong>
                  </div>
                  <div>
                    <span className="text-graphite-500 block text-[10px]">AMOUNT & METHOD</span>
                    <strong className="text-ivory-50 font-bold">{paymentDetail.currency} {paymentDetail.amount.toFixed(2)} ({paymentDetail.payment_method})</strong>
                  </div>
                  <div>
                    <span className="text-graphite-500 block text-[10px]">CURRENT STATUS</span>
                    <strong className="text-gold-400 font-bold">{paymentDetail.status}</strong>
                  </div>
                  <div>
                    <span className="text-graphite-500 block text-[10px]">FAILURE REASON</span>
                    <strong className="text-copper-400 font-semibold">{paymentDetail.failure_reason || 'None'}</strong>
                  </div>
                </div>
              )}

              {/* DECISION METADATA CALLOUT */}
              {paymentDetail?.recovery_action && (
                <div className="bg-graphite-900/90 p-4 rounded-xl border border-emerald-500/30 space-y-2 text-xs font-mono">
                  <div className="flex items-center justify-between border-b border-graphite-800 pb-2">
                    <span className="text-emerald-400 font-bold flex items-center gap-1.5">
                      <Cpu className="w-4 h-4" /> RECLAIM Decision Engine Metadata
                    </span>
                    <span className="text-[10px] bg-emerald-950 text-emerald-300 px-2 py-0.5 rounded border border-emerald-800">
                      POLICY {paymentDetail.recovery_action.policy_version}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-[11px]">
                    <div>Decision: <strong className="text-ivory-50">{paymentDetail.recovery_action.decision}</strong></div>
                    <div>Score: <strong className="text-ivory-50">{paymentDetail.recovery_action.model_score.toFixed(4)}</strong></div>
                    <div>Semantics: <strong className="text-gold-400">{paymentDetail.recovery_action.score_semantics}</strong></div>
                  </div>
                </div>
              )}

              {/* TIMELINE EVENTS LIST */}
              <div className="space-y-4">
                <h3 className="text-xs font-mono text-graphite-500 uppercase tracking-wider font-bold">
                  Chronological State Machine Timeline
                </h3>

                <div className="relative pl-6 space-y-6 before:absolute before:left-2.5 before:top-2 before:bottom-2 before:w-0.5 before:bg-graphite-800">
                  {timelineData?.timeline?.map((item, idx) => (
                    <div key={idx} className="relative flex items-start space-x-4 group">
                      {/* NODE ICON */}
                      <div className="absolute -left-6 top-0.5 w-5 h-5 rounded-full bg-obsidian-950 border-2 border-emerald-500 flex items-center justify-center text-emerald-400 shadow-md">
                        <CheckCircle2 className="w-3 h-3" />
                      </div>

                      {/* EVENT DETAIL CONTENT */}
                      <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 flex-1 space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-ivory-50 font-mono">{item.title}</span>
                          <span className="text-[10px] font-mono text-graphite-500">{new Date(item.timestamp).toLocaleString()}</span>
                        </div>
                        <p className="text-xs text-graphite-400 font-mono">{item.detail}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>

        {/* MODAL FOOTER */}
        <div className="p-4 border-t border-graphite-800 bg-obsidian-950 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-graphite-800 hover:bg-graphite-700 text-ivory-50 rounded-xl text-xs font-mono font-bold transition-colors"
          >
            Close Audit Timeline
          </button>
        </div>
      </div>
    </div>
  )
}
