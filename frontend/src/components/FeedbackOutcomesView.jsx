import React, { useState, useEffect } from 'react'
import { CheckCircle2, AlertTriangle, MessageSquare, Star, RefreshCw, Layers, Clock, ShieldAlert, Plus, Send } from 'lucide-react'

export default function FeedbackOutcomesView() {
  const [outcomes, setOutcomes] = useState([])
  const [summary, setSummary] = useState(null)
  const [feedbackList, setFeedbackList] = useState([])
  const [loading, setLoading] = useState(true)
  const [showFeedbackModal, setShowFeedbackModal] = useState(false)
  
  // Feedback form state
  const [formTxId, setFormTxId] = useState('')
  const [formCustomerId, setFormCustomerId] = useState('')
  const [formRating, setFormRating] = useState(5)
  const [formCategory, setFormCategory] = useState('SURVEY_CSAT')
  const [formComment, setFormComment] = useState('')
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    setLoading(true)
    try {
      const [outcomesRes, feedbackRes] = await Promise.all([
        fetch('/api/recovery/outcomes'),
        fetch('/api/recovery/feedback')
      ])

      if (outcomesRes.ok) {
        const data = await outcomesRes.json()
        setOutcomes(data.outcomes || [])
        setSummary(data.summary || null)
      }

      if (feedbackRes.ok) {
        const fData = await feedbackRes.json()
        setFeedbackList(fData.feedback_entries || [])
      }
    } catch (err) {
      console.error('Failed to fetch outcomes/feedback data:', err)
    } font-sans finally {
      setLoading(false)
    }
  }

  const handleReconcileAll = async () => {
    setLoading(true)
    try {
      await fetch('/api/recovery/reconcile', { method: 'POST' })
      await fetchData()
    } catch (err) {
      console.error('Reconciliation failed:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmitFeedback = async (e) => {
    e.preventDefault()
    if (!formTxId || !formCustomerId) return
    setSubmitting(true)
    try {
      const res = await fetch('/api/recovery/feedback/customer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transaction_id: formTxId,
          customer_id: formCustomerId,
          rating: parseInt(formRating, 10),
          feedback_category: formCategory,
          comment: formComment
        })
      })
      if (res.ok) {
        setShowFeedbackModal(false)
        setFormTxId('')
        setFormCustomerId('')
        setFormComment('')
        fetchData()
      }
    } catch (err) {
      console.error('Error submitting feedback:', err)
    } finally {
      setSubmitting(false)
    }
  }

  const getStatusBadge = (status) => {
    switch (status) {
      case 'MATCHED':
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-emerald-950 text-emerald-400 border border-emerald-800 flex items-center gap-1"><CheckCircle2 className="w-3 h-3"/> MATCHED</span>
      case 'PENDING':
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-gold-950 text-gold-400 border border-gold-800/80 flex items-center gap-1"><Clock className="w-3 h-3"/> PENDING</span>
      case 'MISMATCHED':
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-copper-950 text-copper-400 border border-copper-800 flex items-center gap-1"><AlertTriangle className="w-3 h-3"/> MISMATCHED</span>
      case 'MANUAL_REVIEW':
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-copper-900 text-copper-300 border border-copper-700 flex items-center gap-1"><ShieldAlert className="w-3 h-3"/> MANUAL REVIEW</span>
      default:
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-graphite-800 text-graphite-400 border border-graphite-700">UNKNOWN</span>
    }
  }

  return (
    <div className="space-y-6">
      {/* HEADER BAR */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-obsidian-900 p-6 rounded-2xl border border-graphite-800">
        <div>
          <h1 className="text-xl font-extrabold text-ivory-50 flex items-center gap-2 font-mono">
            <Layers className="w-6 h-6 text-gold-400" />
            Outcome Reconciliation & CSAT Feedback
          </h1>
          <p className="text-xs text-graphite-500 mt-1">
            Real-time multi-channel outcome matching, webhook verification & customer experience metrics
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={fetchData}
            className="p-2.5 rounded-xl bg-graphite-800 text-graphite-400 hover:text-ivory-50 border border-graphite-700 transition-colors"
            title="Refresh Data"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>

          <button
            onClick={handleReconcileAll}
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-emerald-700 text-obsidian-950 font-bold text-xs font-mono shadow-md shadow-emerald-950/40 hover:brightness-110 transition-all flex items-center gap-2"
          >
            <CheckCircle2 className="w-4 h-4" />
            Run Reconciliation
          </button>

          <button
            onClick={() => setShowFeedbackModal(true)}
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-copper-600 to-gold-600 text-ivory-50 font-bold text-xs font-mono shadow-md shadow-copper-950/40 hover:brightness-110 transition-all flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Log Feedback
          </button>
        </div>
      </div>

      {/* METRIC CARDS */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-graphite-900 border border-graphite-800 p-5 rounded-2xl space-y-2">
          <div className="flex items-center justify-between text-xs text-graphite-500 font-mono">
            <span>TOTAL RECONCILED</span>
            <Layers className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-ivory-50">
            {summary ? summary.total_reconciled : 0}
          </div>
          <div className="text-[11px] text-emerald-400 font-mono">
            {summary ? `${summary.matched_count} fully matched` : '0 matched'}
          </div>
        </div>

        <div className="bg-graphite-900 border border-graphite-800 p-5 rounded-2xl space-y-2">
          <div className="flex items-center justify-between text-xs text-graphite-500 font-mono">
            <span>RECONCILIATION MATCH RATE</span>
            <CheckCircle2 className="w-4 h-4 text-gold-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-gold-400">
            {summary ? `${summary.match_rate_pct}%` : '0%'}
          </div>
          <div className="text-[11px] text-graphite-500 font-mono">
            Verified across webhook & API channels
          </div>
        </div>

        <div className="bg-graphite-900 border border-graphite-800 p-5 rounded-2xl space-y-2">
          <div className="flex items-center justify-between text-xs text-graphite-500 font-mono">
            <span>AVG CSAT SCORE</span>
            <Star className="w-4 h-4 text-copper-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-copper-400 flex items-center gap-1">
            {summary ? summary.avg_customer_rating : '5.0'}
            <span className="text-xs text-graphite-500 font-normal">/ 5.0</span>
          </div>
          <div className="text-[11px] text-graphite-500 font-mono">
            {feedbackList.length} feedback submissions
          </div>
        </div>

        <div className="bg-graphite-900 border border-graphite-800 p-5 rounded-2xl space-y-2">
          <div className="flex items-center justify-between text-xs text-graphite-500 font-mono">
            <span>PENDING / REVIEW</span>
            <AlertTriangle className="w-4 h-4 text-copper-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-ivory-50">
            {summary ? (summary.pending_count + summary.mismatched_count + summary.manual_review_count) : 0}
          </div>
          <div className="text-[11px] font-mono text-copper-400">
            {summary ? `${summary.manual_review_count} flagged manual review` : '0 manual review'}
          </div>
        </div>
      </div>

      {/* OUTCOME RECONCILIATION TABLE */}
      <div className="bg-graphite-900 border border-graphite-800 rounded-2xl overflow-hidden">
        <div className="p-5 border-b border-graphite-800 flex items-center justify-between">
          <h2 className="text-sm font-bold font-mono text-ivory-50 flex items-center gap-2">
            <Layers className="w-4 h-4 text-emerald-400" />
            RECONCILIATION LEDGER
          </h2>
          <span className="text-xs text-graphite-500 font-mono">
            {outcomes.length} records
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-graphite-800 bg-obsidian-950 text-[11px] font-mono text-graphite-500 uppercase">
                <th className="p-3.5">Payment ID</th>
                <th className="p-3.5">Status</th>
                <th className="p-3.5">System State</th>
                <th className="p-3.5">Gateway Event</th>
                <th className="p-3.5">Discrepancy</th>
                <th className="p-3.5 text-right">Reconciled At</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-graphite-800/60 text-xs font-mono">
              {outcomes.length === 0 ? (
                <tr>
                  <td colSpan="6" className="p-8 text-center text-graphite-500">
                    No reconciliation records found. Click "Run Reconciliation" to match transactions.
                  </td>
                </tr>
              ) : (
                outcomes.map((item) => (
                  <tr key={item.id} className="hover:bg-obsidian-950/50 transition-colors">
                    <td className="p-3.5 text-emerald-400 font-bold">{item.payment_id}</td>
                    <td className="p-3.5">{getStatusBadge(item.status)}</td>
                    <td className="p-3.5 text-ivory-50">{item.system_state || 'N/A'}</td>
                    <td className="p-3.5 text-graphite-400">{item.gateway_status || 'N/A'}</td>
                    <td className="p-3.5 text-copper-400">{item.discrepancy_details || 'None'}</td>
                    <td className="p-3.5 text-right text-graphite-500">
                      {item.reconciled_at ? new Date(item.reconciled_at).toLocaleTimeString() : 'N/A'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* CUSTOMER FEEDBACK SECTION */}
      <div className="bg-graphite-900 border border-graphite-800 rounded-2xl p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-graphite-800 pb-4">
          <h2 className="text-sm font-bold font-mono text-ivory-50 flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-gold-400" />
            CUSTOMER EXPERIENCE FEEDBACK (CSAT)
          </h2>
          <span className="text-xs font-mono text-graphite-500">
            {feedbackList.length} feedback entries
          </span>
        </div>

        {feedbackList.length === 0 ? (
          <div className="p-6 text-center text-graphite-500 text-xs font-mono">
            No customer feedback submitted yet.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {feedbackList.map((item) => (
              <div key={item.id} className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1 text-gold-400">
                    {[...Array(5)].map((_, i) => (
                      <Star
                        key={i}
                        className={`w-3.5 h-3.5 ${i < item.rating ? 'fill-gold-400 text-gold-400' : 'text-graphite-700'}`}
                      />
                    ))}
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-graphite-800 text-graphite-400 border border-graphite-700">
                    {item.feedback_category}
                  </span>
                </div>
                <div className="text-xs text-ivory-50 italic font-sans">
                  "{item.comment || 'No comment provided'}"
                </div>
                <div className="flex items-center justify-between text-[10px] font-mono text-graphite-500 pt-2 border-t border-graphite-800/80">
                  <span>Tx: <strong className="text-emerald-400">{item.transaction_id}</strong></span>
                  <span>Customer: {item.customer_id}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* MODAL: SUBMIT FEEDBACK */}
      {showFeedbackModal && (
        <div className="fixed inset-0 z-50 bg-obsidian-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-graphite-900 border border-graphite-800 w-full max-w-md rounded-2xl p-6 space-y-5 shadow-2xl">
            <div className="flex items-center justify-between border-b border-graphite-800 pb-3">
              <h3 className="text-sm font-bold font-mono text-ivory-50 flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-gold-400" />
                Submit Customer CSAT Feedback
              </h3>
              <button
                onClick={() => setShowFeedbackModal(false)}
                className="text-graphite-500 hover:text-ivory-50 text-xs font-mono"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleSubmitFeedback} className="space-y-4 text-xs font-mono">
              <div className="space-y-1">
                <label className="text-graphite-400">Transaction ID</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. TX_REC_000001"
                  value={formTxId}
                  onChange={(e) => setFormTxId(e.target.value)}
                  className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl px-3 py-2 text-ivory-50 focus:border-emerald-500 outline-none"
                />
              </div>

              <div className="space-y-1">
                <label className="text-graphite-400">Customer ID</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. CUST_0042"
                  value={formCustomerId}
                  onChange={(e) => setFormCustomerId(e.target.value)}
                  className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl px-3 py-2 text-ivory-50 focus:border-emerald-500 outline-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-graphite-400">Rating (1 to 5)</label>
                  <select
                    value={formRating}
                    onChange={(e) => setFormRating(e.target.value)}
                    className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl px-3 py-2 text-ivory-50 focus:border-emerald-500 outline-none"
                  >
                    <option value={5}>5 Stars - Excellent</option>
                    <option value={4}>4 Stars - Good</option>
                    <option value={3}>3 Stars - Neutral</option>
                    <option value={2}>2 Stars - Poor</option>
                    <option value={1}>1 Star - Dissatisfied</option>
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="text-graphite-400">Category</label>
                  <select
                    value={formCategory}
                    onChange={(e) => setFormCategory(e.target.value)}
                    className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl px-3 py-2 text-ivory-50 focus:border-emerald-500 outline-none"
                  >
                    <option value="SURVEY_CSAT">SURVEY_CSAT</option>
                    <option value="UNSOLICITED_COMPLAINT">UNSOLICITED_COMPLAINT</option>
                    <option value="GATEWAY_DESERIALIZATION_FEEDBACK">GATEWAY_FEEDBACK</option>
                    <option value="RECOVERY_DELAYS">RECOVERY_DELAYS</option>
                  </select>
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-graphite-400">Comments</label>
                <textarea
                  rows={3}
                  placeholder="Details about recovery experience..."
                  value={formComment}
                  onChange={(e) => setFormComment(e.target.value)}
                  className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl px-3 py-2 text-ivory-50 focus:border-emerald-500 outline-none"
                />
              </div>

              <div className="flex items-center justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowFeedbackModal(false)}
                  className="px-4 py-2 rounded-xl bg-graphite-800 text-graphite-400 hover:text-ivory-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  className="px-4 py-2 rounded-xl bg-gradient-to-r from-emerald-600 to-emerald-700 text-obsidian-950 font-bold flex items-center gap-2 hover:brightness-110"
                >
                  <Send className="w-3.5 h-3.5" />
                  {submitting ? 'Submitting...' : 'Submit Feedback'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
