import React, { useState, useEffect } from 'react'
import { 
  CreditCard, 
  Search, 
  Filter, 
  RefreshCw, 
  ChevronLeft, 
  ChevronRight, 
  Eye, 
  AlertTriangle, 
  CheckCircle2, 
  Clock, 
  ArrowUpRight,
  ShieldAlert,
  Play
} from 'lucide-react'
import PaymentDetailModal from './PaymentDetailModal'

const API_BASE = ''

export default function PaymentOperations() {
  const [payments, setPayments] = useState([])
  const [pagination, setPagination] = useState({ page: 1, limit: 15, total_pages: 1, total_matching: 0 })
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [methodFilter, setMethodFilter] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [selectedPaymentId, setSelectedPaymentId] = useState(null)

  useEffect(() => {
    fetchPayments(1)
  }, [statusFilter, methodFilter])

  const fetchPayments = async (page = 1) => {
    setIsLoading(true)
    try {
      let url = `${API_BASE}/api/payments?page=${page}&limit=15`
      if (search) url += `&search=${encodeURIComponent(search)}`
      if (statusFilter) url += `&status=${encodeURIComponent(statusFilter)}`
      if (methodFilter) url += `&payment_method=${encodeURIComponent(methodFilter)}`

      const res = await fetch(url)
      if (res.ok) {
        const data = await res.json()
        setPayments(data.payments || [])
        setPagination(data.pagination || { page: 1, limit: 15, total_pages: 1, total_matching: 0 })
      }
    } catch (e) {
      console.error('Failed fetching payments', e)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSearchSubmit = (e) => {
    e.preventDefault()
    fetchPayments(1)
  }

  const getStatusBadge = (status) => {
    switch (status) {
      case 'RECOVERED':
      case 'CAPTURED':
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-emerald-950 text-emerald-400 border border-emerald-800 flex items-center gap-1 w-fit"><CheckCircle2 className="w-3 h-3" /> {status}</span>
      case 'FAILED':
      case 'RECOVERY_FAILED':
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-copper-950 text-copper-400 border border-copper-800 flex items-center gap-1 w-fit"><AlertTriangle className="w-3 h-3" /> {status}</span>
      case 'RETRY_SCHEDULED':
      case 'RETRY_PROCESSING':
      case 'RECOVERY_ELIGIBLE':
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-gold-950 text-gold-400 border border-gold-800/80 flex items-center gap-1 w-fit"><RefreshCw className="w-3 h-3 animate-spin-slow" /> {status}</span>
      case 'SUPPRESSED':
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-graphite-800 text-graphite-400 border border-graphite-700 flex items-center gap-1 w-fit"><ShieldAlert className="w-3 h-3" /> {status}</span>
      default:
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-graphite-900 text-ivory-50 border border-graphite-800 flex items-center gap-1 w-fit"><Clock className="w-3 h-3" /> {status}</span>
    }
  }

  return (
    <div className="space-y-6">
      {/* HEADER BAR */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-obsidian-900 p-6 rounded-2xl border border-graphite-800">
        <div>
          <h1 className="text-xl font-bold text-ivory-50 flex items-center gap-2">
            <CreditCard className="w-6 h-6 text-emerald-400" />
            Payment Operations & Lifecycle Console
          </h1>
          <p className="text-xs text-graphite-500 font-mono mt-1">
            Real-time payment state machine tracking, attempt history, and recovery execution
          </p>
        </div>

        <button
          onClick={() => fetchPayments(pagination.page)}
          className="px-4 py-2 bg-graphite-800 hover:bg-graphite-700 text-ivory-50 rounded-xl text-xs font-mono font-semibold transition-colors flex items-center gap-2 w-fit"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh Operations
        </button>
      </div>

      {/* FILTERS & SEARCH */}
      <div className="bg-graphite-900/60 p-4 rounded-xl border border-graphite-800/80 flex flex-col md:flex-row items-center justify-between gap-4">
        <form onSubmit={handleSearchSubmit} className="flex items-center space-x-2 w-full md:w-96">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-graphite-500 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search Payment ID, Order ID, Customer..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl pl-9 pr-4 py-2 text-xs font-mono text-ivory-50 focus:outline-none focus:border-emerald-500"
            />
          </div>
          <button type="submit" className="px-3.5 py-2 bg-emerald-600 hover:bg-emerald-500 text-obsidian-950 font-bold rounded-xl text-xs transition-colors">
            Search
          </button>
        </form>

        <div className="flex items-center space-x-3 w-full md:w-auto">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-obsidian-950 border border-graphite-800 text-graphite-300 rounded-xl px-3 py-2 text-xs font-mono focus:outline-none"
          >
            <option value="">All Payment States</option>
            <option value="CREATED">CREATED</option>
            <option value="FAILED">FAILED</option>
            <option value="RECOVERY_ELIGIBLE">RECOVERY_ELIGIBLE</option>
            <option value="RETRY_SCHEDULED">RETRY_SCHEDULED</option>
            <option value="RECOVERED">RECOVERED</option>
            <option value="RECOVERY_FAILED">RECOVERY_FAILED</option>
            <option value="SUPPRESSED">SUPPRESSED</option>
          </select>

          <select
            value={methodFilter}
            onChange={(e) => setMethodFilter(e.target.value)}
            className="bg-obsidian-950 border border-graphite-800 text-graphite-300 rounded-xl px-3 py-2 text-xs font-mono focus:outline-none"
          >
            <option value="">All Payment Methods</option>
            <option value="CARD">CARD</option>
            <option value="UPI">UPI</option>
            <option value="NETBANKING">NETBANKING</option>
          </select>
        </div>
      </div>

      {/* PAYMENTS TABLE */}
      <div className="bg-obsidian-900 border border-graphite-800 rounded-2xl overflow-hidden shadow-lg">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-obsidian-950/80 border-b border-graphite-800 text-[11px] font-mono text-graphite-500 uppercase tracking-wider">
                <th className="p-4">Payment ID / Order ID</th>
                <th className="p-4">Customer ID</th>
                <th className="p-4">Amount</th>
                <th className="p-4">Method</th>
                <th className="p-4">Current Status</th>
                <th className="p-4">Failure Reason</th>
                <th className="p-4">Created Time</th>
                <th className="p-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-graphite-800/60 text-xs font-mono">
              {payments.length === 0 ? (
                <tr>
                  <td colSpan="8" className="p-8 text-center text-graphite-500">
                    No payment operations found matching criteria.
                  </td>
                </tr>
              ) : (
                payments.map((p) => (
                  <tr key={p.id} className="hover:bg-graphite-900/50 transition-colors">
                    <td className="p-4 font-bold text-ivory-50">
                      <div>{p.id}</div>
                      <div className="text-[10px] text-graphite-500 font-normal">{p.order_id}</div>
                    </td>
                    <td className="p-4 text-emerald-400 font-semibold">{p.masked_customer_id || p.customer_id}</td>
                    <td className="p-4 font-bold text-ivory-50">{p.currency} {p.amount.toFixed(2)}</td>
                    <td className="p-4">
                      <span className="bg-graphite-800 text-ivory-50 px-2 py-0.5 rounded text-[10px]">
                        {p.payment_method}
                      </span>
                    </td>
                    <td className="p-4">{getStatusBadge(p.status)}</td>
                    <td className="p-4 text-copper-400">{p.failure_reason || '—'}</td>
                    <td className="p-4 text-graphite-500 text-[11px]">
                      {new Date(p.created_at).toLocaleString()}
                    </td>
                    <td className="p-4 text-right">
                      <button
                        onClick={() => setSelectedPaymentId(p.id)}
                        className="px-3 py-1.5 bg-emerald-950 hover:bg-emerald-900 border border-emerald-800 text-emerald-400 rounded-lg text-xs font-bold transition-colors flex items-center gap-1 ml-auto"
                      >
                        <Eye className="w-3.5 h-3.5" /> Timeline
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* PAGINATION */}
        <div className="p-4 bg-obsidian-950 border-t border-graphite-800 flex items-center justify-between text-xs font-mono text-graphite-500">
          <span>Showing Page {pagination.page} of {pagination.total_pages} ({pagination.total_matching} total)</span>
          <div className="flex items-center space-x-2">
            <button
              disabled={pagination.page <= 1}
              onClick={() => fetchPayments(pagination.page - 1)}
              className="p-2 rounded-lg bg-graphite-800 hover:bg-graphite-700 disabled:opacity-40 text-ivory-50 transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              disabled={pagination.page >= pagination.total_pages}
              onClick={() => fetchPayments(pagination.page + 1)}
              className="p-2 rounded-lg bg-graphite-800 hover:bg-graphite-700 disabled:opacity-40 text-ivory-50 transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* DETAIL TIMELINE MODAL */}
      {selectedPaymentId && (
        <PaymentDetailModal
          paymentId={selectedPaymentId}
          onClose={() => setSelectedPaymentId(null)}
        />
      )}
    </div>
  )
}
