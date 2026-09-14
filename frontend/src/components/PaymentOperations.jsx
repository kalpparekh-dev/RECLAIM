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
  Play,
  Zap
} from 'lucide-react'
import PaymentDetailModal from './PaymentDetailModal'
import RazorpayCheckoutModal from './RazorpayCheckoutModal'

const API_BASE = ''

export default function PaymentOperations() {
  const [payments, setPayments] = useState([])
  const [pagination, setPagination] = useState({ page: 1, limit: 15, total_pages: 1, total_matching: 0 })
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [methodFilter, setMethodFilter] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [selectedPaymentId, setSelectedPaymentId] = useState(null)
  const [checkoutPayment, setCheckoutPayment] = useState(null)

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
      case 'CREATED':
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-graphite-800 text-ivory-50 border border-graphite-700">CREATED</span>
      case 'PROCESSING':
      case 'RETRY_PROCESSING':
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-gold-950 text-gold-400 border border-gold-800/80 flex items-center gap-1"><Clock className="w-3 h-3"/> {status}</span>
      case 'CAPTURED':
      case 'RECOVERED':
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-emerald-950 text-emerald-400 border border-emerald-800 flex items-center gap-1"><CheckCircle2 className="w-3 h-3"/> {status}</span>
      case 'FAILED':
      case 'RECOVERY_FAILED':
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-copper-950 text-copper-400 border border-copper-800 flex items-center gap-1"><AlertTriangle className="w-3 h-3"/> {status}</span>
      case 'SUPPRESSED':
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-copper-900 text-copper-300 border border-copper-700 flex items-center gap-1"><ShieldAlert className="w-3 h-3"/> SUPPRESSED</span>
      default:
        return <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-graphite-800 text-graphite-400 border border-graphite-700">{status}</span>
    }
  }

  return (
    <div className="space-y-6 font-sans">
      {/* HEADER */}
      <div className="bg-obsidian-900 p-6 rounded-2xl border border-graphite-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-extrabold text-ivory-50 flex items-center gap-2 font-mono">
            <CreditCard className="w-6 h-6 text-emerald-400" />
            Payment Operations & Lifecycle Ledger
          </h1>
          <p className="text-xs text-graphite-500 mt-1">
            Real-time multi-gateway payment state machine tracking, attempts history, and timeline audit
          </p>
        </div>

        <button
          onClick={() => fetchPayments(pagination.page)}
          className="p-2.5 rounded-xl bg-graphite-800 text-graphite-400 hover:text-ivory-50 border border-graphite-700 transition-colors flex items-center gap-2 text-xs font-mono"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* FILTER & SEARCH BAR */}
      <div className="bg-graphite-900 border border-graphite-800 p-4 rounded-2xl flex flex-col md:flex-row gap-4 items-center justify-between font-mono text-xs">
        <form onSubmit={handleSearchSubmit} className="flex-1 w-full flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-graphite-500 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search by Payment ID, Order ID, or Customer ID..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl pl-9 pr-4 py-2 text-ivory-50 placeholder:text-graphite-500 outline-none focus:border-emerald-500"
            />
          </div>
          <button
            type="submit"
            className="px-4 py-2 bg-graphite-800 hover:bg-graphite-700 text-ivory-50 rounded-xl font-bold transition-colors"
          >
            Search
          </button>
        </form>

        <div className="flex items-center gap-3 w-full md:w-auto">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-obsidian-950 border border-graphite-800 rounded-xl px-3 py-2 text-ivory-50 outline-none"
          >
            <option value="">All Statuses</option>
            <option value="CREATED">CREATED</option>
            <option value="FAILED">FAILED</option>
            <option value="RETRY_SCHEDULED">RETRY_SCHEDULED</option>
            <option value="RETRY_PROCESSING">RETRY_PROCESSING</option>
            <option value="RECOVERED">RECOVERED</option>
            <option value="CAPTURED">CAPTURED</option>
          </select>

          <select
            value={methodFilter}
            onChange={(e) => setMethodFilter(e.target.value)}
            className="bg-obsidian-950 border border-graphite-800 rounded-xl px-3 py-2 text-ivory-50 outline-none"
          >
            <option value="">All Methods</option>
            <option value="CARD">CARD</option>
            <option value="UPI">UPI</option>
            <option value="NETBANKING">NETBANKING</option>
          </select>
        </div>
      </div>

      {/* PAYMENTS TABLE */}
      <div className="bg-graphite-900 border border-graphite-800 rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse font-mono text-xs">
            <thead>
              <tr className="border-b border-graphite-800 bg-obsidian-950 text-[11px] text-graphite-500 uppercase">
                <th className="p-4">Payment ID / Order</th>
                <th className="p-4">Customer</th>
                <th className="p-4">Amount</th>
                <th className="p-4">Method</th>
                <th className="p-4">Status</th>
                <th className="p-4">Failure Reason</th>
                <th className="p-4">Created At</th>
                <th className="p-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-graphite-800/60">
              {payments.length === 0 ? (
                <tr>
                  <td colSpan="8" className="p-8 text-center text-graphite-500">
                    No payment records found. Use Gateway Sandbox to create test payments.
                  </td>
                </tr>
              ) : (
                payments.map((p) => (
                  <tr key={p.id} className="hover:bg-obsidian-950/50 transition-colors">
                    <td className="p-4 font-bold text-emerald-400">
                      <div>{p.id}</div>
                      <div className="text-[10px] text-graphite-500 font-normal">{p.order_id || p.external_id}</div>
                    </td>
                    <td className="p-4 text-ivory-50">{p.customer_id}</td>
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
                    <td className="p-4 text-right flex items-center justify-end gap-2">
                      <button
                        onClick={() => setCheckoutPayment(p)}
                        className="px-2.5 py-1.5 bg-gold-950 hover:bg-gold-900 border border-gold-800/80 text-gold-400 rounded-lg text-xs font-bold transition-colors flex items-center gap-1"
                        title="Razorpay Test Checkout"
                      >
                        <Zap className="w-3.5 h-3.5" /> Checkout
                      </button>
                      <button
                        onClick={() => setSelectedPaymentId(p.id)}
                        className="px-2.5 py-1.5 bg-emerald-950 hover:bg-emerald-900 border border-emerald-800 text-emerald-400 rounded-lg text-xs font-bold transition-colors flex items-center gap-1"
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

      {/* RAZORPAY CHECKOUT MODAL */}
      {checkoutPayment && (
        <RazorpayCheckoutModal
          isOpen={Boolean(checkoutPayment)}
          onClose={() => setCheckoutPayment(null)}
          paymentItem={checkoutPayment}
        />
      )}
    </div>
  )
}
