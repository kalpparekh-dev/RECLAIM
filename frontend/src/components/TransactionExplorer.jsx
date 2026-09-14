import React, { useState, useEffect } from 'react'
import { 
  Search, 
  ChevronLeft, 
  ChevronRight, 
  ExternalLink, 
  PlayCircle, 
  X, 
  Info,
  CheckCircle2,
  AlertCircle,
  SlidersHorizontal,
  Clock,
  ShieldCheck
} from 'lucide-react'

export default function TransactionExplorer({ onSelectConsole, onSelectDemo }) {
  const [transactions, setTransactions] = useState([])
  const [pagination, setPagination] = useState({ page: 1, limit: 15, total_matching: 0, total_pages: 1 })
  const [search, setSearch] = useState('')
  const [actionFilter, setActionFilter] = useState('')
  const [loading, setLoading] = useState(false)

  // Drawer state
  const [selectedTx, setSelectedTx] = useState(null)

  useEffect(() => {
    fetchTransactions(1)
  }, [actionFilter])

  const fetchTransactions = async (page = 1) => {
    setLoading(true)
    try {
      let url = `/api/transactions?page=${page}&limit=15`
      if (search.trim()) url += `&search=${encodeURIComponent(search.trim())}`
      if (actionFilter) url += `&action=${actionFilter}`

      const res = await fetch(url)
      if (res.ok) {
        const data = await res.json()
        setTransactions(data.transactions || [])
        setPagination(data.pagination || { page: 1, limit: 15, total_matching: 0, total_pages: 1 })
      }
    } catch (e) {
      console.error('Failed to fetch transactions', e)
    } finally {
      setLoading(false)
    }
  }

  const handleSearchSubmit = (e) => {
    e.preventDefault()
    fetchTransactions(1)
  }

  return (
    <div className="space-y-6 font-sans">
      {/* Search and Filters Card */}
      <div className="bg-graphite-900/90 border border-graphite-800 rounded-3xl p-6 space-y-4 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-ivory-50 flex items-center gap-2 font-mono">
              <Search className="w-5 h-5 text-emerald-400" />
              Production Transaction Workspace
            </h2>
            <p className="text-xs text-graphite-500">Query and inspect 10,000 production transactions in policy V10.2</p>
          </div>

          <div className="text-xs font-mono text-graphite-500 bg-obsidian-950 px-4 py-2 rounded-xl border border-graphite-800">
            Matching: <strong className="text-emerald-400">{pagination.total_matching.toLocaleString()}</strong> / 10,000
          </div>
        </div>

        {/* Filter Bar */}
        <form onSubmit={handleSearchSubmit} className="grid grid-cols-1 sm:grid-cols-12 gap-3">
          <div className="sm:col-span-7 relative">
            <Search className="w-4 h-4 text-graphite-500 absolute left-3.5 top-3.5" />
            <input
              type="text"
              placeholder="Search Transaction ID or Customer ID (e.g. RECLAIM-V2-000001, C00233)..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl pl-10 pr-4 py-2.5 text-xs text-ivory-50 placeholder-graphite-500 focus:outline-none focus:border-emerald-500 font-mono"
            />
          </div>

          <div className="sm:col-span-3">
            <select
              value={actionFilter}
              onChange={(e) => setActionFilter(e.target.value)}
              className="w-full bg-obsidian-950 border border-graphite-800 rounded-xl px-3 py-2.5 text-xs text-ivory-200 focus:outline-none focus:border-emerald-500 font-mono"
            >
              <option value="">All Actions (TARGET & RETRY_ALL)</option>
              <option value="TARGET">TARGET (Selected Top 5%)</option>
              <option value="RETRY_ALL">RETRY_ALL (Fallback)</option>
            </select>
          </div>

          <div className="sm:col-span-2 flex gap-2">
            <button
              type="submit"
              className="w-full bg-emerald-600 hover:bg-emerald-500 text-obsidian-950 font-bold text-xs px-4 py-2.5 rounded-xl transition-all shadow-md shadow-emerald-950 uppercase tracking-wider font-mono"
            >
              Filter
            </button>
          </div>
        </form>
      </div>

      {/* Transaction Data Table */}
      <div className="bg-graphite-900/90 border border-graphite-800 rounded-3xl overflow-hidden shadow-2xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-ivory-200">
            <thead className="bg-obsidian-950 border-b border-graphite-800 text-[11px] font-mono text-graphite-500 uppercase tracking-wider">
              <tr>
                <th className="py-4 px-5 font-bold">Transaction ID</th>
                <th className="py-4 px-5 font-bold">Customer ID</th>
                <th className="py-4 px-5 font-bold">Decision Badge</th>
                <th className="py-4 px-5 font-bold text-right">P(Treatment)</th>
                <th className="py-4 px-5 font-bold text-right">P(Control)</th>
                <th className="py-4 px-5 font-bold text-right">Causal Uplift</th>
                <th className="py-4 px-5 font-bold text-center">Console</th>
              </tr>
            </thead>

            <tbody className="divide-y divide-graphite-800/60 font-mono text-xs">
              {loading ? (
                <tr>
                  <td colSpan="7" className="py-12 text-center text-graphite-500">
                    <div className="flex items-center justify-center space-x-2">
                      <div className="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
                      <span>Querying policy V10.2 transaction records...</span>
                    </div>
                  </td>
                </tr>
              ) : transactions.length === 0 ? (
                <tr>
                  <td colSpan="7" className="py-12 text-center text-graphite-500 font-sans">
                    No transactions found matching search criteria.
                  </td>
                </tr>
              ) : (
                transactions.map((tx) => {
                  const isSelected = tx.intervention_selected === 1
                  const uplift = tx.predicted_probability_uplift

                  return (
                    <tr 
                      key={tx.transaction_id}
                      onClick={() => setSelectedTx(tx)}
                      className="hover:bg-graphite-800/50 transition-colors cursor-pointer"
                    >
                      <td className="py-3.5 px-5 font-bold text-emerald-400">
                        {tx.transaction_id}
                      </td>
                      <td className="py-3.5 px-5 text-ivory-200">
                        {tx.customer_id}
                      </td>
                      <td className="py-3.5 px-5">
                        {isSelected ? (
                          <span className="bg-emerald-950 text-emerald-400 border border-emerald-700/80 px-3 py-1 rounded-full text-[10px] font-bold flex items-center w-max gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                            TARGET (TOP 5%)
                          </span>
                        ) : (
                          <span className="bg-obsidian-950 text-graphite-500 border border-graphite-800 px-3 py-1 rounded-full text-[10px] font-medium w-max inline-block">
                            RETRY_ALL
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 px-5 text-right text-emerald-400 font-bold">
                        {(tx.predicted_treatment_probability * 100).toFixed(2)}%
                      </td>
                      <td className="py-3.5 px-5 text-right text-graphite-500">
                        {(tx.predicted_control_probability * 100).toFixed(2)}%
                      </td>
                      <td className={`py-3.5 px-5 text-right font-bold ${
                        uplift > 0 ? 'text-gold-400' : 'text-graphite-500'
                      }`}>
                        {uplift > 0 ? `+${(uplift * 100).toFixed(2)}%` : `${(uplift * 100).toFixed(2)}%`}
                      </td>
                      <td className="py-3.5 px-5 text-center">
                        <div className="flex items-center justify-center space-x-2" onClick={(e) => e.stopPropagation()}>
                          <button
                            onClick={() => onSelectConsole(tx.transaction_id)}
                            title="Evaluate in Decision Console"
                            className="p-1.5 bg-obsidian-950 hover:bg-graphite-800 text-emerald-400 border border-emerald-900 rounded-lg transition-colors"
                          >
                            <ExternalLink className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => onSelectDemo(tx.transaction_id)}
                            title="Simulate Operator Action"
                            className="p-1.5 bg-obsidian-950 hover:bg-graphite-800 text-copper-400 border border-copper-900 rounded-lg transition-colors"
                          >
                            <PlayCircle className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        <div className="bg-obsidian-950 border-t border-graphite-800 px-6 py-4 flex items-center justify-between">
          <div className="text-xs text-graphite-500 font-mono">
            Page <strong className="text-ivory-50">{pagination.page}</strong> of <strong className="text-ivory-50">{pagination.total_pages}</strong>
          </div>

          <div className="flex items-center space-x-2 font-mono">
            <button
              disabled={pagination.page <= 1}
              onClick={() => fetchTransactions(pagination.page - 1)}
              className="px-3.5 py-1.5 bg-graphite-900 hover:bg-graphite-800 disabled:opacity-40 border border-graphite-800 rounded-lg text-ivory-200 text-xs transition-colors flex items-center space-x-1"
            >
              <ChevronLeft className="w-4 h-4" />
              <span>Previous</span>
            </button>

            <button
              disabled={pagination.page >= pagination.total_pages}
              onClick={() => fetchTransactions(pagination.page + 1)}
              className="px-3.5 py-1.5 bg-graphite-900 hover:bg-graphite-800 disabled:opacity-40 border border-graphite-800 rounded-lg text-ivory-200 text-xs transition-colors flex items-center space-x-1"
            >
              <span>Next</span>
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Transaction Detail Drawer Modal */}
      {selectedTx && (
        <div className="fixed inset-0 bg-obsidian-950/80 backdrop-blur-sm z-50 flex justify-end">
          <div className="bg-obsidian-900 border-l border-graphite-800 w-full max-w-lg h-full p-6 flex flex-col justify-between overflow-y-auto space-y-6">
            <div className="space-y-6">
              <div className="flex items-center justify-between border-b border-graphite-800 pb-4">
                <div>
                  <span className="text-[10px] font-mono text-emerald-400 uppercase font-bold tracking-wider">Transaction Inspection</span>
                  <h3 className="text-xl font-bold font-mono text-ivory-50">{selectedTx.transaction_id}</h3>
                </div>
                <button
                  onClick={() => setSelectedTx(null)}
                  className="p-2 text-graphite-500 hover:text-ivory-50 rounded-lg bg-graphite-800"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Policy Action Banner */}
              <div className={`p-5 rounded-2xl border ${
                selectedTx.intervention_selected === 1
                  ? 'bg-emerald-950/40 border-emerald-800 text-emerald-300'
                  : 'bg-obsidian-950 border-graphite-800 text-ivory-200'
              }`}>
                <div className="text-xs font-mono text-graphite-500 uppercase">Policy Decision</div>
                <div className="text-2xl font-black font-mono mt-1 flex items-center gap-2">
                  <span>{selectedTx.selected_policy}</span>
                  {selectedTx.intervention_selected === 1 && (
                    <span className="bg-emerald-500 text-obsidian-950 text-xs px-2.5 py-0.5 rounded font-bold uppercase">
                      TOP 5% TARGETED
                    </span>
                  )}
                </div>
              </div>

              {/* Probabilities Grid */}
              <div className="grid grid-cols-2 gap-3 font-mono">
                <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800">
                  <span className="text-[11px] text-graphite-500">P(Treatment)</span>
                  <div className="text-xl font-bold text-emerald-400 mt-1">
                    {(selectedTx.predicted_treatment_probability * 100).toFixed(2)}%
                  </div>
                </div>

                <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800">
                  <span className="text-[11px] text-graphite-500">P(Control)</span>
                  <div className="text-xl font-bold text-ivory-200 mt-1">
                    {(selectedTx.predicted_control_probability * 100).toFixed(2)}%
                  </div>
                </div>
              </div>

              {/* Causal Uplift */}
              <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 space-y-1 font-mono">
                <span className="text-[11px] text-graphite-500">Predicted Causal Uplift (\(\Delta P\))</span>
                <div className={`text-2xl font-black ${
                  selectedTx.predicted_probability_uplift > 0 ? 'text-gold-400' : 'text-graphite-500'
                }`}>
                  {selectedTx.predicted_probability_uplift > 0 ? '+' : ''}
                  {(selectedTx.predicted_probability_uplift * 100).toFixed(4)}%
                </div>
              </div>

              {/* Transaction Evaluation Timeline */}
              <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 space-y-3 font-mono text-xs">
                <div className="font-bold text-ivory-50 border-b border-graphite-800 pb-2">Evaluation Timeline</div>
                <div className="space-y-2 text-[11px]">
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-copper-500"></span>
                    <span className="text-graphite-500">Failed Payment Received</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-gold-400"></span>
                    <span className="text-graphite-500">V10.2 Uplift Scored</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                    <span className="text-emerald-400 font-bold">Policy Assigned: {selectedTx.selected_policy}</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex gap-3 pt-4 border-t border-graphite-800 font-mono">
              <button
                onClick={() => {
                  setSelectedTx(null)
                  onSelectConsole(selectedTx.transaction_id)
                }}
                className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-obsidian-950 font-bold text-xs py-3 rounded-xl transition-colors flex items-center justify-center gap-1.5 uppercase"
              >
                <ExternalLink className="w-4 h-4" />
                <span>Console</span>
              </button>

              <button
                onClick={() => {
                  setSelectedTx(null)
                  onSelectDemo(selectedTx.transaction_id)
                }}
                className="bg-copper-600 hover:bg-copper-500 text-ivory-50 font-bold text-xs px-5 py-3 rounded-xl transition-colors flex items-center gap-1.5 uppercase"
              >
                <PlayCircle className="w-4 h-4" />
                <span>Demo</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
