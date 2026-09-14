import React from 'react'
import { ShieldCheck, X, CheckCircle2, AlertTriangle, Lock, Database } from 'lucide-react'

export default function GovernanceDrawer({ isOpen, onClose }) {
  if (!isOpen) return null

  const mandates = [
    {
      title: "Immutable Production Policy",
      desc: "V10.2 targeting policy is locked and read-only. No re-ranking of production transactions is permitted.",
      status: "ENFORCED"
    },
    {
      title: "Targeting Allocation Integrity",
      desc: "Exactly 500 out of 10,000 production transactions (5.0%) are selected for TARGET. 9,500 receive RETRY_ALL.",
      status: "ENFORCED"
    },
    {
      title: "Zero Lineage Fabrication",
      desc: "No artificial transaction IDs, artificial outcomes, fuzzy matching, or customer-only lineage joining.",
      status: "ENFORCED"
    },
    {
      title: "Isolation of Historical Outcomes",
      desc: "Historical recovery datasets (recovery_outcomes.csv) are segregated for research and never joined to V10.2 production transactions.",
      status: "ENFORCED"
    },
    {
      title: "Isolated Demo Execution Layer",
      desc: "Interactive operator simulations are stored in a dedicated demo store and explicitly tagged 'SIMULATED — NOT AN OBSERVED RECOVERY'.",
      status: "ENFORCED"
    },
    {
      title: "Oracle Field Protection",
      desc: "Oracle counterfactual fields (oracle_control_value, oracle_treatment_value, oracle_incremental_value) are excluded from production decision making.",
      status: "ENFORCED"
    }
  ]

  return (
    <div className="fixed inset-0 bg-obsidian-950/80 backdrop-blur-sm z-50 flex justify-end font-sans">
      <div className="bg-obsidian-900 border-l border-graphite-800 w-full max-w-md h-full p-6 flex flex-col justify-between overflow-y-auto space-y-6">
        <div className="space-y-6">
          <div className="flex items-center justify-between border-b border-graphite-800 pb-4">
            <div className="flex items-center space-x-2">
              <ShieldCheck className="w-5 h-5 text-emerald-400" />
              <h2 className="text-lg font-bold text-ivory-50 font-mono">Governance & Compliance</h2>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 text-graphite-500 hover:text-ivory-50 rounded-lg bg-graphite-800"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <p className="text-xs text-graphite-500 leading-relaxed font-mono">
            RECLAIM enforces strict data governance constraints to prevent policy drift, artificial performance claims, and invalid lineage matching.
          </p>

          <div className="space-y-3 font-mono">
            {mandates.map((m, idx) => (
              <div key={idx} className="bg-obsidian-950 border border-graphite-800 rounded-xl p-4 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-ivory-200 flex items-center gap-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    {m.title}
                  </span>
                  <span className="bg-emerald-950 text-emerald-400 border border-emerald-800 text-[9px] px-2 py-0.5 rounded font-bold">
                    {m.status}
                  </span>
                </div>
                <p className="text-[11px] text-graphite-500 leading-normal pl-5">{m.desc}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="pt-4 border-t border-graphite-800 space-y-2 font-mono">
          <div className="bg-obsidian-950 border border-graphite-800 p-3.5 rounded-xl text-[11px] text-graphite-500">
            Audit Trail: <span className="text-emerald-400">V21 Monitoring Pipeline</span> <br />
            Status: <span className="text-emerald-400 font-bold">PASSED (All 10,000 records)</span>
          </div>

          <button
            onClick={onClose}
            className="w-full bg-graphite-800 hover:bg-graphite-700 text-ivory-50 text-xs font-bold py-3 rounded-xl transition-colors uppercase tracking-wider"
          >
            Close Governance Overview
          </button>
        </div>
      </div>
    </div>
  )
}
