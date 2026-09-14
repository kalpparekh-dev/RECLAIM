import React, { useState, useEffect } from 'react'
import { ShieldCheck, CheckCircle2, AlertTriangle, CreditCard, RefreshCw, X, Zap } from 'lucide-react'

export default function RazorpayCheckoutModal({ isOpen, onClose, paymentItem, keyId }) {
  const [loading, setLoading] = useState(false)
  const [verifying, setVerifying] = useState(false)
  const [status, setStatus] = useState('IDLE') // IDLE, CHECKOUT_OPEN, VERIFYING, SUCCESS, FAILED
  const [errorMessage, setErrorMessage] = useState('')
  const [verificationResult, setVerificationResult] = useState(null)

  useEffect(() => {
    if (isOpen) {
      setStatus('IDLE')
      setErrorMessage('')
      setVerificationResult(null)
    }
  }, [isOpen])

  if (!isOpen || !paymentItem) return null

  const loadRazorpayScript = () => {
    return new Promise((resolve) => {
      if (window.Razorpay) {
        resolve(true)
        return
      }
      const script = document.createElement('script')
      script.src = 'https://checkout.razorpay.com/v1/checkout.js'
      script.onload = () => resolve(true)
      script.onerror = () => resolve(false)
      document.body.appendChild(script)
    })
  }

  const handleLaunchCheckout = async () => {
    setLoading(true)
    setErrorMessage('')

    const loaded = await loadRazorpayScript()
    if (!loaded) {
      setErrorMessage('Failed to load Razorpay Checkout SDK. Please check your internet connection.')
      setLoading(false)
      return
    }

    const effectiveKeyId = keyId || paymentItem.key_id || 'rzp_test_reclaim_demo'

    const options = {
      key: effectiveKeyId,
      amount: Math.round(paymentItem.amount * 100),
      currency: paymentItem.currency || 'INR',
      name: 'RECLAIM Payment Recovery',
      description: `Payment Checkout - Tx ${paymentItem.id}`,
      order_id: paymentItem.external_id?.startsWith('order_') ? paymentItem.external_id : undefined,
      prefill: {
        name: 'RECLAIM Customer',
        email: 'customer@reclaim.finance',
        contact: '9999999999'
      },
      theme: {
        color: '#10b981' // Emerald accent
      },
      handler: async function (response) {
        // Razorpay modal success callback
        setStatus('VERIFYING')
        setVerifying(true)
        try {
          const verifyRes = await fetch('/api/gateway/razorpay/verify-payment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              razorpay_order_id: response.razorpay_order_id || paymentItem.external_id || 'order_demo',
              razorpay_payment_id: response.razorpay_payment_id,
              razorpay_signature: response.razorpay_signature,
              reclaim_payment_id: paymentItem.id
            })
          })

          const data = await verifyRes.json()

          if (verifyRes.ok && data.verified) {
            setStatus('SUCCESS')
            setVerificationResult(data)
          } else {
            setStatus('FAILED')
            setErrorMessage(data.detail || 'Signature verification failed.')
          }
        } catch (err) {
          setStatus('FAILED')
          setErrorMessage(`Verification error: ${err.message}`)
        } finally {
          setVerifying(false)
        }
      },
      modal: {
        ondismiss: function () {
          setLoading(false)
          if (status === 'CHECKOUT_OPEN') {
            setStatus('IDLE')
          }
        }
      }
    }

    try {
      const rzp = new window.Razorpay(options)
      rzp.on('payment.failed', function (response) {
        setStatus('FAILED')
        setErrorMessage(response.error?.description || 'Payment execution failed on gateway.')
        setLoading(false)
      })
      setStatus('CHECKOUT_OPEN')
      rzp.open()
    } catch (err) {
      setErrorMessage(`Failed to open Razorpay modal: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 bg-obsidian-950/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-graphite-900 border border-graphite-800 w-full max-w-lg rounded-2xl p-6 space-y-6 shadow-2xl font-sans">
        {/* HEADER */}
        <div className="flex items-center justify-between border-b border-graphite-800 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-emerald-950 border border-emerald-800/80 flex items-center justify-center">
              <CreditCard className="w-4 h-4 text-emerald-400" />
            </div>
            <div>
              <h3 className="text-base font-bold font-mono text-ivory-50">
                Razorpay Checkout (Test Mode)
              </h3>
              <p className="text-[11px] text-graphite-500 font-mono">
                Secure HMAC SHA256 Signature Verification
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-graphite-800 text-graphite-400 hover:text-ivory-50 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* DETAILS BODY */}
        <div className="space-y-4 text-xs font-mono">
          <div className="bg-obsidian-950 p-4 rounded-xl border border-graphite-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-graphite-500">RECLAIM Payment ID:</span>
              <span className="text-emerald-400 font-bold">{paymentItem.id}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-graphite-500">Customer ID:</span>
              <span className="text-ivory-100">{paymentItem.customer_id}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-graphite-500">Amount:</span>
              <span className="text-gold-400 font-bold text-sm">
                ₹{paymentItem.amount?.toLocaleString()} {paymentItem.currency || 'INR'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-graphite-500">Public Key ID:</span>
              <span className="text-graphite-400">{keyId || paymentItem.key_id || 'rzp_test_mode'}</span>
            </div>
          </div>

          {/* STATUS DISPLAY */}
          {status === 'VERIFYING' && (
            <div className="bg-gold-950/60 border border-gold-800 p-4 rounded-xl flex items-center gap-3 text-gold-400">
              <RefreshCw className="w-5 h-5 animate-spin shrink-0" />
              <div>
                <p className="font-bold">Verifying HMAC Signature...</p>
                <p className="text-[11px] text-graphite-400">Verifying signature on backend via constant-time SHA256.</p>
              </div>
            </div>
          )}

          {status === 'SUCCESS' && (
            <div className="bg-emerald-950/60 border border-emerald-800 p-4 rounded-xl space-y-2 text-emerald-400">
              <div className="flex items-center gap-2 font-bold text-sm">
                <CheckCircle2 className="w-5 h-5 shrink-0" />
                Payment Signature Verified & Captured!
              </div>
              <div className="text-[11px] text-graphite-300 space-y-1 pt-1 border-t border-emerald-900/60">
                <div>Razorpay Order ID: <strong>{verificationResult?.razorpay_order_id}</strong></div>
                <div>Razorpay Payment ID: <strong>{verificationResult?.razorpay_payment_id}</strong></div>
                <div>Status: <strong className="text-emerald-400">CAPTURED / RECOVERED</strong></div>
              </div>
            </div>
          )}

          {status === 'FAILED' && (
            <div className="bg-copper-950/60 border border-copper-800 p-4 rounded-xl space-y-1 text-copper-400">
              <div className="flex items-center gap-2 font-bold text-sm">
                <AlertTriangle className="w-5 h-5 shrink-0" />
                Payment / Verification Failed
              </div>
              <p className="text-[11px] text-graphite-300">{errorMessage}</p>
            </div>
          )}
        </div>

        {/* ACTIONS FOOTER */}
        <div className="flex items-center justify-end gap-3 pt-2 border-t border-graphite-800">
          <button
            onClick={onClose}
            className="px-4 py-2.5 rounded-xl bg-graphite-800 text-graphite-400 hover:text-ivory-50 text-xs font-mono"
          >
            {status === 'SUCCESS' ? 'Close' : 'Cancel'}
          </button>

          {status !== 'SUCCESS' && (
            <button
              onClick={handleLaunchCheckout}
              disabled={loading || verifying}
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-emerald-700 text-obsidian-950 font-bold text-xs font-mono shadow-md shadow-emerald-950/40 hover:brightness-110 transition-all flex items-center gap-2"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Loading SDK...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Launch Test Checkout
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
