import uuid
import time
import unittest
import concurrent.futures
from starlette.testclient import TestClient
from backend.app.main import app
from backend.app.policy_service import get_policy_service
from backend.app.idempotency_store import get_idempotency_store

class TestPerformanceLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy_svc = get_policy_service()
        cls.idempotency_store = get_idempotency_store()
        cls.client = TestClient(app)

    def test_01_in_memory_policy_lookup_latency(self):
        sample_tx_ids = [f"RECLAIM-V2-{i:06d}" for i in range(1, 1001)]
        latencies = []

        for tx_id in sample_tx_ids:
            start_time = time.perf_counter()
            decision = self.policy_svc.get_decision(tx_id)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000.0) # convert to ms

        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        print(f"\n[PERF] In-Memory Policy Benchmark (1,000 Lookups): p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms")
        self.assertLess(p50, 1.0, "p50 latency should be under 1ms for in-memory lookup")
        self.assertLess(p95, 3.0, "p95 latency should be under 3ms")

    def test_02_concurrent_policy_lookups(self):
        sample_tx_ids = [f"RECLAIM-V2-{i:06d}" for i in range(1, 501)]

        def lookup_worker(tx_id):
            return self.policy_svc.get_decision(tx_id)

        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lookup_worker, sample_tx_ids))
        duration = time.perf_counter() - start

        rps = len(sample_tx_ids) / duration if duration > 0 else 0
        print(f"[PERF] Concurrent In-Memory Benchmark (500 requests, 10 threads): {rps:.1f} RPS in {duration:.3f}s")
        self.assertEqual(len(results), 500)

    def test_03_http_endpoint_load_benchmarks(self):
        """P1-14: Measures full HTTP endpoint response times, throughput, and error rate."""
        endpoints = [
            ("/health", "GET", None),
            ("/readiness", "GET", None),
            ("/decision/evaluate", "POST", {"transaction_id": "RECLAIM-V2-000001", "customer_id": "CUST_0001"}),
            ("/events/payment-failed", "POST", {
                "event_id": f"evt-bench-{uuid.uuid4()}",
                "transaction_id": "RECLAIM-V2-000001",
                "customer_id": "CUST_0001",
                "amount": 100.0,
                "payment_method": "CARD"
            }),
            ("/action/preview", "POST", {"transaction_id": "RECLAIM-V2-000001", "action_type": "SMART_RETRY"})
        ]

        latencies = []
        errors = 0

        start_total = time.perf_counter()
        for i in range(100):
            ep_path, method, body = endpoints[i % len(endpoints)]
            if method == "POST" and ep_path == "/events/payment-failed":
                body["event_id"] = f"evt-bench-{uuid.uuid4()}"

            t0 = time.perf_counter()
            if method == "GET":
                res = self.client.get(ep_path)
            else:
                res = self.client.post(ep_path, json=body)
            t1 = time.perf_counter()

            if res.status_code != 200:
                errors += 1
            latencies.append((t1 - t0) * 1000.0)

        total_duration = time.perf_counter() - start_total
        sorted_lats = sorted(latencies)
        p50 = sorted_lats[int(len(sorted_lats) * 0.50)]
        p95 = sorted_lats[int(len(sorted_lats) * 0.95)]
        p99 = sorted_lats[int(len(sorted_lats) * 0.99)]
        rps = len(latencies) / total_duration if total_duration > 0 else 0

        print(f"[PERF] Full HTTP Endpoint Benchmark (100 requests): p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms | Throughput={rps:.1f} RPS | Error Rate={errors}%")
        self.assertEqual(errors, 0)

if __name__ == "__main__":
    unittest.main()
