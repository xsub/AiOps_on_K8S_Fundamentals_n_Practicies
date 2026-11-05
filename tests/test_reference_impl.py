"""
End-to-End Local Integration Tests for the Runnable Reference Implementation.
Exercises the Paged KV-Cache allocator, GQA Transformer engine, OpenAI SSE server,
Bearer API auth, Prometheus telemetry scraping, live AIOps controller loop,
and SSE streaming benchmark harness over localhost TCP sockets.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = REPO_ROOT / "reference-impl"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Map reference-impl/app as `ref_app` matching container layout
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "ref_app",
    str(REF_DIR / "app" / "__init__.py"),
    submodule_search_locations=[str(REF_DIR / "app")],
)
if "ref_app" not in sys.modules and _spec and _spec.loader:
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["ref_app"] = _mod
    _spec.loader.exec_module(_mod)

from ref_app.k8s_aiops_agent import evaluate_live_cycle, parse_prometheus_text
from ref_app.server import create_server
from ref_app.transformer_engine import KVCacheOverflowError, PagedKVCacheManager

# Import benchmark helpers
_bench_spec = importlib.util.spec_from_file_location(
    "run_benchmark",
    str(REF_DIR / "benchmark" / "run_benchmark.py"),
)
assert _bench_spec and _bench_spec.loader
run_benchmark = importlib.util.module_from_spec(_bench_spec)
sys.modules["run_benchmark"] = run_benchmark
_bench_spec.loader.exec_module(run_benchmark)


class TestReferenceImplementationLiveHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shm_tmp = tempfile.TemporaryDirectory()
        cls.api_key = "TEST_BEARER_KEY_SAFE"
        cls.server, cls.state = create_server(
            host="127.0.0.1",
            port=0,  # OS assigns ephemeral port
            api_key=cls.api_key,
            num_blocks=32,
            block_size=8,
            gpu_memory_utilization=0.90,
            shm_dir=cls.shm_tmp.name,
        )
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.shm_tmp.cleanup()

    def test_01_health_and_shm_readiness_probes(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/readyz", timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            body = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(body["status"], "ready")
            self.assertTrue(body["shm_ipc_mounted"])

    def test_02_unauthenticated_inference_rejected_401(self) -> None:
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=b'{"messages":[{"role":"user","content":"hello"}]}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=3.0)
        self.assertEqual(ctx.exception.code, 401)
        ctx.exception.close()

    def test_03_streaming_sse_inference_and_metrics(self) -> None:
        sample = run_benchmark.run_single_sse_request(
            base_url=self.base_url,
            api_key=self.api_key,
            model="micro-gqa-llama-ref",
            prompt_tokens=16,
            max_tokens=10,
            suite="unit_test",
            concurrency=1,
        )
        self.assertEqual(sample.status_code, 200)
        self.assertEqual(sample.output_tokens, 10)
        self.assertGreater(sample.ttft_ms, 0.0)
        self.assertEqual(self.state.kv_manager.active_requests, 0)  # Freed cleanly after stream!

        with urllib.request.urlopen(f"{self.base_url}/metrics", timeout=3.0) as resp:
            metrics_text = resp.read().decode("utf-8")
        pts = parse_prometheus_text(metrics_text)
        metric_names = {p.metric for p in pts}
        self.assertIn("vllm:gpu_cache_usage_perc", metric_names)
        self.assertIn("DCGM_FI_DEV_FB_USED", metric_names)
        self.assertIn("DCGM_FI_DEV_XID_ERRORS", metric_names)

    def test_04_kv_cache_overflow_guard(self) -> None:
        kv = PagedKVCacheManager(num_blocks=8, block_size=4, shm_dir=self.shm_tmp.name)
        with self.assertRaises(KVCacheOverflowError):
            kv.allocate_sequence("oversized", prompt_tokens=100, timeout_s=0.05)
        self.assertEqual(kv.oom_rejections_total, 1)

    def test_05_live_aiops_closed_loop_fault_injection(self) -> None:
        # Inject critical silicon XID 79 via authenticated fault endpoint
        inject_req = urllib.request.Request(
            f"{self.base_url}/admin/fault-inject",
            data=json.dumps({"mode": "inject", "xid_code": 79, "kv_pressure_ratio": 0.92}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(inject_req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)

        try:
            report = evaluate_live_cycle(
                metrics_url=f"{self.base_url}/metrics",
                samples=4,
                sample_interval_s=0.02,
            )
            self.assertGreaterEqual(len(report["correlated_incidents"]), 1)
            top = report["correlated_incidents"][0]
            self.assertEqual(top["incident"]["root_cause_class"], "GPU_SILICON_XID_FAULT")
            self.assertEqual(top["remediation_plan"]["approval_status"], "PENDING_APPROVAL")
        finally:
            reset_req = urllib.request.Request(
                f"{self.base_url}/admin/fault-inject",
                data=json.dumps({"mode": "reset"}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            urllib.request.urlopen(reset_req, timeout=3.0).close()


if __name__ == "__main__":
    unittest.main()
