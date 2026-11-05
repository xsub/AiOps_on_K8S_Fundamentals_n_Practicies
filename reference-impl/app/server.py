"""
OpenAI-Compatible Streaming Inference & Prometheus Telemetry Server.

Serves:
  - GET  /healthz, /readyz          -> Liveness & /dev/shm IPC readiness probes
  - GET  /metrics                   -> Prometheus text exposition (vllm:* & DCGM_FI_DEV_*)
  - GET  /v1/models                 -> OpenAI model listing (Bearer auth required)
  - POST /v1/chat/completions       -> Streaming SSE & JSON inference (Bearer auth required)
  - POST /admin/fault-inject        -> Deterministic chaos fault injection (Bearer auth required)
"""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

from ref_app.transformer_engine import (
    KVCacheOverflowError,
    MicroGQATransformer,
    PagedKVCacheManager,
)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class RuntimeState:
    def __init__(
        self,
        api_key: str,
        num_blocks: int = 128,
        block_size: int = 16,
        gpu_memory_utilization: float = 0.90,
        shm_dir: str = "/dev/shm",
    ) -> None:
        self.started_at = time.time()
        self.api_key = api_key
        self.model_id = "micro-gqa-llama-ref"
        self.pod_name = os.environ.get("POD_NAME", "inference-engine-0")
        self.pod_namespace = os.environ.get("POD_NAMESPACE", "ai-inference")
        self.node_name = os.environ.get("NODE_NAME", "aiops-ref-worker")
        self.gpu_uuid = os.environ.get("GPU_UUID", "GPU-ref-h100-80gb-0")
        self.total_vram_mib = 81920.0

        self.kv_manager = PagedKVCacheManager(
            num_blocks=num_blocks,
            block_size=block_size,
            shm_dir=shm_dir,
            max_utilization=gpu_memory_utilization,
        )
        self.engine = MicroGQATransformer(self.kv_manager)

        self._metrics_lock = threading.Lock()
        self.requests_total = 0
        self.prompt_tokens_total = 0
        self.generation_tokens_total = 0
        self.last_ttft_seconds = 0.008
        self.last_tpot_seconds = 0.003
        self.injected_xid_code = 0
        self.injected_nvlink_crc_errors = 0
        self.injected_gpu_temp_c = 54.0

    def record_latency(self, ttft_s: float, tpot_s: float, prompt_toks: int, gen_toks: int) -> None:
        with self._metrics_lock:
            self.requests_total += 1
            self.prompt_tokens_total += prompt_toks
            self.generation_tokens_total += gen_toks
            self.last_ttft_seconds = ttft_s
            self.last_tpot_seconds = tpot_s

    def render_prometheus_metrics(self) -> str:
        kv_ratio = self.kv_manager.usage_ratio
        # Map weights (~38 GiB) + KV cache usage onto an 80 GiB HBM telemetry twin
        weights_mib = 38912.0
        kv_pool_mib = (self.total_vram_mib * self.kv_manager.max_utilization - weights_mib) * kv_ratio
        fb_used_mib = min(self.total_vram_mib, weights_mib + max(0.0, kv_pool_mib))
        fb_free_mib = max(0.0, self.total_vram_mib - fb_used_mib)
        active = self.kv_manager.active_requests
        gpu_util = min(100.0, 12.0 + active * 22.0 + kv_ratio * 35.0)

        labels_vllm = (
            f'model_name="{self.model_id}",'
            f'namespace="{self.pod_namespace}",'
            f'pod="{self.pod_name}",'
            f'node="{self.node_name}"'
        )
        labels_dcgm = (
            f'gpu="0",UUID="{self.gpu_uuid}",'
            f'Hostname="{self.node_name}",'
            f'namespace="{self.pod_namespace}",'
            f'pod="{self.pod_name}",'
            f'container="inference-engine"'
        )
        with self._metrics_lock:
            ttft = self.last_ttft_seconds
            tpot = self.last_tpot_seconds
            req_total = self.requests_total
            p_toks = self.prompt_tokens_total
            g_toks = self.generation_tokens_total
            xid = self.injected_xid_code
            nvlink_crc = self.injected_nvlink_crc_errors
            temp_c = self.injected_gpu_temp_c

        lines = [
            "# HELP vllm:gpu_cache_usage_perc Fraction of Paged KV-Cache blocks currently allocated.",
            "# TYPE vllm:gpu_cache_usage_perc gauge",
            f"vllm:gpu_cache_usage_perc{{{labels_vllm}}} {kv_ratio:.4f}",
            "# HELP vllm:num_requests_running Number of requests actively executing inference.",
            "# TYPE vllm:num_requests_running gauge",
            f"vllm:num_requests_running{{{labels_vllm}}} {active}",
            "# HELP vllm:num_requests_waiting Number of requests queued awaiting KV blocks.",
            "# TYPE vllm:num_requests_waiting gauge",
            f"vllm:num_requests_waiting{{{labels_vllm}}} {self.kv_manager.waiting_requests}",
            "# HELP vllm:num_preemptions_total Cumulative KV-Cache block preemptions.",
            "# TYPE vllm:num_preemptions_total counter",
            f"vllm:num_preemptions_total{{{labels_vllm}}} {self.kv_manager.preemptions_total}",
            "# HELP vllm:time_to_first_token_seconds Most recent prefill TTFT in seconds.",
            "# TYPE vllm:time_to_first_token_seconds gauge",
            f"vllm:time_to_first_token_seconds{{{labels_vllm}}} {ttft:.6f}",
            "# HELP vllm:time_per_output_token_seconds Most recent decode TPOT in seconds.",
            "# TYPE vllm:time_per_output_token_seconds gauge",
            f"vllm:time_per_output_token_seconds{{{labels_vllm}}} {tpot:.6f}",
            "# HELP vllm:request_success_total Completed inference requests.",
            "# TYPE vllm:request_success_total counter",
            f"vllm:request_success_total{{{labels_vllm}}} {req_total}",
            "# HELP vllm:prompt_tokens_total Total prefill tokens processed.",
            "# TYPE vllm:prompt_tokens_total counter",
            f"vllm:prompt_tokens_total{{{labels_vllm}}} {p_toks}",
            "# HELP vllm:generation_tokens_total Total generated output tokens.",
            "# TYPE vllm:generation_tokens_total counter",
            f"vllm:generation_tokens_total{{{labels_vllm}}} {g_toks}",
            "# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used in MiB.",
            "# TYPE DCGM_FI_DEV_FB_USED gauge",
            f"DCGM_FI_DEV_FB_USED{{{labels_dcgm}}} {fb_used_mib:.2f}",
            "# HELP DCGM_FI_DEV_FB_FREE Framebuffer memory free in MiB.",
            "# TYPE DCGM_FI_DEV_FB_FREE gauge",
            f"DCGM_FI_DEV_FB_FREE{{{labels_dcgm}}} {fb_free_mib:.2f}",
            "# HELP DCGM_FI_DEV_GPU_UTIL GPU compute utilization percentage.",
            "# TYPE DCGM_FI_DEV_GPU_UTIL gauge",
            f"DCGM_FI_DEV_GPU_UTIL{{{labels_dcgm}}} {gpu_util:.2f}",
            "# HELP DCGM_FI_DEV_GPU_TEMP GPU temperature in Celsius.",
            "# TYPE DCGM_FI_DEV_GPU_TEMP gauge",
            f"DCGM_FI_DEV_GPU_TEMP{{{labels_dcgm}}} {temp_c:.1f}",
            "# HELP DCGM_FI_DEV_XID_ERRORS Last observed NVIDIA XID error code.",
            "# TYPE DCGM_FI_DEV_XID_ERRORS gauge",
            f"DCGM_FI_DEV_XID_ERRORS{{{labels_dcgm}}} {xid}",
            "# HELP DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL NVLink CRC flit error counter.",
            "# TYPE DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL counter",
            f"DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL{{{labels_dcgm}}} {nvlink_crc}",
            "",
        ]
        return "\n".join(lines)


def make_handler(state: RuntimeState):
    class InferenceRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep server stdout concise during high-concurrency benchmarks
            if os.environ.get("REF_VERBOSE_HTTP") == "1":
                super().log_message(fmt, *args)

        def _check_auth(self) -> bool:
            if not state.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            expected = f"Bearer {state.api_key}"
            if auth != expected:
                self._send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {"error": {"message": "Unauthorized: valid Bearer token required", "type": "auth_error"}},
                )
                return False
            return True

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def _read_json_body(self) -> Optional[Dict[str, Any]]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body"})
                return None

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send_json(HTTPStatus.OK, {"status": "ok", "uptime_s": round(time.time() - state.started_at, 2)})
                return

            if self.path == "/readyz":
                shm_ok = state.kv_manager.check_shm_ready()
                code = HTTPStatus.OK if shm_ok else HTTPStatus.SERVICE_UNAVAILABLE
                self._send_json(
                    code,
                    {
                        "status": "ready" if shm_ok else "degraded",
                        "backend": state.engine.backend,
                        "shm_ipc_mounted": shm_ok,
                        "effective_kv_blocks": state.kv_manager.effective_blocks,
                        "used_kv_blocks": state.kv_manager.used_blocks,
                    },
                )
                return

            if self.path == "/metrics":
                text = state.render_prometheus_metrics().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(text)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(text)
                self.wfile.flush()
                return

            if self.path == "/v1/models":
                if not self._check_auth():
                    return
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": state.model_id,
                                "object": "model",
                                "owned_by": "aiops-reference-impl",
                                "backend": state.engine.backend,
                            }
                        ],
                    },
                )
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint {self.path}"})

        def do_POST(self) -> None:
            if not self._check_auth():
                return

            if self.path == "/admin/fault-inject":
                payload = self._read_json_body()
                if payload is None:
                    return
                mode = payload.get("mode", "reset")
                if mode == "reset":
                    state.kv_manager.set_pinned_pressure_ratio(0.0)
                    with state._metrics_lock:
                        state.injected_xid_code = 0
                        state.injected_nvlink_crc_errors = 0
                        state.injected_gpu_temp_c = 54.0
                else:
                    if "kv_pressure_ratio" in payload:
                        state.kv_manager.set_pinned_pressure_ratio(float(payload["kv_pressure_ratio"]))
                    if "preemptions_add" in payload:
                        state.kv_manager.preemptions_total += int(payload["preemptions_add"])
                    with state._metrics_lock:
                        if "xid_code" in payload:
                            state.injected_xid_code = int(payload["xid_code"])
                        if "nvlink_crc_errors" in payload:
                            state.injected_nvlink_crc_errors = int(payload["nvlink_crc_errors"])
                        if "ttft_seconds" in payload:
                            state.last_ttft_seconds = float(payload["ttft_seconds"])
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "applied",
                        "mode": mode,
                        "kv_usage_ratio": round(state.kv_manager.usage_ratio, 4),
                        "xid_code": state.injected_xid_code,
                    },
                )
                return

            if self.path in ("/v1/chat/completions", "/v1/completions"):
                payload = self._read_json_body()
                if payload is None:
                    return

                messages = payload.get("messages")
                if isinstance(messages, list) and messages:
                    prompt = " ".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
                else:
                    prompt = str(payload.get("prompt", "diagnose kubernetes gpu pod health"))

                max_tokens = int(payload.get("max_tokens", 32))
                stream = bool(payload.get("stream", False))
                req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                created_ts = int(time.time())
                t_start = time.perf_counter()

                try:
                    if stream:
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()

                        t_first: Optional[float] = None
                        prompt_len = 0
                        gen_count = 0
                        for tok_str, is_first, p_len in state.engine.generate_stream(req_id, prompt, max_tokens):
                            now = time.perf_counter()
                            if is_first:
                                t_first = now
                                prompt_len = p_len
                            gen_count += 1
                            chunk = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": state.model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": tok_str},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                            self.wfile.flush()

                        done_chunk = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": state.model_id,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                        }
                        self.wfile.write(f"data: {json.dumps(done_chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
                        self.wfile.flush()

                        t_end = time.perf_counter()
                        ttft_s = (t_first - t_start) if t_first is not None else (t_end - t_start)
                        tpot_s = ((t_end - t_first) / max(1, gen_count - 1)) if (t_first and gen_count > 1) else 0.001
                        state.record_latency(ttft_s, tpot_s, prompt_len, gen_count)
                    else:
                        pieces = []
                        t_first = None
                        prompt_len = 0
                        for tok_str, is_first, p_len in state.engine.generate_stream(req_id, prompt, max_tokens):
                            if is_first:
                                t_first = time.perf_counter()
                                prompt_len = p_len
                            pieces.append(tok_str)
                        t_end = time.perf_counter()
                        gen_count = len(pieces)
                        ttft_s = (t_first - t_start) if t_first is not None else (t_end - t_start)
                        tpot_s = ((t_end - t_first) / max(1, gen_count - 1)) if (t_first and gen_count > 1) else 0.001
                        state.record_latency(ttft_s, tpot_s, prompt_len, gen_count)
                        self._send_json(
                            HTTPStatus.OK,
                            {
                                "id": req_id,
                                "object": "chat.completion",
                                "created": created_ts,
                                "model": state.model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {"role": "assistant", "content": "".join(pieces).strip()},
                                        "finish_reason": "length",
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": prompt_len,
                                    "completion_tokens": gen_count,
                                    "total_tokens": prompt_len + gen_count,
                                },
                                "x_telemetry": {
                                    "ttft_ms": round(ttft_s * 1000.0, 3),
                                    "tpot_ms": round(tpot_s * 1000.0, 3),
                                    "backend": state.engine.backend,
                                },
                            },
                        )
                except KVCacheOverflowError as exc:
                    self._send_json(
                        HTTPStatus.TOO_MANY_REQUESTS,
                        {"error": {"message": str(exc), "type": "kv_cache_saturated"}},
                    )
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint {self.path}"})

    return InferenceRequestHandler


def create_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    api_key: str = "",
    num_blocks: int = 128,
    block_size: int = 16,
    gpu_memory_utilization: float = 0.90,
    shm_dir: str = "/dev/shm",
) -> Tuple[ThreadingHTTPServer, RuntimeState]:
    state = RuntimeState(
        api_key=api_key,
        num_blocks=num_blocks,
        block_size=block_size,
        gpu_memory_utilization=gpu_memory_utilization,
        shm_dir=shm_dir,
    )
    server = ThreadingHTTPServer((host, port), make_handler(state))
    return server, state


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference Paged KV-Cache OpenAI Inference Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--num-kv-blocks", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--shm-dir", default="/dev/shm")
    args = parser.parse_args()

    srv, state = create_server(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        num_blocks=args.num_kv_blocks,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        shm_dir=args.shm_dir,
    )
    print(
        f"[inference-engine] Ready on {args.host}:{args.port} | "
        f"backend={state.engine.backend} | effective_kv_blocks={state.kv_manager.effective_blocks}"
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.server_close()


if __name__ == "__main__":
    main()
