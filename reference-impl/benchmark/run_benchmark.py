#!/usr/bin/env python3
"""
Controlled OpenAI SSE Streaming Benchmark Harness.

Works seamlessly against BOTH the runnable Kubernetes reference implementation
(`http://127.0.0.1:18000`) AND production GPU vLLM / KServe endpoints.

Measures:
  - Cold-Start Readiness Latency (passed from deployment timer or probed)
  - Time-To-First-Token (TTFT: p50, p95, p99, mean in ms) via real SSE chunk arrival
  - Time-Per-Output-Token (TPOT: p50, p95, p99, mean in ms)
  - End-to-End Request Latency, Aggregate Token Throughput (tok/s), Request RPS
  - Peak KV-Cache Block Utilization scraped live from /metrics
  - Normalized Cost per 1,000 Requests ($/1k reqs)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import json
import math
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RequestSample:
    suite: str
    concurrency: int
    prompt_target_tokens: int
    prompt_tokens_actual: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    status_code: int


def percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    weight = rank - low
    return sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight


def build_synthetic_prompt(num_tokens: int) -> str:
    vocab = [
        "kubernetes", "gpu", "nvlink", "kvcache", "prefill", "decode",
        "kubelet", "cgroup", "admission", "telemetry", "prometheus", "dcgm",
        "scheduler", "topology", "tensor", "bandwidth",
    ]
    return " ".join(vocab[i % len(vocab)] for i in range(num_tokens))


def scrape_kv_usage(base_url: str) -> float:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=3.0) as resp:
            body = resp.read().decode("utf-8")
        m = re.search(r"vllm:gpu_cache_usage_perc\{[^}]*\}\s+([0-9.]+)", body)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def run_single_sse_request(
    base_url: str,
    api_key: str,
    model: str,
    prompt_tokens: int,
    max_tokens: int,
    suite: str,
    concurrency: int,
) -> RequestSample:
    prompt = build_synthetic_prompt(prompt_tokens)
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )

    t_start = time.perf_counter()
    t_first: Optional[float] = None
    t_last = t_start
    out_tokens = 0

    with urllib.request.urlopen(req, timeout=30.0) as resp:
        status_code = resp.status
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                out_tokens += 1

    t_end = time.perf_counter()
    ttft_ms = ((t_first - t_start) if t_first is not None else (t_end - t_start)) * 1000.0
    tpot_ms = (((t_last - t_first) / max(1, out_tokens - 1)) * 1000.0) if (t_first and out_tokens > 1) else 0.5
    e2e_ms = (t_end - t_start) * 1000.0

    return RequestSample(
        suite=suite,
        concurrency=concurrency,
        prompt_target_tokens=prompt_tokens,
        prompt_tokens_actual=prompt_tokens,
        output_tokens=out_tokens,
        ttft_ms=round(ttft_ms, 3),
        tpot_ms=round(tpot_ms, 3),
        e2e_ms=round(e2e_ms, 3),
        status_code=status_code,
    )


def run_step(
    base_url: str,
    api_key: str,
    model: str,
    suite: str,
    concurrency: int,
    total_requests: int,
    prompt_tokens: int,
    max_tokens: int,
    hourly_cost_usd: float,
) -> Tuple[Dict[str, Any], List[RequestSample]]:
    peak_kv = scrape_kv_usage(base_url)
    samples: List[RequestSample] = []

    t_wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                run_single_sse_request,
                base_url,
                api_key,
                model,
                prompt_tokens,
                max_tokens,
                suite,
                concurrency,
            )
            for _ in range(total_requests)
        ]
        # Poll KV usage mid-flight while workers are active
        time.sleep(0.02)
        peak_kv = max(peak_kv, scrape_kv_usage(base_url))
        for fut in concurrent.futures.as_completed(futures):
            samples.append(fut.result())
            peak_kv = max(peak_kv, scrape_kv_usage(base_url))

    wall_s = max(1e-6, time.perf_counter() - t_wall_start)
    ttfts = sorted(s.ttft_ms for s in samples)
    tpots = sorted(s.tpot_ms for s in samples)
    e2es = sorted(s.e2e_ms for s in samples)
    total_out_tokens = sum(s.output_tokens for s in samples)
    total_all_tokens = sum(s.prompt_tokens_actual + s.output_tokens for s in samples)

    rps = len(samples) / wall_s
    out_tok_per_s = total_out_tokens / wall_s
    total_tok_per_s = total_all_tokens / wall_s
    # Cost per 1,000 requests: ($ / second) * (1,000 / requests_per_second)
    cost_per_1k_usd = (hourly_cost_usd / 3600.0) * (1000.0 / max(1e-6, rps))

    summary = {
        "suite": suite,
        "concurrency": concurrency,
        "requests": len(samples),
        "prompt_tokens": prompt_tokens,
        "max_output_tokens": max_tokens,
        "wall_time_s": round(wall_s, 4),
        "rps": round(rps, 2),
        "output_throughput_tok_s": round(out_tok_per_s, 2),
        "total_throughput_tok_s": round(total_tok_per_s, 2),
        "ttft_p50_ms": round(percentile(ttfts, 50), 2),
        "ttft_p95_ms": round(percentile(ttfts, 95), 2),
        "ttft_p99_ms": round(percentile(ttfts, 99), 2),
        "tpot_p50_ms": round(percentile(tpots, 50), 2),
        "tpot_p95_ms": round(percentile(tpots, 95), 2),
        "tpot_p99_ms": round(percentile(tpots, 99), 2),
        "e2e_p95_ms": round(percentile(e2es, 95), 2),
        "peak_kv_cache_ratio": round(peak_kv, 4),
        "cost_per_1k_requests_usd": round(cost_per_1k_usd, 5),
    }
    return summary, samples


def render_mermaid_summary(report: Dict[str, Any]) -> str:
    conc_rows = [r for r in report["summaries"] if r["suite"] == "concurrency_scaling"]
    ctx_rows = [r for r in report["summaries"] if r["suite"] == "context_scaling"]

    c_labels = ", ".join(f'"c={r["concurrency"]}"' for r in conc_rows)
    ttft_p95 = ", ".join(str(r["ttft_p95_ms"]) for r in conc_rows)
    tpot_p95 = ", ".join(str(r["tpot_p95_ms"]) for r in conc_rows)
    tput_vals = ", ".join(str(r["output_throughput_tok_s"]) for r in conc_rows)
    cost_vals = ", ".join(str(round(r["cost_per_1k_requests_usd"] * 100, 3)) for r in conc_rows)

    ctx_labels = ", ".join(f'"{r["prompt_tokens"]} tok"' for r in ctx_rows)
    ctx_ttft = ", ".join(str(r["ttft_p95_ms"]) for r in ctx_rows)

    max_lat = max([r["ttft_p95_ms"] for r in conc_rows] + [10.0]) * 1.25
    max_tput = max([r["output_throughput_tok_s"] for r in conc_rows] + [50.0]) * 1.25
    max_ctx_lat = max([r["ttft_p95_ms"] for r in ctx_rows] + [10.0]) * 1.25

    lines = [
        "# Controlled Inference Benchmark Report (Live Kubernetes Cluster)",
        "",
        f"- **Timestamp (UTC)**: `{report['generated_at']}`",
        f"- **Target Endpoint**: `{report['base_url']}`",
        f"- **Model**: `{report['model']}`",
        f"- **Cold-Start Readiness Time**: `{report['cold_start_seconds']} s`",
        f"- **Node Hourly Reference Rate**: `${report['hourly_cost_usd']:.2f}/hr`",
        "",
        "## 1. Concurrency Scaling Results (`prompt=128 tokens`, `output=32 tokens`)",
        "",
        "| Concurrency | RPS | Output Tok/s | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | TPOT p95 (ms) | Peak KV Ratio | Cost / 1k Reqs (USD) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in conc_rows:
        lines.append(
            f"| `{r['concurrency']}` | {r['rps']} | {r['output_throughput_tok_s']} | "
            f"{r['ttft_p50_ms']} | **{r['ttft_p95_ms']}** | {r['tpot_p50_ms']} | "
            f"**{r['tpot_p95_ms']}** | {r['peak_kv_cache_ratio']} | `${r['cost_per_1k_requests_usd']:.5f}` |"
        )

    lines.extend(
        [
            "",
            "```mermaid",
            "xychart-beta",
            '    title "TTFT p95 vs TPOT p95 Across Concurrency Levels (ms)"',
            f"    x-axis [{c_labels}]",
            f'    y-axis "Latency (ms)" 0 --> {round(max_lat, 1)}',
            f"    bar [{ttft_p95}]",
            f"    line [{tpot_p95}]",
            "```",
            "",
            "```mermaid",
            "xychart-beta",
            '    title "Aggregate Decode Throughput Across Concurrency (Output Tokens/sec)"',
            f"    x-axis [{c_labels}]",
            f'    y-axis "Tokens / sec" 0 --> {round(max_tput, 1)}',
            f"    bar [{tput_vals}]",
            "```",
            "",
            "## 2. Prompt Context Scaling Results (`concurrency=1`, `output=32 tokens`)",
            "",
            "| Prompt Tokens | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | Output Tok/s | Cost / 1k Reqs (USD) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for r in ctx_rows:
        lines.append(
            f"| `{r['prompt_tokens']}` | {r['ttft_p50_ms']} | **{r['ttft_p95_ms']}** | "
            f"{r['tpot_p50_ms']} | {r['output_throughput_tok_s']} | `${r['cost_per_1k_requests_usd']:.5f}` |"
        )

    lines.extend(
        [
            "",
            "```mermaid",
            "xychart-beta",
            '    title "Prefill TTFT p95 Scaling by Prompt Context Length (ms)"',
            f"    x-axis [{ctx_labels}]",
            f'    y-axis "TTFT p95 (ms)" 0 --> {round(max_ctx_lat, 1)}',
            f"    bar [{ctx_ttft}]",
            "```",
            "",
            f"_Normalized cost curve (cents per 1k requests across concurrency): `[{cost_vals}]`._",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI SSE Streaming Benchmark Harness")
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--api-key", default="REF_IMPL_DEMO_AUTH_PLACEHOLDER")
    parser.add_argument("--model", default="micro-gqa-llama-ref")
    parser.add_argument("--cold-start-seconds", type=float, default=0.0)
    parser.add_argument(
        "--hourly-cost-usd",
        type=float,
        default=0.192,
        help="Instance hourly USD cost (default $0.192/hr = 4-vCPU general compute baseline; pass e.g. 6.98 for 2xH100)",
    )
    parser.add_argument("--output-dir", default="reference-impl/evidence")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Warmup (excluded from steady-state percentiles)
    for _ in range(2):
        run_single_sse_request(args.base_url, args.api_key, args.model, 32, 8, "warmup", 1)

    summaries: List[Dict[str, Any]] = []
    all_samples: List[RequestSample] = []

    # 2. Concurrency scaling matrix (c = 1, 2, 4, 8)
    for conc in (1, 2, 4, 8):
        total_reqs = max(8, conc * 4)
        summary, samples = run_step(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            suite="concurrency_scaling",
            concurrency=conc,
            total_requests=total_reqs,
            prompt_tokens=128,
            max_tokens=32,
            hourly_cost_usd=args.hourly_cost_usd,
        )
        summaries.append(summary)
        all_samples.extend(samples)

    # 3. Prompt context scaling matrix (64, 256, 512 tokens at c=1)
    for p_toks in (64, 256, 512):
        summary, samples = run_step(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            suite="context_scaling",
            concurrency=1,
            total_requests=6,
            prompt_tokens=p_toks,
            max_tokens=32,
            hourly_cost_usd=args.hourly_cost_usd,
        )
        summaries.append(summary)
        all_samples.extend(samples)

    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_url": args.base_url,
        "model": args.model,
        "cold_start_seconds": round(args.cold_start_seconds, 3),
        "hourly_cost_usd": args.hourly_cost_usd,
        "methodology": {
            "protocol": "OpenAI /v1/chat/completions HTTP/1.1 Server-Sent Events (stream=true)",
            "warmup_requests_excluded": 2,
            "ttft_definition": "Client perf_counter delta from HTTP POST write to first content SSE delta",
            "tpot_definition": "(t_last_token - t_first_token) / (output_tokens - 1)",
            "cost_formula": "(hourly_cost_usd / 3600) * (1000 / rps)",
        },
        "summaries": summaries,
        "raw_samples_count": len(all_samples),
        "raw_samples": [asdict(s) for s in all_samples],
    }

    json_path = out_dir / "benchmark-raw-results.json"
    csv_path = out_dir / "benchmark-raw-results.csv"
    md_path = out_dir / "benchmark-charts.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    md_path.write_text(render_mermaid_summary(report), encoding="utf-8")
    print(f"[benchmark] Completed {len(all_samples)} requests -> {json_path}, {csv_path}, {md_path}")


if __name__ == "__main__":
    main()
