"""
Statistical and Mathematical Anomaly Detection Engine for Kubernetes GPU Telemetry.
Implements:
1. EWMA + Rolling Z-Score Detector for sudden spikes (TTFT, TPOT, NVLink CRC flits, GPU Temp).
2. Monotonic Saturation Slope Estimator (OLS Linear Regression over sliding window)
   for VRAM MiB/min growth rate and Time-To-Exhaustion (TTE) prediction.
3. Benign Prefill Burst Filter for single-sample GPU compute spikes without KV saturation.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from aiops.models import AnomalySignal, TelemetryPoint


class RollingZScoreDetector:
    """
    Online statistical anomaly detector combining Exponentially Weighted Moving Average (EWMA)
    and rolling sample standard deviation over a sliding window.
    """

    def __init__(
        self,
        window_size: int = 15,
        alpha: float = 0.25,
        z_threshold: float = 3.0,
        min_samples: int = 3,
        epsilon: float = 1e-5,
    ) -> None:
        self.window_size = window_size
        self.alpha = alpha
        self.z_threshold = z_threshold
        self.min_samples = min_samples
        self.epsilon = epsilon
        # Key: (node, pod, metric) -> deque of (timestamp, value)
        self.history: Dict[Tuple[str, str, str], deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        # Key: (node, pod, metric) -> current EWMA mean
        self.ewma_mean: Dict[Tuple[str, str, str], float] = {}

    def process(self, point: TelemetryPoint) -> Optional[AnomalySignal]:
        key = (point.node, point.pod, point.metric)
        dq = self.history[key]

        current_val = point.value
        # If we have enough history, compute baseline statistics from prior observations
        anomaly = None
        if len(dq) >= self.min_samples:
            values = [v for _, v in dq]
            n = len(values)
            base_mean = sum(values) / n
            variance = sum((v - base_mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
            std_dev = math.sqrt(variance)

            # Rolling Z-score against pre-sample baseline
            effective_std = max(std_dev, self.epsilon)
            z = (current_val - base_mean) / effective_std

            if z >= self.z_threshold:
                anomaly = AnomalySignal(
                    metric=point.metric,
                    node=point.node,
                    pod=point.pod,
                    gpu_uuid=point.gpu_uuid,
                    observed_value=round(current_val, 4),
                    baseline_mean=round(base_mean, 4),
                    z_score=round(z, 2),
                    slope_per_min=0.0,
                    anomaly_type="SPIKE",
                    timestamp=point.timestamp,
                    details={
                        "window_samples": n,
                        "sample_std_dev": round(std_dev, 4),
                        "detector": "RollingZScoreDetector",
                    },
                )

        # Update EWMA and history
        dq.append((point.timestamp, current_val))
        if key not in self.ewma_mean:
            self.ewma_mean[key] = current_val
        else:
            self.ewma_mean[key] = self.alpha * current_val + (1.0 - self.alpha) * self.ewma_mean[key]

        return anomaly


class SaturationSlopeEstimator:
    """
    Sliding window ordinary least squares (OLS) linear regression estimator.
    Computes VRAM growth rate (MiB/min), evaluates monotonic linearity (R^2),
    and predicts Time-To-Exhaustion (TTE in minutes) before OOM.
    """

    def __init__(
        self,
        default_capacity_mib: float = 81920.0,  # 80 GiB (A100 / H100)
        min_slope_mib_per_min: float = 200.0,
        r2_threshold: float = 0.80,
        tte_alert_threshold_minutes: float = 15.0,
        window_size: int = 10,
    ) -> None:
        self.default_capacity_mib = default_capacity_mib
        self.min_slope_mib_per_min = min_slope_mib_per_min
        self.r2_threshold = r2_threshold
        self.tte_alert_threshold_minutes = tte_alert_threshold_minutes
        self.window_size = window_size
        # Key: (node, pod, gpu_uuid) -> deque of (timestamp, value)
        self.history: Dict[Tuple[str, str, Optional[str]], deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def process(
        self, point: TelemetryPoint, capacity_mib: Optional[float] = None
    ) -> Optional[AnomalySignal]:
        if point.metric != "DCGM_FI_DEV_FB_USED":
            return None

        cap = capacity_mib or self.default_capacity_mib
        key = (point.node, point.pod, point.gpu_uuid)
        dq = self.history[key]
        dq.append((point.timestamp, point.value))

        if len(dq) < 3:
            return None

        pts = list(dq)
        t0 = pts[0][0]
        # Express delta_t in minutes
        t_minutes = [(t - t0) / 60.0 for t, _ in pts]
        y_values = [v for _, v in pts]
        n = len(pts)

        # OLS regression: y = m * t + c
        mean_t = sum(t_minutes) / n
        mean_y = sum(y_values) / n

        s_tt = sum((t - mean_t) ** 2 for t in t_minutes)
        s_ty = sum((t - mean_t) * (y - mean_y) for t, y in zip(t_minutes, y_values))
        s_yy = sum((y - mean_y) ** 2 for y in y_values)

        if s_tt <= 1e-7:
            return None

        slope = s_ty / s_tt  # MiB per minute

        # R^2 calculation
        r_squared = (s_ty ** 2) / (s_tt * s_yy) if s_yy > 1e-7 else 1.0

        current_val = point.value
        headroom_mib = max(0.0, cap - current_val)
        tte_minutes = (headroom_mib / slope) if slope > 1e-5 else float("inf")

        # Check conditions for critical saturation slope
        is_high_util = (current_val / cap) >= 0.85
        is_fast_growth = (slope >= self.min_slope_mib_per_min) and (r_squared >= self.r2_threshold)

        if (is_fast_growth and tte_minutes <= self.tte_alert_threshold_minutes) or (
            is_high_util and slope > 100.0 and r_squared >= 0.70
        ):
            # Baseline mean before rapid climb
            baseline_val = pts[0][1]
            z_est = (current_val - baseline_val) / max(100.0, abs(mean_y * 0.05))

            return AnomalySignal(
                metric=point.metric,
                node=point.node,
                pod=point.pod,
                gpu_uuid=point.gpu_uuid,
                observed_value=round(current_val, 2),
                baseline_mean=round(mean_y, 2),
                z_score=round(z_est, 2),
                slope_per_min=round(slope, 2),
                anomaly_type="SATURATION_SLOPE",
                timestamp=point.timestamp,
                details={
                    "r_squared": round(r_squared, 4),
                    "capacity_mib": round(cap, 2),
                    "headroom_mib": round(headroom_mib, 2),
                    "tte_minutes": round(tte_minutes, 2),
                    "utilization_pct": round((current_val / cap) * 100.0, 2),
                    "detector": "SaturationSlopeEstimator",
                },
            )

        return None


class BenignBurstFilter:
    """
    Evaluates whether a metric surge (e.g. DCGM_FI_DEV_GPU_UTIL or transient TTFT) is a
    benign prompt prefill burst or a genuine production failure.
    Transient bursts show:
    - High compute utilization (e.g. 90-100% for 1-2 points)
    - Zero KV cache preemption (vllm:num_preemptions_total == 0)
    - Stable or low VRAM usage (< 80%)
    - KV cache usage perc < 0.85
    - Pod healthy and running
    """

    @staticmethod
    def is_benign_burst(
        util_value: float,
        preemptions: float,
        vram_pct: float,
        kv_cache_usage_pct: float,
        pod_status: str = "Running",
    ) -> bool:
        if pod_status.lower() not in ("running", ""):
            return False

        # If compute spiked but KV cache and VRAM are well within safe thresholds
        # and preemption count is zero, this is standard prompt prefill matrix-multiplication.
        if (
            util_value > 70.0
            and preemptions <= 0.0
            and vram_pct < 85.0
            and kv_cache_usage_pct < 0.85
        ):
            return True

        return False


class UnifiedAnomalyDetector:
    """
    Orchestrates statistical detectors across all streaming telemetry points.
    Applies benign prefill filtering to suppress false positives.
    """

    def __init__(self, default_capacity_mib: float = 81920.0) -> None:
        self.z_detector = RollingZScoreDetector(window_size=15, z_threshold=3.0)
        self.slope_estimator = SaturationSlopeEstimator(default_capacity_mib=default_capacity_mib)
        self.benign_filter = BenignBurstFilter()

        # Track contextual metrics per (node, pod) to correlate during filtering
        self.context_cache: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)

    def process_point(self, point: TelemetryPoint) -> Optional[AnomalySignal]:
        ctx_key = (point.node, point.pod)
        self.context_cache[ctx_key][point.metric] = point.value

        # 1. Check for Monotonic VRAM Saturation
        if point.metric == "DCGM_FI_DEV_FB_USED":
            slope_signal = self.slope_estimator.process(point)
            if slope_signal:
                return slope_signal

        # 2. Check for GPU compute util spikes (apply Benign Burst filter)
        if point.metric in ("DCGM_FI_DEV_GPU_UTIL", "gpu_utilization_perc"):
            preemptions = self.context_cache[ctx_key].get("vllm:num_preemptions_total", 0.0)
            fb_used = self.context_cache[ctx_key].get("DCGM_FI_DEV_FB_USED", 0.0)
            vram_pct = (fb_used / 81920.0) * 100.0 if fb_used > 0 else 0.0
            kv_pct = self.context_cache[ctx_key].get("vllm:gpu_cache_usage_perc", 0.0)

            if self.benign_filter.is_benign_burst(point.value, preemptions, vram_pct, kv_pct):
                # Suppress benign prefill burst
                return None

        # 3. Rolling Z-Score on latency, preemptions, hardware errors, temp
        monitored_spike_metrics = {
            "vllm:num_preemptions_total",
            "vllm:time_to_first_token_seconds",
            "vllm:time_per_output_token_seconds",
            "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL",
            "DCGM_FI_DEV_GPU_TEMP",
            "DCGM_FI_DEV_XID_ERRORS",
            "container_memory_working_set_bytes",
            "cgroup_memory_working_set_bytes",
        }

        if point.metric in monitored_spike_metrics:
            spike_signal = self.z_detector.process(point)
            if spike_signal:
                return spike_signal

        return None

    def process_stream(self, stream: List[TelemetryPoint]) -> List[AnomalySignal]:
        """
        Process a list of ordered telemetry points and return all detected non-benign anomalies.
        """
        anomalies: List[AnomalySignal] = []
        for pt in stream:
            sig = self.process_point(pt)
            if sig:
                anomalies.append(sig)
        return anomalies
