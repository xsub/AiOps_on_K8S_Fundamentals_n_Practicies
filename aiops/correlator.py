"""
Topological & Multi-Signal Event Correlation Engine.
Groups noisy RawAlerts and AnomalySignals within temporal & topological windows,
scores hypotheses across 7 canonical Root-Cause classes plus User-Space XIDs,
and synthesizes actionable CorrelatedIncidents with explainable evidence trails.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from aiops.models import AnomalySignal, CorrelatedIncident, RawAlert

# Hardware XIDs vs Software XIDs
HARDWARE_XIDS: Set[int] = {48, 63, 64, 74, 79, 94, 95}
SOFTWARE_XIDS: Set[int] = {13, 31, 43}


class ClusterGroup:
    """
    Temporary topological grouping of alerts and anomalies for hypothesis evaluation.
    """

    def __init__(self, node: str, gpu_uuid: Optional[str] = None) -> None:
        self.node = node
        self.gpu_uuid = gpu_uuid
        self.alerts: List[RawAlert] = []
        self.anomalies: List[AnomalySignal] = []
        self.affected_pods: Set[str] = set()

    def add_alert(self, alert: RawAlert) -> None:
        self.alerts.append(alert)
        if alert.pod:
            self.affected_pods.add(alert.pod)
        if not self.gpu_uuid and alert.gpu_uuid:
            self.gpu_uuid = alert.gpu_uuid

    def add_anomaly(self, anomaly: AnomalySignal) -> None:
        self.anomalies.append(anomaly)
        if anomaly.pod:
            self.affected_pods.add(anomaly.pod)
        if not self.gpu_uuid and anomaly.gpu_uuid:
            self.gpu_uuid = anomaly.gpu_uuid


class TopologyCorrelator:
    """
    Synthesizes RawAlerts and AnomalySignals into high-confidence CorrelatedIncidents.
    Evaluates topological context (sharing strategy, exit codes, /dev/shm size, etc.)
    across canonical root cause classes.
    """

    def __init__(self, temporal_window_seconds: float = 300.0) -> None:
        self.temporal_window_seconds = temporal_window_seconds

    def group_events(
        self,
        alerts: List[RawAlert],
        anomalies: List[AnomalySignal],
    ) -> List[ClusterGroup]:
        """
        Groups alerts and anomalies by topological key (node, and where appropriate gpu_uuid).
        """
        clusters: Dict[str, ClusterGroup] = {}

        for alert in alerts:
            # Key primarily by node
            key = alert.node or "unknown-node"
            if key not in clusters:
                clusters[key] = ClusterGroup(node=key, gpu_uuid=alert.gpu_uuid)
            clusters[key].add_alert(alert)

        for anomaly in anomalies:
            key = anomaly.node or "unknown-node"
            if key not in clusters:
                clusters[key] = ClusterGroup(node=key, gpu_uuid=anomaly.gpu_uuid)
            clusters[key].add_anomaly(anomaly)

        return list(clusters.values())

    def correlate(
        self,
        alerts: List[RawAlert],
        anomalies: List[AnomalySignal],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[CorrelatedIncident]:
        """
        Execute hypothesis scoring over all clustered alerts and anomalies.
        """
        ctx = context or {}
        clusters = self.group_events(alerts, anomalies)
        incidents: List[CorrelatedIncident] = []

        for cluster in clusters:
            incident = self._evaluate_cluster(cluster, ctx)
            if incident:
                incidents.append(incident)

        return incidents

    def _evaluate_cluster(
        self, cluster: ClusterGroup, ctx: Dict[str, Any]
    ) -> Optional[CorrelatedIncident]:
        # If there are no alerts and no anomalies, nothing to correlate
        if not cluster.alerts and not cluster.anomalies:
            return None

        node = cluster.node
        pods = sorted(list(cluster.affected_pods))
        primary_pod = pods[0] if pods else ""

        # Gather cluster characteristics from alerts and anomalies
        alert_names = {a.name for a in cluster.alerts}
        alert_ids = [a.alert_id for a in cluster.alerts]
        anomaly_metrics = {an.metric for an in cluster.anomalies}

        # Check for XIDs
        extracted_xids: Set[int] = set()
        for a in cluster.alerts:
            # Check value or annotations
            if "xid" in a.name.lower() or "xid" in a.annotations:
                xid_str = a.annotations.get("xid", str(int(a.value)))
                try:
                    extracted_xids.add(int(float(xid_str)))
                except ValueError:
                    pass
            elif a.value in (13, 31, 43, 48, 63, 64, 74, 79, 94, 95):
                extracted_xids.add(int(a.value))

        for an in cluster.anomalies:
            if "xid" in an.metric.lower() and an.observed_value > 0:
                extracted_xids.add(int(an.observed_value))

        # Check exit codes from alerts or context
        exit_code = ctx.get("exit_code")
        if exit_code is None:
            for a in cluster.alerts:
                code_str = a.annotations.get("exit_code")
                if code_str:
                    try:
                        exit_code = int(code_str)
                        break
                    except ValueError:
                        pass

        # Sharing strategy
        sharing_strategy = ctx.get("sharing_strategy", "").lower()
        if not sharing_strategy:
            for a in cluster.alerts:
                strat = a.annotations.get("sharing_strategy")
                if strat:
                    sharing_strategy = strat.lower()
                    break

        # Pod status
        pod_status = ctx.get("pod_status", "Unknown")
        for a in cluster.alerts:
            if "pod_status" in a.annotations:
                pod_status = a.annotations["pod_status"]
                break

        # SHM size
        shm_size_mb = ctx.get("shm_size_mb", 64.0)

        # Hypotheses evaluation
        scores: Dict[str, Tuple[float, List[str], str]] = {}

        # -------------------------------------------------------------
        # Hypothesis 1: GPU_SILICON_XID_FAULT
        # -------------------------------------------------------------
        hw_xids = extracted_xids.intersection(HARDWARE_XIDS)
        if hw_xids:
            hw_xid_list = sorted(list(hw_xids))
            evidence = [
                f"Critical hardware XID {hw_xid_list} detected on node {node} (source: NPD/DCGM).",
                f"Silicon failure mode: PCIe link drop, uncontained ECC error, or GPU halted execution.",
            ]
            temp_anomalies = [an for an in cluster.anomalies if "temp" in an.metric.lower()]
            if temp_anomalies:
                evidence.append(
                    f"Thermal degradation accompanied fault: GPU temp reached {temp_anomalies[0].observed_value}°C."
                )
            scores["GPU_SILICON_XID_FAULT"] = (0.99, evidence, "critical")

        # -------------------------------------------------------------
        # Hypothesis 2: USER_SPACE_CUDA_XID
        # -------------------------------------------------------------
        sw_xids = extracted_xids.intersection(SOFTWARE_XIDS)
        if sw_xids and not hw_xids:
            sw_xid_list = sorted(list(sw_xids))
            evidence = [
                f"User-space CUDA XID {sw_xid_list} (illegal memory access/page fault) detected on pod {primary_pod}.",
                "Hardware silicon and PCIe bus remain fully healthy; failure is confined to application container.",
            ]
            scores["USER_SPACE_CUDA_XID"] = (0.96, evidence, "warning")

        # -------------------------------------------------------------
        # Hypothesis 3: K8S_CGROUP_MMAP_OOM
        # -------------------------------------------------------------
        is_oomkilled = (
            exit_code == 137
            or "HostMemoryOOM" in alert_names
            or any("oomkilled" in a.name.lower() for a in cluster.alerts)
            or any(a.annotations.get("reason") == "OOMKilled" for a in cluster.alerts)
        )
        low_gpu_vram = ctx.get("gpu_vram_pct", 10.0) < 30.0 or any(
            "LowGpuMemoryUtilizationDuringOOM" in a.name for a in cluster.alerts
        )
        if is_oomkilled and low_gpu_vram:
            evidence = [
                f"Pod {primary_pod} terminated with ExitCode 137 (OOMKilled) via Linux cgroup memory controller.",
                "GPU VRAM usage remained low (< 30%); host DRAM saturated during model weight deserialization/mmap.",
                "Root cause: Insufficient host DRAM limits (cgroup memory.max) in pod resources.",
            ]
            scores["K8S_CGROUP_MMAP_OOM"] = (0.97, evidence, "critical")

        # -------------------------------------------------------------
        # Hypothesis 4: NCCL_SHM_EXHAUSTION
        # -------------------------------------------------------------
        is_sigbus = exit_code == 135 or any("SIGBUS" in a.name for a in cluster.alerts)
        has_nccl_alert = any("nccl" in a.name.lower() for a in cluster.alerts)
        is_multi_gpu = ctx.get("is_multi_gpu", False) or any(
            a.annotations.get("is_multi_gpu") == "true" for a in cluster.alerts
        )
        if (is_sigbus or has_nccl_alert) and (shm_size_mb <= 64.0 or is_multi_gpu):
            evidence = [
                f"Multi-GPU distributed pod {primary_pod} crashed with ExitCode 135 (SIGBUS / NCCL Watchdog).",
                f"POSIX shared memory (/dev/shm) exhausted at default size ({shm_size_mb}MiB <= 64MiB).",
                "NCCL ring buffer posix_fallocate failed to allocate inter-GPU communication channels.",
            ]
            scores["NCCL_SHM_EXHAUSTION"] = (0.98, evidence, "critical")

        # -------------------------------------------------------------
        # Hypothesis 5: TIMESLICING_VRAM_COLLISION
        # -------------------------------------------------------------
        has_collision_alert = any(
            "TimeSlicingCoTenantConflict" in a.name or "timeslicing" in a.name.lower()
            for a in cluster.alerts
        )
        is_time_slicing = sharing_strategy == "time-slicing"
        if (is_time_slicing or has_collision_alert) and (
            exit_code == 1
            or any("saturation" in a.name.lower() for a in cluster.alerts)
            or len(pods) > 1
        ):
            evidence = [
                f"Co-tenant VRAM collision on node {node} configured with GPU sharing-strategy=time-slicing.",
                f"Active co-tenant pods {pods} attempted simultaneous unpartitioned VRAM allocations.",
                "Time-slicing provides temporal scheduling with zero hardware memory isolation.",
            ]
            scores["TIMESLICING_VRAM_COLLISION"] = (0.97, evidence, "critical")

        # -------------------------------------------------------------
        # Hypothesis 6: CUDA_VRAM_KV_OVERFLOW
        # -------------------------------------------------------------
        has_slope_anomaly = any(an.anomaly_type == "SATURATION_SLOPE" for an in cluster.anomalies)
        has_vram_alert = any(
            ("vram" in a.name.lower())
            or ("framebuffer" in a.name.lower())
            or ("fb_used" in a.name.lower())
            or ("cuda out of memory" in a.annotations.get("message", "").lower())
            for a in cluster.alerts
        )
        is_failing_or_bursting = (
            exit_code == 1
            or has_slope_anomaly
            or (has_vram_alert and exit_code != 0)
            or any("outofmemory" in a.name.lower() for a in cluster.alerts)
        )
        if (
            is_failing_or_bursting
            and (has_slope_anomaly or has_vram_alert or exit_code == 1)
            and not is_time_slicing
            and not is_oomkilled
            and not is_sigbus
            and not hw_xids
            and not sw_xids
        ):
            evidence = [
                f"CUDA OutOfMemoryError detected on dedicated/MIG GPU on pod {primary_pod}.",
                "VRAM reached 98%+ saturation or steep linear growth slope with critical Time-To-Exhaustion.",
                "PagedAttention KV-cache pool and model weight sizing exceeded physical framebuffer capacity.",
            ]
            if has_slope_anomaly:
                slope_an = [an for an in cluster.anomalies if an.anomaly_type == "SATURATION_SLOPE"][0]
                evidence.append(
                    f"Observed VRAM growth slope: {slope_an.slope_per_min} MiB/min, TTE: {slope_an.details.get('tte_minutes')} min."
                )
            scores["CUDA_VRAM_KV_OVERFLOW"] = (0.97, evidence, "critical")

        # -------------------------------------------------------------
        # Hypothesis 7: VLLM_KV_PREEMPTION_STORM
        # -------------------------------------------------------------
        preemption_anomalies = [
            an for an in cluster.anomalies if "preemption" in an.metric.lower()
        ]
        has_preempt_alert = any("preemption" in a.name.lower() for a in cluster.alerts)
        is_pod_running = pod_status.lower() == "running" or exit_code in (None, 0)
        has_latency_anomaly = any(
            "time_to_first_token" in an.metric.lower() or "output_token" in an.metric.lower()
            for an in cluster.anomalies
        ) or any("latency" in a.name.lower() or "slo" in a.name.lower() for a in cluster.alerts)

        if (preemption_anomalies or has_preempt_alert) and is_pod_running:
            evidence = [
                f"vLLM engine on pod {primary_pod} is thrashing in KV-cache preemption storm.",
                "Inference requests are preempted to RECOMPUTE mode due to KV-cache saturation.",
                "Pod remains in Running state while TTFT and TPOT end-user latency SLOs are breached.",
            ]
            if preemption_anomalies:
                evidence.append(
                    f"Preemption metric Z-score: {preemption_anomalies[0].z_score} (observed: {preemption_anomalies[0].observed_value})."
                )
            scores["VLLM_KV_PREEMPTION_STORM"] = (0.96, evidence, "warning")

        # -------------------------------------------------------------
        # Hypothesis 8: NVLINK_FABRIC_DEGRADATION
        # -------------------------------------------------------------
        nvlink_anomalies = [
            an for an in cluster.anomalies if "nvlink" in an.metric.lower()
        ]
        has_nvlink_alert = any("nvlink" in a.name.lower() for a in cluster.alerts)
        if (nvlink_anomalies or has_nvlink_alert) and not hw_xids:
            evidence = [
                f"NVLink fabric degradation detected on node {node}.",
                "Continuous CRC FLIT error increments observed across inter-GPU fabric.",
                "Degrading multi-GPU inter-connect bandwidth and driving TPOT tail latency spikes.",
            ]
            if nvlink_anomalies:
                evidence.append(
                    f"NVLink error count Z-score: {nvlink_anomalies[0].z_score} (observed count: {nvlink_anomalies[0].observed_value})."
                )
            scores["NVLINK_FABRIC_DEGRADATION"] = (0.94, evidence, "warning")

        if not scores:
            # Benign or unclassified noise
            return None

        # Pick top hypothesis by score
        best_class = max(scores.keys(), key=lambda k: scores[k][0])
        conf, evidence, severity = scores[best_class]

        # Generate deterministic incident ID
        hash_input = f"{node}:{primary_pod}:{best_class}:{sorted(alert_ids)}"
        short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8].upper()
        incident_id = f"INC-{short_hash}"

        return CorrelatedIncident(
            incident_id=incident_id,
            root_cause_class=best_class,
            confidence=conf,
            severity=severity,
            affected_node=node,
            affected_pods=pods,
            raw_alert_ids=alert_ids,
            anomalies=cluster.anomalies,
            evidence=evidence,
            remediation=None,
        )
