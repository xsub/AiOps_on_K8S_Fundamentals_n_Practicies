"""
Live In-Cluster Closed-Loop AIOps Controller.

Integrates the `aiops` statistical anomaly detector, topological correlator, and
PDB/approval-gated remediation planner with a running Kubernetes cluster:
  1. Scrapes Prometheus text metrics (/metrics) from the live inference Service.
  2. Queries the Kubernetes API (PodDisruptionBudget `status.disruptionsAllowed` & Pods)
     using the mounted in-cluster ServiceAccount token and CA bundle.
  3. Runs UnifiedAnomalyDetector -> TopologyCorrelator -> RemediationEngine.
  4. Emits a native Kubernetes `Event` (`reason=AIOpsClosedLoopDiagnosis`) into the
     workload namespace and writes structured audit JSON reports.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aiops.correlator import HARDWARE_XIDS, SOFTWARE_XIDS, TopologyCorrelator
from aiops.detector import UnifiedAnomalyDetector
from aiops.models import RawAlert, TelemetryPoint
from aiops.remediator import RemediationEngine

METRIC_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$"
)
LABEL_PAIR_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')


def parse_prometheus_text(raw_text: str, timestamp: Optional[float] = None) -> List[TelemetryPoint]:
    """Parse Prometheus 0.0.4 exposition format into TelemetryPoint instances."""
    ts = timestamp if timestamp is not None else time.time()
    points: List[TelemetryPoint] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = METRIC_LINE_RE.match(line)
        if not m:
            continue
        metric_name, raw_labels, raw_val = m.groups()
        labels = dict(LABEL_PAIR_RE.findall(raw_labels or ""))
        node = labels.get("node") or labels.get("Hostname") or "aiops-ref-worker"
        pod = labels.get("pod", "")
        namespace = labels.get("namespace", "ai-inference")
        gpu_uuid = labels.get("UUID")
        points.append(
            TelemetryPoint(
                timestamp=ts,
                node=node,
                metric=metric_name,
                value=float(raw_val),
                gpu_uuid=gpu_uuid,
                namespace=namespace,
                pod=pod,
                labels=labels,
            )
        )
    return points


def synthesize_threshold_alerts(points: List[TelemetryPoint]) -> List[RawAlert]:
    """Generate Prometheus/NPD equivalent RawAlerts when live telemetry breaches thresholds."""
    alerts: List[RawAlert] = []
    for pt in points:
        if pt.metric == "DCGM_FI_DEV_XID_ERRORS" and int(pt.value) > 0:
            xid = int(pt.value)
            is_hw = xid in HARDWARE_XIDS
            alerts.append(
                RawAlert(
                    alert_id=f"live-xid-{xid}-{int(pt.timestamp)}",
                    timestamp=pt.timestamp,
                    source="npd" if is_hw else "prometheus",
                    name="GPUCriticalSiliconXIDFault" if is_hw else "GPUUserSpaceCUDAXIDWarning",
                    severity="critical" if is_hw else "warning",
                    node=pt.node,
                    pod=pt.pod,
                    namespace=pt.namespace,
                    gpu_uuid=pt.gpu_uuid,
                    value=float(xid),
                    annotations={"xid_code": str(xid)},
                )
            )
        elif pt.metric == "vllm:gpu_cache_usage_perc" and pt.value >= 0.90:
            alerts.append(
                RawAlert(
                    alert_id=f"live-kv-sat-{int(pt.timestamp)}",
                    timestamp=pt.timestamp,
                    source="prometheus",
                    name="VLLMKVCacheSaturationHigh",
                    severity="critical" if pt.value >= 0.95 else "warning",
                    node=pt.node,
                    pod=pt.pod,
                    namespace=pt.namespace,
                    gpu_uuid=pt.gpu_uuid,
                    value=pt.value,
                    annotations={"summary": f"KV-Cache utilization at {pt.value * 100:.1f}%"},
                )
            )
        elif pt.metric == "vllm:num_preemptions_total" and pt.value >= 3:
            alerts.append(
                RawAlert(
                    alert_id=f"live-preempt-{int(pt.timestamp)}",
                    timestamp=pt.timestamp,
                    source="prometheus",
                    name="VLLMKVPreemptionSurge",
                    severity="warning",
                    node=pt.node,
                    pod=pt.pod,
                    namespace=pt.namespace,
                    gpu_uuid=pt.gpu_uuid,
                    value=pt.value,
                    annotations={"summary": f"KV block preemptions={int(pt.value)}"},
                )
            )
    return alerts


class InClusterK8sClient:
    """Minimal zero-dependency Kubernetes REST client using ServiceAccount credentials."""

    SA_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")

    def __init__(self, api_url: Optional[str] = None, namespace: Optional[str] = None) -> None:
        host = os.environ.get("KUBERNETES_SERVICE_HOST", "")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        self.api_url = api_url or (f"https://{host}:{port}" if host else "")
        self.token = ""
        self.ssl_ctx = ssl.create_default_context()

        token_file = self.SA_DIR / "token"
        ca_file = self.SA_DIR / "ca.crt"
        ns_file = self.SA_DIR / "namespace"

        if token_file.exists():
            self.token = token_file.read_text(encoding="utf-8").strip()
        if ca_file.exists():
            self.ssl_ctx.load_verify_locations(cafile=str(ca_file))
        self.namespace = namespace or (
            ns_file.read_text(encoding="utf-8").strip() if ns_file.exists() else "ai-inference"
        )

    @property
    def available(self) -> bool:
        return bool(self.api_url and self.token)

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.api_url.rstrip('/')}{path}"
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        ctx = self.ssl_ctx if url.startswith("https://") else None
        with urllib.request.urlopen(req, timeout=5.0, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_inference_pod_metrics_urls(
        self,
        label_selector: str = "app.kubernetes.io/name=inference-engine",
        port: int = 8000,
    ) -> List[str]:
        if not self.available:
            return []
        encoded_sel = urllib.parse.quote(label_selector, safe="")
        path = f"/api/v1/namespaces/{self.namespace}/pods?labelSelector={encoded_sel}"
        data = self._request("GET", path)
        urls: List[str] = []
        for item in data.get("items", []):
            status = item.get("status", {})
            pod_ip = status.get("podIP")
            conds = status.get("conditions", [])
            is_ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conds)
            if pod_ip and is_ready:
                urls.append(f"http://{pod_ip}:{port}/metrics")
        return urls

    def get_pdb_disruptions_allowed(self, pdb_name: str) -> Tuple[int, Dict[str, Any]]:
        if not self.available:
            return 1, {"source": "fallback-offline"}
        data = self._request("GET", f"/apis/policy/v1/namespaces/{self.namespace}/poddisruptionbudgets/{pdb_name}")
        status = data.get("status", {})
        allowed = int(status.get("disruptionsAllowed", 0))
        return allowed, {
            "source": "kubernetes-api-live",
            "pdb_name": pdb_name,
            "currentHealthy": status.get("currentHealthy", 0),
            "desiredHealthy": status.get("desiredHealthy", 0),
            "disruptionsAllowed": allowed,
        }

    def emit_diagnosis_event(
        self,
        deployment_name: str,
        root_cause: str,
        approval_status: str,
        action_type: str,
        pdb_allowed: int,
    ) -> bool:
        if not self.available:
            return False
        now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ev_name = f"aiops-rca-{int(time.time() * 1000)}"
        event_body = {
            "apiVersion": "v1",
            "kind": "Event",
            "metadata": {"name": ev_name, "namespace": self.namespace},
            "involvedObject": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": deployment_name,
                "namespace": self.namespace,
            },
            "reason": "AIOpsClosedLoopDiagnosis",
            "message": (
                f"RootCause={root_cause} | Action={action_type} | "
                f"Gate={approval_status} | PDB_disruptionsAllowed={pdb_allowed}"
            ),
            "type": "Warning" if approval_status != "AUTO_APPROVED" else "Normal",
            "firstTimestamp": now_iso,
            "lastTimestamp": now_iso,
            "count": 1,
            "source": {"component": "aiops-closed-loop-controller"},
        }
        try:
            self._request("POST", f"/api/v1/namespaces/{self.namespace}/events", event_body)
            return True
        except urllib.error.URLError:
            return False


def evaluate_live_cycle(
    metrics_url: str,
    pdb_name: str = "inference-pdb",
    deployment_name: str = "inference-engine",
    namespace: str = "ai-inference",
    samples: int = 5,
    sample_interval_s: float = 0.4,
    k8s_client: Optional[InClusterK8sClient] = None,
) -> Dict[str, Any]:
    """
    Collect `samples` consecutive telemetry scrapes, run statistical anomaly detection,
    correlate incidents, inspect live Kubernetes PDB state, and return full closed-loop output.
    """
    detector = UnifiedAnomalyDetector(default_capacity_mib=81920.0)
    correlator = TopologyCorrelator()
    remediator = RemediationEngine()
    k8s = k8s_client or InClusterK8sClient(namespace=namespace)

    all_points: List[TelemetryPoint] = []
    latest_points: List[TelemetryPoint] = []
    target_urls = k8s.list_inference_pod_metrics_urls() or [metrics_url]

    # Prime baseline and observe current window across all discovered Pod endpoints
    for idx in range(samples):
        ts = time.time() + idx * 15.0  # 15s logical scrape cadence for slope calculation
        step_pts: List[TelemetryPoint] = []
        for url in target_urls:
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                raw = resp.read().decode("utf-8")
            step_pts.extend(parse_prometheus_text(raw, timestamp=ts))
        all_points.extend(step_pts)
        latest_points = step_pts
        if idx + 1 < samples:
            time.sleep(sample_interval_s)

    anomalies = detector.process_stream(all_points)
    raw_alerts = synthesize_threshold_alerts(latest_points)
    pdb_allowed, pdb_meta = k8s.get_pdb_disruptions_allowed(pdb_name)

    incidents = correlator.correlate(raw_alerts, anomalies, context={"namespace": namespace})
    results = []
    for inc in incidents:
        plan = remediator.plan_remediation(
            inc,
            cluster_context={
                "pdb_disruptions_allowed": pdb_allowed,
                "namespace": namespace,
            },
        )
        ev_emitted = k8s.emit_diagnosis_event(
            deployment_name=deployment_name,
            root_cause=inc.root_cause_class,
            approval_status=plan.approval_status,
            action_type=plan.action_type,
            pdb_allowed=pdb_allowed,
        )
        results.append(
            {
                "incident": inc.to_dict(),
                "remediation_plan": plan.to_dict(),
                "k8s_event_emitted": ev_emitted,
            }
        )

    return {
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metrics_url": metrics_url,
        "telemetry_points_ingested": len(all_points),
        "anomalies_detected": len(anomalies),
        "raw_alerts_triggered": len(raw_alerts),
        "pdb_live_state": pdb_meta,
        "correlated_incidents": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Live In-Cluster Closed-Loop AIOps Controller")
    parser.add_argument(
        "--metrics-url",
        default=os.environ.get(
            "METRICS_URL",
            "http://inference-engine.ai-inference.svc.cluster.local:8000/metrics",
        ),
    )
    parser.add_argument("--namespace", default=os.environ.get("POD_NAMESPACE", "ai-inference"))
    parser.add_argument("--pdb-name", default=os.environ.get("PDB_NAME", "inference-pdb"))
    parser.add_argument("--deployment-name", default="inference-engine")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--once", action="store_true", help="Run a single diagnosis cycle and exit")
    parser.add_argument("--output-json", default="", help="Optional path to write JSON report")
    args = parser.parse_args()

    while True:
        try:
            report = evaluate_live_cycle(
                metrics_url=args.metrics_url,
                pdb_name=args.pdb_name,
                deployment_name=args.deployment_name,
                namespace=args.namespace,
            )
            summary = json.dumps(report, indent=2)
            if args.output_json:
                Path(args.output_json).write_text(summary, encoding="utf-8")
            print(summary, flush=True)
        except Exception as exc:
            print(json.dumps({"status": "watch_error", "error": str(exc)}), flush=True)
            if args.once:
                raise

        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
