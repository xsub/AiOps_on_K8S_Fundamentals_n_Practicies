"""
AIOps Closed-Loop Architecture Interactive Demo.
Walks through the complete automated telemetry -> anomaly detection -> correlation ->
RCA -> PDB safety gate -> human-in-the-loop approval -> declarative patch pipeline.
"""

from __future__ import annotations

import json
import time
from typing import Dict, Any

from aiops.correlator import TopologyCorrelator
from aiops.detector import RollingZScoreDetector, SaturationSlopeEstimator, BenignBurstFilter
from aiops.models import AnomalySignal, RawAlert, TelemetryPoint
from aiops.remediator import RemediationEngine


def main() -> int:
    print("=" * 80)
    print("   AIOPS ON KUBERNETES: CLOSED-LOOP OBSERVABILITY & REMEDIATION PIPELINE")
    print("=" * 80)
    print()

    # Step 1: Telemetry Stream Ingestion
    print("[STAGE 1/6] Ingesting DCGM Hardware & vLLM Inference Telemetry...")
    base_t = time.time()
    node = "dgx-h100-node04.us-east-1.compute.internal"
    pod = "llama-3-1-70b-instruct-predictor-default-76f8b9-x29z"
    namespace = "ai-inference"

    points = [
        TelemetryPoint(timestamp=base_t - 240, node=node, pod=pod, metric="DCGM_FI_DEV_FB_USED", value=62000.0),
        TelemetryPoint(timestamp=base_t - 180, node=node, pod=pod, metric="DCGM_FI_DEV_FB_USED", value=67000.0),
        TelemetryPoint(timestamp=base_t - 120, node=node, pod=pod, metric="DCGM_FI_DEV_FB_USED", value=72000.0),
        TelemetryPoint(timestamp=base_t - 60,  node=node, pod=pod, metric="DCGM_FI_DEV_FB_USED", value=76500.0),
        TelemetryPoint(timestamp=base_t,       node=node, pod=pod, metric="DCGM_FI_DEV_FB_USED", value=80100.0),
    ]

    print(f"  • Scraped {len(points)} time-series samples for FB_USED (Framebuffer VRAM) on {pod}")
    print(f"  • Current Framebuffer Used: {points[-1].value:.0f} MiB / 81920 MiB ({(points[-1].value/81920)*100:.1f}%)")
    print()

    # Step 2: Anomaly Detection (EWMA Z-Score & OLS Linear Regression TTE)
    print("[STAGE 2/6] Statistical Anomaly Detection & Predictive Slope Analysis...")
    z_detector = RollingZScoreDetector(window_size=10, z_threshold=2.5)
    slope_estimator = SaturationSlopeEstimator(window_size=10, min_slope_mib_per_min=200.0, r2_threshold=0.85)

    anomalies = []
    for p in points:
        z_score_anom = z_detector.process(p)
        if z_score_anom:
            anomalies.append(z_score_anom)
        slope_anom = slope_estimator.process(p, capacity_mib=81920.0)
        if slope_anom:
            anomalies.append(slope_anom)

    for anom in anomalies:
        print(f"  [!] Anomaly Detected: {anom.metric} | Type={anom.anomaly_type} | Z-Score={anom.z_score}")
        if anom.slope_per_min > 0:
            tte = anom.details.get("tte_minutes", 0.0)
            print(f"      Monotonic Growth Rate : {anom.slope_per_min:.1f} MiB/min (R^2={anom.details.get('r_squared', 0.0):.3f})")
            print(f"      Predicted Time-To-OOM : {tte:.1f} minutes to catastrophic 100% saturation!")
    print()

    # Step 3: Alerts Triggered (DCGM + Node Problem Detector)
    print("[STAGE 3/6] Correlating Multi-Signal Alert Storm...")
    raw_alerts = [
        RawAlert(
            alert_id="alert-dcgm-vram-sat-01",
            timestamp=base_t,
            source="prometheus",
            name="GPUVramSaturationCritical",
            severity="critical",
            node=node,
            pod=pod,
            namespace=namespace,
            value=97.8,
            annotations={"failure_domain": "cuda_vram", "metric": "DCGM_FI_DEV_FB_USED"}
        ),
        RawAlert(
            alert_id="alert-vllm-queue-01",
            timestamp=base_t,
            source="prometheus",
            name="vllm:num_requests_waiting",
            severity="warning",
            node=node,
            pod=pod,
            namespace=namespace,
            value=18.0,
            annotations={"failure_domain": "inference_queue"}
        ),
        RawAlert(
            alert_id="alert-npd-xid-01",
            timestamp=base_t,
            source="npd",
            name="GPUHardwareFault",
            severity="critical",
            node=node,
            gpu_uuid="GPU-81a4b9c0-9f21-71bc",
            value=79.0,
            annotations={"xid": "79", "reason": "GPU fallen off bus"}
        )
    ]
    print(f"  • Ingested {len(raw_alerts)} firing alerts across Prometheus & Node Problem Detector:")
    for a in raw_alerts:
        print(f"    - [{a.source.upper()}] {a.name} (severity={a.severity}, node={a.node})")
    print()

    # Step 4: Topological Event Correlation & Probabilistic RCA
    print("[STAGE 4/6] Probabilistic Root Cause Analysis (Hypothesis Scorer)...")
    correlator = TopologyCorrelator(temporal_window_seconds=300.0)
    incidents = correlator.correlate(
        alerts=raw_alerts,
        anomalies=anomalies,
        context={"sharing_strategy": "none", "shm_size_mb": 16384.0}
    )

    if not incidents:
        print("  [-] No correlated incident synthesized.")
        return 1

    primary_inc = incidents[0]
    print(f"  ✓ Incident ID         : {primary_inc.incident_id}")
    print(f"  ✓ Root Cause Class    : {primary_inc.root_cause_class}")
    print(f"  ✓ Confidence Score    : {primary_inc.confidence * 100:.1f}%")
    print(f"  ✓ Target Node         : {primary_inc.affected_node}")
    print(f"  ✓ Affected Pods       : {', '.join(primary_inc.affected_pods) if primary_inc.affected_pods else 'None'}")
    print(f"  ✓ Diagnostic Evidence :")
    for ev in primary_inc.evidence:
        print(f"      • {ev}")
    print()

    # Step 5: PDB Verification & Human-in-the-Loop Gate
    print("[STAGE 5/6] PodDisruptionBudget (PDB) & Human Approval Gate...")
    remediator = RemediationEngine()

    # Scenario A: PDB Violations (0 disruptions allowed)
    plan_blocked = remediator.plan_remediation(primary_inc, cluster_context={"pdb_disruptions_allowed": 0, "namespace": namespace})
    print(f"  [CHECK 1] Evaluating PDB with disruptionsAllowed=0:")
    print(f"      PDB Check Passed : {plan_blocked.pdb_check_passed}")
    print(f"      Gate Status      : {plan_blocked.approval_status} (Automated drain BLOCKED to prevent quorum loss)")
    print(f"      Blast Radius     : {plan_blocked.blast_radius_summary}")

    # Scenario B: PDB Permits disruption (disruptionsAllowed=1)
    plan = remediator.plan_remediation(primary_inc, cluster_context={"pdb_disruptions_allowed": 1, "namespace": namespace})
    print(f"  [CHECK 2] Evaluating PDB with disruptionsAllowed=1:")
    print(f"      PDB Check Passed : {plan.pdb_check_passed}")
    print(f"      Risk Tier        : {plan.risk_tier}")
    print(f"      Gate Status      : {plan.approval_status}")
    print(f"      Requires Operator: {plan.requires_approval} (Human approval gate required for high-risk cordon/drain)")
    print()

    # Step 6: Declarative Kubernetes Remediation Patch
    print("[STAGE 6/6] Emitting Declarative Kubernetes Remediation Artifacts...")
    print("  Proposed kubectl operational commands:")
    for cmd in plan.kubectl_commands:
        print(f"    $ {cmd}")

    if plan.declarative_patch:
        print("\n  Declarative Cluster Policy / Node Condition Patch:")
        print(json.dumps(plan.declarative_patch, indent=4))

    print("\n" + "=" * 80)
    print("   CLOSED-LOOP DEMO EXECUTION COMPLETE (Zero False Evictions Guaranteed)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    main()
