"""
Command-Line Interface (CLI) for the AIOps Engine.
Supports:
  python3 -m aiops evaluate [--format markdown|json]
  python3 -m aiops demo [--approve-id INC-...] [--scenario SCENARIO_ID]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from aiops.correlator import TopologyCorrelator
from aiops.detector import UnifiedAnomalyDetector
from aiops.evaluate import format_evaluation_markdown, load_dataset, run_evaluation
from aiops.models import AnomalySignal, CorrelatedIncident, RawAlert, TelemetryPoint
from aiops.remediator import RemediationEngine


def run_demo(
    scenario_id: Optional[str] = None,
    approve_id: Optional[str] = None,
    approver: str = "oncall-sre@infrastructure.internal",
    reason: str = "Approved via SRE ChatOps after reviewing blast radius and cluster capacity.",
) -> int:
    """
    Step-by-step interactive walkthrough of the closed-loop AIOps engine:
    Telemetry -> Anomaly Detection -> Topology Correlation -> Approval Gate -> Remediation Patch.
    """
    dataset = load_dataset()
    scenarios = dataset.get("scenarios", [])

    # Pick scenario (default to SCENARIO-01-CGROUP-OOM if not specified)
    target_sc = None
    if scenario_id:
        for sc in scenarios:
            if sc["scenario_id"].lower() == scenario_id.lower():
                target_sc = sc
                break
        if not target_sc:
            print(f"Error: Scenario '{scenario_id}' not found.")
            return 1
    else:
        target_sc = scenarios[0]  # SCENARIO-01-CGROUP-OOM

    print("=" * 80)
    print("  AIOPS ON KUBERNETES: CLOSED-LOOP REMEDIATION DEMO")
    print("=" * 80)
    print(f"Scenario: {target_sc['scenario_id']}")
    print(f"Title:    {target_sc['title']}")
    print("-" * 80)

    # -------------------------------------------------------------
    # Step 1: Telemetry & Alert Stream Ingestion
    # -------------------------------------------------------------
    raw_alerts = [RawAlert.from_dict(a) for a in target_sc.get("raw_alerts", [])]
    telemetry_pts = [TelemetryPoint.from_dict(p) for p in target_sc.get("telemetry_points", [])]
    ctx = target_sc.get("cluster_context", {})

    print("\n[STEP 1] Ingesting Noisy Alert Storm & Telemetry Points...")
    print(f"  * Raw Alerts Ingested:    {len(raw_alerts)}")
    for alert in raw_alerts:
        print(f"    - [{alert.source.upper()}] {alert.name} ({alert.severity}) on {alert.node}/{alert.pod}")
    print(f"  * Telemetry Points:       {len(telemetry_pts)}")

    # -------------------------------------------------------------
    # Step 2: Statistical Anomaly Detection
    # -------------------------------------------------------------
    print("\n[STEP 2] Running Statistical Anomaly Detection Engine...")
    detector = UnifiedAnomalyDetector()
    anomalies: List[AnomalySignal] = detector.process_stream(telemetry_pts)
    print(f"  * Anomalies Detected:     {len(anomalies)}")
    for an in anomalies:
        slope_info = f", slope={an.slope_per_min} MiB/min" if an.slope_per_min else ""
        print(
            f"    - [{an.anomaly_type}] {an.metric} (observed={an.observed_value}, z_score={an.z_score}{slope_info})"
        )

    # -------------------------------------------------------------
    # Step 3: Multi-Signal Topological Correlation
    # -------------------------------------------------------------
    print("\n[STEP 3] Performing Topological & Multi-Signal Event Correlation...")
    correlator = TopologyCorrelator(temporal_window_seconds=300.0)
    incidents: List[CorrelatedIncident] = correlator.correlate(raw_alerts, anomalies, context=ctx)

    if not incidents:
        print("  * Result: No actionable incident emitted (benign burst suppressed).")
        return 0

    incident = incidents[0]
    print(f"  * Correlated Incident ID: {incident.incident_id}")
    print(f"  * Root-Cause Class:       {incident.root_cause_class} (Confidence: {incident.confidence * 100:.1f}%)")
    print(f"  * Severity:               {incident.severity.upper()}")
    print(f"  * Affected Node:          {incident.affected_node}")
    print(f"  * Affected Pods:          {', '.join(incident.affected_pods)}")
    print("  * Evidence Trail:")
    for ev in incident.evidence:
        print(f"    + {ev}")

    # -------------------------------------------------------------
    # Step 4: Remediation Planning & PDB Safety Gate
    # -------------------------------------------------------------
    print("\n[STEP 4] Evaluating PodDisruptionBudget & Generating Remediation Plan...")
    remediator = RemediationEngine()
    remediator.attach_remediation(incident, cluster_context=ctx)
    plan = incident.remediation

    print(f"  * Action Type:            {plan.action_type}")
    print(f"  * Risk Tier:              {plan.risk_tier}")
    print(f"  * Requires Approval:      {plan.requires_approval}")
    print(f"  * PDB Check Passed:       {plan.pdb_check_passed}")
    print(f"  * Initial Status:         {plan.approval_status}")
    print(f"  * Blast Radius Summary:   {plan.blast_radius_summary}")

    # -------------------------------------------------------------
    # Step 5: Approval-Gated Execution Workflow
    # -------------------------------------------------------------
    print("\n[STEP 5] Human-in-the-Loop Operator Approval Gate...")
    if plan.approval_status == "BLOCKED_BY_PDB":
        print("  [BLOCKED] Automated execution halted by PodDisruptionBudget (disruptions_allowed=0).")
        print("  Cluster safety invariant enforced: zero service disruptions permitted.")
        return 0

    if plan.approval_status == "AUTO_APPROVED":
        print("  [AUTO_APPROVED] Low-risk read-only action executed automatically.")
        return 0

    # Mutating action requiring approval
    target_approve_id = approve_id or incident.incident_id
    if approve_id and approve_id != incident.incident_id:
        print(f"  [WAITING] Specified --approve-id ({approve_id}) does not match current incident ({incident.incident_id}).")
        print(f"  To approve this incident, pass: --approve-id {incident.incident_id}")
        return 0

    print(f"  Operator '{approver}' reviewing incident {target_approve_id}...")
    remediator.approve_remediation(incident, approver=approver, reason=reason)
    print(f"  * New Approval Status:    {incident.remediation.approval_status}")
    print(f"  * Approved By:            {incident.remediation.approved_by}")
    print("  * Immutable Audit Log:")
    for log_entry in incident.remediation.audit_log:
        print(f"    - {log_entry}")

    print("\n[STEP 6] Executing Declarative Kubectl Patch & Commands:")
    for cmd in incident.remediation.kubectl_commands:
        print(f"  $ {cmd}")

    if incident.remediation.declarative_patch:
        print("\n  Declarative Patch Payload:")
        print("  " + json.dumps(incident.remediation.declarative_patch, indent=4).replace("\n", "\n  "))

    print("\n" + "=" * 80)
    print("  CLOSED-LOOP REMEDIATION SUCCESSFUL")
    print("=" * 80 + "\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m aiops",
        description="Reference AIOps Closed-Loop Engine for Kubernetes GPU Clusters",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'evaluate' command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run offline evaluation benchmark across labeled multi-step incident scenarios",
    )
    eval_parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format: markdown table (default) or raw json",
    )
    eval_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to custom labeled incidents json file",
    )

    # 'demo' command
    demo_parser = subparsers.add_parser(
        "demo",
        help="Walk through live closed-loop triage: telemetry -> anomaly -> correlation -> remediation",
    )
    demo_parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario ID to execute in demo (e.g., SCENARIO-01-CGROUP-OOM, SCENARIO-06-SILICON-XID-79)",
    )
    demo_parser.add_argument(
        "--approve-id",
        type=str,
        default=None,
        help="Approve the specific incident ID through the human-in-the-loop gate",
    )
    demo_parser.add_argument(
        "--approver",
        type=str,
        default="oncall-sre@infrastructure.internal",
        help="Operator ID or email providing remediation approval",
    )
    demo_parser.add_argument(
        "--reason",
        type=str,
        default="Approved via SRE ChatOps after reviewing blast radius.",
        help="Operator audit justification reason",
    )

    args = parser.parse_args(argv)

    if args.command == "evaluate":
        eval_data = run_evaluation(dataset_path=args.dataset)
        if args.format == "json":
            print(json.dumps(eval_data, indent=2))
        else:
            print(format_evaluation_markdown(eval_data))
        return 0

    elif args.command == "demo":
        return run_demo(
            scenario_id=args.scenario,
            approve_id=args.approve_id,
            approver=args.approver,
            reason=args.reason,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
