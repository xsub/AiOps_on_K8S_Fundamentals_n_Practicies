"""
Unit and Integration Test Suite for Reference AIOps Closed-Loop Engine.
Tests:
- Data model serialization & deserialization (TelemetryPoint, RawAlert, AnomalySignal, RemediationPlan, CorrelatedIncident)
- Statistical Anomaly Detectors (EWMA Z-score, Monotonic Saturation Linear Regression, Benign Prefill Filter)
- Multi-Signal Topology Correlator (7 Canonical Root Causes + User-Space XID)
- Approval-Gated Remediation Engine & PDB Invariant Checks
- Offline Benchmark Evaluation & Noise Reduction Metrics
"""

import unittest
from aiops.models import (
    TelemetryPoint,
    RawAlert,
    AnomalySignal,
    RemediationPlan,
    CorrelatedIncident,
)
from aiops.detector import (
    RollingZScoreDetector,
    SaturationSlopeEstimator,
    BenignBurstFilter,
    UnifiedAnomalyDetector,
)
from aiops.correlator import TopologyCorrelator
from aiops.remediator import RemediationEngine
from aiops.evaluate import run_evaluation, evaluate_scenario, load_dataset


class TestAIOpsModels(unittest.TestCase):
    def test_telemetry_point_roundtrip(self):
        pt = TelemetryPoint(
            timestamp=1730764800.0,
            node="worker-1",
            metric="DCGM_FI_DEV_FB_USED",
            value=45000.0,
            gpu_uuid="GPU-123",
            namespace="prod",
            pod="vllm-0",
            labels={"app": "vllm"},
        )
        d = pt.to_dict()
        reconstructed = TelemetryPoint.from_dict(d)
        self.assertEqual(pt.timestamp, reconstructed.timestamp)
        self.assertEqual(pt.node, reconstructed.node)
        self.assertEqual(pt.value, reconstructed.value)
        self.assertEqual(pt.labels, reconstructed.labels)

    def test_raw_alert_roundtrip(self):
        alert = RawAlert(
            alert_id="ALT-100",
            timestamp=1730764800.0,
            source="npd",
            name="NpdGpuHardwareFault",
            severity="critical",
            node="worker-1",
            pod="vllm-0",
            value=79.0,
            annotations={"xid": "79"},
        )
        d = alert.to_dict()
        reconstructed = RawAlert.from_dict(d)
        self.assertEqual(alert.alert_id, reconstructed.alert_id)
        self.assertEqual(alert.source, reconstructed.source)
        self.assertEqual(alert.annotations, reconstructed.annotations)

    def test_remediation_plan_roundtrip(self):
        plan = RemediationPlan(
            action_type="CORDON_AND_DRAIN_NODE",
            risk_tier="CRITICAL_NODE_DRAIN",
            requires_approval=True,
            pdb_check_passed=True,
            approval_status="PENDING_APPROVAL",
            kubectl_commands=["kubectl cordon worker-1"],
            blast_radius_summary="Node drain",
        )
        d = plan.to_dict()
        reconstructed = RemediationPlan.from_dict(d)
        self.assertEqual(plan.action_type, reconstructed.action_type)
        self.assertEqual(plan.risk_tier, reconstructed.risk_tier)
        self.assertTrue(reconstructed.requires_approval)


class TestAIOpsDetectors(unittest.TestCase):
    def test_rolling_z_score_spike_detection(self):
        detector = RollingZScoreDetector(window_size=10, z_threshold=3.0)
        # Establish baseline: 0.12, 0.13, 0.12, 0.12
        for t in range(4):
            pt = TelemetryPoint(
                timestamp=float(t * 10),
                node="worker-1",
                metric="vllm:time_to_first_token_seconds",
                value=0.12,
                pod="pod-1",
            )
            sig = detector.process(pt)
            self.assertIsNone(sig)

        # Spike to 4.80 seconds
        spike_pt = TelemetryPoint(
            timestamp=50.0,
            node="worker-1",
            metric="vllm:time_to_first_token_seconds",
            value=4.80,
            pod="pod-1",
        )
        sig = detector.process(spike_pt)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.anomaly_type, "SPIKE")
        self.assertGreater(sig.z_score, 3.0)

    def test_monotonic_saturation_slope_estimator(self):
        estimator = SaturationSlopeEstimator(
            default_capacity_mib=81920.0,
            min_slope_mib_per_min=200.0,
            r2_threshold=0.80,
            tte_alert_threshold_minutes=15.0,
        )
        # Simulate steep linear rise: 70000, 74000, 78000, 81000 MiB over 3 minutes
        # Slope = ~3666 MiB/min
        pts = [
            (0.0, 70000.0),
            (60.0, 74000.0),
            (120.0, 78000.0),
            (180.0, 81000.0),
        ]
        sig = None
        for t, val in pts:
            pt = TelemetryPoint(
                timestamp=t,
                node="worker-2",
                metric="DCGM_FI_DEV_FB_USED",
                value=val,
                gpu_uuid="GPU-1",
                pod="vllm-pod",
            )
            sig = estimator.process(pt)

        self.assertIsNotNone(sig)
        self.assertEqual(sig.anomaly_type, "SATURATION_SLOPE")
        self.assertGreater(sig.slope_per_min, 3000.0)
        self.assertLess(sig.details["tte_minutes"], 5.0)

    def test_benign_prefill_burst_suppression(self):
        # High compute util (98%), but zero preemptions, low VRAM (45%), healthy KV cache (42%)
        is_benign = BenignBurstFilter.is_benign_burst(
            util_value=98.0,
            preemptions=0.0,
            vram_pct=45.0,
            kv_cache_usage_pct=0.42,
            pod_status="Running",
        )
        self.assertTrue(is_benign)

        # Same compute spike but with active preemptions -> NOT benign
        is_not_benign = BenignBurstFilter.is_benign_burst(
            util_value=98.0,
            preemptions=15.0,
            vram_pct=45.0,
            kv_cache_usage_pct=0.42,
            pod_status="Running",
        )
        self.assertFalse(is_not_benign)


class TestAIOpsCorrelator(unittest.TestCase):
    def setUp(self):
        self.correlator = TopologyCorrelator(temporal_window_seconds=300.0)

    def test_correlate_cgroup_mmap_oom(self):
        alerts = [
            RawAlert(
                alert_id="A-1",
                timestamp=100.0,
                source="kubelet",
                name="KubePodCrashLooping",
                severity="critical",
                node="gpu-node-1",
                pod="llama-1",
                annotations={"exit_code": "137", "reason": "OOMKilled"},
            ),
            RawAlert(
                alert_id="A-2",
                timestamp=105.0,
                source="prometheus",
                name="LowGpuMemoryUtilizationDuringOOM",
                severity="info",
                node="gpu-node-1",
                pod="llama-1",
                value=12.0,
            ),
        ]
        incidents = self.correlator.correlate(
            alerts, [], context={"exit_code": 137, "gpu_vram_pct": 12.0}
        )
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0].root_cause_class, "K8S_CGROUP_MMAP_OOM")

    def test_correlate_silicon_xid_vs_userspace_xid(self):
        # Test Hardware XID 79
        hw_alert = [
            RawAlert(
                alert_id="A-HW",
                timestamp=100.0,
                source="npd",
                name="NpdGpuHardwareFault",
                severity="critical",
                node="gpu-node-hw",
                pod="worker-1",
                value=79.0,
                annotations={"xid": "79"},
            )
        ]
        hw_incidents = self.correlator.correlate(hw_alert, [])
        self.assertEqual(len(hw_incidents), 1)
        self.assertEqual(hw_incidents[0].root_cause_class, "GPU_SILICON_XID_FAULT")

        # Test User-space XID 31
        sw_alert = [
            RawAlert(
                alert_id="A-SW",
                timestamp=100.0,
                source="npd",
                name="DcgmXidAlert",
                severity="warning",
                node="gpu-node-sw",
                pod="worker-2",
                value=31.0,
                annotations={"xid": "31"},
            )
        ]
        sw_incidents = self.correlator.correlate(sw_alert, [])
        self.assertEqual(len(sw_incidents), 1)
        self.assertEqual(sw_incidents[0].root_cause_class, "USER_SPACE_CUDA_XID")


class TestAIOpsRemediator(unittest.TestCase):
    def setUp(self):
        self.remediator = RemediationEngine()

    def test_mutating_actions_require_approval(self):
        incident = CorrelatedIncident(
            incident_id="INC-TEST1",
            root_cause_class="K8S_CGROUP_MMAP_OOM",
            confidence=0.97,
            severity="critical",
            affected_node="node-1",
            affected_pods=["pod-1"],
        )
        self.remediator.attach_remediation(incident, cluster_context={"namespace": "prod"})
        plan = incident.remediation
        self.assertIsNotNone(plan)
        self.assertTrue(plan.requires_approval)
        self.assertEqual(plan.approval_status, "PENDING_APPROVAL")
        self.assertIn("kubectl patch deployment", plan.kubectl_commands[0])

        # Operator approval flow
        self.remediator.approve_remediation(
            incident, approver="sre@example.com", reason="Memory bump approved"
        )
        self.assertEqual(plan.approval_status, "APPROVED")
        self.assertEqual(plan.approved_by, "sre@example.com")
        self.assertTrue(any("APPROVED by sre@example.com" in log for log in plan.audit_log))

    def test_pdb_enforcement_blocks_drain(self):
        incident = CorrelatedIncident(
            incident_id="INC-PDB",
            root_cause_class="GPU_SILICON_XID_FAULT",
            confidence=0.99,
            severity="critical",
            affected_node="node-pdb",
            affected_pods=["api-pod"],
        )
        # pdb_disruptions_allowed == 0
        self.remediator.attach_remediation(
            incident, cluster_context={"pdb_disruptions_allowed": 0}
        )
        plan = incident.remediation
        self.assertFalse(plan.pdb_check_passed)
        self.assertEqual(plan.approval_status, "BLOCKED_BY_PDB")

        # Attempting to approve must raise PermissionError
        with self.assertRaises(PermissionError):
            self.remediator.approve_remediation(
                incident, approver="sre@example.com", reason="Force approve"
            )


class TestAIOpsEvaluationBenchmark(unittest.TestCase):
    def test_full_benchmark_metrics(self):
        eval_data = run_evaluation()
        summary = eval_data["summary"]

        self.assertEqual(summary["total_scenarios"], 12)
        self.assertGreaterEqual(summary["total_raw_alerts"], 50)
        self.assertEqual(summary["total_incidents_emitted"], 10)
        self.assertGreater(summary["alert_noise_reduction_pct"], 75.0)
        self.assertEqual(summary["top1_accuracy_pct"], 100.0)
        self.assertEqual(summary["anomaly_precision"], 1.0)
        self.assertEqual(summary["anomaly_recall"], 1.0)
        self.assertEqual(summary["anomaly_f1"], 1.0)
        self.assertEqual(summary["safety_invariant_pass_rate_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
