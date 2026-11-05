"""
Unit & Integration Test Suite for Reference AIOps Closed-Loop Pipeline.

Verifies:
1. Anomaly Detection Mathematics (EWMA, Rolling Z-Score, OLS Linear Regression, TTE, Prefill Filter)
2. Multi-Signal Topology Correlation across 7 Canonical Root-Cause Classes
3. Remediation Safety Engine, PodDisruptionBudget (PDB) Blocking & Human Approval Gates
4. Offline Evaluation Benchmark Performance Thresholds (accuracy >= 0.85, noise_reduction >= 0.50)
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


class TestAnomalyMath(unittest.TestCase):
    def test_rolling_z_score_detector(self):
        detector = RollingZScoreDetector(window_size=10, z_threshold=3.0)
        # Establish stable latency baseline around 0.15s
        for t in range(5):
            pt = TelemetryPoint(
                timestamp=float(t * 10),
                node="gpu-worker-1",
                metric="vllm:time_to_first_token_seconds",
                value=0.15,
                pod="llama-0",
            )
            sig = detector.process(pt)
            self.assertIsNone(sig)

        # Incur significant latency anomaly: 5.2 seconds
        spike_pt = TelemetryPoint(
            timestamp=60.0,
            node="gpu-worker-1",
            metric="vllm:time_to_first_token_seconds",
            value=5.20,
            pod="llama-0",
        )
        sig = detector.process(spike_pt)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.anomaly_type, "SPIKE")
        self.assertGreaterEqual(sig.z_score, 3.0)
        self.assertEqual(sig.node, "gpu-worker-1")

    def test_ols_slope_and_time_to_exhaustion(self):
        """
        Verify Ordinary Least Squares (OLS) regression slope and Time-To-Exhaustion (TTE).
        Rate: +3,000 MiB/min on an 80GB H100 (81,920 MiB capacity).
        At 75,000 MiB, remaining capacity is 6,920 MiB -> TTE = 6,920 / 3,000 = 2.3 minutes.
        """
        estimator = SaturationSlopeEstimator(
            default_capacity_mib=81920.0,
            min_slope_mib_per_min=200.0,
            r2_threshold=0.80,
            tte_alert_threshold_minutes=15.0,
        )
        pts = [
            (0.0, 66000.0),
            (60.0, 69000.0),
            (120.0, 72000.0),
            (180.0, 75000.0),
        ]
        sig = None
        for t, val in pts:
            pt = TelemetryPoint(
                timestamp=t,
                node="worker-h100-1",
                metric="DCGM_FI_DEV_FB_USED",
                value=val,
                gpu_uuid="GPU-001",
                pod="vllm-70b-0",
            )
            sig = estimator.process(pt)

        self.assertIsNotNone(sig)
        self.assertEqual(sig.anomaly_type, "SATURATION_SLOPE")
        self.assertAlmostEqual(sig.slope_per_min, 3000.0, places=1)
        self.assertGreaterEqual(sig.details["r_squared"], 0.99)
        self.assertLess(sig.details["tte_minutes"], 3.0)

    def test_benign_prefill_burst_suppression(self):
        """
        Ensure compute saturation during prompt prefill without memory leak or preemptions
        is filtered as benign and does not fire false alarms.
        """
        benign = BenignBurstFilter.is_benign_burst(
            util_value=99.0,
            preemptions=0.0,
            vram_pct=48.0,
            kv_cache_usage_pct=0.35,
            pod_status="Running",
        )
        self.assertTrue(benign)

        # Same utilization but with active KV cache preemptions -> Anomaly!
        non_benign = BenignBurstFilter.is_benign_burst(
            util_value=99.0,
            preemptions=12.0,
            vram_pct=48.0,
            kv_cache_usage_pct=0.98,
            pod_status="Running",
        )
        self.assertFalse(non_benign)


class TestMultiSignalTopologyCorrelator(unittest.TestCase):
    def setUp(self):
        self.correlator = TopologyCorrelator(temporal_window_seconds=300.0)

    def test_cgroup_host_ram_oom_vs_vram_oom(self):
        # Scenario 1: Exit 137 (OOMKilled) on GPU pod with Low VRAM (<50%) -> K8S_CGROUP_MMAP_OOM
        alerts = [
            RawAlert(
                alert_id="A1",
                timestamp=100.0,
                source="kubelet",
                name="KubePodCrashLooping",
                severity="critical",
                node="node-1",
                pod="vllm-pod",
                annotations={"exit_code": "137", "reason": "OOMKilled"},
            )
        ]
        incidents = self.correlator.correlate(alerts, [], context={"exit_code": 137, "gpu_vram_pct": 20.0})
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0].root_cause_class, "K8S_CGROUP_MMAP_OOM")

        # Scenario 2: High VRAM saturation with Slope Anomaly -> CUDA_VRAM_KV_OVERFLOW
        leak_signal = [
            AnomalySignal(
                metric="DCGM_FI_DEV_FB_USED",
                node="node-1",
                pod="vllm-pod",
                observed_value=75000.0,
                slope_per_min=2500.0,
                anomaly_type="SATURATION_SLOPE",
                details={"r_squared": 0.98, "tte_minutes": 4.2},
            )
        ]
        incidents2 = self.correlator.correlate([], leak_signal, context={"gpu_vram_pct": 96.0})
        self.assertEqual(len(incidents2), 1)
        self.assertEqual(incidents2[0].root_cause_class, "CUDA_VRAM_KV_OVERFLOW")

    def test_hardware_silicon_xid_vs_user_space_mmu(self):
        # Critical hardware XID 79 (Fell off bus)
        hw_alerts = [
            RawAlert(
                alert_id="AHW",
                timestamp=100.0,
                source="npd",
                name="NpdGpuHardwareFault",
                severity="critical",
                node="node-hw",
                value=79.0,
                annotations={"xid": "79"},
            )
        ]
        hw_inc = self.correlator.correlate(hw_alerts, [])
        self.assertEqual(hw_inc[0].root_cause_class, "GPU_SILICON_XID_FAULT")

        # Non-fatal user-space MMU XID 31
        sw_alerts = [
            RawAlert(
                alert_id="ASW",
                timestamp=100.0,
                source="npd",
                name="DcgmXidAlert",
                severity="warning",
                node="node-sw",
                value=31.0,
                annotations={"xid": "31"},
            )
        ]
        sw_inc = self.correlator.correlate(sw_alerts, [])
        self.assertEqual(sw_inc[0].root_cause_class, "USER_SPACE_CUDA_XID")

    def test_nccl_distributed_timeout_hang(self):
        alerts = [
            RawAlert(
                alert_id="AN1",
                timestamp=100.0,
                source="vllm",
                name="NcclCommunicationTimeout",
                severity="critical",
                node="node-1",
                pod="vllm-head-0",
                annotations={"nccl_error": "watchdog timeout 1800s", "is_multi_gpu": "true"},
            )
        ]
        inc = self.correlator.correlate(alerts, [], context={"shm_size_mb": 64.0, "is_multi_gpu": True})
        self.assertEqual(inc[0].root_cause_class, "NCCL_SHM_EXHAUSTION")


class TestRemediationEngineSafetyAndPDB(unittest.TestCase):
    def setUp(self):
        self.remediator = RemediationEngine()

    def test_pdb_safety_blocks_drain_when_disruptions_zero(self):
        incident = CorrelatedIncident(
            incident_id="INC-PDB-ZERO",
            root_cause_class="GPU_SILICON_XID_FAULT",
            confidence=0.99,
            severity="critical",
            affected_node="node-prod-1",
            affected_pods=["api-gpu-0"],
        )
        self.remediator.attach_remediation(
            incident, cluster_context={"pdb_disruptions_allowed": 0}
        )
        plan = incident.remediation
        self.assertIsNotNone(plan)
        self.assertFalse(plan.pdb_check_passed)
        self.assertEqual(plan.approval_status, "BLOCKED_BY_PDB")

        # Force-approval attempt must be rejected
        with self.assertRaises(PermissionError):
            self.remediator.approve_remediation(
                incident, approver="admin@cluster.local", reason="Emergency bypass attempt"
            )

    def test_human_approval_state_machine(self):
        incident = CorrelatedIncident(
            incident_id="INC-DRAIN-OK",
            root_cause_class="GPU_SILICON_XID_FAULT",
            confidence=0.99,
            severity="critical",
            affected_node="node-prod-2",
            affected_pods=["worker-1"],
        )
        self.remediator.attach_remediation(
            incident, cluster_context={"pdb_disruptions_allowed": 1}
        )
        plan = incident.remediation
        self.assertTrue(plan.pdb_check_passed)
        self.assertTrue(plan.requires_approval)
        self.assertEqual(plan.approval_status, "PENDING_APPROVAL")

        # Approve remediation
        self.remediator.approve_remediation(
            incident, approver="sre-lead@company.com", reason="Node RMA verified by hardware telemetry"
        )
        self.assertEqual(plan.approval_status, "APPROVED")
        self.assertEqual(plan.approved_by, "sre-lead@company.com")
        self.assertTrue(any("APPROVED by sre-lead@company.com" in log for log in plan.audit_log))


class TestEvaluationBenchmarkThresholds(unittest.TestCase):
    def test_benchmark_passes_production_acceptance_thresholds(self):
        """
        Assert benchmark evaluation meets all required quality thresholds:
        - Accuracy >= 0.85 (85%)
        - Alert Noise Reduction >= 0.50 (50%)
        - Safety Invariant Pass Rate == 1.0 (100%)
        """
        eval_result = run_evaluation()
        summary = eval_result["summary"]

        accuracy = summary["top1_accuracy_pct"] / 100.0
        noise_reduction = summary["alert_noise_reduction_pct"] / 100.0
        safety_rate = summary["safety_invariant_pass_rate_pct"] / 100.0

        self.assertGreaterEqual(accuracy, 0.85, f"Accuracy {accuracy:.2f} below required 0.85 threshold")
        self.assertGreaterEqual(noise_reduction, 0.50, f"Noise reduction {noise_reduction:.2f} below required 0.50 threshold")
        self.assertEqual(safety_rate, 1.0, f"Safety invariant pass rate {safety_rate:.2f} must be 1.0 (100%)")


if __name__ == "__main__":
    unittest.main()
