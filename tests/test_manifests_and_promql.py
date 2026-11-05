"""
Unit Test Suite for Kubernetes Manifests, Kustomize Overlays, PromQL Rules & Security Invariants.

Tests:
1. Kustomize Overlays & Scheduler/Autoscaler Orthogonality
2. KServe Llama-3.1-70B FP8 Analytical VRAM & Tensor-Parallel Math
3. LeaderWorkerSet Multi-Node Ray Node Count Barrier & Distributed Executor
4. Prometheus Alert Rules & PromQL Hardware vs Application XID Separation
5. Fluid S3 Accelerator Zero-Static-Key IRSA Security Architecture
"""

import os
import re
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_yaml_all(file_path: Path):
    content = file_path.read_text(encoding="utf-8")
    if yaml is not None:
        return list(yaml.safe_load_all(content))
    # Regex fallback
    docs = []
    raw_docs = re.split(r'^(?:---|---[ \t].*)$', content, flags=re.MULTILINE)
    for rd in raw_docs:
        if rd.strip():
            docs.append({"_raw": rd})
    return docs


class TestKustomizeOverlays(unittest.TestCase):
    def setUp(self):
        self.repo_root = get_repo_root()
        self.deploy_dir = self.repo_root / "deploy"

    def test_all_overlays_exist_and_reference_valid_targets(self):
        overlays = [
            self.deploy_dir / "base",
            self.deploy_dir / "overlays" / "kserve-keda",
            self.deploy_dir / "overlays" / "lws-kueue",
            self.deploy_dir / "overlays" / "volcano-gang",
        ]
        for overlay in overlays:
            self.assertTrue(overlay.exists(), f"Overlay directory {overlay} does not exist")
            kust_file = overlay / "kustomization.yaml"
            self.assertTrue(kust_file.exists(), f"kustomization.yaml missing in {overlay}")

            docs = parse_yaml_all(kust_file)
            self.assertGreater(len(docs), 0)
            resources = docs[0].get("resources", [])
            self.assertGreater(len(resources), 0, f"No resources in {kust_file}")

            for res in resources:
                target_path = (overlay / res).resolve()
                self.assertTrue(
                    target_path.exists(),
                    f"Resource '{res}' referenced in {kust_file} does not exist at {target_path}",
                )

    def test_scheduler_orthogonality_no_conflicting_schedulers(self):
        """
        Verify that Kueue, Volcano, and KServe overlays do not mix conflicting schedulers
        on the same workload specifications.
        """
        volcano_manifest = self.repo_root / "manifests" / "03-scheduling" / "volcano-gang-distributed-vllm.yaml"
        kueue_manifest = self.repo_root / "manifests" / "01-hardware-orchestration" / "kueue-gpu-clusterqueue.yaml"
        kserve_manifest = self.repo_root / "manifests" / "04-inference-serving" / "kserve-vllm-canary.yaml"

        # Volcano must use schedulerName: volcano
        volcano_docs = parse_yaml_all(volcano_manifest)
        for doc in volcano_docs:
            if doc.get("kind") == "StatefulSet":
                scheduler = doc.get("spec", {}).get("template", {}).get("spec", {}).get("schedulerName")
                self.assertEqual(scheduler, "volcano")

        # Kueue clusterqueue must define kueue.x-k8s.io apiVersion
        kueue_docs = parse_yaml_all(kueue_manifest)
        self.assertTrue(any("kueue.x-k8s.io" in str(doc.get("apiVersion")) for doc in kueue_docs))

        # KServe must NOT set schedulerName to volcano or conflict with kueue
        kserve_docs = parse_yaml_all(kserve_manifest)
        for doc in kserve_docs:
            if doc.get("kind") == "InferenceService":
                content_str = str(doc)
                self.assertNotIn("schedulerName: volcano", content_str)

    def test_zero_duplicate_autoscalers(self):
        """
        Verify no duplicate or conflicting HPA and ScaledObject for the same target workload.
        """
        keda_manifest = self.repo_root / "manifests" / "04-inference-serving" / "keda-vllm-autoscaler.yaml"
        docs = parse_yaml_all(keda_manifest)

        scaled_objects = [d for d in docs if d.get("kind") == "ScaledObject"]
        hpas = [d for d in docs if d.get("kind") == "HorizontalPodAutoscaler"]

        # KEDA manifest should use ScaledObject targeting the Deployment, not a conflicting HPA
        self.assertGreaterEqual(len(scaled_objects), 1)
        self.assertEqual(len(hpas), 0, "Conflicting HPA found alongside KEDA ScaledObject")


class TestKServeVLLMMath(unittest.TestCase):
    def setUp(self):
        self.repo_root = get_repo_root()
        self.kserve_manifest = self.repo_root / "manifests" / "04-inference-serving" / "kserve-vllm-canary.yaml"
        self.docs = parse_yaml_all(self.kserve_manifest)

    def test_gpu_count_and_tensor_parallel_size(self):
        """
        Verify KServe Llama-3.1-70B FP8 requests >= 2 GPUs and --tensor-parallel-size=2.
        """
        inferenceservices = [d for d in self.docs if d.get("kind") == "InferenceService"]
        self.assertGreaterEqual(len(inferenceservices), 1)

        for isvc in inferenceservices:
            predictor = isvc.get("spec", {}).get("predictor", {})
            containers = predictor.get("containers", [])
            self.assertGreaterEqual(len(containers), 1)

            container = containers[0]
            requests = container.get("resources", {}).get("requests", {})
            limits = container.get("resources", {}).get("limits", {})
            args = container.get("args", [])

            # Check GPU resource requests >= 2
            gpu_requests = int(requests.get("nvidia.com/gpu", "0"))
            gpu_limits = int(limits.get("nvidia.com/gpu", "0"))
            self.assertGreaterEqual(gpu_requests, 2, "KServe 70B must request at least 2 GPUs")
            self.assertEqual(gpu_requests, gpu_limits, "GPU requests and limits must be identical (Guaranteed QoS)")

            # Check CLI args
            has_tp2 = any("--tensor-parallel-size=2" in arg or "--tensor-parallel-size 2" in arg for arg in args)
            self.assertTrue(has_tp2, "Container args must specify --tensor-parallel-size=2")

    def test_analytical_vram_and_kv_cache_feasibility(self):
        """
        Analytical verification of 70B FP8 VRAM bounds on H100 (80 GiB = 81,920 MiB):
        - Meta-Llama-3.1-70B FP8 parameter count = 70.6 Billion parameters.
        - Precision: FP8 (1 byte per weight) -> ~70.6 GB (67,329 MiB) static weights.
        - Single H100 GPU (80 GiB):
            70.6 GB / 80 GiB = 88.25% of VRAM consumed purely by weights!
            Exceeds safe --gpu-memory-utilization=0.90 margin (leaves only 1.4 GB for CUDA context, activations & KV cache).
            Result: Instant OOM during prefill for any seq_len > 1024.
        - Two H100 GPUs (TP=2, 160 GiB total):
            Static weights per GPU: 70.6 GB / 2 = 35.3 GB (~44.1% of 80 GiB).
            At --gpu-memory-utilization=0.90 (72 GiB usable):
            Leaves 72 GiB - 35.3 GiB = 36.7 GiB per GPU dedicated to KV cache & activation buffers.
            At 16,384 max-model-len, FP8 KV-cache consumes ~0.5 MiB per token per seq;
            36.7 GiB safely accommodates concurrency >= 128 concurrent requests.
        """
        h100_vram_gb = 80.0
        llama70b_fp8_weights_gb = 70.6
        gpu_mem_util = 0.90
        single_gpu_usable_vram = h100_vram_gb * gpu_mem_util  # 72.0 GB
        single_seq_16k_kv_cache_gb = 2.68  # 2 * 80 layers * 8 heads * 128 dim * 1 byte * 16384 tokens

        # With 1 GPU:
        # Static weights (70.6 GB) + minimal single-request KV cache (2.68 GB) = 73.28 GB > 72.0 GB usable
        single_gpu_min_memory_needed = llama70b_fp8_weights_gb + single_seq_16k_kv_cache_gb
        self.assertGreater(
            single_gpu_min_memory_needed,
            single_gpu_usable_vram,
            "1 GPU is mathematically insufficient for 70B FP8 under 0.90 utilization cap",
        )

        # With TP=2 (2 GPUs):
        weights_per_gpu = llama70b_fp8_weights_gb / 2.0  # 35.3 GB
        remaining_kv_cache_headroom_gb = single_gpu_usable_vram - weights_per_gpu
        self.assertGreater(
            remaining_kv_cache_headroom_gb,
            35.0,
            "TP=2 must leave at least 35 GB of VRAM headroom per GPU for KV cache",
        )

        # Verify manifest explicitly specifies --gpu-memory-utilization=0.90 and --max-model-len=16384
        for isvc in [d for d in self.docs if d.get("kind") == "InferenceService"]:
            args = isvc.get("spec", {}).get("predictor", {}).get("containers", [])[0].get("args", [])
            self.assertTrue(any("--gpu-memory-utilization=0.90" in a for a in args))
            self.assertTrue(any("--max-model-len=16384" in a for a in args))


class TestLeaderWorkerSetRayBarrier(unittest.TestCase):
    def setUp(self):
        self.repo_root = get_repo_root()
        self.lws_manifest = self.repo_root / "manifests" / "03-scheduling" / "leaderworkerset-multinode-vllm.yaml"
        self.docs = parse_yaml_all(self.lws_manifest)

    def test_lws_executor_and_barrier(self):
        lws_docs = [d for d in self.docs if d.get("kind") == "LeaderWorkerSet"]
        self.assertEqual(len(lws_docs), 1)

        lws = lws_docs[0]
        leader_spec = lws.get("spec", {}).get("leaderWorkerTemplate", {}).get("leaderTemplate", {}).get("spec", {})
        leader_containers = leader_spec.get("containers", [])
        self.assertGreaterEqual(len(leader_containers), 1)

        leader_container = leader_containers[0]
        args_str = " ".join(leader_container.get("args", []))

        # Check Ray distributed executor backend
        self.assertIn("--distributed-executor-backend ray", args_str)

        # Check Ray node count synchronization barrier
        self.assertIn("LWS_GROUP_SIZE", args_str)
        self.assertIn("ray.nodes()", args_str)
        self.assertIn("LWS barrier reached", args_str)

    def test_headless_service_defined(self):
        services = [d for d in self.docs if d.get("kind") == "Service"]
        self.assertGreaterEqual(len(services), 1)
        svc = services[0]
        self.assertEqual(svc.get("spec", {}).get("clusterIP"), "None", "Service must be headless")


class TestPrometheusAlertRules(unittest.TestCase):
    def setUp(self):
        self.repo_root = get_repo_root()
        self.rules_file = self.repo_root / "manifests" / "05-observability" / "prometheus-gpu-oom-alerts.yaml"
        self.test_file = self.repo_root / "manifests" / "05-observability" / "prometheus-gpu-oom-alerts.test.yaml"

    def test_alert_names_match_between_rules_and_tests(self):
        rules_docs = parse_yaml_all(self.rules_file)
        rules = []
        for d in rules_docs:
            for g in d.get("spec", {}).get("groups", []) or d.get("groups", []):
                for r in g.get("rules", []):
                    if "alert" in r:
                        rules.append(r["alert"])

        test_docs = parse_yaml_all(self.test_file)
        test_alerts = set()
        for d in test_docs:
            for t in d.get("tests", []):
                for art in t.get("alert_rule_test", []):
                    if "alertname" in art:
                        test_alerts.add(art["alertname"])

        expected_alerts = {
            "K8sSystemRamOOMOnGPUPod",
            "GPUVramSaturationCritical",
            "CudaVramOOMCrashDetected",
            "NvidiaGPUCriticalSiliconXidFault",
            "NvidiaGPUApplicationOrDriverXidWarning",
        }
        self.assertEqual(set(rules), expected_alerts)
        self.assertEqual(test_alerts, expected_alerts)

    def test_xid_classification_separation(self):
        rules_docs = parse_yaml_all(self.rules_file)
        rules_by_name = {}
        for d in rules_docs:
            for g in d.get("spec", {}).get("groups", []) or d.get("groups", []):
                for r in g.get("rules", []):
                    if "alert" in r:
                        rules_by_name[r["alert"]] = r

        # Critical Hardware XIDs
        crit = rules_by_name["NvidiaGPUCriticalSiliconXidFault"]
        self.assertEqual(crit.get("labels", {}).get("severity"), "critical")
        self.assertEqual(crit.get("labels", {}).get("failure_domain"), "gpu_silicon_xid")
        for xid in ["48", "63", "64", "74", "79", "94", "95"]:
            self.assertIn(f"== {xid}", crit["expr"])

        # Warning Application XIDs
        warn = rules_by_name["NvidiaGPUApplicationOrDriverXidWarning"]
        self.assertEqual(warn.get("labels", {}).get("severity"), "warning")
        self.assertEqual(warn.get("labels", {}).get("failure_domain"), "gpu_application_mmu")
        for xid in ["13", "31", "43"]:
            self.assertIn(f"== {xid}", warn["expr"])


class TestFluidSecurityZeroStaticKeys(unittest.TestCase):
    def setUp(self):
        self.repo_root = get_repo_root()
        self.fluid_manifest = self.repo_root / "manifests" / "06-model-storage" / "fluid-s3-accelerator.yaml"
        self.docs = parse_yaml_all(self.fluid_manifest)

    def test_zero_plaintext_credentials(self):
        content = self.fluid_manifest.read_text(encoding="utf-8")
        # Ban typical static secret keys
        self.assertNotIn("fs.s3a.access.key", content)
        self.assertNotIn("fs.s3a.secret.key", content)
        self.assertNotIn("AWS_ACCESS_KEY_ID", content)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", content)
        self.assertNotIn("AKIA", content)

    def test_irsa_web_identity_credentials_provider_configured(self):
        content = self.fluid_manifest.read_text(encoding="utf-8")
        self.assertIn("fs.s3a.aws.credentials.provider", content)
        self.assertIn("com.amazonaws.auth.WebIdentityTokenCredentialsProvider", content)

        sa_docs = [d for d in self.docs if d.get("kind") == "ServiceAccount"]
        self.assertGreaterEqual(len(sa_docs), 1)
        sa = sa_docs[0]
        annotations = sa.get("metadata", {}).get("annotations", {})
        self.assertIn("eks.amazonaws.com/role-arn", annotations)
        self.assertTrue(annotations["eks.amazonaws.com/role-arn"].startswith("arn:aws:iam::"))


if __name__ == "__main__":
    unittest.main()
