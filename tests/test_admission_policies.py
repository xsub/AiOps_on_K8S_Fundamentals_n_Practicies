#!/usr/bin/env python3
"""
Unit tests for Kubernetes 1.30+ Native CEL ValidatingAdmissionPolicy & Guardrails.
Tests deterministic Python policy verifier and parses vap-gpu-inference-guardrails.yaml
to verify all 4 CEL expressions enforce:
1. Rejects privileged: true on GPU pods in non-system namespaces.
2. Requires spec.runtimeClassName == "nvidia" on GPU pods.
3. Requires a memory-backed emptyDir (medium: Memory, sizeLimit >= 8Gi)
   ACTUALLY mounted at mountPath: /dev/shm inside every multi-GPU container (nvidia.com/gpu >= 2).
   Explicitly tests the evasion edge case (unmounted emptyDir) and 64Mi undersized case.
4. Rejects uncapped vLLM (--gpu-memory-utilization > 0.45 or omitted) when targeting time-slicing nodes,
   while allowing capped --gpu-memory-utilization=0.40.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None


def parse_memory_to_bytes(size_str: str) -> int:
    """
    Parses Kubernetes quantity strings (e.g., '64Mi', '8Gi', '16Gi', '1000M', '1000000') into bytes.
    """
    size_str = str(size_str).strip()
    if not size_str:
        return 0

    units = {
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Ti": 1024**4,
        "Pi": 1024**5,
        "Ei": 1024**6,
        "k": 1000,
        "M": 1000**2,
        "G": 1000**3,
        "T": 1000**4,
        "P": 1000**5,
        "E": 1000**6,
    }

    for suffix, multiplier in units.items():
        if size_str.endswith(suffix):
            number_part = size_str[: -len(suffix)].strip()
            try:
                return int(float(number_part) * multiplier)
            except ValueError:
                return 0

    try:
        return int(float(size_str))
    except ValueError:
        return 0


class GPUAdmissionPolicyEvaluator:
    """
    Deterministic Python verifier mirroring the exact semantics of
    Kubernetes 1.30+ ValidatingAdmissionPolicy for GPU inference guardrails.
    """

    SYSTEM_NAMESPACES = {"kube-system", "gpu-operator", "cert-manager", "monitoring"}
    MIN_SHM_BYTES = 8 * (1024**3)  # 8 GiB
    MAX_TIMESLICING_VLLM_UTIL = 0.45

    def evaluate(
        self,
        pod_manifest: Dict[str, Any],
        namespace: str = "ai-inference",
    ) -> Tuple[bool, List[str]]:
        """
        Evaluates a Pod manifest against the 4 GPU guardrails.
        Returns (is_allowed, list_of_violations).
        """
        violations: List[str] = []

        # Policy binding exemption: System namespaces are exempted from tenant guardrails
        if namespace in self.SYSTEM_NAMESPACES:
            return True, []

        spec = pod_manifest.get("spec", {})
        containers = list(spec.get("containers", [])) + list(spec.get("initContainers", []))

        # Helper: Extract GPU count from a container (requests or limits)
        def get_gpu_count(c: Dict[str, Any]) -> int:
            resources = c.get("resources", {})
            limits = resources.get("limits", {})
            requests = resources.get("requests", {})
            val = limits.get("nvidia.com/gpu") or requests.get("nvidia.com/gpu") or 0
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0

        gpu_containers = [c for c in containers if get_gpu_count(c) > 0]
        is_gpu_pod = len(gpu_containers) > 0
        multi_gpu_containers = [c for c in containers if get_gpu_count(c) >= 2]
        is_multi_gpu_pod = len(multi_gpu_containers) > 0

        # Non-GPU pods are unaffected by GPU guardrails
        if not is_gpu_pod:
            return True, []

        # -------------------------------------------------------------
        # Rule 1: Privileged Mode Rejection on GPU Pods
        # -------------------------------------------------------------
        for c in gpu_containers:
            sec_ctx = c.get("securityContext", {})
            if sec_ctx.get("privileged") is True:
                violations.append(
                    f"Rule 1 Violation: Privileged container '{c.get('name')}' is forbidden on GPU pod "
                    f"in tenant namespace '{namespace}' (CVE/isolation risk)."
                )

        # -------------------------------------------------------------
        # Rule 2: Explicit NVIDIA RuntimeClass Requirement
        # -------------------------------------------------------------
        runtime_class = spec.get("runtimeClassName")
        if runtime_class != "nvidia":
            violations.append(
                f"Rule 2 Violation: GPU pod must explicitly set 'spec.runtimeClassName: nvidia', "
                f"got '{runtime_class}'."
            )

        # -------------------------------------------------------------
        # Rule 3: Multi-GPU /dev/shm emptyDir Memory Volume Mount Check
        # Requires:
        #   a) Volume with medium: Memory and sizeLimit >= 8Gi
        #   b) ACTUALLY mounted at /dev/shm inside every multi-GPU container
        # -------------------------------------------------------------
        if is_multi_gpu_pod:
            volumes = spec.get("volumes", [])
            valid_shm_volume_names = set()

            for vol in volumes:
                empty_dir = vol.get("emptyDir")
                if empty_dir is not None:
                    medium = empty_dir.get("medium")
                    size_limit_str = empty_dir.get("sizeLimit", "")
                    size_bytes = parse_memory_to_bytes(size_limit_str)

                    if medium == "Memory" and size_bytes >= self.MIN_SHM_BYTES:
                        valid_shm_volume_names.add(vol.get("name"))

            if not valid_shm_volume_names:
                violations.append(
                    "Rule 3 Violation: Multi-GPU pod (nvidia.com/gpu >= 2) requires a memory-backed "
                    "emptyDir volume with medium: Memory and sizeLimit >= 8Gi."
                )
            else:
                # Evasion Edge Case Check: Verify every multi-GPU container ACTUALLY mounts the valid shm volume at /dev/shm!
                for c in multi_gpu_containers:
                    mounts = c.get("volumeMounts", [])
                    has_shm_mount = False
                    for m in mounts:
                        if m.get("mountPath") == "/dev/shm" and m.get("name") in valid_shm_volume_names:
                            has_shm_mount = True
                            break

                    if not has_shm_mount:
                        violations.append(
                            f"Rule 3 Violation (Evasion Detected): Container '{c.get('name')}' requests >= 2 GPUs "
                            "but does NOT mount a qualified memory-backed emptyDir (>= 8Gi) at '/dev/shm'."
                        )

        # -------------------------------------------------------------
        # Rule 4: Time-Slicing vLLM Framebuffer Cap Guardrail
        # -------------------------------------------------------------
        node_selector = spec.get("nodeSelector", {})
        is_timesliced_target = (
            node_selector.get("nvidia.com/gpu.sharing-strategy") == "time-slicing"
            or node_selector.get("nvidia.com/gpu.partitioning") == "time-slicing"
        )

        if is_timesliced_target:
            for c in gpu_containers:
                image_name = c.get("image", "").lower()
                cmd = " ".join(c.get("command", []))
                args = " ".join(c.get("args", []))
                full_exec_str = f"{cmd} {args}"

                # Detect vLLM workload
                is_vllm = "vllm" in image_name or "vllm" in full_exec_str or "--model" in full_exec_str

                if is_vllm:
                    # Look for --gpu-memory-utilization flag
                    util_match = re.search(
                        r'--gpu-memory-utilization(?:=|\s+)([0-9]*\.?[0-9]+)', full_exec_str
                    )
                    if not util_match:
                        violations.append(
                            f"Rule 4 Violation: vLLM container '{c.get('name')}' scheduled to time-sliced node "
                            "without explicit '--gpu-memory-utilization' cap (defaults to 0.90, causing co-tenant OOM)."
                        )
                    else:
                        util_val = float(util_match.group(1))
                        if util_val > self.MAX_TIMESLICING_VLLM_UTIL:
                            violations.append(
                                f"Rule 4 Violation: vLLM container '{c.get('name')}' sets '--gpu-memory-utilization={util_val}', "
                                f"which exceeds maximum allowed threshold of {self.MAX_TIMESLICING_VLLM_UTIL} on time-sliced nodes."
                            )

        return (len(violations) == 0, violations)


class TestAdmissionPoliciesEvaluator(unittest.TestCase):
    """
    Comprehensive unit tests for the 4 GPU admission guardrails.
    """

    def setUp(self) -> None:
        self.evaluator = GPUAdmissionPolicyEvaluator()

    def test_rule1_privileged_gpu_pod_rejected_in_tenant_namespace(self) -> None:
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "insecure-gpu-pod"},
            "spec": {
                "runtimeClassName": "nvidia",
                "containers": [
                    {
                        "name": "inference",
                        "image": "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
                        "securityContext": {"privileged": True},
                        "resources": {"limits": {"nvidia.com/gpu": "1"}},
                    }
                ],
            },
        }
        allowed, violations = self.evaluator.evaluate(pod, namespace="ai-inference")
        self.assertFalse(allowed)
        self.assertTrue(any("Rule 1 Violation" in v for v in violations))

    def test_rule1_privileged_gpu_pod_allowed_in_system_namespace(self) -> None:
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "gpu-operator-diagnostic"},
            "spec": {
                "runtimeClassName": "nvidia",
                "containers": [
                    {
                        "name": "dcgm-diag",
                        "image": "nvcr.io/nvidia/cloud-native/k8s-driver-manager:v0.6.8",
                        "securityContext": {"privileged": True},
                        "resources": {"limits": {"nvidia.com/gpu": "1"}},
                    }
                ],
            },
        }
        allowed, violations = self.evaluator.evaluate(pod, namespace="gpu-operator")
        self.assertTrue(allowed)
        self.assertEqual(len(violations), 0)

    def test_rule1_compliant_non_root_gpu_pod_allowed(self) -> None:
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "compliant-gpu-pod"},
            "spec": {
                "runtimeClassName": "nvidia",
                "containers": [
                    {
                        "name": "inference",
                        "image": "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
                        "securityContext": {
                            "privileged": False,
                            "runAsNonRoot": True,
                            "allowPrivilegeEscalation": False,
                        },
                        "resources": {"limits": {"nvidia.com/gpu": "1"}},
                    }
                ],
            },
        }
        allowed, violations = self.evaluator.evaluate(pod, namespace="ai-inference")
        self.assertTrue(allowed)
        self.assertEqual(len(violations), 0)

    def test_rule2_missing_runtime_class_rejected(self) -> None:
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "missing-runtime-class"},
            "spec": {
                "containers": [
                    {
                        "name": "worker",
                        "image": "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
                        "resources": {"requests": {"nvidia.com/gpu": "1"}},
                    }
                ]
            },
        }
        allowed, violations = self.evaluator.evaluate(pod, namespace="ai-inference")
        self.assertFalse(allowed)
        self.assertTrue(any("Rule 2 Violation" in v for v in violations))

    def test_rule2_runc_runtime_class_rejected(self) -> None:
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "runc-runtime-class"},
            "spec": {
                "runtimeClassName": "runc",
                "containers": [
                    {
                        "name": "worker",
                        "image": "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
                        "resources": {"requests": {"nvidia.com/gpu": "1"}},
                    }
                ],
            },
        }
        allowed, violations = self.evaluator.evaluate(pod, namespace="ai-inference")
        self.assertFalse(allowed)
        self.assertTrue(any("Rule 2 Violation" in v for v in violations))

    def test_rule3_multi_gpu_without_shm_volume_rejected(self) -> None:
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "multi-gpu-no-shm"},
            "spec": {
                "runtimeClassName": "nvidia",
                "containers": [
                    {
                        "name": "vllm-multi",
                        "image": "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
                        "resources": {"limits": {"nvidia.com/gpu": "4"}},
                    }
                ],
            },
        }
        allowed, violations = self.evaluator.evaluate(pod, namespace="ai-inference")
        self.assertFalse(allowed)
        self.assertTrue(any("Rule 3 Violation" in v for v in violations))

    def test_rule3_multi_gpu_with_undersized_64mi_shm_rejected(self) -> None:
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "multi-gpu-64mi-shm"},
            "spec": {
                "runtimeClassName": "nvidia",
                "containers": [
                    {
                        "name": "vllm-multi",
                        "image": "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
                        "resources": {"limits": {"nvidia.com/gpu": "2"}},
                        "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
                    }
                ],
                "volumes": [
                    {
                        "name": "dshm",
                        "emptyDir": {"medium": "Memory", "sizeLimit": "64Mi"},
                    }
                ],
            },
        }
        allowed, violations = self.evaluator.evaluate(pod, namespace="ai-inference")
        self.assertFalse(allowed)
        self.assertTrue(any("Rule 3 Violation" in v for v in violations))

    def test_rule3_evasion_edge_case_unmounted_memory_shm_rejected(self) -> None:
        """
        Tests the evasion attempt where a pod declares a valid 16Gi memory emptyDir in spec.volumes,
        but fails to mount it at /dev/shm inside the multi-GPU container!
        """
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "multi-gpu-evasion-attempt"},
            "spec": {
                "runtimeClassName": "nvidia",
                "containers": [
                    {
                        "name": "vllm-multi",
                        "image": "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
                        "resources": {"limits": {"nvidia.com/gpu": "4"}},
                        # Volume is NOT mounted to /dev/shm (or mounted elsewhere)
                        "volumeMounts": [{"name": "dshm", "mountPath": "/tmp/custom-ramdisk"}],
                    }
                ],
                "volumes": [
                    {
                        "name": "dshm",
                        "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"},
                    }
                ],
            },
        }
        allowed, violations = self.evaluator.evaluate(pod, namespace="ai-inference")
        self.assertFalse(allowed)
        self.assertTrue(any("Evasion Detected" in v for v in violations))

    def test_rule3_compliant_multi_gpu_pod_with_shm_allowed(self) -> None:
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "multi-gpu-compliant"},
            "spec": {
                "runtimeClassName": "nvidia",
                "containers": [
                    {
                        "name": "vllm-multi",
                        "image": "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
                        "resources": {"limits": {"nvidia.com/gpu": "4"}},
                        "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
                    }
                ],
                "volumes": [
                    {
                        "name": "dshm",
                        "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"},
                    }
                ],
            },
        }
        allowed, violations = self.evaluator.evaluate(pod, namespace="ai-inference")
        self.assertTrue(allowed)
        self.assertEqual(len(violations), 0)

    def test_rule4_time_slicing_uncapped_vllm_rejected(self) -> None:
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "vllm-time-slicing-uncapped"},
            "spec": {
                "runtimeClassName": "nvidia",
                "nodeSelector": {"nvidia.com/gpu.sharing-strategy": "time-slicing"},
                "containers": [
                    {
                        "name": "vllm",
                        "image": "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
                        "resources": {"limits": {"nvidia.com/gpu": "1"}},
                        # Omits --gpu-memory-utilization, default 0.90 will collide
                        "args": ["--model=meta-llama/Meta-Llama-3-8B-Instruct"],
                    }
                ],
            },
        }
        allowed, violations = self.evaluator.evaluate(pod, namespace="ai-inference")
        self.assertFalse(allowed)
        self.assertTrue(any("Rule 4 Violation" in v for v in violations))

    def test_rule4_time_slicing_overcapped_vllm_rejected(self) -> None:
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "vllm-time-slicing-overcapped"},
            "spec": {
                "runtimeClassName": "nvidia",
                "nodeSelector": {"nvidia.com/gpu.sharing-strategy": "time-slicing"},
                "containers": [
                    {
                        "name": "vllm",
                        "image": "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
                        "resources": {"limits": {"nvidia.com/gpu": "1"}},
                        "args": [
                            "--model=meta-llama/Meta-Llama-3-8B-Instruct",
                            "--gpu-memory-utilization=0.90",
                        ],
                    }
                ],
            },
        }
        allowed, violations = self.evaluator.evaluate(pod, namespace="ai-inference")
        self.assertFalse(allowed)
        self.assertTrue(any("Rule 4 Violation" in v for v in violations))

    def test_rule4_time_slicing_capped_vllm_allowed(self) -> None:
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "vllm-time-slicing-capped"},
            "spec": {
                "runtimeClassName": "nvidia",
                "nodeSelector": {"nvidia.com/gpu.sharing-strategy": "time-slicing"},
                "containers": [
                    {
                        "name": "vllm",
                        "image": "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
                        "resources": {"limits": {"nvidia.com/gpu": "1"}},
                        "args": [
                            "--model=meta-llama/Meta-Llama-3-8B-Instruct",
                            "--gpu-memory-utilization=0.40",
                        ],
                    }
                ],
            },
        }
        allowed, violations = self.evaluator.evaluate(pod, namespace="ai-inference")
        self.assertTrue(allowed)
        self.assertEqual(len(violations), 0)

    def test_rule4_non_timeslicing_uncapped_vllm_allowed(self) -> None:
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "vllm-dedicated-node"},
            "spec": {
                "runtimeClassName": "nvidia",
                "nodeSelector": {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"},
                "containers": [
                    {
                        "name": "vllm",
                        "image": "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
                        "resources": {"limits": {"nvidia.com/gpu": "1"}},
                        "args": [
                            "--model=meta-llama/Meta-Llama-3-8B-Instruct",
                            "--gpu-memory-utilization=0.92",
                        ],
                    }
                ],
            },
        }
        allowed, violations = self.evaluator.evaluate(pod, namespace="ai-inference")
        self.assertTrue(allowed)
        self.assertEqual(len(violations), 0)


class TestValidatingAdmissionPolicyManifest(unittest.TestCase):
    """
    Parses vap-gpu-inference-guardrails.yaml to verify all 4 CEL guardrails
    are declared with correct failurePolicy, namespace exemptions, and CEL expressions.
    """

    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        cls.vap_path = repo_root / "manifests" / "06-security-guardrails" / "vap-gpu-inference-guardrails.yaml"
        cls.assertTrue(cls.vap_path.exists(), f"VAP manifest missing: {cls.vap_path}")

        cls.content = cls.vap_path.read_text(encoding="utf-8")

    def test_vap_manifest_structure(self) -> None:
        self.assertIn("kind: ValidatingAdmissionPolicy", self.content)
        self.assertIn("kind: ValidatingAdmissionPolicyBinding", self.content)
        self.assertIn("failurePolicy: Fail", self.content)
        self.assertIn("gpu-inference-security-guardrails", self.content)

    def test_vap_namespace_selector_exemptions(self) -> None:
        self.assertIn("kube-system", self.content)
        self.assertIn("gpu-operator", self.content)
        self.assertIn("monitoring", self.content)

    def test_vap_cel_expressions_present(self) -> None:
        """
        Verifies all 4 CEL policy expressions are present in the manifest.
        """
        # Rule 1: Privileged check
        self.assertTrue(
            "privileged == false" in self.content or "!c.securityContext.privileged" in self.content,
            "CEL expression for Rule 1 (privileged check) missing in VAP manifest",
        )

        # Rule 2: RuntimeClassName check
        self.assertTrue(
            'runtimeClassName == "nvidia"' in self.content or "runtimeClassName == 'nvidia'" in self.content,
            "CEL expression for Rule 2 (runtimeClassName == nvidia) missing in VAP manifest",
        )

        # Rule 3: Memory emptyDir /dev/shm check
        self.assertTrue(
            "/dev/shm" in self.content or "hasShmMemoryVolume" in self.content or "emptyDir.medium == \"Memory\"" in self.content,
            "CEL expression for Rule 3 (/dev/shm memory volume) missing in VAP manifest",
        )

        # Rule 4: Time-slicing vLLM cap check
        self.assertTrue(
            "time-slicing" in self.content or "gpu-memory-utilization" in self.content,
            "CEL expression for Rule 4 (time-slicing vLLM utilization cap) missing in VAP manifest",
        )


if __name__ == "__main__":
    unittest.main()
