"""
Remediation Engine for AIOps on Kubernetes.
Generates safety-classified, approval-gated RemediationPlans for CorrelatedIncidents.
Enforces PodDisruptionBudget (PDB) constraints and provides an immutable audit log
for operator approval workflows.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from aiops.models import CorrelatedIncident, RemediationPlan


class RemediationEngine:
    """
    Evaluates incident root cause, computes blast radius, checks PodDisruptionBudgets,
    and assigns risk tiers and approval gates.
    """

    def __init__(self, default_pdb_disruptions_allowed: int = 1) -> None:
        self.default_pdb_disruptions_allowed = default_pdb_disruptions_allowed

    def plan_remediation(
        self,
        incident: CorrelatedIncident,
        cluster_context: Optional[Dict[str, Any]] = None,
    ) -> RemediationPlan:
        ctx = cluster_context or {}
        pdb_allowed = ctx.get("pdb_disruptions_allowed", self.default_pdb_disruptions_allowed)
        node = incident.affected_node
        pods = incident.affected_pods
        primary_pod = pods[0] if pods else "unknown-pod"
        namespace = ctx.get("namespace", "default")

        # Deduce deployment name from pod name (e.g., llama3-70b-vllm-789cf-1 -> llama3-70b-vllm)
        parts = primary_pod.split("-")
        dep_name = "-".join(parts[:-2]) if len(parts) >= 3 else primary_pod

        root_cause = incident.root_cause_class

        if root_cause == "K8S_CGROUP_MMAP_OOM":
            return RemediationPlan(
                action_type="PATCH_HOST_MEMORY_LIMITS",
                risk_tier="HIGH_MUTATION",
                requires_approval=True,
                pdb_check_passed=True,
                approval_status="PENDING_APPROVAL",
                approved_by=None,
                kubectl_commands=[
                    f"kubectl patch deployment {dep_name} -n {namespace} --type='json' "
                    f"-p='[{{\"op\": \"replace\", \"path\": \"/spec/template/spec/containers/0/resources/limits/memory\", \"value\": \"128Gi\"}}, "
                    f"{{\"op\": \"replace\", \"path\": \"/spec/template/spec/containers/0/resources/requests/memory\", \"value\": \"128Gi\"}}]'"
                ],
                declarative_patch={
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "inference-engine",
                                        "resources": {
                                            "limits": {"memory": "128Gi"},
                                            "requests": {"memory": "128Gi"},
                                        },
                                    }
                                ]
                            }
                        }
                    }
                },
                blast_radius_summary=(
                    f"Rolling restart of deployment {dep_name} in namespace {namespace}. "
                    "Doubles host DRAM limit (64Gi -> 128Gi) to stage uncompressed safetensors model weights."
                ),
                audit_log=[
                    f"[{self._now()}] Remediation plan generated for {incident.incident_id}: PENDING_APPROVAL"
                ],
            )

        elif root_cause == "CUDA_VRAM_KV_OVERFLOW":
            return RemediationPlan(
                action_type="RECONFIGURE_VLLM_VRAM_FOOTPRINT",
                risk_tier="HIGH_MUTATION",
                requires_approval=True,
                pdb_check_passed=True,
                approval_status="PENDING_APPROVAL",
                approved_by=None,
                kubectl_commands=[
                    f"kubectl set args deployment/{dep_name} -n {namespace} -- "
                    "--gpu-memory-utilization=0.88 --max-model-len=16384 --kv-cache-dtype=fp8"
                ],
                declarative_patch={
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "inference-engine",
                                        "args": [
                                            "--gpu-memory-utilization=0.88",
                                            "--max-model-len=16384",
                                            "--kv-cache-dtype=fp8",
                                        ],
                                    }
                                ]
                            }
                        }
                    }
                },
                blast_radius_summary=(
                    f"Rolling restart of deployment {dep_name}. Lowers gpu-memory-utilization to 0.88 "
                    "and enables FP8 KV caching to prevent physical VRAM exhaustion."
                ),
                audit_log=[
                    f"[{self._now()}] Remediation plan generated for {incident.incident_id}: PENDING_APPROVAL"
                ],
            )

        elif root_cause == "TIMESLICING_VRAM_COLLISION":
            co_tenant_dep = ctx.get("co_tenant_deployment", f"{dep_name}-co-tenant")
            return RemediationPlan(
                action_type="ISOLATE_TIME_SLICING_TENANTS",
                risk_tier="HIGH_MUTATION",
                requires_approval=True,
                pdb_check_passed=True,
                approval_status="PENDING_APPROVAL",
                approved_by=None,
                kubectl_commands=[
                    f"kubectl scale deployment/{co_tenant_dep} -n {namespace} --replicas=0",
                    f"kubectl label node {node} nvidia.com/gpu.workload-quarantine=true --overwrite",
                ],
                declarative_patch={
                    "metadata": {
                        "labels": {"nvidia.com/gpu.workload-quarantine": "true"}
                    }
                },
                blast_radius_summary=(
                    f"Scales down conflicting co-tenant deployment {co_tenant_dep} on node {node}. "
                    "Quarantines node from future unisolated time-slicing allocations."
                ),
                audit_log=[
                    f"[{self._now()}] Remediation plan generated for {incident.incident_id}: PENDING_APPROVAL"
                ],
            )

        elif root_cause == "VLLM_KV_PREEMPTION_STORM":
            return RemediationPlan(
                action_type="SCALE_KEDA_REPLICAS_AND_CAP_CONCURRENCY",
                risk_tier="MEDIUM_SCALING",
                requires_approval=True,
                pdb_check_passed=True,
                approval_status="PENDING_APPROVAL",
                approved_by=None,
                kubectl_commands=[
                    f"kubectl scale deployment/{dep_name} -n {namespace} --replicas=4",
                    f"kubectl set env deployment/{dep_name} -n {namespace} VLLM_MAX_NUM_SEQS=64",
                ],
                declarative_patch={
                    "spec": {"replicas": 4}
                },
                blast_radius_summary=(
                    f"Scales deployment {dep_name} horizontally from 2 to 4 replicas and caps "
                    "VLLM_MAX_NUM_SEQS to 64 to extinguish KV preemption storm."
                ),
                audit_log=[
                    f"[{self._now()}] Remediation plan generated for {incident.incident_id}: PENDING_APPROVAL"
                ],
            )

        elif root_cause == "NCCL_SHM_EXHAUSTION":
            return RemediationPlan(
                action_type="MOUNT_POSIX_SHM_VOLUME",
                risk_tier="HIGH_MUTATION",
                requires_approval=True,
                pdb_check_passed=True,
                approval_status="PENDING_APPROVAL",
                approved_by=None,
                kubectl_commands=[
                    f"kubectl patch deployment {dep_name} -n {namespace} --type='json' "
                    "-p='[{\"op\": \"add\", \"path\": \"/spec/template/spec/volumes/-\", \"value\": {\"name\": \"dshm\", \"emptyDir\": {\"medium\": \"Memory\", \"sizeLimit\": \"16Gi\"}}}, "
                    "{\"op\": \"add\", \"path\": \"/spec/template/spec/containers/0/volumeMounts/-\", \"value\": {\"name\": \"dshm\", \"mountPath\": \"/dev/shm\"}}]'"
                ],
                declarative_patch={
                    "spec": {
                        "template": {
                            "spec": {
                                "volumes": [
                                    {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}
                                ],
                                "containers": [
                                    {"name": "inference-engine", "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}]}
                                ],
                            }
                        }
                    }
                },
                blast_radius_summary=(
                    f"Rolling restart of deployment {dep_name}. Mounts 16Gi emptyDir memory volume "
                    "at /dev/shm for multi-GPU NCCL AllReduce ring buffers."
                ),
                audit_log=[
                    f"[{self._now()}] Remediation plan generated for {incident.incident_id}: PENDING_APPROVAL"
                ],
            )

        elif root_cause == "GPU_SILICON_XID_FAULT":
            is_pdb_passed = pdb_allowed > 0
            approval_status = "PENDING_APPROVAL" if is_pdb_passed else "BLOCKED_BY_PDB"

            summary = (
                f"Critical hardware failure on node {node}. Cordon node, taint with NoSchedule, "
                "and drain all pods to healthy nodes for GPU RMA replacement."
            )
            if not is_pdb_passed:
                summary = (
                    f"BLOCKED BY PDB: Drain on node {node} disallowed because pdb_disruptions_allowed=0. "
                    "Automatic drain halted to protect service availability."
                )

            return RemediationPlan(
                action_type="CORDON_AND_DRAIN_NODE",
                risk_tier="CRITICAL_NODE_DRAIN",
                requires_approval=True,
                pdb_check_passed=is_pdb_passed,
                approval_status=approval_status,
                approved_by=None,
                kubectl_commands=[
                    f"kubectl cordon {node}",
                    f"kubectl taint nodes {node} nvidia.com/gpu-hardware-fault=true:NoSchedule",
                    f"kubectl drain {node} --delete-emptydir-data --ignore-daemonsets --force",
                ],
                declarative_patch={
                    "spec": {
                        "taints": [
                            {
                                "key": "nvidia.com/gpu-hardware-fault",
                                "value": "true",
                                "effect": "NoSchedule",
                            }
                        ],
                        "unschedulable": True,
                    }
                },
                blast_radius_summary=summary,
                audit_log=[
                    f"[{self._now()}] Remediation plan generated for {incident.incident_id}: {approval_status}"
                ],
            )

        elif root_cause == "NVLINK_FABRIC_DEGRADATION":
            is_pdb_passed = pdb_allowed > 0
            approval_status = "PENDING_APPROVAL" if is_pdb_passed else "BLOCKED_BY_PDB"
            summary = (
                f"NVLink fabric degradation on node {node}. Cordon node and schedule maintenance drain."
            )
            if not is_pdb_passed:
                summary = (
                    f"BLOCKED BY PDB: Maintenance drain on node {node} blocked because pdb_disruptions_allowed=0."
                )

            return RemediationPlan(
                action_type="SCHEDULE_MAINTENANCE_DRAIN",
                risk_tier="CRITICAL_NODE_DRAIN",
                requires_approval=True,
                pdb_check_passed=is_pdb_passed,
                approval_status=approval_status,
                approved_by=None,
                kubectl_commands=[
                    f"kubectl cordon {node}",
                    f"kubectl drain {node} --delete-emptydir-data --ignore-daemonsets",
                ],
                declarative_patch={"spec": {"unschedulable": True}},
                blast_radius_summary=summary,
                audit_log=[
                    f"[{self._now()}] Remediation plan generated for {incident.incident_id}: {approval_status}"
                ],
            )

        elif root_cause == "USER_SPACE_CUDA_XID":
            return RemediationPlan(
                action_type="NOTIFY_APPLICATION_TEAM",
                risk_tier="LOW_READONLY",
                requires_approval=False,
                pdb_check_passed=True,
                approval_status="AUTO_APPROVED",
                approved_by="AIOps-Autopilot",
                kubectl_commands=[
                    f"kubectl get events -n {namespace} --field-selector involvedObject.name={primary_pod}"
                ],
                declarative_patch={},
                blast_radius_summary=(
                    f"Non-destructive event. Hardware silicon healthy. User-space CUDA page fault "
                    f"on pod {primary_pod} forwarded to application owners."
                ),
                audit_log=[
                    f"[{self._now()}] Remediation AUTO_APPROVED for {incident.incident_id} (read-only notify)."
                ],
            )

        # Fallback / generic read-only plan
        return RemediationPlan(
            action_type="OBSERVE_AND_ALERT",
            risk_tier="LOW_READONLY",
            requires_approval=False,
            pdb_check_passed=True,
            approval_status="AUTO_APPROVED",
            approved_by="AIOps-Autopilot",
            kubectl_commands=[],
            declarative_patch={},
            blast_radius_summary="No mutation required. Continuing telemetry observation.",
            audit_log=[
                f"[{self._now()}] Generic observe plan assigned for {incident.incident_id}."
            ],
        )

    def attach_remediation(
        self,
        incident: CorrelatedIncident,
        cluster_context: Optional[Dict[str, Any]] = None,
    ) -> CorrelatedIncident:
        incident.remediation = self.plan_remediation(incident, cluster_context)
        return incident

    @staticmethod
    def approve_remediation(
        incident: CorrelatedIncident,
        approver: str,
        reason: str,
    ) -> CorrelatedIncident:
        """
        Transition an incident's RemediationPlan from PENDING_APPROVAL to APPROVED.
        Rejects approval if plan is BLOCKED_BY_PDB.
        Appends immutable audit log entry.
        """
        if not incident.remediation:
            raise ValueError(f"Incident {incident.incident_id} has no remediation plan.")

        plan = incident.remediation
        if plan.approval_status == "BLOCKED_BY_PDB":
            raise PermissionError(
                f"Cannot approve remediation {plan.action_type} for {incident.incident_id}: "
                "Action is strictly BLOCKED_BY_PDB (pdb_disruptions_allowed == 0)."
            )

        if plan.approval_status == "APPROVED":
            return incident

        if plan.approval_status != "PENDING_APPROVAL":
            raise ValueError(
                f"Cannot approve remediation in status '{plan.approval_status}'. "
                "Must be 'PENDING_APPROVAL'."
            )

        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        plan.approval_status = "APPROVED"
        plan.approved_by = approver
        plan.audit_log.append(
            f"[{now_str}] APPROVED by {approver}: {reason}. Ready for automated execution."
        )

        return incident

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()
