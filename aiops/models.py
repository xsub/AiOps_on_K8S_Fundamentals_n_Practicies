"""
Core Data Models for AIOps on Kubernetes Engine.
Defines telemetry points, raw alerts, anomaly signals, correlated incidents,
and approval-gated remediation plans.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TelemetryPoint:
    """
    A single raw time-series metric observation from DCGM, vLLM, or Kubelet.
    """
    timestamp: float
    node: str
    metric: str
    value: float
    gpu_uuid: Optional[str] = None
    namespace: str = "default"
    pod: str = ""
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TelemetryPoint:
        return cls(
            timestamp=float(data["timestamp"]),
            node=str(data["node"]),
            metric=str(data["metric"]),
            value=float(data["value"]),
            gpu_uuid=data.get("gpu_uuid"),
            namespace=data.get("namespace", "default"),
            pod=data.get("pod", ""),
            labels=data.get("labels", {}),
        )


@dataclass
class RawAlert:
    """
    An un-correlated alert from Prometheus Alertmanager, Node Problem Detector (NPD),
    or Kubelet event streams.
    """
    alert_id: str
    timestamp: float
    source: str  # "prometheus" | "npd" | "kubelet"
    name: str
    severity: str  # "info" | "warning" | "critical"
    node: str
    pod: str = ""
    namespace: str = "default"
    gpu_uuid: Optional[str] = None
    value: float = 0.0
    annotations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RawAlert:
        return cls(
            alert_id=str(data["alert_id"]),
            timestamp=float(data["timestamp"]),
            source=str(data["source"]),
            name=str(data["name"]),
            severity=str(data["severity"]),
            node=str(data["node"]),
            pod=data.get("pod", ""),
            namespace=data.get("namespace", "default"),
            gpu_uuid=data.get("gpu_uuid"),
            value=float(data.get("value", 0.0)),
            annotations=data.get("annotations", {}),
        )


@dataclass
class AnomalySignal:
    """
    A statistically significant deviation detected by EWMA, Z-Score,
    or monotonic linear regression saturation estimators.
    """
    metric: str
    node: str
    pod: str
    gpu_uuid: Optional[str] = None
    observed_value: float = 0.0
    baseline_mean: float = 0.0
    z_score: float = 0.0
    slope_per_min: float = 0.0
    anomaly_type: str = "SPIKE"  # "SPIKE" | "SATURATION_SLOPE" | "THRESHOLD_BREACH" | "RATE_INCREASE"
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnomalySignal:
        return cls(
            metric=str(data["metric"]),
            node=str(data["node"]),
            pod=str(data.get("pod", "")),
            gpu_uuid=data.get("gpu_uuid"),
            observed_value=float(data.get("observed_value", 0.0)),
            baseline_mean=float(data.get("baseline_mean", 0.0)),
            z_score=float(data.get("z_score", 0.0)),
            slope_per_min=float(data.get("slope_per_min", 0.0)),
            anomaly_type=str(data.get("anomaly_type", "SPIKE")),
            timestamp=float(data.get("timestamp", 0.0)),
            details=data.get("details", {}),
        )


@dataclass
class RemediationPlan:
    """
    Actionable remediation plan with safety classification, PodDisruptionBudget checks,
    and human-in-the-loop approval gating.
    """
    action_type: str
    risk_tier: str  # "LOW_READONLY" | "MEDIUM_SCALING" | "HIGH_MUTATION" | "CRITICAL_NODE_DRAIN"
    requires_approval: bool
    pdb_check_passed: bool
    approval_status: str  # "AUTO_APPROVED" | "PENDING_APPROVAL" | "APPROVED" | "BLOCKED_BY_PDB"
    approved_by: Optional[str] = None
    kubectl_commands: List[str] = field(default_factory=list)
    declarative_patch: Dict[str, Any] = field(default_factory=dict)
    blast_radius_summary: str = ""
    audit_log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RemediationPlan:
        return cls(
            action_type=str(data["action_type"]),
            risk_tier=str(data["risk_tier"]),
            requires_approval=bool(data["requires_approval"]),
            pdb_check_passed=bool(data["pdb_check_passed"]),
            approval_status=str(data["approval_status"]),
            approved_by=data.get("approved_by"),
            kubectl_commands=list(data.get("kubectl_commands", [])),
            declarative_patch=data.get("declarative_patch", {}),
            blast_radius_summary=str(data.get("blast_radius_summary", "")),
            audit_log=list(data.get("audit_log", [])),
        )


@dataclass
class CorrelatedIncident:
    """
    Topologically and temporally correlated incident synthesizing multiple raw alerts,
    statistical anomaly signals, diagnostic evidence, and proposed remediation.
    """
    incident_id: str
    root_cause_class: str
    confidence: float
    severity: str
    affected_node: str
    affected_pods: List[str] = field(default_factory=list)
    raw_alert_ids: List[str] = field(default_factory=list)
    anomalies: List[AnomalySignal] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    remediation: Optional[RemediationPlan] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.remediation:
            d["remediation"] = self.remediation.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CorrelatedIncident:
        remediation_data = data.get("remediation")
        remediation = RemediationPlan.from_dict(remediation_data) if remediation_data else None
        anomalies = [AnomalySignal.from_dict(a) for a in data.get("anomalies", [])]
        return cls(
            incident_id=str(data["incident_id"]),
            root_cause_class=str(data["root_cause_class"]),
            confidence=float(data["confidence"]),
            severity=str(data["severity"]),
            affected_node=str(data["affected_node"]),
            affected_pods=list(data.get("affected_pods", [])),
            raw_alert_ids=list(data.get("raw_alert_ids", [])),
            anomalies=anomalies,
            evidence=list(data.get("evidence", [])),
            remediation=remediation,
        )
