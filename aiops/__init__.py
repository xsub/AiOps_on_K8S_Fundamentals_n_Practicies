"""
AIOps on Kubernetes Engine.
A Python 3.10+ standard-library-only closed-loop engine fulfilling:
Telemetry Stream -> Statistical Anomaly Detection -> Multi-Signal Topological Correlation
-> Root-Cause Diagnosis -> Approval-Gated & PDB-Checked Remediation.
"""

from aiops.correlator import TopologyCorrelator
from aiops.detector import (
    BenignBurstFilter,
    RollingZScoreDetector,
    SaturationSlopeEstimator,
    UnifiedAnomalyDetector,
)
from aiops.evaluate import format_evaluation_markdown, load_dataset, run_evaluation
from aiops.models import (
    AnomalySignal,
    CorrelatedIncident,
    RawAlert,
    RemediationPlan,
    TelemetryPoint,
)
from aiops.remediator import RemediationEngine

__version__ = "1.0.0"

__all__ = [
    "TelemetryPoint",
    "RawAlert",
    "AnomalySignal",
    "RemediationPlan",
    "CorrelatedIncident",
    "RollingZScoreDetector",
    "SaturationSlopeEstimator",
    "BenignBurstFilter",
    "UnifiedAnomalyDetector",
    "TopologyCorrelator",
    "RemediationEngine",
    "run_evaluation",
    "format_evaluation_markdown",
    "load_dataset",
]
