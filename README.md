# AiOps on K8S: Foundations, Reference Architecture & Closed-Loop Engine

[![CI](https://github.com/xsub/AiOps_on_K8S_Fundamentals_n_Practicies/actions/workflows/ci.yml/badge.svg)](https://github.com/xsub/AiOps_on_K8S_Fundamentals_n_Practicies/actions/workflows/ci.yml)

**GPU platform engineering handbook, schema-validated Kubernetes reference overlays, SRE runbooks, and a runnable closed-loop AIOps control-plane engine for LLM inference infrastructure.**

---

## Scope, Positioning & Honest Validation Status

In industry practice, *"AIOps on Kubernetes"* is often conflated with *"Kubernetes for AI workloads"*. This repository deliberately separates and implements **both** disciplines:

1. **Kubernetes for AI / GPU Platform Engineering** (`docs/01`–`10`, `deploy/`, `manifests/`): Hardware abstraction (`Device Plugin`, `DRA/CDI`, `GPU Operator`), silicon partitioning (`MIG`, `MPS`, `Time-Slicing`), gang/topology scheduling (`LeaderWorkerSet`, `Kueue`, `Volcano`), `vLLM`/`KServe`/`KEDA` serving, `DCGM` telemetry, storage cold-start optimization, and CEL admission guardrails.
2. **AIOps for GPU Infrastructure** (`docs/11`, [`aiops/`](./aiops)): An executable closed-loop pipeline (`Telemetry Stream -> EWMA/Z-Score & OLS VRAM Slope Detection -> Multi-Signal Topological Alert Correlation -> Root-Cause Diagnosis -> PDB-Checked & Human-Approval-Gated Remediation`) evaluated against a 12-scenario labeled incident benchmark.

### Artifact Validation Tiers

No manifest in this repository claims unverified bare-metal hardware execution. Every artifact carries an explicit verification tier enforced in [GitHub Actions CI](./.github/workflows/ci.yml):

| Component | Validation Tier | Automated Verification Executed in CI |
| :--- | :--- | :--- |
| **`aiops/` Closed-Loop Engine** | `executed-and-benchmarked` | `47` unit/integration tests + `python3 -m aiops evaluate` (12 scenarios, 56 raw alerts) |
| **`deploy/` Kustomize Overlays** | `schema-and-kustomize-validated` | `kustomize build` (`base`, `kserve-keda`, `lws-kueue`, `volcano-gang`) + `kubeconform -strict` with CRD schemas |
| **`manifests/` Reference Specs** | `schema-and-security-validated` | `kubeconform -strict` (37 resources), `@sha256` digest pinning, `yamllint`, `trivy config`, `gitleaks` |
| **Prometheus / DCGM Alert Rules** | `promtool-unit-tested` | `promtool check rules` + `promtool test rules` ([`prometheus-gpu-oom-alerts.test.yaml`](./manifests/05-observability/prometheus-gpu-oom-alerts.test.yaml)) |
| **CEL Admission Guardrails** | `policy-simulation-tested` | `16` positive, negative, and mount-evasion unit tests ([`test_admission_policies.py`](./tests/test_admission_policies.py)) |
| **GPU Data-Plane Execution** | `requires-nvidia-gpu-nodes` | Requires physical Ampere/Hopper nodes (NVML, MIG silicon, RoCEv2/NVSwitch) |

---

## Reference Stack & Overlay Orthogonality

Rather than mixing incompatible schedulers (`Kueue`, `Volcano`, `Run:ai`) or competing autoscalers (`KServe` vs. manual `KEDA`) into a single cluster state, [`deploy/`](./deploy) isolates mutually exclusive architectures into Kustomize overlays pinned to a single reference matrix:

| Layer | Reference Component & Version | Deployment Overlay / Path |
| :--- | :--- | :--- |
| **Cluster & Runtime** | Kubernetes `1.32.x`, `containerd` + `RuntimeClass: nvidia` | [`deploy/base/`](./deploy/base/) |
| **GPU Lifecycle** | NVIDIA GPU Operator `v25.3.x`, Precompiled Driver Branch `"570"` | [`gpu-operator-clusterpolicy.yaml`](./manifests/01-hardware-orchestration/gpu-operator-clusterpolicy.yaml) |
| **Inference Engine** | `vllm/vllm-openai:v0.33.0` (pinned `@sha256:d825baf...`) | All serving overlays |
| **Single/Multi-GPU Serving + Canary** | KServe `v0.15.x` (`RawDeployment` + `autoscalerClass: keda`) + Gateway API `HTTPRoute` | [`deploy/overlays/kserve-keda/`](./deploy/overlays/kserve-keda/) |
| **Multi-Node Serving + Fair-Share** | LeaderWorkerSet `v0.6.x` (Ray readiness barrier) + Kueue `v0.11.x` | [`deploy/overlays/lws-kueue/`](./deploy/overlays/lws-kueue/) |
| **Gang-Scheduled Distributed Serving** | Volcano `v1.11.x` (`PodGroup` `head` + `worker` + headless `Service`) | [`deploy/overlays/volcano-gang/`](./deploy/overlays/volcano-gang/) |
| **Hardware Fault Telemetry** | DCGM Exporter (`honorLabels: true`) + Node Problem Detector `v1.36.1` (`kmsg`) | [`manifests/05-observability/`](./manifests/05-observability/) |

---

## Repository Structure

```text
AiOps_on_K8S_Fundamentals_n_Practicies/
├── .github/workflows/ci.yml                                     # Automated CI: tests, kubeconform, promtool, trivy, gitleaks
├── Makefile                                                     # Reproducible local build, validation & benchmark targets
├── docs/                                                        # 11-Chapter Engineering Handbook (with primary citations)
│   ├── 00-aiops-on-k8s-foundations-and-practices.md             # Master Architecture, Positioning & Compatibility Matrix
│   ├── 01-hardware-abstraction-orchestration.md                 # 1. Device Plugin, DRA/CDI, GPU Operator, vGPU/VFIO & Quotas
│   ├── 02-gpu-partitioning-strategies.md                        # 2. Hypervisor vGPU, MIG Silicon Placement, MPS & Time-Slicing
│   ├── 03-advanced-workload-scheduling.md                       # 3. Gang Scheduling (Volcano/LWS), NVSwitch & GPUDirect RDMA
│   ├── 04-inference-runtimes-model-serving.md                   # 4. vLLM/TRT-LLM/SGLang, Disaggregated P/D, KServe & KEDA
│   ├── 05-observability-and-telemetry.md                        # 5. DCGM Profiling, LLM SLOs, XID Catalog & 5-Way OOM Triage
│   ├── 06-model-storage-and-cold-start-optimization.md          # 6. safetensors mmap Traps, OCI Modelcars, NVMe Cache & GDS
│   ├── 07-security-multitenancy-and-governance.md               # 7. Least-Privilege GPU Containers, RBAC & CEL Guardrails
│   ├── 08-linux-to-k8s-sysadmin-bridge.md                       # 8. Bare-Metal Linux/CUDA to Kubernetes & PromQL Reference
│   ├── 09-day2-incident-runbooks.md                             # 9. On-Call Incident Playbooks (INC-01 through INC-08)
│   ├── 10-production-readiness-checklist.md                     # 10. Pre-Flight Audit Checklist & GPU FinOps Sizing Worksheet
│   └── 11-aiops-closed-loop-architecture.md                     # 11. Anomaly Detection Math, Topological RCA & PDB Remediation
├── aiops/                                                       # Runnable Closed-Loop AIOps Engine (Python 3.10+ stdlib)
│   ├── models.py                                                # TelemetryPoint, RawAlert, CorrelatedIncident, RemediationPlan
│   ├── detector.py                                              # Rolling EWMA Z-Score, OLS VRAM TTE slope & prefill burst filter
│   ├── correlator.py                                            # Multi-signal topological correlator (7 failure classes + XID)
│   ├── remediator.py                                            # PDB-checked & human-approval-gated remediation planner
│   ├── evaluate.py                                              # Reproducible 12-scenario offline benchmark harness
│   ├── cli.py                                                   # CLI entrypoint (evaluate & interactive demo)
│   └── datasets/labeled_gpu_incidents.json                      # 12 labeled telemetry + alert-storm scenarios (56 raw alerts)
├── deploy/                                                      # Coherent Kustomize Base & Mutually Exclusive Overlays
│   ├── base/                                                    # Namespace, least-privilege SA/RBAC, default-deny NetworkPolicy, PDB, PVC
│   └── overlays/
│       ├── kserve-keda/                                         # TP=2 70B KServe RawDeployment + native KEDA + HTTPRoute 90/10 canary
│       ├── lws-kueue/                                           # Multi-node 405B LeaderWorkerSet + Ray barrier + Kueue ClusterQueue
│       └── volcano-gang/                                        # Volcano PodGroup + headless Service + Head & Worker StatefulSets
├── manifests/                                                   # Schema-Validated Reference Manifests & Promtool Tests
│   ├── 01-hardware-orchestration/                               # ClusterPolicy (branch 570), RuntimeClass, Kueue, DRA Claim
│   ├── 02-gpu-partitioning/                                     # MIG mixed profiles, Time-Slicing & CUDA MPS configs
│   ├── 03-scheduling/                                           # Volcano gang & LeaderWorkerSet reference specs
│   ├── 04-inference-serving/                                    # KServe HTTPRoute canary & standalone Deployment KEDA ScaledObject
│   ├── 05-observability/                                        # DCGM CSV, ServiceMonitor (honorLabels), NPD (kmsg), PromQL + tests
│   ├── 06-model-storage/                                        # Fluid S3 NVMe cache with OIDC Workload Identity (IRSA)
│   └── 06-security-guardrails/                                  # 4-rule Kubernetes CEL ValidatingAdmissionPolicy
├── tests/                                                       # 47 Automated Unit, Policy & Manifest Invariant Tests
└── scripts/validate_repo.py                                     # 10-check semantic validator (digests, services, gangs, NPD, PromQL)
```

---

## Quickstart: Verification, Kustomize Build & AIOps Benchmark

```bash
# Run full local CI pipeline (semantic validator + 47 unit tests + AIOps benchmark + kustomize + promtool)
make ci

# Or run individual verification steps:
make validate          # 10-check semantic & security repository validator
make test              # 47 unit tests (CEL guardrails, VRAM sizing math, PromQL, AIOps pipeline)
make evaluate-aiops    # Run 12-scenario AIOps benchmark (outputs Precision/Recall/F1 & noise reduction)
make demo-aiops        # Walk through live alert storm -> RCA -> approval gate -> kubectl patch
make kustomize-build   # Build deploy/base and all 3 overlays + kubeconform -strict schema check
make promtool-test     # Verify PrometheusRule syntax and run promtool unit tests
```

---

## Engineering Handbook Navigation

Start with [**00 — Foundations, Positioning & Compatibility Matrix**](./docs/00-aiops-on-k8s-foundations-and-practices.md):

1. [**01 — Hardware Abstraction & Orchestration**](./docs/01-hardware-abstraction-orchestration.md) — Device Plugin vs. DRA/CDI, GPU Operator precompiled driver branches vs. GDS, VFIO/vGPU, and Kueue/Run:ai quotas.
2. [**02 — GPU Partitioning & Sharing Strategies**](./docs/02-gpu-partitioning-strategies.md) — MIG binary-buddy silicon placement, post-Volta CUDA MPS address-space vs. fault boundaries, and Time-Slicing OOM math.
3. [**03 — Advanced Workload Scheduling**](./docs/03-advanced-workload-scheduling.md) — Volcano, KubeRay, LeaderWorkerSet Ray synchronization barriers, NUMA/NVSwitch topology, and GPUDirect RDMA.
4. [**04 — Inference Runtimes & Model Serving**](./docs/04-inference-runtimes-model-serving.md) — vLLM/TRT-LLM/SGLang, GQA/MLA KV-cache math, Disaggregated Prefill/Decode, Gateway API canaries, and KEDA scaling.
5. [**05 — Observability & Telemetry**](./docs/05-observability-and-telemetry.md) — `cAdvisor` + DCGM profiling (`honorLabels: true`), TTFT/TPOT SLOs, hardware vs. user-space XID catalog, and 5-way crash correlation.
6. [**06 — Model Storage & Cold-Start Optimization**](./docs/06-model-storage-and-cold-start-optimization.md) — Resolving `safetensors` `mmap()` cgroup page-cache `Exit 137` OOMs, OCI Modelcars, Fluid NVMe caching, and GPUDirect Storage.
7. [**07 — Security, Multi-Tenancy & Governance**](./docs/07-security-multitenancy-and-governance.md) — Least-privilege CDI containers, MIG vs. Confidential Computing, IRSA Workload Identity, NetworkPolicies, and CEL guardrails.
8. [**08 — Linux-to-Kubernetes Sysadmin Bridge**](./docs/08-linux-to-k8s-sysadmin-bridge.md) — Mapping bare-metal Linux/CUDA diagnostics (`nvidia-smi`, `topo -m`, `dcgmi`, `dmesg`, `numactl`, `ipcs`) to Kubernetes & PromQL.
9. [**09 — Day-2 SRE Incident Runbooks**](./docs/09-day2-incident-runbooks.md) — Eight step-by-step on-call playbooks (`INC-01`–`INC-08`) with mitigation and declarative fixes.
10. [**10 — Production Readiness Checklist & FinOps**](./docs/10-production-readiness-checklist.md) — Six-domain pre-flight audit matrix and Little's Law GPU capacity/cost worksheet.
11. [**11 — AIOps Closed-Loop Architecture**](./docs/11-aiops-closed-loop-architecture.md) — EWMA Z-score & OLS VRAM TTE detection math, topological alert correlation, and PDB-governed human approval gates.

---

## Companion CLI Tool: KubeVRAM

For terminal-centric GPU topology mapping, 5-way OOM/XID triage, deterministic `vLLM` KV-cache sizing, and interactive Linux-to-Kubernetes translation, see [**xsub/KubeVRAM**](https://github.com/xsub/KubeVRAM).
