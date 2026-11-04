# AiOps on K8S: Foundations and Practices

**Production engineering handbook, SRE runbooks, and reference Kubernetes manifests for AI inference at scale: GPU Operator, DRA/CDI, MIG/MPS partitioning, Volcano, LeaderWorkerSet, vLLM, KServe, KEDA, and DCGM.**

---

## Repository Structure

```text
AiOps_on_K8S_Fundamentals_n_Practicies/
├── README.md                                                    # Master overview & navigation
├── docs/                                                        # 10-Chapter Engineering Handbook & SRE Playbooks
│   ├── 00-aiops-on-k8s-foundations-and-practices.md             # Master Architecture & Executive Reference
│   ├── 01-hardware-abstraction-orchestration.md                 # 1. Device Plugin, DRA/CDI, GPU Operator, vGPU/VFIO & Quotas
│   ├── 02-gpu-partitioning-strategies.md                        # 2. Hypervisor vGPU, MIG Silicon Placement, MPS & Time-Slicing
│   ├── 03-advanced-workload-scheduling.md                       # 3. Gang Scheduling (Volcano/LWS), NVSwitch & GPUDirect RDMA
│   ├── 04-inference-runtimes-model-serving.md                   # 4. vLLM/TRT-LLM/SGLang, Disaggregated P/D, KServe & KEDA
│   ├── 05-observability-and-telemetry.md                        # 5. DCGM Profiling, LLM SLOs, XID Catalog & 5-Way OOM Triage
│   ├── 06-model-storage-and-cold-start-optimization.md          # 6. safetensors mmap Traps, OCI Modelcars, NVMe Cache & GDS
│   ├── 07-security-multitenancy-and-governance.md               # 7. Least-Privilege GPU Containers, RBAC & CEL Guardrails
│   ├── 08-linux-to-k8s-sysadmin-bridge.md                       # 8. Bare-Metal Linux/CUDA to Kubernetes & PromQL Reference
│   ├── 09-day2-incident-runbooks.md                             # 9. On-Call Incident Playbooks (INC-01 through INC-08)
│   └── 10-production-readiness-checklist.md                     # 10. Pre-Flight Audit Checklist & GPU FinOps Sizing Worksheet
├── manifests/                                                   # Production Reference Kubernetes Manifests
│   ├── 01-hardware-orchestration/
│   │   ├── gpu-operator-clusterpolicy.yaml                      # Precompiled drivers, nvidia runtime, mixed MIG & DCGM
│   │   ├── nvidia-runtimeclass.yaml                             # Explicit nvidia RuntimeClass & Run:ai Fair-Share Queues
│   │   ├── kueue-gpu-clusterqueue.yaml                          # Kueue ResourceFlavors, Cohort Borrowing & Preemption
│   │   └── dra-gpu-resourceclaim.yaml                           # K8s Dynamic Resource Allocation (DRA) DeviceClass & Claim
│   ├── 02-gpu-partitioning/
│   │   ├── mig-parted-configmap.yaml                            # Mixed MIG profiles (3g.40gb, 2g.20gb, 1g.10gb)
│   │   └── time-slicing-and-mps-configmap.yaml                  # Time-Slicing & CUDA MPS Device Plugin configurations
│   ├── 03-scheduling/
│   │   ├── volcano-gang-distributed-vllm.yaml                   # Volcano PodGroup + NVLink P2P + 32Gi /dev/shm StatefulSet
│   │   └── leaderworkerset-multinode-vllm.yaml                  # K8s LeaderWorkerSet (LWS) multi-node 405B RDMA serving
│   ├── 04-inference-serving/
│   │   ├── kserve-vllm-canary.yaml                              # KServe RawDeployment (90/10 Canary, FP8 KV-cache, warm pool)
│   │   └── keda-vllm-autoscaler.yaml                            # KEDA ScaledObject on vLLM queue depth & KV-cache usage
│   ├── 05-observability/
│   │   ├── dcgm-custom-metrics-configmap.yaml                   # DCGM Exporter CSV with SM/Tensor/DRAM/NVLink profiling
│   │   ├── gpu-node-problem-detector.yaml                       # Kernel kmsg XID fault detector & NodeCondition reporter
│   │   └── prometheus-gpu-oom-alerts.yaml                       # PrometheusRule correlating Exit 137 vs CUDA OOM vs XID
│   ├── 06-model-storage/
│   │   └── fluid-s3-accelerator.yaml                            # Fluid AlluxioRuntime NVMe tiered cache for model weights
│   └── 06-security-guardrails/
│       └── vap-gpu-inference-guardrails.yaml                    # Native K8s CEL ValidatingAdmissionPolicy for GPU pods
└── scripts/
    └── validate_repo.py                                         # Markdown link, Mermaid diagram & YAML manifest validator
```

---

## Engineering Handbook Chapters

Start with [**00 — AiOps on K8S: Foundations and Practices**](./docs/00-aiops-on-k8s-foundations-and-practices.md), or jump directly to a specific layer:

### Part I: Core GPU & Inference Infrastructure
1. [**01 — Hardware Abstraction & Orchestration**](./docs/01-hardware-abstraction-orchestration.md): Device Plugin API vs. **Dynamic Resource Allocation (DRA) & CDI**, NVIDIA GPU Operator lifecycle, VFIO passthrough vs. hypervisor vGPU, and multi-tenant quota management (**Kueue** & **Run:ai / KAI**).
2. [**02 — GPU Partitioning & Sharing Strategies**](./docs/02-gpu-partitioning-strategies.md): Hardware **MIG** silicon GI/CI placement rules, **CUDA MPS** pinned memory/thread limits, and why software **Time-Slicing** triggers deterministic `vLLM` pre-allocation OOM crashes.
3. [**03 — Advanced Workload Scheduling**](./docs/03-advanced-workload-scheduling.md): Atomic gang scheduling (**Volcano**, **KubeRay**, **LeaderWorkerSet**), NUMA & PCIe/NVSwitch affinity, **GPUDirect RDMA** (`nvidia-peermem`), and `/dev/shm` + `RLIMIT_MEMLOCK` sizing for NCCL.
4. [**04 — Inference Runtimes & Model Serving**](./docs/04-inference-runtimes-model-serving.md): **vLLM** vs. **TensorRT-LLM** vs. **SGLang**, PagedAttention KV-cache math, Chunked Prefill, **Disaggregated Prefill/Decode** serving, KServe canary routing, and **KEDA** token/queue autoscaling.
5. [**05 — Observability & Telemetry**](./docs/05-observability-and-telemetry.md): Three-layer telemetry (`cAdvisor` + **DCGM Profiling** + runtime metrics), TTFT/TPOT SLO engineering, comprehensive **NVIDIA XID Error Catalog**, and 5-way crash correlation.

### Part II: Day-2 Production Operations, Storage & Security
6. [**06 — Model Storage & Cold-Start Optimization**](./docs/06-model-storage-and-cold-start-optimization.md): Eliminating 4-minute LLM cold starts, resolving the **`safetensors` `mmap()` Linux cgroup page-cache `OOMKilled` (`Exit 137`) trap**, OCI Modelcars, NVMe caching, and **GPUDirect Storage (GDS)**.
7. [**07 — Security, Multi-Tenancy & Governance**](./docs/07-security-multitenancy-and-governance.md): Why `privileged: true` exposes all host GPUs, enforcing `runtimeClassName: nvidia` and `/dev/shm` guardrails via native Kubernetes **CEL `ValidatingAdmissionPolicy`**, VRAM zeroing, and RBAC hardening.
8. [**08 — Linux-to-Kubernetes Sysadmin Bridge**](./docs/08-linux-to-k8s-sysadmin-bridge.md): Practical translation reference mapping bare-metal Linux/CUDA commands (`nvidia-smi`, `topo -m`, `dcgmi`, `dmesg`, `numactl`, `ipcs`, `ulimit`) to `kubectl` API queries and DCGM PromQL.
9. [**09 — Day-2 SRE Incident Runbooks**](./docs/09-day2-incident-runbooks.md): Eight step-by-step on-call playbooks (`INC-01` through `INC-08`) covering cgroup vs. CUDA OOMs, Time-Slicing collisions, NCCL `/dev/shm` `SIGBUS`, `XID 79/95` hardware faults, gang deadlocks, KV-cache preemption storms, and kernel driver mismatches.
10. [**10 — Production Readiness Checklist & FinOps**](./docs/10-production-readiness-checklist.md): Pre-flight audit matrix across 6 operational layers plus a worked **GPU Capacity Planning & FinOps Sizing Worksheet**.

---

## Validating Documentation & Manifests

Run the zero-dependency repository validator to verify all relative Markdown links, Mermaid diagrams, and Kubernetes YAML manifests:

```bash
python3 scripts/validate_repo.py
```

---

## Companion CLI Tool: KubeVRAM

For terminal-centric GPU topology mapping, automated 5-way OOM/XID triage, deterministic `vLLM` KV-cache sizing, and interactive Linux-to-Kubernetes translation, see the companion repository:
- [**xsub/KubeVRAM**](https://github.com/xsub/KubeVRAM)
