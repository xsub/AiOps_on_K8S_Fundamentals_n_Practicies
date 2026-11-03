# AiOps on K8S: Foundations and Practices

**Production engineering handbook and reference Kubernetes manifests for AI inference at scale: GPU Operator, MIG/MPS partitioning, Volcano, vLLM, KServe, and DCGM.**

---

## Repository Structure

```text
AiOps_on_K8S_Fundamentals_n_Practicies/
├── README.md                                          # Master overview & navigation
├── docs/                                              # Engineering Knowledge Base
│   ├── 00-aiops-on-k8s-foundations-and-practices.md   # Complete Unified Reference Document
│   ├── 01-hardware-abstraction-orchestration.md       # 1. Hardware Abstraction, GPU Operator & Run:ai Quotas
│   ├── 02-gpu-partitioning-strategies.md              # 2. GPU Partitioning: Time-Slicing vs MIG vs MPS
│   ├── 03-advanced-workload-scheduling.md             # 3. Gang Scheduling (Volcano/KubeRay) & NVLink Topology
│   ├── 04-inference-runtimes-model-serving.md         # 4. vLLM / Triton Runtimes, KV-Cache Math & KServe Canary
│   └── 05-observability-and-telemetry.md              # 5. DCGM Telemetry & K8s OOMKilled vs CUDA VRAM OOM
└── manifests/                                         # Production Reference Kubernetes Manifests
    ├── 01-hardware-orchestration/
    │   ├── gpu-operator-clusterpolicy.yaml            # Decoupled drivers, nvidia runtime, MIG & DCGM enabled
    │   └── nvidia-runtimeclass.yaml                   # Explicit nvidia RuntimeClass & Run:ai Fair-Share Queues
    ├── 02-gpu-partitioning/
    │   ├── mig-parted-configmap.yaml                  # Balanced mixed MIG profiles (3g.40gb, 2g.20gb, 1g.10gb)
    │   └── time-slicing-and-mps-configmap.yaml        # Time-Slicing & CUDA MPS Device Plugin configurations
    ├── 03-scheduling/
    │   └── volcano-gang-distributed-vllm.yaml         # Volcano PodGroup + NVLink P2P + 32Gi /dev/shm StatefulSet
    ├── 04-inference-serving/
    │   └── kserve-vllm-canary.yaml                    # KServe InferenceService (90/10 Canary, FP8 KV-cache, minReplicas: 2)
    └── 05-observability/
        └── prometheus-gpu-oom-alerts.yaml             # PrometheusRule correlating Exit 137 vs CUDA OOM vs XID faults
```

---

## Engineering Pillars

Start with the unified document [**AiOps on K8S: Foundations and Practices**](./docs/00-aiops-on-k8s-foundations-and-practices.md), or dive directly into each engineering chapter:

1. [**1. Hardware Abstraction & Orchestration**](./docs/01-hardware-abstraction-orchestration.md)
   - Device Plugin API vs. Dynamic Resource Allocation (DRA + CDI), NVIDIA GPU Operator DaemonSets, decoupled precompiled drivers, explicit `nvidia` `RuntimeClass`, and multi-tenant fair-share quotas via **Run:ai (KAI Scheduler)** / **Kueue**.
2. [**2. GPU Partitioning and Sharing Strategies (Hypervisor & Kubernetes)**](./docs/02-gpu-partitioning-strategies.md)
   - Hypervisor GPU virtualization (**VFIO PCIe Passthrough** vs. **NVIDIA vGPU** vs. **SR-IOV MIG-backed vGPU**) alongside Kubernetes sharing (**Time-Slicing**, **MIG**, and **CUDA MPS**)—including why Time-Slicing causes fatal pre-allocation VRAM OOMs in vLLM.
3. [**3. Advanced Workload Scheduling**](./docs/03-advanced-workload-scheduling.md)
   - Overcoming default `kube-scheduler` deadlocks and topology blindness with **Gang Scheduling** (**Volcano** / **KubeRay** `PodGroup`), **Topology Manager** (`single-numa-node`), `nvidia-smi topo -m` NVLink/NVSwitch placement, and `/dev/shm` sizing (`SIGBUS` Exit 135 prevention) for NCCL.
4. [**4. Inference Runtimes and Model Serving**](./docs/04-inference-runtimes-model-serving.md)
   - Separating the **Inference Engine** (**vLLM**, **Triton / TensorRT-LLM**, **SGLang**) from the **Serving Platform** (**KServe + Istio + KEDA**); Prefill vs. Decode roofline physics, PagedAttention KV-Cache math, canary rollouts, and avoiding LLM `scale-to-zero` cold starts.
5. [**5. Observability and Telemetry**](./docs/05-observability-and-telemetry.md)
   - Why cAdvisor fails for GPUs; correlating **DCGM Exporter** hardware counters with **vLLM runtime metrics**; production **NVIDIA XID Error Catalog**; and automated 5-way triage across **K8s `OOMKilled` (Exit 137)**, **CUDA VRAM OOM (Exit 1)**, **Time-Slicing Collisions**, **NCCL `/dev/shm` Exhaustion (Exit 135)**, and **Silicon XID Faults**.

---

## Companion CLI Tool: KubeVRAM

For terminal-centric GPU topology mapping, OOM/XID triage, deterministic vLLM KV-cache tuning, and Linux-to-Kubernetes intent translation, see the companion CLI repository:
- [**xsub/KubeVRAM**](https://github.com/xsub/KubeVRAM)
