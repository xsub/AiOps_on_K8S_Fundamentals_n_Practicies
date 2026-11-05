#!/usr/bin/env bash
# Deploy the Runnable AIOps Reference Implementation onto a local kind Kubernetes cluster.
set -euo pipefail

CLUSTER_NAME="${KIND_CLUSTER_NAME:-aiops-ref}"
IMAGE_TAG="aiops-ref-runtime:v1.0.0"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVIDENCE_DIR="${REPO_ROOT}/reference-impl/evidence"

mkdir -p "${EVIDENCE_DIR}"

echo "==> [1/5] Ensuring kind Kubernetes cluster '${CLUSTER_NAME}' exists..."
if ! kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  kind create cluster --name "${CLUSTER_NAME}" --config "${REPO_ROOT}/reference-impl/k8s/kind-config.yaml" --wait 90s
else
  echo "    Cluster '${CLUSTER_NAME}' already running."
fi

kubectl config use-context "kind-${CLUSTER_NAME}" >/dev/null
kubectl wait --for=condition=Ready node --all --timeout=90s

echo "==> [2/5] Building deterministic runtime container image (${IMAGE_TAG})..."
docker build -t "${IMAGE_TAG}" -f "${REPO_ROOT}/reference-impl/Dockerfile" "${REPO_ROOT}"

echo "==> [3/5] Loading '${IMAGE_TAG}' into kind node containerd..."
kind load docker-image "${IMAGE_TAG}" --name "${CLUSTER_NAME}"

echo "==> [4/5] Applying Namespace, RBAC, NetworkPolicy, PDB, and CEL ValidatingAdmissionPolicy..."
kubectl apply -f "${REPO_ROOT}/reference-impl/k8s/00-namespace-rbac-secret.yaml"
kubectl apply -f "${REPO_ROOT}/reference-impl/k8s/01-guardrails-vap.yaml"
kubectl apply -f "${REPO_ROOT}/reference-impl/k8s/02-networkpolicy-pdb.yaml"

# Wait for kube-apiserver CEL policy compiler to activate enforcement
python3 - "${REPO_ROOT}/reference-impl/k8s/policy-tests/reject-privileged-pod.yaml" <<'PY'
import subprocess, sys, time
fixture = sys.argv[1]
deadline = time.monotonic() + 30.0
while time.monotonic() < deadline:
    res = subprocess.run(
        ["kubectl", "apply", "--dry-run=server", "-f", fixture],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 and "ValidatingAdmissionPolicy" in res.stderr:
        print("    [OK] kube-apiserver CEL ValidatingAdmissionPolicy active and enforcing.")
        sys.exit(0)
    time.sleep(0.5)
print("ERROR: Timed out waiting for ValidatingAdmissionPolicy activation", file=sys.stderr)
sys.exit(1)
PY

echo "==> [5/5] Deploying Inference Engine & Closed-Loop AIOps Controller (measuring Cold-Start)..."
T_START="$(python3 -c 'import time; print(f"{time.time():.6f}")')"

kubectl apply -f "${REPO_ROOT}/reference-impl/k8s/03-inference-deployment.yaml"
kubectl apply -f "${REPO_ROOT}/reference-impl/k8s/04-aiops-controller.yaml"

kubectl rollout status deployment/inference-engine -n ai-inference --timeout=90s
kubectl rollout status deployment/aiops-controller -n ai-inference --timeout=90s

python3 - "${T_START}" "${EVIDENCE_DIR}/cold-start.json" <<'PY'
import json, sys, time, urllib.request
t_start = float(sys.argv[1])
out_path = sys.argv[2]
deadline = time.monotonic() + 30.0
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:18000/readyz", timeout=2.0) as resp:
            if resp.status == 200:
                t_ready = time.time()
                cold_start_s = round(t_ready - t_start, 3)
                body = json.loads(resp.read().decode("utf-8"))
                payload = {"cold_start_seconds": cold_start_s, "readyz": body}
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                print(f"    [OK] Inference Service HTTP /readyz reachable in {cold_start_s}s.")
                sys.exit(0)
    except Exception:
        time.sleep(0.25)
print("ERROR: Timed out waiting for http://127.0.0.1:18000/readyz", file=sys.stderr)
sys.exit(1)
PY

echo "==> Deployment complete."
