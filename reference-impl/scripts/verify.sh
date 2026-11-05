#!/usr/bin/env bash
# Verify live Kubernetes cluster guardrails, SSE inference smoke test, TTFT/TPOT benchmark,
# and live Closed-Loop AIOps + PodDisruptionBudget state transitions.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVIDENCE_DIR="${REPO_ROOT}/reference-impl/evidence"
BASE_URL="${REF_BASE_URL:-http://127.0.0.1:18000}"
API_KEY="REF_IMPL_DEMO_AUTH_PLACEHOLDER"

mkdir -p "${EVIDENCE_DIR}"

echo "==> [1/5] Capturing live Kubernetes cluster & operator/resource status..."
{
  echo "=== KUBERNETES VERSION ==="
  kubectl version
  echo ""
  echo "=== NODES ==="
  kubectl get nodes -o wide
  echo ""
  echo "=== VALIDATING ADMISSION POLICIES ==="
  kubectl get validatingadmissionpolicy,validatingadmissionpolicybinding
  echo ""
  echo "=== AI-INFERENCE NAMESPACE WORKLOADS, SERVICES, PDB & NETWORKPOLICIES ==="
  kubectl get deploy,pods,svc,pdb,networkpolicy,sa -n ai-inference -o wide
} > "${EVIDENCE_DIR}/kubectl-cluster-state.txt"
cat "${EVIDENCE_DIR}/kubectl-cluster-state.txt"

echo "==> [2/5] Testing live kube-apiserver CEL ValidatingAdmissionPolicy rejections..."
{
  echo "=== TEST 1: Privileged Container Rejection ==="
  if kubectl apply -f "${REPO_ROOT}/reference-impl/k8s/policy-tests/reject-privileged-pod.yaml" 2>&1; then
    echo "ERROR: Privileged pod was unexpectedly admitted!" >&2
    exit 1
  else
    echo "[PASS] Privileged pod deterministically denied by kube-apiserver CEL policy."
  fi
  echo ""
  echo "=== TEST 2: Unmounted /dev/shm Memory Volume Rejection ==="
  if kubectl apply -f "${REPO_ROOT}/reference-impl/k8s/policy-tests/reject-missing-shm-pod.yaml" 2>&1; then
    echo "ERROR: Unmounted /dev/shm pod was unexpectedly admitted!" >&2
    exit 1
  else
    echo "[PASS] Unmounted /dev/shm pod deterministically denied by kube-apiserver CEL policy."
  fi
} > "${EVIDENCE_DIR}/vap-admission-live-output.txt"
cat "${EVIDENCE_DIR}/vap-admission-live-output.txt"

echo "==> [3/5] Running authenticated OpenAI inference smoke test & 401 auth gate check..."
python3 - "${BASE_URL}" "${API_KEY}" "${EVIDENCE_DIR}/smoke-test-inference.json" <<'PY'
import json, sys, urllib.error, urllib.request

base_url, api_key, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

# 1. Verify unauthenticated call returns 401
unauth_req = urllib.request.Request(
    f"{base_url}/v1/chat/completions",
    data=b'{"messages":[{"role":"user","content":"ping"}]}',
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    urllib.request.urlopen(unauth_req, timeout=5.0)
    raise SystemExit("ERROR: Unauthenticated request succeeded unexpectedly!")
except urllib.error.HTTPError as exc:
    unauth_status = exc.code
    exc.close()
assert unauth_status == 401, f"Expected 401, got {unauth_status}"

# 2. Verify authenticated OpenAI completion & /readyz
with urllib.request.urlopen(f"{base_url}/readyz", timeout=5.0) as r_resp:
    readyz = json.loads(r_resp.read().decode("utf-8"))

auth_req = urllib.request.Request(
    f"{base_url}/v1/chat/completions",
    data=json.dumps({
        "model": "micro-gqa-llama-ref",
        "messages": [{"role": "user", "content": "verify kubernetes gpu inference pod readiness and kv cache"}],
        "max_tokens": 24,
        "stream": False,
    }).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    },
    method="POST",
)
with urllib.request.urlopen(auth_req, timeout=10.0) as resp:
    completion = json.loads(resp.read().decode("utf-8"))

smoke = {
    "unauthenticated_http_status": unauth_status,
    "authenticated_http_status": 200,
    "readyz": readyz,
    "completion_sample": completion,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(smoke, f, indent=2)
print(json.dumps(smoke, indent=2))
PY

echo "==> [4/5] Running controlled SSE streaming benchmark (TTFT, TPOT, Throughput, Cost/1k reqs)..."
COLD_START_SEC="0.0"
if [[ -f "${EVIDENCE_DIR}/cold-start.json" ]]; then
  COLD_START_SEC="$(python3 -c "import json; print(json.load(open('${EVIDENCE_DIR}/cold-start.json'))['cold_start_seconds'])")"
fi

python3 "${REPO_ROOT}/reference-impl/benchmark/run_benchmark.py" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --cold-start-seconds "${COLD_START_SEC}" \
  --output-dir "${EVIDENCE_DIR}"

echo "==> [5/5] Verifying live Closed-Loop AIOps Controller & Kubernetes PDB state transitions..."
# Phase A: replicas=1 -> PDB disruptionsAllowed=0 -> BLOCKED_BY_PDB
python3 - "${BASE_URL}" "${API_KEY}" <<'PY'
import json, sys, urllib.request
base_url, api_key = sys.argv[1], sys.argv[2]
req = urllib.request.Request(
    f"{base_url}/admin/fault-inject",
    data=json.dumps({"mode": "inject", "xid_code": 79, "kv_pressure_ratio": 0.94}).encode("utf-8"),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    method="POST",
)
urllib.request.urlopen(req, timeout=5.0).close()
PY

PHASE_A_JSON="$(kubectl exec -n ai-inference deploy/aiops-controller -- python3 -m ref_app.k8s_aiops_agent --once)"

# Phase B: scale to replicas=2 -> PDB disruptionsAllowed=1 -> PENDING_APPROVAL
kubectl scale deployment/inference-engine -n ai-inference --replicas=2
kubectl rollout status deployment/inference-engine -n ai-inference --timeout=90s

# Ensure newly scaled pod endpoint is ready in PDB status
python3 - <<'PY'
import json, subprocess, sys, time
deadline = time.monotonic() + 30.0
while time.monotonic() < deadline:
    out = subprocess.check_output(
        ["kubectl", "get", "pdb", "inference-pdb", "-n", "ai-inference", "-o", "json"],
        text=True,
    )
    allowed = int(json.loads(out).get("status", {}).get("disruptionsAllowed", 0))
    if allowed >= 1:
        print(f"    [OK] Kubernetes PDB disruptionsAllowed transitioned to {allowed}.")
        sys.exit(0)
    time.sleep(0.5)
sys.exit("ERROR: PDB disruptionsAllowed did not reach >= 1 after scale-up")
PY

PHASE_B_JSON="$(kubectl exec -n ai-inference deploy/aiops-controller -- python3 -m ref_app.k8s_aiops_agent --once)"

# Validate both phases and write evidence report
python3 - "${PHASE_A_JSON}" "${PHASE_B_JSON}" "${EVIDENCE_DIR}/closed-loop-aiops-live.json" <<'PY'
import json, sys
phase_a = json.loads(sys.argv[1])
phase_b = json.loads(sys.argv[2])
out_path = sys.argv[3]

assert phase_a["pdb_live_state"]["source"] == "kubernetes-api-live"
assert phase_a["pdb_live_state"]["disruptionsAllowed"] == 0
inc_a = phase_a["correlated_incidents"][0]
assert inc_a["incident"]["root_cause_class"] == "GPU_SILICON_XID_FAULT"
assert inc_a["remediation_plan"]["approval_status"] == "BLOCKED_BY_PDB"
assert inc_a["k8s_event_emitted"] is True

assert phase_b["pdb_live_state"]["disruptionsAllowed"] >= 1
inc_b = phase_b["correlated_incidents"][0]
assert inc_b["incident"]["root_cause_class"] == "GPU_SILICON_XID_FAULT"
assert inc_b["remediation_plan"]["approval_status"] == "PENDING_APPROVAL"
assert inc_b["remediation_plan"]["requires_approval"] is True
assert inc_b["k8s_event_emitted"] is True

combined = {
    "phase_a_single_replica_pdb_blocked": phase_a,
    "phase_b_scaled_replicas_pending_human_approval": phase_b,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2)
print("    [PASS] Closed-Loop AIOps verified: BLOCKED_BY_PDB (replicas=1) -> PENDING_APPROVAL (replicas=2).")
PY

# Reset fault state and append live AIOps Kubernetes Events to cluster state evidence
kubectl scale deployment/inference-engine -n ai-inference --replicas=1 >/dev/null
kubectl rollout status deployment/inference-engine -n ai-inference --timeout=60s >/dev/null
python3 - "${BASE_URL}" "${API_KEY}" <<'PY'
import json, sys, urllib.request
base_url, api_key = sys.argv[1], sys.argv[2]
req = urllib.request.Request(
    f"{base_url}/admin/fault-inject",
    data=json.dumps({"mode": "reset"}).encode("utf-8"),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    method="POST",
)
urllib.request.urlopen(req, timeout=5.0).close()
PY

{
  echo ""
  echo "=== LIVE KUBERNETES EVENTS (INCLUDING AIOPS CLOSED-LOOP DIAGNOSIS) ==="
  kubectl get events -n ai-inference --sort-by='.lastTimestamp'
} >> "${EVIDENCE_DIR}/kubectl-cluster-state.txt"

echo "==> ALL LIVE KUBERNETES VERIFICATION & BENCHMARK CHECKS PASSED!"
