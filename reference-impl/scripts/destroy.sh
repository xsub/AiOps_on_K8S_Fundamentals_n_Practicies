#!/usr/bin/env bash
# Tear down the local kind Kubernetes cluster for the Reference Implementation.
set -euo pipefail

CLUSTER_NAME="${KIND_CLUSTER_NAME:-aiops-ref}"

echo "==> Deleting kind Kubernetes cluster '${CLUSTER_NAME}'..."
kind delete cluster --name "${CLUSTER_NAME}"
echo "==> Cluster '${CLUSTER_NAME}' destroyed cleanly."
