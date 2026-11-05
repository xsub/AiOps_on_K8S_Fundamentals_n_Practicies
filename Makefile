.PHONY: help validate test evaluate-aiops demo-aiops kustomize-build promtool-test deploy verify destroy ci clean

PYTHON ?= python3

help:  ## Display this help screen
	@echo "Available Makefile targets for AiOps on K8S:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

validate:  ## Run strict semantic repository validator (YAML, PromQL, Security, Links, Mermaid)
	$(PYTHON) scripts/validate_repo.py

test:  ## Run unit and policy test suites (admission CEL, manifests, PromQL, AIOps engine)
	$(PYTHON) -m unittest discover -s tests -v

evaluate-aiops:  ## Run AIOps benchmark evaluation suite
	$(PYTHON) -m aiops evaluate

demo-aiops:  ## Run interactive AIOps closed-loop incident triage and remediation demo
	$(PYTHON) -m aiops demo

kustomize-build:  ## Build Kustomize overlays and validate schemas if kubeconform is available
	@if command -v kustomize >/dev/null 2>&1; then \
		echo "Building Kustomize overlays using kustomize..."; \
		for dir in deploy/base deploy/overlays/*; do \
			if [ -d "$$dir" ]; then \
				echo "--- Building $$dir ---"; \
				kustomize build --load-restrictor LoadRestrictionsNone "$$dir" >/dev/null && echo "  [OK] $$dir built successfully"; \
			fi; \
		done; \
		if command -v kubeconform >/dev/null 2>&1; then \
			echo "Validating standalone and reference-impl manifests with kubeconform -strict..."; \
			find manifests reference-impl/k8s -name '*.yaml' ! -name '*.test.yaml' ! -name 'kind-config.yaml' | sort | xargs kubeconform -strict -ignore-missing-schemas -skip PodGroup -schema-location default -schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json' -summary; \
		fi; \
	elif command -v kubectl >/dev/null 2>&1; then \
		echo "Building Kustomize overlays using kubectl kustomize..."; \
		for dir in deploy/base deploy/overlays/*; do \
			if [ -d "$$dir" ]; then \
				echo "--- Building $$dir ---"; \
				kubectl kustomize --load-restrictor LoadRestrictionsNone "$$dir" >/dev/null && echo "  [OK] $$dir built successfully"; \
			fi; \
		done; \
	else \
		echo "Kustomize/kubectl not detected in PATH; executing Python semantic Kustomize validator..."; \
		$(PYTHON) scripts/validate_repo.py; \
	fi

promtool-test:  ## Check PromQL syntax and run alert unit tests if promtool is available
	@if command -v promtool >/dev/null 2>&1; then \
		echo "Running promtool check rules..."; \
		promtool check rules --ignore-unknown-fields manifests/05-observability/prometheus-gpu-oom-alerts.yaml; \
		if [ -f manifests/05-observability/prometheus-gpu-oom-alerts.test.yaml ]; then \
			echo "Running promtool test rules..."; \
			promtool test rules --ignore-unknown-fields manifests/05-observability/prometheus-gpu-oom-alerts.test.yaml; \
		fi; \
	else \
		echo "promtool not installed in PATH; skipping promtool checks (run in CI)."; \
	fi

deploy:  ## Boot kind K8s cluster and deploy the runnable reference implementation
	bash reference-impl/scripts/deploy.sh

verify:  ## Verify live CEL admission policies, SSE smoke test, benchmark, and AIOps PDB loop
	bash reference-impl/scripts/verify.sh

destroy:  ## Tear down the local kind Kubernetes cluster cleanly
	bash reference-impl/scripts/destroy.sh

ci: validate test evaluate-aiops kustomize-build promtool-test  ## Run full local CI verification pipeline

clean:  ## Clean temporary Python caches and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
