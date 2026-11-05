#!/usr/bin/env python3
"""
Strict Semantic Repository Validator for AiOps on K8S: Foundations and Practices.

Performs 10 comprehensive architectural, security, syntax, and policy validations:
 1. Relative Markdown Link Integrity (README.md and docs/*.md)
 2. Mermaid Diagram Syntax & Render Compatibility (README.md and docs/*.md)
 3. YAML Syntax, Indentation & Kubernetes Document Structure (manifests/ & deploy/)
 4. Container Image Pinning (@sha256 digest required; ban vulnerable vLLM releases)
 5. StatefulSet Governing Headless Service Check (clusterIP: None)
 6. Volcano Gang Scheduling Consistency (minTaskMember <-> task-spec)
 7. Node Problem Detector Configuration (system-log-monitor & ServiceAccount)
 8. GPU Operator Precompiled Driver Versioning (branch identifier e.g. "570")
 9. PromQL Alerting Hygiene (vector matching, XID classification, no raw exported_pod)
10. Kustomize Reference Integrity (deploy/base & deploy/overlays)

Returns 0 on complete pass, 1 on any failure.
"""

import os
import sys
import re
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "ERROR: PyYAML is required for strict Kubernetes AST validation.\n"
        "Install with: pip install -r requirements-dev.txt\n"
    )
    sys.exit(1)

# Supported Mermaid diagram headers
SUPPORTED_MERMAID_HEADERS = (
    "flowchart",
    "graph",
    "sequenceDiagram",
    "stateDiagram",
    "classDiagram",
    "erDiagram",
    "xychart-beta",
)

# Vulnerable or banned vLLM versions (e.g. unpatched CVE-2024-52285 / unsafe deserialization)
BANNED_VLLM_VERSIONS = [
    r"v0\.[5-9]\.",
    r"v0\.10\.",
    r"v0\.11\.",
    r"v0\.6\.4(?!\+)",
]

# Hardware fatal vs User-space non-fatal XIDs
CRITICAL_HARDWARE_XIDS = {"48", "63", "64", "74", "79", "94", "95"}
NON_FATAL_USER_SPACE_XIDS = {"13", "31", "43"}

# Terminal colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_repo_root() -> Path:
    """Resolve the repository root directory."""
    return Path(__file__).resolve().parent.parent


def parse_yaml_docs(content: str) -> List[Dict[str, Any]]:
    """Parse YAML content using PyYAML if present, or return empty list on failure."""
    if yaml is not None:
        try:
            return [doc for doc in yaml.safe_load_all(content) if isinstance(doc, dict)]
        except Exception:
            return []
    return []


# -----------------------------------------------------------------------------
# Check 1: Relative Markdown Link Integrity
# -----------------------------------------------------------------------------
def validate_markdown_links(repo_root: Path) -> Tuple[int, int, List[str]]:
    md_files = []
    readme = repo_root / "README.md"
    if readme.exists():
        md_files.append(readme)
    md_files.extend(sorted((repo_root / "docs").glob("*.md")))

    total_links = 0
    errors = []
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

    for md_path in md_files:
        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"{md_path.relative_to(repo_root)}: Failed to read file: {e}")
            continue

        base_dir = md_path.parent
        for line_no, line in enumerate(content.splitlines(), 1):
            for match in link_pattern.finditer(line):
                raw_url = match.group(2).strip()
                if " " in raw_url:
                    raw_url = raw_url.split()[0].strip()

                if raw_url.startswith(("http://", "https://", "mailto:", "#", "tel:", "ftp://")):
                    continue

                total_links += 1
                target_path_str = raw_url.split("#")[0].strip()
                if not target_path_str:
                    continue

                resolved = (base_dir / target_path_str).resolve()
                if not resolved.exists():
                    errors.append(
                        f"{md_path.relative_to(repo_root)}:{line_no}: Broken relative link '{raw_url}' "
                        f"-> target does not exist"
                    )

    return len(md_files), total_links, errors


# -----------------------------------------------------------------------------
# Check 2: Mermaid Diagram Blocks
# -----------------------------------------------------------------------------
def validate_mermaid_blocks(repo_root: Path) -> Tuple[int, int, List[str]]:
    md_files = []
    readme = repo_root / "README.md"
    if readme.exists():
        md_files.append(readme)
    md_files.extend(sorted((repo_root / "docs").glob("*.md")))

    total_blocks = 0
    errors = []
    mermaid_pattern = re.compile(r'```mermaid[ \t]*\r?\n(.*?)\r?\n```', re.DOTALL)

    for md_path in md_files:
        rel_path = md_path.relative_to(repo_root)
        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"{rel_path}: Failed to read file: {e}")
            continue

        matches = list(mermaid_pattern.finditer(content))
        for idx, match in enumerate(matches, 1):
            total_blocks += 1
            block_content = match.group(1)

            if re.search(r'<br\s*/?>', block_content, re.IGNORECASE):
                errors.append(
                    f"{rel_path} (mermaid #{idx}): Contains forbidden '<br>' HTML tag (use newline or string split)"
                )

            lines = [
                l.strip() for l in block_content.splitlines()
                if l.strip() and not l.strip().startswith("%%")
            ]
            if not lines:
                errors.append(f"{rel_path} (mermaid #{idx}): Empty diagram block")
                continue

            header_line = lines[0]
            first_token = header_line.split()[0]
            if not any(first_token.startswith(h) or header_line.startswith(h) for h in SUPPORTED_MERMAID_HEADERS):
                errors.append(
                    f"{rel_path} (mermaid #{idx}): Unsupported diagram header '{header_line}'. "
                    f"Must start with one of: {', '.join(SUPPORTED_MERMAID_HEADERS)}"
                )

    return len(md_files), total_blocks, errors


# -----------------------------------------------------------------------------
# Check 3: YAML Syntax, Indentation & Document Structure
# -----------------------------------------------------------------------------
def validate_yaml_syntax_and_indentation(repo_root: Path) -> Tuple[int, int, List[str]]:
    search_dirs = [repo_root / "manifests", repo_root / "deploy"]
    yaml_files = []
    for d in search_dirs:
        if d.exists():
            yaml_files.extend(sorted(d.glob("**/*.yaml")))
            yaml_files.extend(sorted(d.glob("**/*.yml")))

    total_docs = 0
    errors = []

    for yaml_path in yaml_files:
        rel_path = yaml_path.relative_to(repo_root)
        if yaml_path.name.endswith(".test.yaml") or yaml_path.name.endswith(".test.yml"):
            continue

        try:
            content = yaml_path.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"{rel_path}: Failed to read file: {e}")
            continue

        for line_no, line in enumerate(content.splitlines(), 1):
            if "\t" in line:
                errors.append(f"{rel_path}:{line_no}: Tab character detected (YAML forbids tab indentation)")

        docs = re.split(r'^(?:---|---[ \t].*)$', content, flags=re.MULTILINE)
        doc_idx = 0
        for doc in docs:
            doc_strip = doc.strip()
            if not doc_strip:
                continue

            non_comment_lines = [
                l for l in doc_strip.splitlines()
                if l.strip() and not l.strip().startswith("#")
            ]
            if not non_comment_lines:
                continue

            doc_idx += 1
            total_docs += 1

            has_api_version = bool(re.search(r'^[ \t]*apiVersion\s*:', doc, re.MULTILINE))
            has_kind = bool(re.search(r'^[ \t]*kind\s*:', doc, re.MULTILINE))
            is_kustomization = bool(re.search(r'^[ \t]*kind\s*:\s*Kustomization', doc, re.MULTILINE))
            has_metadata = bool(re.search(r'^[ \t]*metadata\s*:', doc, re.MULTILINE))

            missing = []
            if not has_api_version:
                missing.append("apiVersion")
            if not has_kind:
                missing.append("kind")
            if not has_metadata and not is_kustomization:
                missing.append("metadata")

            if missing:
                errors.append(
                    f"{rel_path} (document #{doc_idx}): Missing required Kubernetes fields: {', '.join(missing)}"
                )

        if yaml is not None:
            try:
                list(yaml.safe_load_all(content))
            except Exception as e:
                errors.append(f"{rel_path}: YAML parsing error: {e}")

    return len(yaml_files), total_docs, errors


# -----------------------------------------------------------------------------
# Check 4: Container Image Pinning & CVE Banning
# -----------------------------------------------------------------------------
def validate_container_image_pinning(repo_root: Path) -> Tuple[int, List[str]]:
    manifest_dirs = [repo_root / "manifests", repo_root / "deploy"]
    yaml_files = []
    for d in manifest_dirs:
        if d.exists():
            yaml_files.extend(sorted(d.glob("**/*.yaml")))

    total_images = 0
    errors = []
    image_regex = re.compile(r'^[ \t]*image:\s*["\']?([^"\'\s]+)["\']?', re.MULTILINE)

    for yaml_path in yaml_files:
        rel_path = yaml_path.relative_to(repo_root)
        if yaml_path.name.endswith(".test.yaml"):
            continue

        try:
            content = yaml_path.read_text(encoding="utf-8")
        except Exception:
            continue

        # In GPU Operator ClusterPolicy, 'image: driver' is a configuration directive, not a container spec image
        is_cluster_policy = "gpu-operator-clusterpolicy.yaml" in yaml_path.name

        for line_no, line in enumerate(content.splitlines(), 1):
            m = image_regex.match(line)
            if not m:
                continue

            image_ref = m.group(1).strip()
            if is_cluster_policy and image_ref == "driver":
                continue

            total_images += 1

            # Check 1: Must have SHA256 digest
            if "@sha256:" not in image_ref:
                errors.append(
                    f"{rel_path}:{line_no}: Container image '{image_ref}' is NOT pinned to a cryptographic @sha256 digest"
                )

            # Check 2: Banned vulnerable vLLM versions
            for banned_pat in BANNED_VLLM_VERSIONS:
                if "vllm" in image_ref and re.search(banned_pat, image_ref):
                    errors.append(
                        f"{rel_path}:{line_no}: Container image '{image_ref}' uses vulnerable/banned vLLM version matching '{banned_pat}'"
                    )

    return total_images, errors


# -----------------------------------------------------------------------------
# Check 5: StatefulSet Governing Headless Service Check
# -----------------------------------------------------------------------------
def validate_statefulset_governing_services(repo_root: Path) -> Tuple[int, List[str]]:
    manifest_dir = repo_root / "manifests"
    yaml_files = sorted(manifest_dir.glob("**/*.yaml"))
    checked_count = 0
    errors = []

    for yaml_path in yaml_files:
        rel_path = yaml_path.relative_to(repo_root)
        try:
            content = yaml_path.read_text(encoding="utf-8")
        except Exception:
            continue

        docs = parse_yaml_docs(content)
        services = {}
        statefulsets = []

        for d in docs:
            if not isinstance(d, dict):
                continue
            kind = d.get("kind")
            metadata = d.get("metadata", {})
            name = metadata.get("name")
            if kind == "Service" and name:
                spec = d.get("spec", {})
                services[name] = spec
            elif kind == "StatefulSet":
                statefulsets.append(d)

        for sts in statefulsets:
            checked_count += 1
            sts_name = sts.get("metadata", {}).get("name", "unknown")
            service_name = sts.get("spec", {}).get("serviceName")

            if not service_name:
                errors.append(
                    f"{rel_path}: StatefulSet '{sts_name}' is missing 'spec.serviceName'"
                )
                continue

            if service_name not in services:
                errors.append(
                    f"{rel_path}: StatefulSet '{sts_name}' references governing service '{service_name}' which is not defined in the same manifest"
                )
                continue

            svc_spec = services[service_name]
            cluster_ip = svc_spec.get("clusterIP")
            if cluster_ip != "None":
                errors.append(
                    f"{rel_path}: Governing service '{service_name}' for StatefulSet '{sts_name}' must be headless (spec.clusterIP: None), got: {cluster_ip}"
                )

    return checked_count, errors


# -----------------------------------------------------------------------------
# Check 6: Volcano Gang Scheduling Consistency
# -----------------------------------------------------------------------------
def validate_volcano_gang_consistency(repo_root: Path) -> Tuple[int, List[str]]:
    volcano_manifest = repo_root / "manifests" / "03-scheduling" / "volcano-gang-distributed-vllm.yaml"
    if not volcano_manifest.exists():
        return 0, ["Volcano manifest manifests/03-scheduling/volcano-gang-distributed-vllm.yaml not found"]

    errors = []
    content = volcano_manifest.read_text(encoding="utf-8")
    docs = parse_yaml_docs(content)

    pod_groups = [d for d in docs if d.get("kind") == "PodGroup"]
    if not pod_groups:
        return 0, ["No PodGroup found in Volcano manifest"]

    pg = pod_groups[0]
    min_member = pg.get("spec", {}).get("minMember", 0)
    min_task_member = pg.get("spec", {}).get("minTaskMember", {})

    if not min_task_member:
        errors.append("PodGroup is missing 'spec.minTaskMember' specification")

    total_task_members = sum(int(v) for v in min_task_member.values())
    if min_member > 0 and total_task_members != min_member:
        errors.append(
            f"PodGroup minTaskMember sum ({total_task_members}) does not match spec.minMember ({min_member})"
        )

    task_spec_annotations = set()
    for d in docs:
        spec = d.get("spec", {})
        template = spec.get("template", {})
        metadata = template.get("metadata", {})
        annotations = metadata.get("annotations", {})
        task_spec = annotations.get("volcano.sh/task-spec")
        if task_spec:
            task_spec_annotations.add(task_spec)

    for task_name in min_task_member.keys():
        if task_name not in task_spec_annotations:
            errors.append(
                f"PodGroup minTaskMember task '{task_name}' has no matching template with volcano.sh/task-spec annotation"
            )

    return 1, errors


# -----------------------------------------------------------------------------
# Check 7: Node Problem Detector Configuration
# -----------------------------------------------------------------------------
def validate_npd_configuration(repo_root: Path) -> Tuple[int, List[str]]:
    npd_manifest = repo_root / "manifests" / "05-observability" / "gpu-node-problem-detector.yaml"
    if not npd_manifest.exists():
        return 0, ["NPD manifest manifests/05-observability/gpu-node-problem-detector.yaml not found"]

    errors = []
    content = npd_manifest.read_text(encoding="utf-8")
    docs = parse_yaml_docs(content)

    sa_names = {d.get("metadata", {}).get("name") for d in docs if d.get("kind") == "ServiceAccount"}
    daemonsets = [d for d in docs if d.get("kind") == "DaemonSet"]

    if not daemonsets:
        errors.append("No DaemonSet found in NPD manifest")
        return 0, errors

    ds = daemonsets[0]
    pod_spec = ds.get("spec", {}).get("template", {}).get("spec", {})
    ds_sa = pod_spec.get("serviceAccountName")

    if not ds_sa or ds_sa not in sa_names:
        errors.append(
            f"NPD DaemonSet serviceAccountName '{ds_sa}' does not reference a ServiceAccount defined in the same manifest"
        )

    # Check kmsg monitor flag in command/args
    containers = pod_spec.get("containers", [])
    found_syslog = False
    for c in containers:
        args = c.get("command", []) + c.get("args", [])
        for arg in args:
            if "--config.custom-plugin-monitor" in str(arg):
                errors.append(
                    "NPD kmsg plugin configured with '--config.custom-plugin-monitor' instead of '--config.system-log-monitor'"
                )
            if "--config.system-log-monitor" in str(arg):
                found_syslog = True

    if not found_syslog:
        errors.append("NPD DaemonSet missing '--config.system-log-monitor' argument")

    return 1, errors


# -----------------------------------------------------------------------------
# Check 8: GPU Operator Driver Versioning
# -----------------------------------------------------------------------------
def validate_gpu_operator_driver_version(repo_root: Path) -> Tuple[int, List[str]]:
    cp_manifest = repo_root / "manifests" / "01-hardware-orchestration" / "gpu-operator-clusterpolicy.yaml"
    if not cp_manifest.exists():
        return 0, ["ClusterPolicy manifest not found"]

    errors = []
    content = cp_manifest.read_text(encoding="utf-8")
    docs = parse_yaml_docs(content)
    cp_docs = [d for d in docs if d.get("kind") == "ClusterPolicy"]

    if not cp_docs:
        errors.append("No ClusterPolicy document found in gpu-operator-clusterpolicy.yaml")
        return 0, errors

    driver_cfg = cp_docs[0].get("spec", {}).get("driver", {})
    version_str = str(driver_cfg.get("version", "")).strip()

    # Precompiled drivers require branch series (e.g. "570", "550", "535")
    if not re.match(r'^\d{3}$', version_str):
        errors.append(
            f"GPU Operator precompiled driver version must be a 3-digit branch identifier (e.g. '570'), got: '{version_str}'"
        )

    return 1, errors


# -----------------------------------------------------------------------------
# Check 9: PromQL Hygiene & XID Classification
# -----------------------------------------------------------------------------
def validate_promql_hygiene(repo_root: Path) -> Tuple[int, List[str]]:
    rules_file = repo_root / "manifests" / "05-observability" / "prometheus-gpu-oom-alerts.yaml"
    if not rules_file.exists():
        return 0, ["Prometheus alerts file not found"]

    errors = []
    content = rules_file.read_text(encoding="utf-8")
    docs = parse_yaml_docs(content)

    rules = []
    for d in docs:
        groups = d.get("spec", {}).get("groups", []) or d.get("groups", [])
        for g in groups:
            rules.extend(g.get("rules", []))

    if not rules:
        errors.append("No alert rules parsed from prometheus-gpu-oom-alerts.yaml")
        return 0, errors

    rule_dict = {r.get("alert"): r for r in rules if r.get("alert")}

    # Check 1: Critical XID fault rule exists and covers hardware XIDs
    crit_rule = rule_dict.get("NvidiaGPUCriticalSiliconXidFault")
    if not crit_rule:
        errors.append("Missing critical alert rule 'NvidiaGPUCriticalSiliconXidFault'")
    else:
        expr = crit_rule.get("expr", "")
        for xid in CRITICAL_HARDWARE_XIDS:
            if f"== {xid}" not in expr:
                errors.append(f"Critical Silicon XID alert missing hardware XID {xid}")

    # Check 2: Warning XID alert rule exists and covers application XIDs
    warn_rule = rule_dict.get("NvidiaGPUApplicationOrDriverXidWarning")
    if not warn_rule:
        errors.append("Missing warning alert rule 'NvidiaGPUApplicationOrDriverXidWarning'")
    else:
        expr = warn_rule.get("expr", "")
        for xid in NON_FATAL_USER_SPACE_XIDS:
            if f"== {xid}" not in expr:
                errors.append(f"Application/Driver XID alert missing non-fatal XID {xid}")

    # Check 3: PromQL binary matching and un-joined exported_pod check
    for r in rules:
        alert_name = r.get("alert", "unknown")
        expr = r.get("expr", "")
        if "exported_pod" in expr and "label_replace" not in expr and "label_join" not in expr:
            errors.append(f"Alert '{alert_name}' references raw 'exported_pod' without label_replace/join")

        # Check binary operations between different metrics contain vector matching
        if "kube_pod_container_status_last_terminated_reason" in expr and "kube_pod_container_resource_requests" in expr:
            if "and on(namespace, pod)" not in expr and "and on (namespace, pod)" not in expr:
                errors.append(f"Alert '{alert_name}' missing vector matching 'and on(namespace, pod)'")

    return len(rules), errors


# -----------------------------------------------------------------------------
# Check 10: Kustomize Reference Integrity
# -----------------------------------------------------------------------------
def validate_kustomize_references(repo_root: Path) -> Tuple[int, List[str]]:
    kustomization_files = sorted(repo_root.glob("deploy/**/kustomization.yaml"))
    checked_count = 0
    errors = []

    for kf in kustomization_files:
        checked_count += 1
        rel_path = kf.relative_to(repo_root)
        try:
            content = kf.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"{rel_path}: Failed to read file: {e}")
            continue

        base_dir = kf.parent
        docs = parse_yaml_docs(content)
        if not docs:
            # Fallback regex if PyYAML unparseable
            resources = re.findall(r'^[ \t]*-\s*([^\s#]+)', content, re.MULTILINE)
        else:
            resources = docs[0].get("resources", [])

        for res in resources:
            target = (base_dir / res).resolve()
            if not target.exists():
                errors.append(
                    f"{rel_path}: Referenced resource '{res}' -> '{target}' does not exist on disk"
                )

    return checked_count, errors


# -----------------------------------------------------------------------------
# Main Orchestrator
# -----------------------------------------------------------------------------
def main() -> int:
    repo_root = get_repo_root()
    print(f"\n{BOLD}{CYAN}=== Strict Semantic Repository Validator for AiOps on K8S ==={RESET}")
    print(f"Target repository: {repo_root}\n")

    results: List[Tuple[str, str, List[str]]] = []

    # 1. Relative Markdown Links
    files_scanned, links_checked, errors = validate_markdown_links(repo_root)
    desc = f"{files_scanned} files, {links_checked} links validated"
    results.append(("Markdown Relative Link Integrity", desc, errors))

    # 2. Mermaid Diagram Blocks
    md_files, blocks_checked, errors = validate_mermaid_blocks(repo_root)
    desc = f"{md_files} files, {blocks_checked} diagram blocks validated"
    results.append(("Mermaid Diagram Syntax & Render Compatibility", desc, errors))

    # 3. YAML Syntax & Indentation
    yaml_files, docs_checked, errors = validate_yaml_syntax_and_indentation(repo_root)
    desc = f"{yaml_files} files, {docs_checked} documents validated"
    results.append(("Kubernetes YAML Syntax & Indentation", desc, errors))

    # 4. Container Image Pinning
    images_checked, errors = validate_container_image_pinning(repo_root)
    desc = f"{images_checked} container images verified (@sha256 digest & CVE checks)"
    results.append(("Container Image Cryptographic Digest Pinning", desc, errors))

    # 5. StatefulSet Governing Headless Service
    sts_checked, errors = validate_statefulset_governing_services(repo_root)
    desc = f"{sts_checked} StatefulSets verified with headless Service (clusterIP: None)"
    results.append(("StatefulSet Governing Headless Service Integrity", desc, errors))

    # 6. Volcano Gang Scheduling Consistency
    pg_checked, errors = validate_volcano_gang_consistency(repo_root)
    desc = f"{pg_checked} Volcano gang specs verified (minTaskMember <-> task-spec)"
    results.append(("Volcano Gang Scheduling & PodGroup Invariants", desc, errors))

    # 7. Node Problem Detector Configuration
    npd_checked, errors = validate_npd_configuration(repo_root)
    desc = f"{npd_checked} NPD specs verified (system-log-monitor & ServiceAccount)"
    results.append(("Node Problem Detector DaemonSet & Telemetry Configuration", desc, errors))

    # 8. GPU Operator Precompiled Driver Versioning
    drv_checked, errors = validate_gpu_operator_driver_version(repo_root)
    desc = f"{drv_checked} ClusterPolicy driver verified with branch identifier"
    results.append(("GPU Operator Precompiled Driver Versioning", desc, errors))

    # 9. PromQL Alerting Hygiene & XID Classification
    prom_rules, errors = validate_promql_hygiene(repo_root)
    desc = f"{prom_rules} alert rules verified (vector matching & XID isolation)"
    results.append(("PromQL Alerting Hygiene & Hardware XID Classification", desc, errors))

    # 10. Kustomize Reference Integrity
    kust_checked, errors = validate_kustomize_references(repo_root)
    desc = f"{kust_checked} kustomization overlays verified across deploy/base & overlays"
    results.append(("Kustomize Overlays & Base Reference Integrity", desc, errors))

    # Print summary
    all_errors = []
    for title, desc, errs in results:
        status = f"{GREEN}PASS{RESET}" if not errs else f"{RED}FAIL ({len(errs)} errors){RESET}"
        print(f"  [{status}] {title}: {desc}")
        all_errors.extend(errs)

    print("\n" + "=" * 65)
    if all_errors:
        print(f"\n{BOLD}{RED}VALIDATION FAILED with {len(all_errors)} error(s):{RESET}\n")
        for err in all_errors:
            print(f"  {RED}•{RESET} {err}")
        print()
        return 1
    else:
        print(f"\n{BOLD}{GREEN}ALL 10 REPOSITORY ARCHITECTURAL & POLICY CHECKS PASSED SUCCESSFULLY!{RESET}\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
