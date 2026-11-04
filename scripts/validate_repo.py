#!/usr/bin/env python3
"""
Repository Validation Script for AiOps on K8S: Foundations and Practices.

Scans the repository to ensure production quality:
1. Relative Markdown Link Integrity in README.md and docs/*.md
2. Kubernetes YAML Manifest Structure and Syntax in manifests/**/*.yaml
3. Mermaid Diagram Syntax and Render Compatibility in docs/*.md and README.md

Returns 0 on complete pass, 1 on any error.
"""

import os
import sys
import re
import glob
from pathlib import Path

# Supported Mermaid diagram headers
SUPPORTED_MERMAID_HEADERS = (
    "flowchart",
    "graph",
    "sequenceDiagram",
    "stateDiagram",
    "classDiagram",
    "erDiagram",
    "xychart-beta"
)

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_repo_root() -> Path:
    """Resolve the repository root directory."""
    script_path = Path(__file__).resolve()
    return script_path.parent.parent


def validate_markdown_links(repo_root: Path) -> tuple[int, int, list[str]]:
    """
    Validates all relative Markdown links in README.md and docs/*.md.
    Strips anchors (#L..., #section) before verifying the target exists.
    Ignores external URLs (http://, https://, mailto:, etc.).
    """
    md_files = []
    readme = repo_root / "README.md"
    if readme.exists():
        md_files.append(readme)
    md_files.extend(sorted((repo_root / "docs").glob("*.md")))

    total_links = 0
    errors = []

    # Regex to extract [text](target) - also accounts for optional title e.g. [text](target "title")
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

                # Remove optional title if present (e.g., [text](path "title"))
                if " " in raw_url:
                    raw_url = raw_url.split()[0].strip()

                # Exclude external links and pure local anchors
                if raw_url.startswith(("http://", "https://", "mailto:", "#", "tel:", "ftp://")):
                    continue

                total_links += 1

                # Strip anchor fragment
                target_path_str = raw_url.split("#")[0].strip()
                if not target_path_str:
                    # Link was only an anchor reference e.g. (#...)
                    continue

                resolved = (base_dir / target_path_str).resolve()
                if not resolved.exists():
                    errors.append(
                        f"{md_path.relative_to(repo_root)}:{line_no}: Broken relative link '{raw_url}' "
                        f"-> '{resolved.relative_to(repo_root) if resolved.is_relative_to(repo_root) else resolved}' does not exist"
                    )

    return len(md_files), total_links, errors


def validate_yaml_manifests(repo_root: Path) -> tuple[int, int, list[str]]:
    """
    Validates every .yaml file in manifests/:
    - Verifies no tab indentation.
    - Validates non-empty documents split by '---'.
    - Verifies every non-comment document defines apiVersion:, kind:, and metadata:.
    """
    manifest_dir = repo_root / "manifests"
    yaml_files = sorted(manifest_dir.glob("**/*.yaml")) + sorted(manifest_dir.glob("**/*.yml"))

    total_docs = 0
    errors = []

    for yaml_path in yaml_files:
        rel_path = yaml_path.relative_to(repo_root)
        try:
            content = yaml_path.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"{rel_path}: Failed to read file: {e}")
            continue

        # Check for tab indentation
        for line_no, line in enumerate(content.splitlines(), 1):
            if "\t" in line:
                errors.append(f"{rel_path}:{line_no}: Tab character detected (YAML forbids tab indentation)")

        # Split documents by '---'
        docs = re.split(r'^(?:---|---[ \t].*)$', content, flags=re.MULTILINE)
        doc_idx = 0

        for doc in docs:
            doc_strip = doc.strip()
            if not doc_strip:
                continue

            # Ignore documents that only contain comments
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
            has_metadata = bool(re.search(r'^[ \t]*metadata\s*:', doc, re.MULTILINE))

            missing = []
            if not has_api_version:
                missing.append("apiVersion")
            if not has_kind:
                missing.append("kind")
            if not has_metadata:
                missing.append("metadata")

            if missing:
                errors.append(
                    f"{rel_path} (document #{doc_idx}): Missing required Kubernetes fields: {', '.join(missing)}"
                )

    return len(yaml_files), total_docs, errors


def validate_mermaid_blocks(repo_root: Path) -> tuple[int, int, list[str]]:
    """
    Validates every mermaid code block in docs/*.md and README.md:
    - Uses a supported diagram header (flowchart, graph, sequenceDiagram, etc.)
    - Contains no <br> HTML tags.
    """
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

            # Check for <br> HTML tags
            if re.search(r'<br\s*/?>', block_content, re.IGNORECASE):
                errors.append(
                    f"{rel_path} (mermaid block #{idx}): Contains '<br>' HTML tag (disallowed in strict markdown renderers; use newline or string split)"
                )

            # Check diagram header
            lines = [
                line.strip() for line in block_content.splitlines()
                if line.strip() and not line.strip().startswith("%%")
            ]
            if not lines:
                errors.append(f"{rel_path} (mermaid block #{idx}): Empty diagram block")
                continue

            header_line = lines[0]
            first_token = header_line.split()[0]
            if not any(first_token.startswith(h) or header_line.startswith(h) for h in SUPPORTED_MERMAID_HEADERS):
                errors.append(
                    f"{rel_path} (mermaid block #{idx}): Unsupported diagram header '{header_line}'. "
                    f"Must start with one of: {', '.join(SUPPORTED_MERMAID_HEADERS)}"
                )

    return len(md_files), total_blocks, errors


def main() -> int:
    repo_root = get_repo_root()
    print(f"\n{BOLD}{CYAN}=== AiOps on K8S Repository Validator ==={RESET}")
    print(f"Target repository: {repo_root}\n")

    # 1. Validate Markdown Links
    files_scanned, links_checked, link_errors = validate_markdown_links(repo_root)
    link_status = f"{GREEN}PASS{RESET}" if not link_errors else f"{RED}FAIL ({len(link_errors)} errors){RESET}"
    print(f"  [{link_status}] Markdown Relative Links: {files_scanned} files, {links_checked} links validated")

    # 2. Validate YAML Manifests
    yaml_files_scanned, yaml_docs_scanned, yaml_errors = validate_yaml_manifests(repo_root)
    yaml_status = f"{GREEN}PASS{RESET}" if not yaml_errors else f"{RED}FAIL ({len(yaml_errors)} errors){RESET}"
    print(f"  [{yaml_status}] Kubernetes YAML Manifests: {yaml_files_scanned} files, {yaml_docs_scanned} documents validated")

    # 3. Validate Mermaid Code Blocks
    md_mermaid_files, mermaid_blocks, mermaid_errors = validate_mermaid_blocks(repo_root)
    mermaid_status = f"{GREEN}PASS{RESET}" if not mermaid_errors else f"{RED}FAIL ({len(mermaid_errors)} errors){RESET}"
    print(f"  [{mermaid_status}] Mermaid Diagram Blocks: {md_mermaid_files} files, {mermaid_blocks} blocks validated")

    all_errors = link_errors + yaml_errors + mermaid_errors

    print("\n" + "=" * 50)
    if all_errors:
        print(f"\n{BOLD}{RED}VALIDATION FAILED with {len(all_errors)} error(s):{RESET}\n")
        for err in all_errors:
            print(f"  {RED}•{RESET} {err}")
        print()
        return 1
    else:
        print(f"\n{BOLD}{GREEN}ALL REPOSITORY CHECKS PASSED SUCCESSFULLY!{RESET}\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
