#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AISBench DevFlow local gate preflight.

Read-only: this module inspects the local git repository and renders verdicts.
It never writes to the remote, never installs packages, never runs workflows.

Verdict vocabulary (exactly four values):

    PASS            evidence shows the gate holds
    FAIL            evidence shows the gate does not hold
    UNKNOWN         the gate cannot be decided here (remote/runtime evidence needed)
    NOT_APPLICABLE  the gate does not apply, with an explicit applicability reason

``PASS`` must never be used for "not checked".  Exit code 0 only means "no known
FAIL"; promotion decisions additionally require every gate in
``promotion_blockers`` to be empty.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 1

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"
NOT_APPLICABLE = "NOT_APPLICABLE"

DOCS_ZH_ROOT = "docs/source_zh_cn"
DOCS_EN_ROOT = "docs/source_en"
README_ZH = "README.md"
README_EN = "README_en.md"

TITLE_RE = re.compile(r"^\[(feat|fix|docs|style|refactor|test|chore|perf|build|ci|dep|deps)\](\[[^\]]+\])?.+$")
ISSUE_REF_RE = re.compile(r"(?:fixes|closes|resolves|relates\s+to|ref)\s*#(\d+)|#(\d+)|issues/(\d+)", re.IGNORECASE)
CHECKBOX_TICKED_RE = re.compile(r"^\s*[-*]\s*\[[xX]\]\s*(.+?)\s*$")
CHECKBOX_ANY_RE = re.compile(r"^\s*[-*]\s*\[[ xX]\]\s*(.+?)\s*$")
TRAILER = "<!-- aisbench-devflow -->"

PR_TYPES = (
    "feature",
    "bugfix",
    "docs",
    "ci-cd",
    "refactor",
    "perf",
    "dependency",
    "test-cases",
    "other",
)


class GateError(RuntimeError):
    """Raised for environment problems (missing repo, bad revision, ...)."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def configure_stdout() -> None:
    """Force UTF-8 output so Windows GBK consoles do not mangle Chinese reports."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


class Repo:
    """Thin read-only git wrapper bound to a work tree."""

    def __init__(self, root: Path) -> None:
        self.root = root
        top = self._run("rev-parse", "--show-toplevel", allow_failure=True)
        if not top:
            raise GateError(f"not a git work tree: {root}")
        self.top = Path(top.strip())

    def _run(self, *args: str, allow_failure: bool = False) -> str:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            raise GateError(f"cannot run git in {self.root}: {exc}") from exc
        if proc.returncode != 0:
            if allow_failure:
                return ""
            raise GateError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout

    def rev(self, ref: str) -> str:
        out = self._run("rev-parse", "--verify", f"{ref}^{{commit}}", allow_failure=True)
        if not out.strip():
            raise GateError(f"cannot resolve revision: {ref}")
        return out.strip()

    def merge_base(self, base: str, head: str) -> str:
        out = self._run("merge-base", base, head, allow_failure=True)
        if not out.strip():
            raise GateError(f"no merge base between {base} and {head}")
        return out.strip()

    def is_dirty(self) -> bool:
        return bool(self._run("status", "--porcelain").strip())

    def changed_files(self, base: str, head: str) -> list[tuple[str, str]]:
        """Return [(status, path)] using rename-aware name-status output."""
        out = self._run("diff", "--name-status", "-z", "--no-renames", base, head)
        fields = out.split("\0")
        rows: list[tuple[str, str]] = []
        i = 0
        while i + 1 < len(fields):
            status = fields[i].strip()
            path = fields[i + 1].strip()
            if status:
                rows.append((status[:1], path))
            i += 2
        return rows

    def numstat(self, base: str, head: str) -> tuple[int, int, int]:
        """Return (additions, deletions, binary_files)."""
        out = self._run("diff", "--numstat", "--no-renames", base, head)
        additions = deletions = binary = 0
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            added, removed = parts[0], parts[1]
            if added == "-" or removed == "-":
                binary += 1
                continue
            if added.isdigit():
                additions += int(added)
            if removed.isdigit():
                deletions += int(removed)
        return additions, deletions, binary

    def commits(self, base: str, head: str) -> list[tuple[str, str]]:
        """Return [(sha, full message)] for base..head."""
        out = self._run("log", "--format=%H%x1f%B%x1e", f"{base}..{head}")
        rows: list[tuple[str, str]] = []
        for chunk in out.split("\x1e"):
            chunk = chunk.strip("\n")
            if not chunk.strip():
                continue
            sha, _, body = chunk.partition("\x1f")
            rows.append((sha.strip(), body))
        return rows

    def exists_at(self, rev: str, path: str) -> bool:
        out = self._run("rev-parse", "--verify", "--quiet", f"{rev}:{path}", allow_failure=True)
        return bool(out.strip())

    def default_branch(self) -> str | None:
        out = self._run("symbolic-ref", "--short", "refs/remotes/origin/HEAD", allow_failure=True).strip()
        if out:
            return out
        for candidate in ("origin/master", "origin/main", "master", "main"):
            if self._run("rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}", allow_failure=True).strip():
                return candidate
        return None


class Profile:
    """Repository profile read from the checkout; never cached across HEAD changes."""

    def __init__(self, repo: Repo) -> None:
        self.repo = repo
        self.warnings: list[str] = []
        self.workflows = repo.top / ".github" / "workflows"
        self.template_path = self._find_template()
        self.template_options = self._read_template_options()
        self.label_map = self._read_label_map()
        self.size_limit = self._read_size_limit()
        self.ut_prefix = self._read_ut_prefix()
        self.bugfix_ut_threshold = self._read_bugfix_threshold()
        self.coverage_thresholds = self._read_coverage_thresholds()
        self.ut_dirs = self._read_ut_dirs()
        self.docs_roots = self._read_docs_roots()

    # -- discovery -----------------------------------------------------
    def _find_template(self) -> Path | None:
        candidates = [
            self.repo.top / ".github" / "pull_request_template.md",
            self.repo.top / ".github" / "PULL_REQUEST_TEMPLATE.md",
            self.repo.top / "docs" / "pull_request_template.md",
        ]
        for path in candidates:
            if path.is_file():
                return path
        self.warnings.append("pull_request_template_missing: 未找到 PR 模板，标签与正文检查将降级")
        return None

    def _read_template_options(self) -> list[str]:
        """Options from the PR Type block only (not the Checklist section)."""
        if not self.template_path:
            return []
        lines = self.template_path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = None
        for index, line in enumerate(lines):
            if re.search(r"PR\s*Type", line, re.IGNORECASE):
                start = index + 1
                break
        if start is None:
            self.warnings.append("pr_type_block_missing: 未定位到 PR Type 区块，按全文首个复选框区块解析")
            start = 0
        options: list[str] = []
        for line in lines[start:]:
            stripped = line.strip()
            if stripped.startswith("**") or stripped.startswith("##"):
                break
            match = CHECKBOX_ANY_RE.match(line)
            if match:
                options.append(match.group(1))
        return options

    def _workflow(self, name: str) -> Path | None:
        path = self.workflows / name
        if not path.is_file():
            self.warnings.append(f"workflow_missing: .github/workflows/{name} 不存在，相关阈值使用默认值")
            return None
        return path

    def _read_label_map(self) -> dict[str, str]:
        path = self._workflow("auto_label_pr.yml")
        if not path:
            return {}
        text = path.read_text(encoding="utf-8", errors="replace")
        block = re.search(r"const\s+typeToLabel\s*=\s*\{(?P<body>.*?)\n\s*\};", text, re.S)
        if not block:
            self.warnings.append("label_map_unreadable: auto_label_pr.yml 中未找到 typeToLabel 映射")
            return {}
        mapping: dict[str, str] = {}
        for key, value in re.findall(r"'([^']+)'\s*:\s*'([^']+)'", block.group("body")):
            mapping[key] = value
        if not mapping:
            self.warnings.append("label_map_empty: typeToLabel 解析结果为空")
        return mapping

    def _read_size_limit(self) -> int:
        path = self._workflow("pr_quality_check.yml")
        if path:
            text = path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"if\s*\(\s*additions\s*>\s*(\d+)\s*\)", text)
            if match:
                return int(match.group(1))
            self.warnings.append("size_limit_unreadable: 未解析到 additions 阈值，使用 1000")
        return 1000

    def _read_ut_prefix(self) -> str:
        path = self._workflow("pr_quality_check.yml")
        if path:
            text = path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"UT_DIR_PREFIX\s*=\s*'([^']+)'", text)
            if match:
                return match.group(1)
            self.warnings.append("ut_prefix_unreadable: 未解析到 UT_DIR_PREFIX，使用 tests/UT/")
        return "tests/UT/"

    def _read_bugfix_threshold(self) -> int:
        path = self._workflow("pr_quality_check.yml")
        if path:
            text = path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"hasBugfix\s*&&\s*additions\s*>\s*(\d+)", text)
            if match:
                return int(match.group(1))
            self.warnings.append("bugfix_threshold_unreadable: 未解析到 bugfix UT 阈值，使用 50")
        return 50

    def _read_coverage_thresholds(self) -> dict[str, int]:
        path = self._workflow("run-ut-on-pr-py.yml")
        thresholds = {"total": 80, "line": 80, "branch": 60}
        if path:
            text = path.read_text(encoding="utf-8", errors="replace")
            for name, value in re.findall(r"\$(total|line|branch)_coverage'\)\s*<\s*(\d+)", text):
                thresholds[name] = int(value)
        pytest_ini = self.repo.top / "tests" / "pytest.ini"
        if pytest_ini.is_file():
            match = re.search(
                r"fail_under\s*=\s*(\d+)",
                pytest_ini.read_text(encoding="utf-8", errors="replace"),
            )
            if match:
                thresholds["fail_under"] = int(match.group(1))
        return thresholds

    def _read_ut_dirs(self) -> list[str]:
        ut = self.repo.top / "tests" / "UT"
        if not ut.is_dir():
            return []
        return sorted(p.name for p in ut.iterdir() if p.is_dir())

    def _read_docs_roots(self) -> dict[str, str]:
        roots = {}
        if (self.repo.top / DOCS_ZH_ROOT).is_dir():
            roots["zh"] = DOCS_ZH_ROOT
        if (self.repo.top / DOCS_EN_ROOT).is_dir():
            roots["en"] = DOCS_EN_ROOT
        return roots

    # -- derived -------------------------------------------------------
    def predicted_label(self, template_text: str) -> str | None:
        return self.label_map.get(template_text)

    def option_for_type(self, pr_type: str) -> str | None:
        """Map a CLI pr-type to the exact template option text."""
        keyword_map = {
            "feature": "Feature",
            "bugfix": "Bugfix",
            "docs": "Docs",
            "ci-cd": "CI/CD",
            "refactor": "Refactor",
            "perf": "Perf",
            "dependency": "Dependency",
            "test-cases": "Test-Cases",
            "other": "Other",
        }
        keyword = keyword_map.get(pr_type)
        if not keyword:
            return None
        for option in self.template_options:
            if option.startswith(keyword):
                return option
        return None

    def drift(self) -> list[dict[str, str | None]]:
        """Template options that cannot be mapped to a label key."""
        rows = []
        for option in self.template_options:
            if option in self.label_map:
                continue
            rows.append({"template_text": option, "matched_key": None})
        return rows

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "head": self.repo.rev("HEAD"),
            "default_branch": self.repo.default_branch(),
            "template": str(self.template_path.relative_to(self.repo.top)) if self.template_path else None,
            "template_options": self.template_options,
            "label_map": self.label_map,
            "template_label_drift": self.drift(),
            "thresholds": {
                "size_limit": self.size_limit,
                "ut_prefix": self.ut_prefix,
                "bugfix_ut_threshold": self.bugfix_ut_threshold,
                "coverage": self.coverage_thresholds,
            },
            "ut_dirs": self.ut_dirs,
            "docs_roots": self.docs_roots,
            "warnings": self.warnings,
        }


def gate(status: str, **payload) -> dict:
    row = {"status": status}
    row.update(payload)
    return row


def read_body(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        raise GateError(f"pr body file not found: {path}")
    return candidate.read_text(encoding="utf-8", errors="replace")


def issue_numbers(text: str) -> list[int]:
    numbers = []
    for groups in ISSUE_REF_RE.findall(text):
        for value in groups:
            if value:
                numbers.append(int(value))
    return sorted(set(numbers))


def check_size(repo: Repo, profile: Profile, base: str, head: str) -> dict:
    additions, deletions, binary = repo.numstat(base, head)
    rows = {
        "additions": additions,
        "deletions": deletions,
        "binary_files": binary,
        "limit": profile.size_limit,
        "evidence": f"git diff --numstat {base}..{head}",
    }
    if additions > profile.size_limit:
        return gate(
            FAIL,
            value=rows,
            remedy=f"新增 {additions} 行超过上限 {profile.size_limit}：按模块或 [type] 拆分 PR",
        )
    return gate(PASS, value=rows)


def check_issue_link(body: str | None, issue: int | None) -> dict:
    if body is None:
        return gate(UNKNOWN, reason="未提供 --pr-body-file，无法判定正文关联")
    numbers = issue_numbers(body)
    if issue is not None:
        if issue in numbers:
            return gate(PASS, value={"issue": issue, "references": numbers})
        return gate(
            FAIL,
            value={"issue": issue, "references": numbers},
            remedy=f"在 PR 正文加入 `Fixes #{issue}` 或 `Relates to #{issue}`",
        )
    if numbers:
        return gate(PASS, value={"references": numbers})
    return gate(FAIL, value={"references": []}, remedy="在 PR 正文关联 Issue，例如 `Fixes #123`")


def check_ut_changed(
    repo: Repo,
    profile: Profile,
    base: str,
    head: str,
    pr_type: str | None,
    additions: int,
) -> dict:
    changed = [path for _, path in repo.changed_files(base, head)]
    hits = [path for path in changed if path.startswith(profile.ut_prefix)]
    required = False
    reason = "PR 类型未触发 UT 要求"
    if pr_type == "feature":
        required = True
        reason = "feature 变更必须包含 UT"
    elif pr_type == "bugfix" and additions > profile.bugfix_ut_threshold:
        required = True
        reason = f"bugfix 且新增 {additions} 行 > {profile.bugfix_ut_threshold}，必须包含 UT"
    if not required:
        return gate(NOT_APPLICABLE, value={"ut_files": hits}, reason=reason)
    if hits:
        return gate(PASS, value={"ut_files": hits}, reason=reason)
    return gate(
        FAIL,
        value={"ut_files": []},
        reason=reason,
        remedy=f"在 {profile.ut_prefix} 下补充用例；可运行 ut_map.py suggest 获取建议路径",
    )


def check_dco(repo: Repo, base: str, head: str) -> dict:
    commits = repo.commits(base, head)
    if not commits:
        return gate(NOT_APPLICABLE, reason="比较范围内没有提交")
    missing = [sha for sha, body in commits if "Signed-off-by:" not in body]
    if missing:
        return gate(
            FAIL,
            value={"commits": len(commits), "missing": missing},
            remedy="对未推送的提交执行 `git commit -s --amend`；已推送的提交不要改写历史，需先与用户确认",
        )
    return gate(PASS, value={"commits": len(commits), "missing": []})


def check_title(title: str | None) -> dict:
    if not title:
        return gate(UNKNOWN, reason="未提供 --title")
    if TITLE_RE.match(title):
        return gate(PASS, value=title)
    return gate(
        FAIL,
        value=title,
        remedy="按贡献规范改写为 `[type][scope]subject`，例如 `[feat][datasets]支持 X`",
    )


def check_labels(
    profile: Profile,
    body: str | None,
    pr_type: str | None,
) -> dict:
    if body is None:
        return gate(UNKNOWN, reason="未提供 --pr-body-file，无法判定标签")
    if not profile.label_map:
        return gate(UNKNOWN, reason="无法解析 auto_label_pr.yml 的 typeToLabel 映射")
    ticked = [m.group(1).strip() for line in body.splitlines() if (m := CHECKBOX_TICKED_RE.match(line))]
    predicted: list[str] = []
    unmapped: list[str] = []
    for text in ticked:
        label = profile.label_map.get(text)
        if label:
            predicted.append(label)
        else:
            unmapped.append(text)
    desired = profile.option_for_type(pr_type) if pr_type else None
    payload = {
        "ticked": ticked,
        "predicted_labels": sorted(set(predicted)),
        "unmapped_options": unmapped,
    }
    if unmapped:
        return gate(
            FAIL,
            value=payload,
            remedy="所选类型无法命中 auto_label 映射：保留模板文案，按告警说明人工确认标签；不要为迎合脚本改写模板",
            drift=True,
        )
    if desired is None:
        return gate(UNKNOWN, value=payload, reason="未指定 --pr-type 或模板中无对应类型")
    if len(ticked) != 1:
        return gate(
            FAIL,
            value=payload,
            remedy=f"PR Type 必须且只能勾选一项，当前勾选 {len(ticked)} 项",
        )
    if predicted and predicted[0] != profile.label_map.get(desired):
        return gate(FAIL, value=payload, remedy=f"勾选类型与 --pr-type={pr_type} 不一致")
    return gate(PASS, value=payload, drift=True)


def _counterpart(path: str, src_root: str, dst_root: str) -> str | None:
    if path.startswith(src_root + "/"):
        return dst_root + path[len(src_root):]
    return None


def check_docs_parity(repo: Repo, profile: Profile, base: str, head: str) -> dict:
    changes = repo.changed_files(base, head)
    doc_changes = [(s, p) for s, p in changes if p.startswith((DOCS_ZH_ROOT + "/", DOCS_EN_ROOT + "/"))]
    readme_changes = [p for _, p in changes if p in (README_ZH, README_EN)]
    if not doc_changes and not readme_changes:
        return gate(NOT_APPLICABLE, reason="本次变更不涉及文档")

    missing: list[str] = []
    deleted_without_pair: list[str] = []
    for status, path in doc_changes:
        pair = _counterpart(path, DOCS_ZH_ROOT, DOCS_EN_ROOT) or _counterpart(path, DOCS_EN_ROOT, DOCS_ZH_ROOT)
        if pair is None:
            continue
        if status == "D":
            if repo.exists_at(head, pair):
                deleted_without_pair.append(f"{path} 删除但 {pair} 仍存在")
        else:
            if not repo.exists_at(head, pair):
                missing.append(f"{path} 缺少对侧 {pair}")

    readme_unpaired: list[str] = []
    if README_ZH in readme_changes and README_EN not in readme_changes:
        readme_unpaired.append(f"{README_ZH} 变更但 {README_EN} 未同步")
    if README_EN in readme_changes and README_ZH not in readme_changes:
        readme_unpaired.append(f"{README_EN} 变更但 {README_ZH} 未同步")

    payload = {
        "policy": "all-bilingual",
        "doc_changes": len(doc_changes),
        "readme_changes": readme_changes,
        "missing_counterparts": missing,
        "deleted_without_counterpart": deleted_without_pair,
        "readme_unpaired": readme_unpaired,
    }
    problems = missing + deleted_without_pair + readme_unpaired
    if problems:
        return gate(
            FAIL,
            value=payload,
            remedy="补齐对侧语言文件；同一 PR 内成对提交",
        )
    return gate(PASS, value=payload)


def check_docs_content(doc_change_count: int) -> dict:
    if doc_change_count == 0:
        return gate(NOT_APPLICABLE, reason="本次变更不涉及文档")
    return gate(UNKNOWN, reason="路径配对不能证明内容同步，需逐项人工/Agent 复核并记录证据")


def build_report(
    repo: Repo,
    profile: Profile,
    args: argparse.Namespace,
    base: str,
    head: str,
) -> dict:
    additions, _, _ = repo.numstat(base, head)
    body = read_body(args.pr_body_file)

    gates: dict[str, dict] = {}
    gates["size"] = check_size(repo, profile, base, head)
    gates["issue_link"] = check_issue_link(body, args.issue)
    gates["ut_changed"] = check_ut_changed(repo, profile, base, head, args.pr_type, additions)
    gates["dco"] = check_dco(repo, base, head)
    gates["title_format"] = check_title(args.title)
    gates["labels"] = check_labels(profile, body, args.pr_type)
    gates["docs_parity"] = check_docs_parity(repo, profile, base, head)
    doc_change_count = sum(
        1
        for _, path in repo.changed_files(base, head)
        if path.startswith((DOCS_ZH_ROOT + "/", DOCS_EN_ROOT + "/")) or path in (README_ZH, README_EN)
    )
    gates["docs_content"] = check_docs_content(doc_change_count)

    gates["coverage_scope"] = gate(
        UNKNOWN,
        scope="subset",
        reason="本地只跑目标子集，不能证明全量覆盖率；全量由 pr_run_test 给出",
    )
    gates["review_present"] = gate(UNKNOWN, reason="需要远端 review 证据，本地不可得")
    gates["threads_resolved"] = gate(UNKNOWN, reason="需要远端 PR 与 review thread 状态")
    gates["rtd_and_smoke"] = gate(UNKNOWN, reason="外部构建与自托管冒烟环境不可本地等价")

    warnings = list(profile.warnings)
    drift = profile.drift()
    if drift:
        warnings.append(
            "pr_template_label_drift: " + ", ".join(row["template_text"] for row in drift)
        )

    promotion_blockers = [
        {
            "gate": name,
            "status": row["status"],
            "reason": row.get("reason") or row.get("remedy") or "",
        }
        for name, row in gates.items()
        if row["status"] in (FAIL, UNKNOWN)
    ]

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "range": {
            "base": args.base,
            "head_ref": args.head,
            "merge_base": base,
            "head": head,
            "dirty": repo.is_dirty(),
        },
        "inputs": {
            "pr_type": args.pr_type,
            "issue": args.issue,
            "title": args.title,
            "pr_body_file": args.pr_body_file,
            "docs_policy": args.docs_policy,
        },
        "gates": gates,
        "warnings": warnings,
        "unverified": sorted(name for name, row in gates.items() if row["status"] == UNKNOWN),
        "promotion_blockers": promotion_blockers,
        "exit": 1 if any(row["status"] == FAIL for row in gates.values()) else 0,
    }


def render_text(report: dict) -> str:
    lines = [
        f"base={report['range']['base']} merge_base={report['range']['merge_base'][:12]} "
        f"head={report['range']['head'][:12]} dirty={report['range']['dirty']}",
        "",
        f"{'gate':<18}{'status':<16}detail",
        "-" * 78,
    ]
    for name, row in report["gates"].items():
        detail = row.get("reason") or row.get("remedy") or ""
        if not detail and "value" in row:
            value = row["value"]
            detail = json.dumps(value, ensure_ascii=False)[:60] if not isinstance(value, str) else str(value)[:60]
        lines.append(f"{name:<18}{row['status']:<16}{detail}")
    if report["warnings"]:
        lines.append("")
        lines.append("warnings:")
        lines.extend(f"  - {w}" for w in report["warnings"])
    if report["unverified"]:
        lines.append("")
        lines.append("unverified: " + ", ".join(report["unverified"]))
    lines.append("")
    lines.append(f"exit={report['exit']}  (0 只表示无已知 FAIL，不代表可晋级)")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gate_check.py",
        description="AISBench DevFlow 本地门禁预检（只读）",
    )
    parser.add_argument("--repo-root", default=".", help="仓库工作树路径，默认当前目录")
    parser.add_argument("--base", default=None, help="比较基线，默认 origin/HEAD 或 origin/master")
    parser.add_argument("--head", default="HEAD", help="比较终点，默认 HEAD")
    parser.add_argument("--pr-type", default=None, choices=list(PR_TYPES) + [None], help="PR 类型")
    parser.add_argument("--issue", type=int, default=None, help="期望关联的 Issue 编号")
    parser.add_argument("--title", default=None, help="拟用 PR 标题")
    parser.add_argument("--pr-body-file", default=None, help="已渲染的 PR 正文本地路径")
    parser.add_argument(
        "--docs-policy",
        default="all-bilingual",
        choices=["all-bilingual"],
        help="文档双语策略；当前仅支持强制全量双语",
    )
    parser.add_argument("--format", default="json", choices=["json", "text"], help="输出格式")
    parser.add_argument("--out", default=None, help="把 JSON 报告写入指定文件")
    parser.add_argument("--profile", action="store_true", help="只输出仓库画像")
    parser.add_argument("--offline", action="store_true", help="保留参数，当前实现本就只读本地 git")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    args = parse_args(argv)
    try:
        repo = Repo(Path(args.repo_root).resolve())
        profile = Profile(repo)

        if args.profile:
            payload = profile.as_dict()
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            if args.out:
                Path(args.out).write_text(text + "\n", encoding="utf-8")
            print(text)
            return 0

        base_ref = args.base or profile.repo.default_branch() or "master"
        base = repo.merge_base(base_ref, args.head)
        head = repo.rev(args.head)
        report = build_report(repo, profile, args, base, head)

        if args.format == "text":
            rendered = render_text(report)
        else:
            rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        print(rendered)
        return report["exit"]
    except GateError as exc:
        print(json.dumps({"schema": SCHEMA, "error": str(exc), "exit": 2}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
