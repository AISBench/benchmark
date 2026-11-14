"""Shared fixtures for aisbench-devflow unit tests.

Every test builds a throwaway git repository so the preflight logic is exercised
against real ``git`` output instead of mocked strings.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
FALLBACK_TMP = SKILL_ROOT / ".devflow-tmp"


def _probe(base: Path) -> bool:
    probe = base / f".probe-{os.getpid()}"
    try:
        (probe / "child").mkdir(parents=True)
        (probe / "child" / "file").write_text("ok", encoding="utf-8")
        shutil.rmtree(probe, ignore_errors=True)
        return True
    except OSError:
        return False


def temp_base() -> Path:
    """Pick a writable temp root.

    Some sandboxes deny nested ``mkdir`` inside their own temp directory, so the
    skill falls back to a directory beside the tests.  CI (Linux) uses the system
    temp dir.
    """
    override = os.environ.get("AISBENCH_DEVFLOW_TMPDIR")
    if override:
        base = Path(override)
        base.mkdir(parents=True, exist_ok=True)
        return base
    for base in (Path(tempfile.gettempdir()), FALLBACK_TMP):
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if _probe(base):
            return base
    raise RuntimeError("no writable temp directory found for aisbench-devflow tests")


def new_workdir() -> Path:
    """Create an isolated directory without ``mkdtemp`` (its 0700 mode is unusable in some sandboxes)."""
    base = temp_base()
    for _ in range(100):
        candidate = base / f"devflow-{os.getpid()}-{os.urandom(4).hex()}"
        try:
            candidate.mkdir(parents=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("cannot create a working directory for aisbench-devflow tests")


def cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)

PR_TEMPLATE = """Thanks for your contribution.

**PR Type / PR类型**
- [ ] Feature（功能新增）
- [ ] Bugfix（Bug 修复）
- [ ] Docs（文档更新）
- [ ] CI/CD（持续集成/持续部署）
- [ ] Refactor（代码重构）
- [ ] Perf（性能优化）
- [ ] Dependency（依赖项更新）
- [ ] Test-Cases（测试用例更新）
- [ ] Other（其他）

**Related Issue | 关联 Issue**
Fixes #(issue ID / issue 编号) / Relates to #(issue ID / issue 编号)

## 🔍 Motivation / 变更动机

Please describe the motivation.

## 📝 Modification / 修改内容

Please briefly describe the modification.

## ✅ Checklist / 检查列表

- [ ] The modification is covered by complete unit tests.
- [ ] CLA has been signed and all committers have signed the CLA in this PR.
"""

AUTO_LABEL_WORKFLOW = """name: Auto Label PR
on:
  pull_request_target:
    types: [opened, edited]
jobs:
  auto-label:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@v7
        with:
          script: |
            const typeToLabel = {
              'Feature（功能新增）': 'feature',
              'Bugfix（Bug 修复）': 'bugfix',
              'Docs（文档更新）': 'docs',
              'CI/CD（流水线工程）': 'ci/cd',
              'Refactor（代码重构）': 'refactor',
              'Perf（性能优化）': 'perf',
              'Dependency（依赖更新）': 'dependency',
              'Test-Cases（测试用例更新）': 'test-cases',
              'Other（其他）': 'other'
            };
"""

QUALITY_WORKFLOW = """name: PR Quality Check
jobs:
  pr-quality-gate:
    steps:
      - uses: actions/github-script@v7
        with:
          script: |
            if (additions > 1000) {
              failures.push('large_pr');
            }
            const UT_DIR_PREFIX = 'tests/UT/';
            if (hasFeature) {
              needUT = true;
            } else if (hasBugfix && additions > 50) {
              needUT = true;
            }
"""

UT_WORKFLOW = """name: run_ut_on_pr
jobs:
  pr_run_test:
    runs-on: [self-hosted, Linux, run]
    steps:
      - run: python3 tests/run_tests.py tests/UT -p 16
      - run: |
          if python3 -c "exit(0 if float('$total_coverage') < 80 else 1)"; then exit 1; fi
          if python3 -c "exit(0 if float('$line_coverage') < 80 else 1)"; then exit 1; fi
          if python3 -c "exit(0 if float('$branch_coverage') < 60 else 1)"; then exit 1; fi
"""

PYTEST_INI = """[coverage:run]
source = ais_bench

[coverage:report]
fail_under = 80
"""


def git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout


def init_repo(root: Path, with_profile: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    git(root, "config", "user.name", "DevFlow Test")
    git(root, "config", "user.email", "devflow@example.com")
    git(root, "config", "commit.gpgsign", "false")
    if with_profile:
        (root / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
        (root / "tests").mkdir(parents=True, exist_ok=True)
        (root / ".github" / "pull_request_template.md").write_text(PR_TEMPLATE, encoding="utf-8")
        (root / ".github" / "workflows" / "auto_label_pr.yml").write_text(
            AUTO_LABEL_WORKFLOW, encoding="utf-8"
        )
        (root / ".github" / "workflows" / "pr_quality_check.yml").write_text(
            QUALITY_WORKFLOW, encoding="utf-8"
        )
        (root / ".github" / "workflows" / "run-ut-on-pr-py.yml").write_text(
            UT_WORKFLOW, encoding="utf-8"
        )
        (root / "tests" / "pytest.ini").write_text(PYTEST_INI, encoding="utf-8")
    (root / "app.py").write_text("print('base')\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root


def write_files(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def commit(root: Path, message: str, files: dict[str, str] | None = None, sign: bool = False) -> str:
    if files:
        write_files(root, files)
    git(root, "add", "-A")
    args = ["commit", "-q", "-m", message]
    if sign:
        args.insert(2, "-s")
    git(root, *args)
    return git(root, "rev-parse", "HEAD").strip()
