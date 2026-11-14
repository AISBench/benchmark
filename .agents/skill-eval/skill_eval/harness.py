"""AISBench SKILL 测评框架：通用执行器（与具体 SKILL 解耦）。

设计约束
--------
* 被测对象只有两样：SKILL 目录 + suite 描述（suite.json + cases/*.json）。
* 断言只读被测 SKILL 的**外部行为**（退出码、stdout/stderr、产物文件、产物内容），
  不 import 被测代码，避免把测试绑死在内部函数签名上。
* 离线可跑：默认只调用 SKILL 自带的离线入口（本 SKILL 为 ``run.py --mode scripts``），
  不需要 SSH / Docker / NPU。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FIXTURE_FILES = {"job_spec.json", "bindings.json"}


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pointer_get(document: Any, pointer: str) -> Any:
    """RFC6901 子集：/a/0/b（支持 ~0 ~1 转义）。"""
    if pointer in ("", "/"):
        return document
    node = document
    for raw in pointer.lstrip("/").split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(token)]
        else:
            if token not in node:
                raise KeyError(f"pointer not found: {pointer} (at {token})")
            node = node[token]
    return node


def pointer_set(document: Any, pointer: str, value: Any) -> None:
    tokens = [t.replace("~1", "/").replace("~0", "~") for t in pointer.lstrip("/").split("/")]
    node = document
    for token in tokens[:-1]:
        if isinstance(node, list):
            node = node[int(token)]
        else:
            node = node.setdefault(token, {})
    last = tokens[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def pointer_pop(document: Any, pointer: str) -> None:
    tokens = [t.replace("~1", "/").replace("~0", "~") for t in pointer.lstrip("/").split("/")]
    node = document
    for token in tokens[:-1]:
        node = node[int(token)] if isinstance(node, list) else node[token]
    last = tokens[-1]
    if isinstance(node, list):
        node.pop(int(last))
    else:
        node.pop(last)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_files(root: Path) -> list[str]:
    return sorted(
        str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()
    )


# --------------------------------------------------------------------------
# 数据模型
# --------------------------------------------------------------------------
@dataclass
class Check:
    ok: bool
    kind: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {"ok": self.ok, "kind": self.kind, "detail": self.detail}


@dataclass
class CaseResult:
    case: dict
    checks: list[Check] = field(default_factory=list)
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    run_dir: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def weight(self) -> float:
        return float(self.case.get("weight", 1))

    def as_dict(self) -> dict:
        return {
            "id": self.case["id"],
            "title": self.case.get("title", ""),
            "dimension": self.case.get("dimension", ""),
            "weight": self.weight,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration, 3),
            "checks": [c.as_dict() for c in self.checks],
            "run_dir": self.run_dir,
        }


# --------------------------------------------------------------------------
# suite 装载
# --------------------------------------------------------------------------
class Suite:
    def __init__(self, root: Path, skill_dir: Path, deps: str | None):
        self.root = Path(root).resolve()
        skill = Path(skill_dir)
        self.skill_dir = (skill if skill.is_absolute() else (self.root / skill)).resolve()
        self.deps = deps
        self.spec = read_json(self.root / "suite.json")
        self.cases = self._load_cases()

    def _load_cases(self) -> list[dict]:
        cases = []
        seen = set()
        case_dir = self.root / "cases"
        for path in sorted(case_dir.glob("*.json")):
            case = read_json(path)
            case["_file"] = str(path)
            if case["id"] in seen:
                raise ValueError(f"duplicate case id: {case['id']}")
            seen.add(case["id"])
            cases.append(case)
        if not cases:
            raise ValueError(f"no cases under {case_dir}")
        return cases

    def fixture(self, name: str) -> tuple[dict, dict]:
        base = self.root / "fixtures" / name
        return read_json(base / "job_spec.json"), read_json(base / "bindings.json")


# --------------------------------------------------------------------------
# 用例执行
# --------------------------------------------------------------------------
class Runner:
    def __init__(self, suite: Suite, work_dir: Path, python: str | None = None):
        self.suite = suite
        self.work = Path(work_dir).resolve()
        self.python = python or sys.executable
        self.base_env = dict(os.environ)
        deps = str(Path(suite.deps).resolve()) if suite.deps else ""
        self.base_env["PYTHONPATH"] = os.pathsep.join(
            [p for p in [deps, os.environ.get("PYTHONPATH", "")] if p]
        )

    # -- 输入准备 ---------------------------------------------------------
    def materialize(self, case: dict, case_dir: Path) -> tuple[Path, Path]:
        fixture = case.get("fixture", "base")
        spec, bindings = self.suite.fixture(fixture)
        for patch in case.get("patch", []):
            target = spec if patch["file"] == "job_spec.json" else bindings
            op = patch.get("op", "set")
            if op == "set":
                pointer_set(target, patch["pointer"], copy.deepcopy(patch["value"]))
            elif op == "remove":
                pointer_pop(target, patch["pointer"])
            elif op == "append":
                container = pointer_get(target, patch["pointer"])
                container.append(copy.deepcopy(patch["value"]))
            else:
                raise ValueError(f"unknown patch op: {op}")
        inputs = case_dir / "input"
        job_spec = inputs / "job_spec.json"
        binding_file = inputs / "bindings.json"
        dump_json(job_spec, spec)
        dump_json(binding_file, bindings)
        return job_spec, binding_file

    # -- 调用被测 SKILL ---------------------------------------------------
    def invoke(self, case: dict, case_dir: Path, job_spec: Path, bindings: Path) -> tuple[int, str, str, float]:
        adapter = dict(self.suite.spec["adapter"])
        if case.get("adapter"):
            merged_env = dict(adapter.get("env", {}))
            merged_env.update(case["adapter"].get("env", {}))
            adapter.update(case["adapter"])
            if merged_env:
                adapter["env"] = merged_env
        subs = {
            "{python}": self.python,
            "{skill_dir}": str(self.suite.skill_dir),
            "{suite_dir}": str(self.suite.root),
            "{job_spec}": str(job_spec),
            "{bindings}": str(bindings),
            "{run_dir}": str(case_dir / "run"),
            "{case_dir}": str(case_dir),
            "{mode}": case.get("mode", adapter.get("mode", "scripts")),
        }

        def expand(value: str) -> str:
            for key, replacement in subs.items():
                value = value.replace(key, replacement)
            return value

        command = [expand(part) for part in adapter["command"]]
        env = dict(self.base_env)
        for key, value in adapter.get("env", {}).items():
            env[key] = expand(value)
        started = time.time()
        try:
            process = subprocess.run(
                command,
                cwd=expand(adapter.get("workdir", "{skill_dir}")),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=float(case.get("timeout_seconds", 180)),
            )
            return process.returncode, process.stdout, process.stderr, time.time() - started
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return 124, stdout, stderr, time.time() - started

    # -- 断言 -------------------------------------------------------------
    def check(self, assertion: dict, ctx: dict) -> Check:
        kind = assertion["type"]
        handler = getattr(self, f"_check_{kind}", None)
        if handler is None:
            raise ValueError(f"unknown assertion type: {kind}")
        return handler(assertion, ctx)

    def _path(self, ctx: dict, value: str) -> Path:
        if value.startswith("skill:"):
            return self.suite.skill_dir / value[len("skill:"):]
        if value.startswith("run:"):
            return Path(ctx["run_dir"]) / value[len("run:"):]
        if value.startswith("{run_dir}"):
            return Path(ctx["run_dir"]) / value[len("{run_dir}"):].lstrip("/\\")
        return Path(ctx["case_dir"]) / value

    def _text(self, ctx: dict, value: str) -> str:
        if value.startswith("skill:"):
            return (self.suite.skill_dir / value[len("skill:"):]).read_text(encoding="utf-8", errors="replace")
        path = Path(ctx["run_dir"]) / value
        return path.read_text(encoding="utf-8", errors="replace")

    # 输出通道
    def _check_exit_code(self, a: dict, ctx: dict) -> Check:
        expected = int(a["value"])
        return Check(ctx["exit_code"] == expected, "exit_code",
                     f"expected {expected}, got {ctx['exit_code']}")

    def _check_stdout_contains(self, a: dict, ctx: dict) -> Check:
        ok = a["value"] in ctx["stdout"]
        return Check(ok, "stdout_contains", "" if ok else f"stdout missing {a['value']!r}")

    def _check_stderr_contains(self, a: dict, ctx: dict) -> Check:
        ok = a["value"] in ctx["stderr"]
        return Check(ok, "stderr_contains",
                     "" if ok else f"stderr missing {a['value']!r}; stderr tail={ctx['stderr'][-400:]!r}")

    def _check_stderr_not_contains(self, a: dict, ctx: dict) -> Check:
        ok = a["value"] not in ctx["stderr"]
        return Check(ok, "stderr_not_contains", "" if ok else f"stderr unexpectedly contains {a['value']!r}")

    def _check_stderr_regex(self, a: dict, ctx: dict) -> Check:
        ok = re.search(a["value"], ctx["stderr"]) is not None
        return Check(ok, "stderr_regex", "" if ok else f"stderr does not match /{a['value']}/")

    def _check_stderr_any_of(self, a: dict, ctx: dict) -> Check:
        hit = [v for v in a["value"] if v in ctx["stderr"]]
        return Check(bool(hit), "stderr_any_of",
                     "" if hit else f"none of {a['value']} in stderr tail={ctx['stderr'][-400:]!r}")

    # 文件系统
    def _check_file_exists(self, a: dict, ctx: dict) -> Check:
        missing = [v for v in a["value"] if not (Path(ctx["run_dir"]) / v).is_file()]
        return Check(not missing, "file_exists", "" if not missing else f"missing: {missing}")

    def _check_file_absent(self, a: dict, ctx: dict) -> Check:
        present = [v for v in a["value"] if (Path(ctx["run_dir"]) / v).exists()]
        return Check(not present, "file_absent", "" if not present else f"unexpected: {present}")

    def _check_file_set_equals(self, a: dict, ctx: dict) -> Check:
        actual = relative_files(Path(ctx["run_dir"]))
        expected = sorted(a["value"])
        if actual == expected:
            return Check(True, "file_set_equals")
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        return Check(False, "file_set_equals", f"missing={missing} extra={extra}")

    def _check_dir_absent(self, a: dict, ctx: dict) -> Check:
        target = self._path(ctx, a["value"])
        return Check(not target.exists(), "dir_absent", "" if not target.exists() else f"unexpected dir: {target}")

    def _check_no_temp_leftover(self, a: dict, ctx: dict) -> Check:
        parent = Path(ctx["run_dir"]).parent
        leftovers = [p.name for p in parent.iterdir() if p.name.startswith((".generate-", ".JOB"))]
        return Check(not leftovers, "no_temp_leftover", "" if not leftovers else f"leftovers: {leftovers}")

    # 产物内容
    def _check_json_equals(self, a: dict, ctx: dict) -> Check:
        try:
            document = read_json(Path(ctx["run_dir"]) / a["path"])
        except FileNotFoundError:
            return Check(False, "json_equals", f"missing file: {a['path']}")
        try:
            actual = pointer_get(document, a["pointer"])
        except (KeyError, IndexError) as exc:
            return Check(False, "json_equals", f"{a['path']}{a['pointer']} unreadable: {exc}")
        ok = actual == a["value"]
        return Check(ok, "json_equals",
                     "" if ok else f"{a['path']}{a['pointer']}: expected {a['value']!r}, got {actual!r}")

    def _check_json_present(self, a: dict, ctx: dict) -> Check:
        try:
            document = read_json(Path(ctx["run_dir"]) / a["path"])
        except FileNotFoundError:
            return Check(False, "json_present", f"missing file: {a['path']}")
        try:
            pointer_get(document, a["pointer"])
            return Check(True, "json_present")
        except (KeyError, IndexError):
            return Check(False, "json_present", f"missing {a['path']}{a['pointer']}")

    def _check_text_contains(self, a: dict, ctx: dict) -> Check:
        try:
            text = self._text(ctx, a["path"])
        except FileNotFoundError:
            return Check(False, "text_contains", f"missing file: {a['path']}")
        missing = [v for v in (a["value"] if isinstance(a["value"], list) else [a["value"]]) if v not in text]
        return Check(not missing, "text_contains", "" if not missing else f"{a['path']} missing {missing}")

    def _check_text_not_contains(self, a: dict, ctx: dict) -> Check:
        try:
            text = self._text(ctx, a["path"])
        except FileNotFoundError:
            return Check(False, "text_not_contains", f"missing file: {a['path']}")
        present = [v for v in (a["value"] if isinstance(a["value"], list) else [a["value"]]) if v in text]
        return Check(not present, "text_not_contains", "" if not present else f"{a['path']} contains {present}")

    def _check_manifest_integrity(self, a: dict, ctx: dict) -> Check:
        root = Path(ctx["run_dir"])
        manifest_path = a.get("path", ".runtime/manifest.json")
        manifest = read_json(root / manifest_path)
        recorded = manifest.get("files", {})
        excluded = {manifest_path, *a.get("exclude", [])}
        actual_files = [p for p in root.rglob("*") if p.is_file()
                        and str(p.relative_to(root)).replace("\\", "/") not in excluded]
        actual = {str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in actual_files}
        missing = sorted(set(actual) - set(recorded))
        stale = sorted(k for k, v in recorded.items() if k not in actual or actual[k] != v)
        if missing or stale:
            return Check(False, "manifest_integrity", f"unrecorded={missing[:5]} mismatched={stale[:5]}")
        return Check(True, "manifest_integrity")

    # 探针（模块级契约检查，离线替代需要 SSH 的入口）
    def _check_probe_json(self, a: dict, ctx: dict) -> Check:
        payload = json.loads(ctx["stdout"])
        try:
            actual = pointer_get(payload, a["pointer"])
        except (KeyError, IndexError) as exc:
            return Check(False, "probe_json", f"{a['pointer']} unreadable: {exc}; stdout={ctx['stdout'][:300]!r}")
        if "equals" in a:
            ok = actual == a["equals"]
            return Check(ok, "probe_json", "" if ok else f"{a['pointer']}: expected {a['equals']!r}, got {actual!r}")
        ok = str(a["contains"]) in json.dumps(actual, ensure_ascii=False)
        return Check(ok, "probe_json", "" if ok else f"{a['pointer']} does not contain {a['contains']!r}: {actual!r}")

    # 静态检查（不执行 SKILL）
    def _check_frontmatter(self, a: dict, ctx: dict) -> Check:
        text = (self.suite.skill_dir / a.get("path", "SKILL.md")).read_text(encoding="utf-8", errors="replace")
        match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        if not match:
            return Check(False, "frontmatter", "missing YAML frontmatter block")
        block = match.group(1)
        keys = set(re.findall(r"^([A-Za-z_][A-Za-z0-9_-]*):", block, re.M))
        missing = [k for k in a.get("require", []) if k not in keys]
        if missing:
            return Check(False, "frontmatter", f"missing keys: {missing}")
        if a.get("name_equals_dir"):
            name = re.search(r"^name:\s*(.+)$", block, re.M)
            if not name or name.group(1).strip() != self.suite.skill_dir.name:
                return Check(False, "frontmatter", "frontmatter name != skill directory name")
        return Check(True, "frontmatter")

    def _check_relative_links_exist(self, a: dict, ctx: dict) -> Check:
        text = (self.suite.skill_dir / a["path"]).read_text(encoding="utf-8", errors="replace")
        broken = []
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"^[a-z]+://", target) or target.startswith(("#", "mailto:")):
                continue
            clean = target.split("#", 1)[0]
            if not clean:
                continue
            if not (self.suite.skill_dir / clean).exists():
                broken.append(target)
        return Check(not broken, "relative_links_exist", "" if not broken else f"broken links: {sorted(set(broken))}")

    def _check_cli_flags(self, a: dict, ctx: dict) -> Check:
        text = (self.suite.skill_dir / a["script"]).read_text(encoding="utf-8", errors="replace")
        missing = [flag for flag in a["flags"] if f"'{flag}'" not in text and f'"{flag}"' not in text]
        return Check(not missing, "cli_flags", "" if not missing else f"{a['script']} missing flags: {missing}")

    # -- 单个用例 ---------------------------------------------------------
    def run_case(self, case: dict) -> CaseResult:
        case_dir = self.work / "runs" / case["id"]
        if case_dir.exists():
            shutil.rmtree(case_dir, ignore_errors=True)
        case_dir.mkdir(parents=True, exist_ok=True)
        run_dir = case_dir / "run"
        result = CaseResult(case=case, run_dir=str(run_dir))
        job_spec, bindings = self.materialize(case, case_dir)

        if case.get("adapter", {}).get("type") == "static":
            ctx = {"case_dir": case_dir, "run_dir": run_dir, "exit_code": 0, "stdout": "", "stderr": ""}
        else:
            code, stdout, stderr, duration = self.invoke(case, case_dir, job_spec, bindings)
            result.exit_code, result.stdout, result.stderr, result.duration = code, stdout, stderr, duration
            ctx = {"case_dir": case_dir, "run_dir": run_dir, "exit_code": code,
                   "stdout": stdout, "stderr": stderr}

        for assertion in case.get("expect", []):
            try:
                result.checks.append(self.check(assertion, ctx))
            except Exception as exc:  # 断言自身出错也算失败，不能中断整个测评
                result.checks.append(Check(False, assertion.get("type", "unknown"),
                                           f"assertion error: {type(exc).__name__}: {exc}"))
        (case_dir / "result.json").write_text(
            json.dumps(result.as_dict() | {"stdout": result.stdout, "stderr": result.stderr},
                       indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return result
