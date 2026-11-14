"""命令行入口：python -m skill_eval --suite <suite dir> [...]"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .harness import Runner, Suite
from .render import render_markdown, summarize


def git_commit(path: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def python_label(python: str) -> str:
    """报告里只显示解释器名与版本，不泄露本机绝对路径。"""
    version = ""
    try:
        out = subprocess.run([python, "-c", "import sys;print(sys.version.split()[0])"],
                             capture_output=True, text=True, timeout=20)
        version = out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        pass
    return f"{Path(python).name} {version}".strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SKILL 测评执行器（离线、黑盒、可复现）")
    parser.add_argument("--suite", required=True, type=Path, help="测评集目录（含 suite.json 与 cases/）")
    parser.add_argument("--skill-dir", type=Path, help="被测 SKILL 目录；默认取 suite.json 的 skill.local_dir")
    parser.add_argument("--work", type=Path, default=Path(".skill-eval-work"), help="工作目录（用例输入与产物）")
    parser.add_argument("--python", default=sys.executable, help="执行被测 SKILL 的解释器（需 >=3.10）")
    parser.add_argument("--deps", default=None, help="被测 SKILL 依赖的额外 PYTHONPATH（如 jsonschema 的安装目录）")
    parser.add_argument("--json", type=Path, help="机器可读结果输出路径")
    parser.add_argument("--md", type=Path, help="Markdown 报告输出路径")
    parser.add_argument("--filter", default="", help="只跑 id/title/dimension 匹配该子串的用例")
    parser.add_argument("--list", action="store_true", help="只列出用例")
    parser.add_argument("--keep-going", action="store_true", default=True, help="单用例失败不影响其他用例")
    args = parser.parse_args(argv)

    suite = Suite(args.suite, Path(args.skill_dir or suite_skill_dir(args.suite)), args.deps)
    if args.list:
        for case in suite.cases:
            print(f"{case['id']}\t{case.get('dimension','')}\t{case.get('weight',1)}\t{case.get('title','')}")
        return 0

    selected = [c for c in suite.cases
                if not args.filter or args.filter in json.dumps(c, ensure_ascii=False)]
    if not selected:
        print(f"no case matches filter {args.filter!r}", file=sys.stderr)
        return 2

    if not (suite.skill_dir / "SKILL.md").is_file():
        print(f"SKILL.md not found under {suite.skill_dir}", file=sys.stderr)
        return 2

    runner = Runner(suite, args.work, args.python)
    print(f"target : {suite.skill_dir}")
    print(f"suite  : {suite.root}  ({len(selected)} cases)")
    results = []
    for case in selected:
        result = runner.run_case(case)
        results.append(result)
        mark = "PASS" if result.passed else "FAIL"
        detail = ""
        if not result.passed:
            detail = " | " + (result.failures[0].detail[:160] if result.failures else "")
        print(f"[{mark}] {case['id']:<10} {case.get('title','')}{detail}")

    summary = summarize(results, suite.spec)
    commit = git_commit(suite.skill_dir)
    relative_run = "python -m skill_eval " + " ".join(
        shlex.quote(part) for part in [
            "--suite", _rel(args.suite),
            "--skill-dir", _rel(suite.skill_dir),
            "--python", _label(Path(args.python)),
            *(["--deps", _label(Path(args.deps))] if args.deps else []),
            "--work", _rel(args.work),
            "--md", _rel(args.md) if args.md else "reports/report.md",
            "--json", _rel(args.json) if args.json else "reports/result.json",
        ]
    )
    report = render_markdown(
        summary, suite.spec, suite.skill_dir,
        command=relative_run,
        deps=_label(Path(args.deps) if args.deps else None) or None,
        target_python=python_label(args.python),
        commit=commit,
    )
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(report, encoding="utf-8")
        print(f"report : {args.md}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = summary | {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "target": str(suite.skill_dir),
            "commit": commit,
            "python": args.python,
        }
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"json   : {args.json}")

    print(f"score  : {summary['score']}/100 ({summary['passed']}/{summary['cases']} cases)")
    return 0 if summary["failed"] == 0 else 1


def suite_skill_dir(suite_dir: Path) -> str:
    spec = json.loads((Path(suite_dir) / "suite.json").read_text(encoding="utf-8-sig"))
    return spec["skill"]["local_dir"]


def _rel(path: Path) -> str:
    try:
        return os.path.relpath(Path(path).resolve(), Path.cwd()).replace("\\", "/")
    except Exception:
        return str(path)


def _label(path: Path | None) -> str:
    """报告里显示依赖目录：仓内用相对路径，仓外用目录名。"""
    if not path:
        return ""
    relative = _rel(path)
    return Path(path).name if relative.startswith("..") else relative


if __name__ == "__main__":
    raise SystemExit(main())
