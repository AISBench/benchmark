"""测评报告渲染：Markdown + 机器可读 JSON。"""
from __future__ import annotations

import datetime
import json
import platform
import sys
from pathlib import Path


def summarize(results, suite_spec):
    total_weight = sum(r.weight for r in results) or 1.0
    passed_weight = sum(r.weight for r in results if r.passed)
    score = round(100.0 * passed_weight / total_weight, 1)

    dimensions = {}
    for result in results:
        name = result.case.get("dimension", "未分类")
        bucket = dimensions.setdefault(name, {"cases": 0, "passed": 0, "weight": 0.0, "passed_weight": 0.0})
        bucket["cases"] += 1
        bucket["weight"] += result.weight
        if result.passed:
            bucket["passed"] += 1
            bucket["passed_weight"] += result.weight

    for bucket in dimensions.values():
        bucket["score"] = round(100.0 * bucket["passed_weight"] / (bucket["weight"] or 1), 1)

    return {
        "suite": suite_spec.get("name", ""),
        "score": score,
        "cases": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "weight": total_weight,
        "passed_weight": passed_weight,
        "dimensions": dimensions,
        "results": [r.as_dict() for r in results],
    }


def render_markdown(summary, suite_spec, skill_dir: Path, command: str, deps: str | None,
                    target_python: str = "", commit: str = "") -> str:
    now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    target = suite_spec.get("skill", {})
    anchor = commit or target.get("commit", "")
    lines = [
        f"# SKILL 测评报告：{target.get('name', suite_spec.get('name',''))}",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 被测 SKILL | `{target.get('repo','')}::{target.get('path','')}` |",
        f"| 版本锚点 | ref `{target.get('ref','')}` / commit `{anchor or '未固定（工作区快照）'}` |",
        f"| 测评集版本 | `{suite_spec.get('version','')}` |",
        f"| 执行时间 | {now} |",
        f"| 被测解释器 | `{target_python or '未指定'}`（执行器 Python {sys.version.split()[0]} / {platform.system()}） |",
        "| 执行方式 | 离线黑盒：只调用 SKILL 自带入口，不需要 SSH、Docker、NPU |",
        f"| 依赖路径 | `{deps or '无（使用解释器内置环境）'}` |",
        "",
        "## 总评",
        "",
        f"**综合得分 {summary['score']} / 100**（{summary['passed']}/{summary['cases']} 用例通过，"
        f"权重 {summary['passed_weight']:.0f}/{summary['weight']:.0f}）",
        "",
        "| 维度 | 用例 | 通过 | 权重 | 得分 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, bucket in sorted(summary["dimensions"].items()):
        lines.append(
            f"| {name} | {bucket['cases']} | {bucket['passed']} | {bucket['weight']:.0f} | {bucket['score']} |"
        )

    lines += ["", "## 用例明细", "", "| 用例 | 名称 | 维度 | 权重 | 结果 | 失败首因 |", "|---|---|---|---:|---|---|"]
    for result in summary["results"]:
        failure = result["checks"] and [c for c in result["checks"] if not c["ok"]]
        first = failure[0]["detail"] if failure else ""
        first = first.replace("|", "\\|")[:160]
        status = "PASS" if result["passed"] else "**FAIL**"
        lines.append(
            f"| {result['id']} | {result['title'].replace('|','\\|')} | {result['dimension']} | "
            f"{result['weight']:.0f} | {status} | {first} |"
        )

    failed = [r for r in summary["results"] if not r["passed"]]
    lines += ["", "## 失败详情", ""]
    if not failed:
        lines.append("无失败用例。")
    for result in failed:
        lines.append(f"### {result['id']} {result['title']}")
        lines.append("")
        lines.append(f"- 期望依据：{result.get('why') or '见用例文件'}")
        for check in result["checks"]:
            if not check["ok"]:
                lines.append(f"- `{check['kind']}`：{check['detail']}")
        lines.append(f"- 现场：`{result['run_dir']}`")
        lines.append("")

    limits = suite_spec.get("limits", [])
    if limits:
        lines += ["", "## 覆盖边界（未纳入自动判定的部分）", ""]
        lines += [f"- {item}" for item in limits]

    lines += [
        "",
        "## 复现",
        "",
        "```sh",
        command,
        "```",
        "",
        "用例输入（job_spec / bindings）与断言全部在测评集内，执行不修改被测 SKILL 目录。",
        "",
    ]
    return "\n".join(lines)
