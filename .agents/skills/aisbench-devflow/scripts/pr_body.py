#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render and self-check an AISBench pull request body.

Two sub-commands:

    render  fill a copy of the repository PR template, tick exactly one PR type,
            and append the audit trailer once
    check   verify an existing body (trailer, single type selection, mapped label,
            issue reference, optional title)

This module never edits the template and never touches the network.  When the
template option text does not match an ``auto_label_pr`` mapping key, the drift
is reported as a failure instead of being papered over by rewriting the
template wording.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = 1
TRAILER = "<!-- aisbench-devflow -->"
CHECKBOX_ANY_RE = re.compile(r"^\s*[-*]\s*\[[ xX]\]\s*(.+?)\s*$")
CHECKBOX_TICKED_RE = re.compile(r"^\s*[-*]\s*\[[xX]\]\s*(.+?)\s*$")
HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
ISSUE_REF_RE = re.compile(r"#(\d+)")
TITLE_RE = re.compile(r"^\[(feat|fix|docs|style|refactor|test|chore|perf|build|ci|dep|deps)\](\[[^\]]+\])?.+$")
RELATED_LINE_RE = re.compile(r"^\s*Fixes\s*#.*$", re.IGNORECASE)

TYPE_KEYWORD = {
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


class BodyError(RuntimeError):
    pass


def load_label_map(workflow: Path) -> dict[str, str]:
    if not workflow.is_file():
        raise BodyError(f"workflow not found: {workflow}")
    text = workflow.read_text(encoding="utf-8-sig", errors="replace")
    block = re.search(r"const\s+typeToLabel\s*=\s*\{(?P<body>.*?)\n\s*\};", text, re.S)
    if not block:
        raise BodyError("typeToLabel block not found in auto_label_pr.yml")
    return dict(re.findall(r"'([^']+)'\s*:\s*'([^']+)'", block.group("body")))


def read_template(path: Path) -> str:
    if not path.is_file():
        raise BodyError(f"template not found: {path}")
    return path.read_text(encoding="utf-8-sig", errors="replace")


def template_options(text: str) -> list[str]:
    """Options from the PR Type block only, so the Checklist section is ignored."""
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if re.search(r"PR\s*Type", line, re.IGNORECASE):
            start = index + 1
            break
    options = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("**") or stripped.startswith("##"):
            break
        match = CHECKBOX_ANY_RE.match(line)
        if match:
            options.append(match.group(1))
    return options


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def option_for_type(options: list[str], pr_type: str) -> str:
    keyword = TYPE_KEYWORD.get(pr_type)
    if not keyword:
        raise BodyError(f"unknown pr type: {pr_type}")
    for option in options:
        if option.startswith(keyword):
            return option
    raise BodyError(f"template has no option starting with {keyword!r}")


def normalize(text: str) -> str:
    """Strip emoji/punctuation/spaces so headings can be matched loosely."""
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).lower()


def tick_option(body: str, option: str) -> str:
    lines = body.splitlines()
    hit = False
    for index, line in enumerate(lines):
        match = CHECKBOX_ANY_RE.match(line)
        if not match:
            continue
        if match.group(1) != option:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        lines[index] = f"{indent}- [x] {option}"
        hit = True
        break
    if not hit:
        raise BodyError(f"template option not found while ticking: {option}")
    return "\n".join(lines)


def set_issue_line(body: str, issue: int) -> str:
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if RELATED_LINE_RE.match(line):
            lines[index] = f"Fixes #{issue}"
            return "\n".join(lines)
    return "\n".join(lines) + f"\n\nFixes #{issue}"


def fill_sections(body: str, sections: dict[str, str]) -> str:
    """Insert section content under matching headings.

    Headings left with template placeholder prose are reported in a trailer
    comment instead of being silently presented as filled content.  Checkbox-only
    blocks (the checklist) are not treated as placeholders.
    """
    normalized_sections = {normalize(key): value for key, value in sections.items()}
    used: set[str] = set()
    unfilled: list[str] = []
    lines = body.splitlines()
    out: list[str] = []
    index = 0
    total = len(lines)
    while index < total:
        line = lines[index]
        out.append(line)
        heading = HEADING_RE.match(line)
        index += 1
        if not heading:
            continue
        normalized_heading = normalize(heading.group(2))
        matched_key = None
        for key in normalized_sections:
            if key and (key in normalized_heading or normalized_heading in key):
                matched_key = key
                break
        # collect the template body of this heading
        block: list[str] = []
        while index < total and not HEADING_RE.match(lines[index]):
            block.append(lines[index])
            index += 1
        if matched_key is not None and matched_key not in used:
            used.add(matched_key)
            out.append("")
            out.append(normalized_sections[matched_key].strip())
            out.append("")
            continue
        if any(
            item.strip() and not CHECKBOX_ANY_RE.match(item)
            for item in block
        ):
            unfilled.append(heading.group(2))
        out.extend(block)
    # sections supplied but not matched to any heading
    unmatched = sorted(set(normalized_sections) - used)
    text = "\n".join(out)
    if unfilled or unmatched:
        parts = []
        if unfilled:
            parts.append("unfilled headings: " + ", ".join(unfilled))
        if unmatched:
            parts.append("unmatched sections: " + ", ".join(unmatched))
        text += "\n\n<!-- aisbench-devflow:unfilled-sections: " + "; ".join(parts) + " -->"
    return text


def ensure_trailer(body: str) -> str:
    stripped = body.rstrip()
    count = stripped.count(TRAILER)
    if count == 1:
        return stripped + "\n"
    if count > 1:
        stripped = stripped.replace(TRAILER, "").rstrip()
    return stripped + "\n\n" + TRAILER + "\n"


def cmd_render(args: argparse.Namespace) -> int:
    template = read_template(Path(args.template))
    options = template_options(template)
    option = option_for_type(options, args.pr_type)
    label_map = load_label_map(Path(args.workflow))

    solution = tick_option(template, option)
    if args.issue is not None:
        solution = set_issue_line(solution, args.issue)
    sections = {}
    if args.sections:
        try:
            sections = json.loads(Path(args.sections).read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise BodyError(f"sections file is not valid JSON: {exc}") from exc
        if not isinstance(sections, dict):
            raise BodyError("sections file must contain a JSON object of {heading: text}")
    solution = fill_sections(solution, sections)
    solution = ensure_trailer(solution)

    predicted = label_map.get(option)
    report = {
        "schema": SCHEMA,
        "pr_type": args.pr_type,
        "template_option": option,
        "predicted_label": predicted,
        "label_mapping_ok": predicted is not None,
        "issue": args.issue,
        "trailer": TRAILER,
    }
    if predicted is None:
        report["warning"] = (
            f"pr_template_label_drift: 模板选项 {option!r} 不在 auto_label 映射中，"
            "该类型不会自动获得标签；保留模板文案并人工确认，不要改写模板"
        )
    if args.out:
        Path(args.out).write_text(solution, encoding="utf-8", newline="\n")
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.print_body:
        print(solution)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    path = Path(args.body_file)
    if not path.is_file():
        raise BodyError(f"body file not found: {path}")
    body = path.read_text(encoding="utf-8-sig", errors="replace")
    label_map = load_label_map(Path(args.workflow))

    checks: dict[str, dict] = {}
    trailer_count = body.count(TRAILER)
    checks["trailer"] = {
        "status": "PASS" if trailer_count == 1 else "FAIL",
        "value": trailer_count,
        "remedy": "" if trailer_count == 1 else "正文末尾必须恰好出现一次 " + TRAILER,
    }
    if trailer_count == 0:
        checks["trailer"]["remedy"] = "正文末尾追加一行 " + TRAILER

    ticked = [m.group(1).strip() for line in body.splitlines() if (m := CHECKBOX_TICKED_RE.match(line))]
    if len(ticked) == 1:
        checks["pr_type_selection"] = {"status": "PASS", "value": ticked}
    else:
        checks["pr_type_selection"] = {
            "status": "FAIL",
            "value": ticked,
            "remedy": f"PR Type 必须且只能勾选一项，当前 {len(ticked)} 项",
        }

    if ticked:
        labels = []
        unmapped = []
        for text in ticked:
            label = label_map.get(text)
            if label:
                labels.append(label)
            else:
                unmapped.append(text)
        checks["label_prediction"] = {
            "status": "FAIL" if unmapped else "PASS",
            "value": {"labels": labels, "unmapped": unmapped},
            "remedy": (
                "所选类型不能命中 auto_label 映射（模板文案与 workflow 不一致）；"
                "保留模板文案并人工确认标签，不要为迎合脚本改写模板"
                if unmapped
                else ""
            ),
        }
    else:
        checks["label_prediction"] = {"status": "UNKNOWN", "reason": "未发现已勾选类型"}

    if args.issue is not None:
        numbers = {int(n) for n in ISSUE_REF_RE.findall(body)}
        checks["issue_link"] = {
            "status": "PASS" if args.issue in numbers else "FAIL",
            "value": {"expected": args.issue, "found": sorted(numbers)},
            "remedy": "" if args.issue in numbers else f"正文加入 `Fixes #{args.issue}`",
        }
    else:
        numbers = {int(n) for n in ISSUE_REF_RE.findall(body)}
        checks["issue_link"] = {
            "status": "PASS" if numbers else "FAIL",
            "value": {"found": sorted(numbers)},
            "remedy": "" if numbers else "正文需关联 Issue，例如 `Fixes #123`",
        }

    if args.title:
        checks["title_format"] = {
            "status": "PASS" if TITLE_RE.match(args.title) else "FAIL",
            "value": args.title,
            "remedy": "" if TITLE_RE.match(args.title) else "改写为 `[type][scope]subject`",
        }
    else:
        checks["title_format"] = {"status": "UNKNOWN", "reason": "未提供 --title"}

    exit_code = 1 if any(row["status"] == "FAIL" for row in checks.values()) else 0
    report = {"schema": SCHEMA, "checks": checks, "exit": exit_code}
    if args.format == "text":
        lines = [f"{name:<20}{row['status']:<10}{row.get('remedy') or row.get('reason') or ''}" for name, row in checks.items()]
        print("\n".join(lines))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pr_body.py", description="渲染/自检 AISBench PR 正文")
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="按模板渲染正文")
    render.add_argument("--template", required=True)
    render.add_argument("--workflow", required=True, help="auto_label_pr.yml 路径")
    render.add_argument("--pr-type", required=True, choices=sorted(TYPE_KEYWORD))
    render.add_argument("--issue", type=int, default=None)
    render.add_argument("--sections", default=None, help="章节内容 JSON: {标题片段: 正文}")
    render.add_argument("--out", default=None)
    render.add_argument("--print-body", action="store_true")
    render.add_argument("--format", default="json", choices=["json", "text"])
    render.set_defaults(func=cmd_render)

    check = sub.add_parser("check", help="自检已有正文")
    check.add_argument("--body-file", required=True)
    check.add_argument("--workflow", required=True)
    check.add_argument("--issue", type=int, default=None)
    check.add_argument("--title", default=None)
    check.add_argument("--format", default="json", choices=["json", "text"])
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BodyError as exc:
        print(json.dumps({"schema": SCHEMA, "error": str(exc), "exit": 2}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
