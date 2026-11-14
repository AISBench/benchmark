#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Action driver: judge an action boundary, persist it to the ledger, say what is next.

This is the glue between the judging scripts and the resume cache.  It never
performs remote writes: pushing, opening a PR, commenting and merging stay with
the agent/human as described in SKILL.md.

    python devflow.py develop --pr-type feature --issue 651
    python devflow.py pr --pr-type feature --issue 651 --title "..." --body-file body.md
    python devflow.py status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import gate_check
    import ledger as ledger_mod
    import pr_body
    import ut_map
except ImportError as exc:  # pragma: no cover - packaging guard
    raise SystemExit(f"aisbench-devflow scripts must stay in one directory: {exc}")

SCHEMA = 1

# Which gates must hold before an action may advance to its completion state.
# Anything not listed here is recorded for the report but does not block.
REQUIRED_GATES = {
    "locally-verified": ("ut_changed", "docs_parity", "docs_content"),
    "pr-open": (
        "size",
        "issue_link",
        "ut_changed",
        "docs_parity",
        "docs_content",
        "title_format",
        "labels",
        "dco",
    ),
}

NEXT_ACTION = {
    None: "issue",
    "discovered": "issue",
    "verified": "plan",
    "planned": "develop",
    "locally-verified": "pr",
    "pr-open": "ci",
    "ci-passed": "feedback-or-wait",
    "waiting-for-review": "feedback",
    "feedback-resolved": "ci",
    "review-approved": "merge",
    "ready-to-merge": "human-merge",
}

GATE_EVIDENCE_KEYS = ("evidence", "reason", "remedy")


def summarize_gate(row: dict) -> str:
    for key in GATE_EVIDENCE_KEYS:
        if row.get(key):
            return str(row[key])[:200]
    value = row.get("value")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)[:200]
    return str(value)[:200] if value is not None else ""


def parse_confirmations(items: list[str] | None) -> dict[str, str]:
    confirmations: dict[str, str] = {}
    for item in items or []:
        name, _, evidence = item.partition("=")
        name = name.strip()
        if not name:
            raise ValueError("--confirm 需要 gate 名称，例如 --confirm docs_content")
        confirmations[name] = evidence.strip() or "caller confirmation"
    return confirmations


def load_ledger(git_dir: Path, repo_root: Path, repo: str | None) -> dict:
    try:
        return ledger_mod.read_ledger(git_dir)
    except ledger_mod.LedgerNotFound:
        return ledger_mod.empty_ledger(git_dir, repo_root, repo)


def record(
    git_dir: Path,
    payload: dict,
    report: dict,
    *,
    action: str,
    state: str | None,
    issue: int | None,
    pr_number: int | None = None,
    dry_run: bool,
) -> str | None:
    payload["requested_action"] = action
    if issue is not None:
        payload["issue_number"] = issue
    if pr_number is not None:
        payload["pr_number"] = pr_number
    head = report.get("range", {}).get("head")
    if head:
        payload["head_sha"] = head
    if state:
        payload["last_completed_state"] = state
    payload["unverified"] = list(report.get("unverified", []))
    gates = payload.setdefault("gates", {})
    for name, row in report.get("gates", {}).items():
        entry = {
            "status": row["status"],
            "evidence": summarize_gate(row),
            "head_sha": head,
            "at": report.get("generated_at"),
        }
        if row.get("confirmed"):
            entry["confirmed"] = True
        gates[name] = entry
    if dry_run:
        return None
    with ledger_mod.Lock(git_dir):
        written = ledger_mod.write_ledger(git_dir, payload)
    return written.as_posix()


def body_checks(args: argparse.Namespace, repo_root: Path) -> dict[str, dict]:
    """Run pr_body checks when a rendered body exists; return name -> verdict row."""
    if not args.body_file:
        return {}
    body_path = Path(args.body_file)
    if not body_path.is_file():
        raise SystemExit(json.dumps({"schema": SCHEMA, "error": f"body file not found: {body_path}"}, ensure_ascii=False))
    workflow = Path(args.workflow)
    if not workflow.is_absolute():
        workflow = repo_root / workflow
    body = body_path.read_text(encoding="utf-8-sig", errors="replace")
    label_map = pr_body.load_label_map(workflow)
    results: dict[str, dict] = {}

    trailer = body.count(pr_body.TRAILER)
    results["body_trailer"] = {"status": "PASS" if trailer == 1 else "FAIL", "evidence": f"trailer x{trailer}"}

    ticked = [m.group(1).strip() for line in body.splitlines() if (m := pr_body.CHECKBOX_TICKED_RE.match(line))]
    results["body_type_selection"] = {
        "status": "PASS" if len(ticked) == 1 else "FAIL",
        "evidence": f"{len(ticked)} type(s) ticked",
    }

    unmapped = [text for text in ticked if text not in label_map]
    results["body_label_mapping"] = {
        "status": "FAIL" if unmapped else ("PASS" if ticked else "UNKNOWN"),
        "evidence": "unmapped: " + ", ".join(unmapped) if unmapped else "all ticked options map to a label",
    }
    return results


def merge_body_checks(report: dict, extras: dict[str, dict]) -> None:
    """Fold PR-body findings into the report so they block like any other gate."""
    for name, row in extras.items():
        report["gates"][name] = gate_check.gate(row["status"], evidence=row.get("evidence"))
    report["unverified"] = sorted(n for n, r in report["gates"].items() if r["status"] == gate_check.UNKNOWN)
    report["promotion_blockers"] = [
        {"gate": n, "status": r["status"], "reason": r.get("reason") or r.get("remedy") or r.get("evidence") or ""}
        for n, r in report["gates"].items()
        if r["status"] in (gate_check.FAIL, gate_check.UNKNOWN)
    ]
    report["exit"] = 1 if any(r["status"] == gate_check.FAIL for r in report["gates"].values()) else 0


def evaluate(report: dict, target_state: str) -> list[dict]:
    """Blockers = required gates that are not satisfied + anything that outright failed.

    A known FAIL always blocks advancement, even for gates that are merely
    advisory for this state (for example a malformed PR body).
    """
    blockers: list[dict] = []
    seen: set[str] = set()
    for name in REQUIRED_GATES[target_state]:
        row = report["gates"].get(name)
        if row is None:
            blockers.append({"gate": name, "status": "MISSING", "reason": "门禁未产生结论"})
            seen.add(name)
            continue
        if row["status"] in (gate_check.FAIL, gate_check.UNKNOWN):
            blockers.append(
                {
                    "gate": name,
                    "status": row["status"],
                    "reason": row.get("remedy") or row.get("reason") or summarize_gate(row),
                }
            )
            seen.add(name)
    for name, row in report["gates"].items():
        if name in seen or row["status"] != gate_check.FAIL:
            continue
        blockers.append(
            {
                "gate": name,
                "status": row["status"],
                "reason": row.get("remedy") or row.get("reason") or summarize_gate(row),
            }
        )
    return blockers


def changed_sources(repo_root: Path, base: str | None) -> list[str]:
    try:
        repo = gate_check.Repo(repo_root)
        base_ref = base or repo.default_branch() or "master"
        merge_base = repo.merge_base(base_ref, "HEAD")
        return [path for status, path in repo.changed_files(merge_base, repo.rev("HEAD")) if status != "D"]
    except gate_check.GateError:
        return []


def emit(payload: dict, fmt: str) -> None:
    if fmt == "text":
        lines = [f"action: {payload.get('action')}", f"state : {payload.get('state')}"]
        for blocker in payload.get("blockers", []):
            lines.append(f"  blocker [{blocker['gate']}] {blocker['status']}: {blocker['reason']}")
        for test in payload.get("ut_suggestions", []):
            lines.append(f"  add test: {test['suggested_test_file']} (exists={test['suggested_test_file_exists']})")
        if payload.get("unverified"):
            lines.append("  unverified: " + ", ".join(payload["unverified"]))
        lines.append(f"next  : {payload.get('next_action')}")
        print("\n".join(lines))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def do_action(args: argparse.Namespace, action: str, target_state: str) -> int:
    repo_root = Path(args.repo_root).resolve()
    git_dir = ledger_mod.resolve_git_dir(repo_root)
    confirmations = parse_confirmations(args.confirm)

    options = gate_check.Options(
        base=args.base,
        head=args.head,
        pr_type=args.pr_type,
        issue=args.issue,
        title=args.title,
        pr_body_file=getattr(args, "body_file", None),
    )
    report = gate_check.run_checks(repo_root, options, confirmations)
    merge_body_checks(report, body_checks(args, repo_root) if action == "pr" else {})

    blockers = evaluate(report, target_state)
    advanced = not blockers
    payload = load_ledger(git_dir, repo_root, args.repo)
    written = record(
        git_dir,
        payload,
        report,
        action=action,
        state=target_state if advanced else None,
        issue=args.issue,
        dry_run=args.dry_run,
    )

    state = (target_state if advanced else payload.get("last_completed_state"))
    result = {
        "schema": SCHEMA,
        "action": action,
        "state": state,
        "advanced": advanced,
        "head_sha": report.get("range", {}).get("head"),
        "dirty": report.get("range", {}).get("dirty"),
        "blockers": blockers,
        "unverified": report.get("unverified", []),
        "next_action": NEXT_ACTION.get(state, "unknown"),
        "ledger": written,
        "dry_run": args.dry_run,
    }
    if action == "develop":
        sources = [p for p in changed_sources(repo_root, args.base) if p.startswith("ais_bench/")]
        if sources:
            result["ut_suggestions"] = [
                row
                for row in ut_map.suggest(sources, repo_root)["rows"]
                if row.get("supported") and not row.get("suggested_test_file_exists")
            ]
    emit(result, args.format)
    return 0 if advanced else 1


def do_status(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    git_dir = ledger_mod.resolve_git_dir(repo_root)
    facts = ledger_mod.current_facts(repo_root)
    try:
        payload = ledger_mod.read_ledger(git_dir)
        stale = ledger_mod.staleness(payload, facts)
        state = payload.get("last_completed_state")
        result = {
            "schema": SCHEMA,
            "exists": True,
            "state": state,
            "requested_action": payload.get("requested_action"),
            "head_sha": payload.get("head_sha"),
            "facts": facts,
            "staleness": stale,
            "unverified": payload.get("unverified", []),
            "next_action": NEXT_ACTION.get(state, "unknown") if not stale.get("stale") else "re-verify",
            "ledger": payload,
        }
        code = 0
    except ledger_mod.LedgerNotFound:
        result = {
            "schema": SCHEMA,
            "exists": False,
            "state": None,
            "facts": facts,
            "next_action": NEXT_ACTION[None],
            "note": "尚无账本；从 action 起点开始",
        }
        code = 0
    emit(result, args.format)
    return code


def common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--format", default="json", choices=["json", "text"])
    parser.add_argument("--dry-run", action="store_true", help="只判定，不写账本")
    parser.add_argument("--base", default=None)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--pr-type", default=None, choices=list(gate_check.PR_TYPES))
    parser.add_argument("--issue", type=int, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--body-file", default=None, help="已渲染的 PR 正文")
    parser.add_argument("--repo", default=None, help="canonical owner/repo")
    parser.add_argument(
        "--confirm",
        action="append",
        metavar="GATE[=EVIDENCE]",
        help="为无法脚本判定的门禁提供显式复核结论（如 docs_content）",
    )
    parser.add_argument("--workflow", default=".github/workflows/auto_label_pr.yml")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devflow.py", description="aisbench-devflow action 驱动（仅本地）"
    )
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("develop", "pr", "status"):
        sub.add_parser(name, parents=[common_parser()])
    return parser


def main(argv: list[str] | None = None) -> int:
    gate_check.configure_stdout()
    args = build_parser().parse_args(argv)
    try:
        if args.action == "status":
            return do_status(args)
        if args.action == "develop":
            return do_action(args, "develop", "locally-verified")
        if args.action == "pr":
            if not getattr(args, "body_file", None):
                args.body_file = None
            return do_action(args, "pr", "pr-open")
        raise SystemExit(2)
    except (gate_check.GateError, ledger_mod.LedgerError, ValueError) as exc:
        print(json.dumps({"schema": SCHEMA, "error": str(exc), "exit": 2}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
