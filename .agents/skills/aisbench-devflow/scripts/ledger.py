#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session ledger for aisbench-devflow.

The ledger is a **cache for resuming**, never an authority: every action re-reads
git and remote facts, then cross-checks this file.  It therefore contains only
identifiers and verdicts — no tokens, no credentials, no absolute paths, no logs.

Storage location follows the real Git directory, resolved with
``git rev-parse --absolute-git-dir`` so worktrees (where ``.git`` is a file) and
submodules behave correctly:

    <absolute-git-dir>/aisbench-devflow/ledger.json
    <absolute-git-dir>/aisbench-devflow/ledger.lock
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:  # single source of truth for verdicts
    from gate_check import FAIL, NOT_APPLICABLE, PASS, UNKNOWN
except ImportError:  # pragma: no cover - standalone fallback
    PASS, FAIL, UNKNOWN, NOT_APPLICABLE = "PASS", "FAIL", "UNKNOWN", "NOT_APPLICABLE"

SCHEMA = 1
VERDICTS = (PASS, FAIL, UNKNOWN, NOT_APPLICABLE)
SUBDIR = "aisbench-devflow"
LEDGER_NAME = "ledger.json"
LOCK_NAME = "ledger.lock"
LOCK_STALE_SECONDS = 3600

# Fields a caller may set; anything else is rejected to keep the file honest.
SCALAR_FIELDS = {
    "run_id": str,
    "mode": str,
    "repo": str,
    "canonical_default_branch": str,
    "fork_remote": str,
    "work_branch": str,
    "issue_number": int,
    "pr_number": int,
    "head_sha": str,
    "review_head_sha": str,
    "requested_action": str,
    "last_completed_state": str,
}
REQUIRED_FIELDS = ("run_id", "repo", "updated_at", "head_sha", "last_completed_state")
FORBIDDEN_SUBSTRINGS = ("token", "password", "secret", "authorization", "credential", "private_key")


class LedgerError(RuntimeError):
    """Environment, identity or integrity problem that must stop the caller."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class LedgerNotFound(LedgerError):
    """No ledger has been written for this work tree yet."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def git(repo_root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise LedgerError(f"cannot run git in {repo_root}: {exc}", 2) from exc
    if proc.returncode != 0:
        raise LedgerError(f"git {' '.join(args)} failed: {proc.stderr.strip()}", 2)
    return proc.stdout


def resolve_git_dir(repo_root: Path) -> Path:
    out = git(repo_root, "rev-parse", "--absolute-git-dir").strip()
    if not out:
        raise LedgerError("cannot resolve the Git directory", 2)
    path = Path(out)
    if not path.is_dir():
        raise LedgerError("resolved Git directory does not exist", 2)
    return path


def ledger_dir(git_dir: Path) -> Path:
    return git_dir / SUBDIR


def ledger_path(git_dir: Path) -> Path:
    return ledger_dir(git_dir) / LEDGER_NAME


def identity_of(git_dir: Path) -> str:
    """Stable identifier for this work tree without storing the absolute path."""
    import hashlib

    normalized = os.path.normcase(os.path.realpath(str(git_dir)))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def empty_ledger(git_dir: Path, repo_root: Path, repo: str | None) -> dict:
    head = None
    try:
        head = git(repo_root, "rev-parse", "HEAD").strip() or None
    except LedgerError:
        head = None
    return {
        "schema": SCHEMA,
        "identity": identity_of(git_dir),
        "run_id": None,
        "mode": "guided",
        "repo": repo,
        "canonical_default_branch": None,
        "fork_remote": None,
        "work_branch": None,
        "issue_number": None,
        "pr_number": None,
        "head_sha": head,
        "review_head_sha": None,
        "requested_action": None,
        "last_completed_state": None,
        "gates": {},
        "unverified": [],
        "updated_at": utc_now(),
    }


def _forbidden_keys(node, trail: str = "") -> list[str]:
    """Recursively collect key names that look like secrets.

    Only key names are checked: evidence text may legitimately mention words such
    as "token" when explaining that no token was read.
    """
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = str(key).lower()
            if any(needle in lowered for needle in FORBIDDEN_SUBSTRINGS):
                found.append(f"{trail}{key}")
            found.extend(_forbidden_keys(value, f"{trail}{key}."))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_forbidden_keys(value, f"{trail}{index}."))
    return found


def check_payload_shape(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise LedgerError("ledger must be a JSON object")
    if payload.get("schema") != SCHEMA:
        raise LedgerError(f"unsupported ledger schema: {payload.get('schema')!r} (expected {SCHEMA})")
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise LedgerError("ledger is missing required fields: " + ", ".join(missing))
    bad_keys = _forbidden_keys(payload)
    if bad_keys:
        raise LedgerError("ledger contains forbidden field names: " + ", ".join(sorted(bad_keys)))


def read_ledger(git_dir: Path) -> dict:
    path = ledger_path(git_dir)
    if not path.is_file():
        raise LedgerNotFound("ledger does not exist yet; run `set` first", 1)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise LedgerError(f"ledger is not valid JSON ({exc}); refusing to overwrite it") from exc
    check_payload_shape(payload)
    stored = payload.get("identity")
    current = identity_of(git_dir)
    if stored and stored != current:
        raise LedgerError(
            "ledger belongs to another work tree (identity mismatch); use --reinit to replace it"
        )
    return payload


class Lock:
    """Exclusive lock file next to the ledger; never silently overwrites."""

    def __init__(self, git_dir: Path, break_stale: bool = False) -> None:
        self.path = ledger_dir(git_dir) / LOCK_NAME
        self.break_stale = break_stale
        self.fd: int | None = None

    def __enter__(self) -> "Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = time.time() - self.path.stat().st_mtime if self.path.exists() else 0
            if self.break_stale and age > LOCK_STALE_SECONDS:
                self.path.unlink(missing_ok=True)
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            else:
                raise LedgerError(
                    f"another run holds the ledger lock ({self.path.name}, age {int(age)}s); "
                    "wait, or pass --break-lock when it is provably stale"
                )
        os.write(self.fd, f"pid={os.getpid()} at={utc_now()}\n".encode("utf-8"))
        return self

    def __exit__(self, *_exc) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.path.unlink(missing_ok=True)


def write_ledger(git_dir: Path, payload: dict) -> Path:
    payload["schema"] = SCHEMA
    payload["identity"] = identity_of(git_dir)
    payload["updated_at"] = utc_now()
    check_payload_shape(payload)
    path = ledger_path(git_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{LEDGER_NAME}.{os.getpid()}.tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def current_facts(repo_root: Path) -> dict:
    def safe(*args: str) -> str | None:
        try:
            return git(repo_root, *args).strip() or None
        except LedgerError:
            return None

    return {
        "head_sha": safe("rev-parse", "HEAD"),
        "branch": safe("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(safe("status", "--porcelain")),
    }


def staleness(payload: dict, facts: dict) -> dict:
    stored = payload.get("head_sha")
    current = facts.get("head_sha")
    if not stored or not current:
        return {"stale": None, "reason": "缺少 head_sha，无法比较"}
    if stored != current:
        return {
            "stale": True,
            "reason": f"ledger head {stored[:12]} != current head {current[:12]}",
            "stale_gates": sorted(payload.get("gates", {})),
        }
    return {"stale": False, "reason": "head 一致"}


def coerce(field: str, raw: str):
    if raw == "null":
        return None
    expected = SCALAR_FIELDS[field]
    if expected is int:
        try:
            return int(raw)
        except ValueError as exc:
            raise LedgerError(f"{field} expects an integer, got {raw!r}", 2) from exc
    return raw


def cmd_path(args: argparse.Namespace) -> int:
    git_dir = resolve_git_dir(Path(args.repo_root).resolve())
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "git_dir_kind": "worktree-file" if (Path(args.repo_root) / ".git").is_file() else "directory",
                "ledger": str(ledger_path(git_dir).as_posix()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    git_dir = resolve_git_dir(repo_root)
    facts = current_facts(repo_root)
    try:
        payload = read_ledger(git_dir)
    except LedgerNotFound:
        report = {
            "schema": SCHEMA,
            "exists": False,
            "facts": facts,
            "resume_from": None,
            "note": "尚无账本；按事实从 action 起点重新开始",
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    report = {
        "schema": SCHEMA,
        "exists": True,
        "ledger": payload,
        "facts": facts,
        "staleness": staleness(payload, facts),
        "resume_from": payload.get("last_completed_state"),
        "unverified": payload.get("unverified", []),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _load_base(git_dir: Path, repo_root: Path, args: argparse.Namespace) -> dict:
    try:
        return read_ledger(git_dir)
    except LedgerNotFound:
        return empty_ledger(git_dir, repo_root, getattr(args, "repo", None))
    except LedgerError:
        if getattr(args, "reinit", False):
            return empty_ledger(git_dir, repo_root, getattr(args, "repo", None))
        raise


def cmd_set(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    git_dir = resolve_git_dir(repo_root)
    with Lock(git_dir, break_stale=args.break_lock):
        payload = _load_base(git_dir, repo_root, args)
        if args.repo:
            payload["repo"] = args.repo
        if args.head:
            payload["head_sha"] = current_facts(repo_root).get("head_sha")
        for item in args.field or []:
            if "=" not in item:
                raise LedgerError(f"--field expects key=value, got {item!r}", 2)
            key, _, raw = item.partition("=")
            key = key.strip()
            if key not in SCALAR_FIELDS:
                raise LedgerError(
                    f"unknown or unwritable field {key!r}; allowed: {', '.join(sorted(SCALAR_FIELDS))}",
                    2,
                )
            payload[key] = coerce(key, raw.strip())
        if args.patch:
            patch = json.loads(Path(args.patch).read_text(encoding="utf-8-sig"))
            if not isinstance(patch, dict):
                raise LedgerError("--patch file must contain a JSON object", 2)
            for key, value in patch.items():
                if key not in SCALAR_FIELDS:
                    raise LedgerError(f"patch contains unwritable field {key!r}", 2)
                payload[key] = value
        if args.unverified is not None:
            payload["unverified"] = sorted({item.strip() for item in args.unverified.split(",") if item.strip()})
        path = write_ledger(git_dir, payload)
    print(json.dumps({"schema": SCHEMA, "written": path.as_posix(), "state": payload}, ensure_ascii=False, indent=2))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    git_dir = resolve_git_dir(repo_root)
    if args.status not in VERDICTS:
        raise LedgerError(f"status must be one of {', '.join(VERDICTS)}", 2)
    facts = current_facts(repo_root)
    with Lock(git_dir, break_stale=args.break_lock):
        payload = _load_base(git_dir, repo_root, args)
        if facts.get("head_sha"):
            payload["head_sha"] = facts["head_sha"]
        payload.setdefault("gates", {})
        payload["gates"][args.name] = {
            "status": args.status,
            "evidence": args.evidence,
            "head_sha": facts.get("head_sha"),
            "at": utc_now(),
        }
        if args.status == UNKNOWN:
            unverified = set(payload.get("unverified", []))
            unverified.add(args.name)
            payload["unverified"] = sorted(unverified)
        else:
            payload["unverified"] = [item for item in payload.get("unverified", []) if item != args.name]
        path = write_ledger(git_dir, payload)
    print(
        json.dumps(
            {"schema": SCHEMA, "written": path.as_posix(), "gate": args.name, "status": args.status},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    if not args.yes:
        raise LedgerError("clearing the ledger requires --yes", 2)
    git_dir = resolve_git_dir(Path(args.repo_root).resolve())
    path = ledger_path(git_dir)
    removed = path.is_file()
    if removed:
        path.unlink()
    print(json.dumps({"schema": SCHEMA, "removed": removed, "path": path.as_posix()}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ledger.py", description="aisbench-devflow 恢复账本（仅本地）")
    parser.add_argument("--repo-root", default=".", help="仓库工作树路径")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("path", help="打印账本与 Git 目录的解析结果").set_defaults(func=cmd_path)

    show = sub.add_parser("show", help="显示账本并与当前事实交叉校验")
    show.set_defaults(func=cmd_show)

    set_cmd = sub.add_parser("set", help="写入字段（原子写 + 加锁）")
    set_cmd.add_argument("--field", action="append", metavar="key=value")
    set_cmd.add_argument("--patch", help="JSON 补丁文件")
    set_cmd.add_argument("--repo", help="canonical 仓库 owner/repo")
    set_cmd.add_argument("--head", action="store_true", help="用当前 HEAD 刷新 head_sha")
    set_cmd.add_argument("--unverified", help="逗号分隔的未验证项")
    set_cmd.add_argument("--reinit", action="store_true", help="身份不匹配时重建账本")
    set_cmd.add_argument("--break-lock", action="store_true", help="锁已明显过期时强制接管")
    set_cmd.set_defaults(func=cmd_set)

    gate = sub.add_parser("gate", help="记录单条门禁结论")
    gate.add_argument("--name", required=True)
    gate.add_argument("--status", required=True, choices=list(VERDICTS))
    gate.add_argument("--evidence", default=None)
    gate.add_argument("--reinit", action="store_true")
    gate.add_argument("--break-lock", action="store_true")
    gate.set_defaults(func=cmd_gate)

    clear = sub.add_parser("clear", help="删除账本")
    clear.add_argument("--yes", action="store_true")
    clear.set_defaults(func=cmd_clear)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LedgerError as exc:
        print(json.dumps({"schema": SCHEMA, "error": str(exc), "exit": exc.exit_code}, ensure_ascii=False))
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
