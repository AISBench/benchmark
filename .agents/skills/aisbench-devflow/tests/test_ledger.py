"""Unit tests for scripts/ledger.py (resume cache: location, atomicity, identity)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ledger  # noqa: E402
import support  # noqa: E402


def run_raw(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "ledger.py"), "--repo-root", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
def run(repo: Path, *args: str) -> tuple[int, dict]:
    proc = run_raw(repo, *args)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:  # pragma: no cover - debugging aid
        raise AssertionError(f"non-JSON output: {proc.stdout!r} {proc.stderr!r}")
    return proc.returncode, payload


class LedgerCase(unittest.TestCase):
    def setUp(self) -> None:
        self.work = support.new_workdir()
        self.repo = support.init_repo(self.work / "repo", with_profile=False)
        self.head = support.git(self.repo, "rev-parse", "HEAD").strip()

    def tearDown(self) -> None:
        support.cleanup(self.work)

    def ledger_file(self) -> Path:
        return Path(self.repo, ".git", "aisbench-devflow", "ledger.json")


class Location(LedgerCase):
    def test_path_is_inside_git_dir(self) -> None:
        code, payload = run(self.repo, "path")
        self.assertEqual(code, 0)
        self.assertEqual(Path(payload["ledger"]).resolve(), self.ledger_file().resolve())
        self.assertEqual(payload["git_dir_kind"], "directory")

    def test_worktree_uses_file_git_dir(self) -> None:
        linked = self.work / "linked"
        support.git(self.repo, "worktree", "add", "-q", "-b", "wt-branch", str(linked))
        code, payload = run(linked, "path")
        self.assertEqual(code, 0)
        self.assertEqual(payload["git_dir_kind"], "worktree-file")
        self.assertIn("worktrees", payload["ledger"])

    def test_show_without_ledger_is_not_an_error(self) -> None:
        code, payload = run(self.repo, "show")
        self.assertEqual(code, 0)
        self.assertFalse(payload["exists"])
        self.assertIsNone(payload["resume_from"])


class RoundTrip(LedgerCase):
    def test_set_and_show_persist_fields(self) -> None:
        code, payload = run(
            self.repo,
            "set",
            "--repo",
            "AISBench/benchmark",
            "--field",
            "run_id=pr-651-20261114-a1b2",
            "--field",
            "requested_action=develop",
            "--field",
            "last_completed_state=locally-verified",
            "--field",
            "issue_number=651",
        )
        self.assertEqual(code, 0)
        self.assertEqual(payload["state"]["issue_number"], 651)

        code, shown = run(self.repo, "show")
        self.assertEqual(code, 0)
        self.assertTrue(shown["exists"])
        self.assertEqual(shown["ledger"]["repo"], "AISBench/benchmark")
        self.assertEqual(shown["resume_from"], "locally-verified")
        self.assertFalse(shown["staleness"]["stale"])

    def test_head_flag_binds_current_commit(self) -> None:
        support.commit(self.repo, "second", {"app.py": "print('2')\n"}, sign=True)
        new_head = support.git(self.repo, "rev-parse", "HEAD").strip()
        code, payload = run(self.repo, "set", "--field", "run_id=r1", "--head")
        self.assertEqual(code, 0)
        self.assertEqual(payload["state"]["head_sha"], new_head)
        self.assertNotEqual(new_head, self.head)

    def test_stale_detection_after_new_commit(self) -> None:
        run(self.repo, "set", "--field", "run_id=r1", "--head")
        support.commit(self.repo, "move head", {"app.py": "print('3')\n"}, sign=True)
        code, shown = run(self.repo, "show")
        self.assertEqual(code, 0)
        self.assertTrue(shown["staleness"]["stale"])
        self.assertIn("head", shown["staleness"]["reason"])

    def test_unknown_field_is_rejected(self) -> None:
        code, payload = run(self.repo, "set", "--field", "access_token=abc")
        self.assertEqual(code, 2)
        self.assertIn("unknown or unwritable field", payload["error"])

    def test_invalid_integer_field_is_rejected(self) -> None:
        code, payload = run(self.repo, "set", "--field", "issue_number=abc")
        self.assertEqual(code, 2)
        self.assertIn("expects an integer", payload["error"])


class Gates(LedgerCase):
    def test_unknown_gate_enters_unverified(self) -> None:
        code, _ = run(self.repo, "gate", "--name", "review_present", "--status", "UNKNOWN")
        self.assertEqual(code, 0)
        _, shown = run(self.repo, "show")
        self.assertIn("review_present", shown["unverified"])
        self.assertEqual(shown["ledger"]["gates"]["review_present"]["status"], "UNKNOWN")

    def test_pass_gate_clears_unverified(self) -> None:
        run(self.repo, "gate", "--name", "size", "--status", "UNKNOWN")
        code, _ = run(self.repo, "gate", "--name", "size", "--status", "PASS", "--evidence", "412 lines")
        self.assertEqual(code, 0)
        _, shown = run(self.repo, "show")
        self.assertNotIn("size", shown["unverified"])
        self.assertEqual(shown["ledger"]["gates"]["size"]["evidence"], "412 lines")

    def test_invalid_status_rejected(self) -> None:
        proc = run_raw(self.repo, "gate", "--name", "size", "--status", "OK")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("invalid choice", proc.stderr)
        self.assertFalse(self.ledger_file().exists())


class Integrity(LedgerCase):
    def test_corrupt_ledger_is_not_overwritten(self) -> None:
        self.ledger_file().parent.mkdir(parents=True, exist_ok=True)
        self.ledger_file().write_text("{not json", encoding="utf-8")
        code, payload = run(self.repo, "set", "--field", "run_id=r1")
        self.assertEqual(code, 1)
        self.assertIn("not valid JSON", payload["error"])
        self.assertEqual(self.ledger_file().read_text(encoding="utf-8"), "{not json")

    def test_identity_mismatch_blocks_write(self) -> None:
        run(self.repo, "set", "--field", "run_id=r1")
        other = support.init_repo(self.work / "other", with_profile=False)
        target = Path(other, ".git", "aisbench-devflow")
        target.mkdir(parents=True, exist_ok=True)
        target.joinpath("ledger.json").write_text(self.ledger_file().read_text(encoding="utf-8"), encoding="utf-8")
        code, payload = run(other, "set", "--field", "run_id=r2")
        self.assertEqual(code, 1)
        self.assertIn("identity mismatch", payload["error"])

    def test_reinit_replaces_foreign_ledger(self) -> None:
        run(self.repo, "set", "--field", "run_id=r1")
        other = support.init_repo(self.work / "other", with_profile=False)
        target = Path(other, ".git", "aisbench-devflow")
        target.mkdir(parents=True, exist_ok=True)
        target.joinpath("ledger.json").write_text(self.ledger_file().read_text(encoding="utf-8"), encoding="utf-8")
        code, payload = run(other, "set", "--field", "run_id=r2", "--reinit")
        self.assertEqual(code, 0)
        self.assertEqual(payload["state"]["run_id"], "r2")

    def test_lock_blocks_concurrent_write(self) -> None:
        run(self.repo, "set", "--field", "run_id=r1")
        lock = self.ledger_file().with_name("ledger.lock")
        lock.write_text("pid=1\n", encoding="utf-8")
        code, payload = run(self.repo, "set", "--field", "run_id=r2")
        self.assertEqual(code, 1)
        self.assertIn("holds the ledger lock", payload["error"])
        self.assertEqual(lock.read_text(encoding="utf-8"), "pid=1\n")

    def test_stale_lock_requires_explicit_break(self) -> None:
        run(self.repo, "set", "--field", "run_id=r1")
        lock = self.ledger_file().with_name("ledger.lock")
        lock.write_text("pid=1\n", encoding="utf-8")
        code, _ = run(self.repo, "set", "--field", "run_id=r2", "--break-lock")
        self.assertEqual(code, 1)  # fresh lock is not stale, breaking is refused

    def test_no_temp_files_left_behind(self) -> None:
        run(self.repo, "set", "--field", "run_id=r1")
        leftovers = list(self.ledger_file().parent.glob("*.tmp"))
        self.assertEqual(leftovers, [])
        self.assertFalse(self.ledger_file().with_name("ledger.lock").exists())

    def test_clear_requires_confirmation(self) -> None:
        run(self.repo, "set", "--field", "run_id=r1")
        code, payload = run(self.repo, "clear")
        self.assertEqual(code, 2)
        self.assertIn("--yes", payload["error"])
        self.assertTrue(self.ledger_file().exists())

        code, payload = run(self.repo, "clear", "--yes")
        self.assertEqual(code, 0)
        self.assertTrue(payload["removed"])
        self.assertFalse(self.ledger_file().exists())


class Hygiene(LedgerCase):
    def test_ledger_contains_no_absolute_paths(self) -> None:
        run(self.repo, "set", "--repo", "AISBench/benchmark", "--field", "run_id=r1", "--head")
        text = self.ledger_file().read_text(encoding="utf-8")
        self.assertNotIn(str(self.repo).replace("\\", "/"), text)
        self.assertNotIn(str(self.repo), text)
        self.assertNotIn("dsh-", text)

    def test_evidence_may_mention_token_word_without_being_rejected(self) -> None:
        code, _ = run(
            self.repo,
            "gate",
            "--name",
            "review_present",
            "--status",
            "UNKNOWN",
            "--evidence",
            "no token was read or printed",
        )
        self.assertEqual(code, 0)
        _, shown = run(self.repo, "show")
        self.assertEqual(
            shown["ledger"]["gates"]["review_present"]["evidence"], "no token was read or printed"
        )

    def test_schema_guards_shape(self) -> None:
        with self.assertRaises(ledger.LedgerError):
            ledger.check_payload_shape({"schema": 99})
        with self.assertRaises(ledger.LedgerError):
            ledger.check_payload_shape({"schema": 1})
        with self.assertRaises(ledger.LedgerError):
            ledger.check_payload_shape(
                {"schema": 1, "run_id": "r", "repo": "a/b", "updated_at": "t", "head_sha": "s",
                 "last_completed_state": "x", "api_token": "leak"}
            )


class ResumeSemantics(LedgerCase):
    def test_resume_reports_state_and_unverified(self) -> None:
        run(
            self.repo,
            "set",
            "--field",
            "run_id=r1",
            "--field",
            "last_completed_state=pr-open",
            "--field",
            "requested_action=ci",
        )
        run(self.repo, "gate", "--name", "rtd_and_smoke", "--status", "UNKNOWN")
        _, shown = run(self.repo, "show")
        self.assertEqual(shown["resume_from"], "pr-open")
        self.assertEqual(shown["ledger"]["requested_action"], "ci")
        self.assertEqual(shown["unverified"], ["rtd_and_smoke"])


if __name__ == "__main__":
    unittest.main()
