"""Unit tests for scripts/devflow.py (action driver: judge + persist + next step)."""

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

import support  # noqa: E402

TRAILER = "<!-- aisbench-devflow -->"
GOOD_BODY = f"- [x] Feature（功能新增）\n\nFixes #651\n\n{TRAILER}\n"


def devflow(repo: Path, *args: str) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "devflow.py"), args[0], "--repo-root", str(repo), *args[1:]],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:  # pragma: no cover - debugging aid
        raise AssertionError(f"non-JSON output: {proc.stdout!r} {proc.stderr!r}")
    return proc.returncode, payload


class DriverCase(unittest.TestCase):
    def setUp(self) -> None:
        self.work = support.new_workdir()
        self.repo = support.init_repo(self.work / "repo")
        self.base = support.git(self.repo, "rev-parse", "HEAD").strip()

    def tearDown(self) -> None:
        support.cleanup(self.work)

    def ledger_file(self) -> Path:
        return self.repo / ".git" / "aisbench-devflow" / "ledger.json"

    def write_body(self, text: str) -> str:
        path = self.work / "body.md"
        path.write_text(text, encoding="utf-8")
        return str(path)


class Develop(DriverCase):
    def test_feature_without_test_is_blocked(self) -> None:
        support.commit(self.repo, "feat no test", {"app.py": "print('a')\n"}, sign=True)
        code, payload = devflow(self.repo, "develop", "--base", self.base, "--pr-type", "feature")
        self.assertEqual(code, 1)
        self.assertFalse(payload["advanced"])
        gates = {b["gate"] for b in payload["blockers"]}
        self.assertIn("ut_changed", gates)
        self.assertIn("tests/UT/", json.dumps(payload, ensure_ascii=False))

    def test_feature_with_test_advances(self) -> None:
        support.commit(
            self.repo,
            "feat with test",
            {"app.py": "print('b')\n", "tests/UT/datasets/test_app.py": "def test_ok():\n    assert True\n"},
            sign=True,
        )
        code, payload = devflow(self.repo, "develop", "--base", self.base, "--pr-type", "feature")
        self.assertEqual(code, 0)
        self.assertTrue(payload["advanced"])
        self.assertEqual(payload["state"], "locally-verified")
        self.assertEqual(payload["next_action"], "pr")
        self.assertTrue(self.ledger_file().is_file())

    def test_docs_change_requires_explicit_content_confirmation(self) -> None:
        support.commit(
            self.repo,
            "docs pair",
            {"docs/source_zh_cn/x.md": "# x\n", "docs/source_en/x.md": "# x\n"},
            sign=True,
        )
        code, payload = devflow(self.repo, "develop", "--base", self.base, "--pr-type", "docs")
        self.assertEqual(code, 1)
        self.assertIn("docs_content", {b["gate"] for b in payload["blockers"]})

        code, payload = devflow(
            self.repo,
            "develop",
            "--base",
            self.base,
            "--pr-type",
            "docs",
            "--confirm",
            "docs_content=逐段核对中英一致",
        )
        self.assertEqual(code, 0)
        ledger_text = self.ledger_file().read_text(encoding="utf-8")
        self.assertIn("逐段核对中英一致", ledger_text)
        self.assertIn('"confirmed": true', ledger_text)

    def test_dry_run_does_not_touch_ledger(self) -> None:
        support.commit(
            self.repo,
            "feat",
            {"app.py": "print('c')\n", "tests/UT/datasets/test_app.py": "def test_ok():\n    assert True\n"},
            sign=True,
        )
        code, payload = devflow(
            self.repo, "develop", "--base", self.base, "--pr-type", "feature", "--dry-run"
        )
        self.assertEqual(code, 0)
        self.assertTrue(payload["dry_run"])
        self.assertIsNone(payload["ledger"])
        self.assertFalse(self.ledger_file().exists())

    def test_suggests_missing_test_path(self) -> None:
        support.commit(self.repo, "touch dataset", {"ais_bench/benchmark/datasets/hle.py": "x = 1\n"}, sign=True)
        code, payload = devflow(self.repo, "develop", "--base", self.base, "--pr-type", "feature")
        self.assertEqual(code, 1)
        suggestions = payload.get("ut_suggestions", [])
        self.assertTrue(suggestions)
        self.assertEqual(suggestions[0]["suggested_test_file"], "tests/UT/datasets/test_hle.py")


class PrAction(DriverCase):
    def setUp(self) -> None:
        super().setUp()
        support.commit(
            self.repo,
            "feat with test",
            {"app.py": "print('d')\n", "tests/UT/datasets/test_app.py": "def test_ok():\n    assert True\n"},
            sign=True,
        )

    def test_advances_when_every_gate_holds(self) -> None:
        body = self.write_body(GOOD_BODY)
        code, payload = devflow(
            self.repo,
            "pr",
            "--base",
            self.base,
            "--pr-type",
            "feature",
            "--issue",
            "651",
            "--title",
            "[feat][datasets]支持 X",
            "--body-file",
            body,
        )
        self.assertEqual(code, 0)
        self.assertEqual(payload["state"], "pr-open")
        self.assertEqual(payload["next_action"], "ci")

    def test_oversize_pr_is_blocked(self) -> None:
        support.commit(self.repo, "grow", {"big.txt": "\n".join(str(i) for i in range(1200)) + "\n"}, sign=True)
        body = self.write_body(GOOD_BODY)
        code, payload = devflow(
            self.repo,
            "pr",
            "--base",
            self.base,
            "--pr-type",
            "feature",
            "--issue",
            "651",
            "--title",
            "[feat][datasets]支持 X",
            "--body-file",
            body,
        )
        self.assertEqual(code, 1)
        self.assertIn("size", {b["gate"] for b in payload["blockers"]})

    def test_missing_trailer_blocks(self) -> None:
        body = self.write_body("- [x] Feature（功能新增）\n\nFixes #651\n")
        code, payload = devflow(
            self.repo,
            "pr",
            "--base",
            self.base,
            "--pr-type",
            "feature",
            "--issue",
            "651",
            "--title",
            "[feat][datasets]支持 X",
            "--body-file",
            body,
        )
        self.assertEqual(code, 1)
        self.assertIn("body_trailer", {b["gate"] for b in payload["blockers"]})

    def test_two_ticked_types_block(self) -> None:
        body = self.write_body(f"- [x] Feature（功能新增）\n- [x] Docs（文档更新）\n\nFixes #651\n\n{TRAILER}\n")
        code, payload = devflow(
            self.repo,
            "pr",
            "--base",
            self.base,
            "--pr-type",
            "feature",
            "--issue",
            "651",
            "--title",
            "[feat][datasets]支持 X",
            "--body-file",
            body,
        )
        self.assertEqual(code, 1)
        self.assertIn("body_type_selection", {b["gate"] for b in payload["blockers"]})

    def test_wrong_issue_blocks(self) -> None:
        body = self.write_body(GOOD_BODY.replace("#651", "#9"))
        code, payload = devflow(
            self.repo,
            "pr",
            "--base",
            self.base,
            "--pr-type",
            "feature",
            "--issue",
            "651",
            "--title",
            "[feat][datasets]支持 X",
            "--body-file",
            body,
        )
        self.assertEqual(code, 1)
        self.assertIn("issue_link", {b["gate"] for b in payload["blockers"]})

    def test_bad_title_blocks(self) -> None:
        body = self.write_body(GOOD_BODY)
        code, payload = devflow(
            self.repo,
            "pr",
            "--base",
            self.base,
            "--pr-type",
            "feature",
            "--issue",
            "651",
            "--title",
            "add support",
            "--body-file",
            body,
        )
        self.assertEqual(code, 1)
        self.assertIn("title_format", {b["gate"] for b in payload["blockers"]})


class Persistence(DriverCase):
    def test_ledger_records_head_and_gates(self) -> None:
        support.commit(
            self.repo,
            "feat",
            {"app.py": "print('e')\n", "tests/UT/datasets/test_app.py": "def test_ok():\n    assert True\n"},
            sign=True,
        )
        head = support.git(self.repo, "rev-parse", "HEAD").strip()
        devflow(self.repo, "develop", "--base", self.base, "--pr-type", "feature")
        data = json.loads(self.ledger_file().read_text(encoding="utf-8"))
        self.assertEqual(data["head_sha"], head)
        self.assertEqual(data["last_completed_state"], "locally-verified")
        self.assertEqual(data["requested_action"], "develop")
        for name, row in data["gates"].items():
            self.assertEqual(row["head_sha"], head, name)
            self.assertIn(row["status"], {"PASS", "FAIL", "UNKNOWN", "NOT_APPLICABLE"})

    def test_status_reports_state_and_next_action(self) -> None:
        support.commit(
            self.repo,
            "feat",
            {"app.py": "print('f')\n", "tests/UT/datasets/test_app.py": "def test_ok():\n    assert True\n"},
            sign=True,
        )
        devflow(self.repo, "develop", "--base", self.base, "--pr-type", "feature")
        code, payload = devflow(self.repo, "status")
        self.assertEqual(code, 0)
        self.assertEqual(payload["state"], "locally-verified")
        self.assertEqual(payload["next_action"], "pr")
        self.assertFalse(payload["staleness"]["stale"])

    def test_status_flags_stale_after_new_commit(self) -> None:
        support.commit(
            self.repo,
            "feat",
            {"app.py": "print('g')\n", "tests/UT/datasets/test_app.py": "def test_ok():\n    assert True\n"},
            sign=True,
        )
        devflow(self.repo, "develop", "--base", self.base, "--pr-type", "feature")
        support.commit(self.repo, "later", {"app.py": "print('h')\n"}, sign=True)
        code, payload = devflow(self.repo, "status")
        self.assertEqual(code, 0)
        self.assertTrue(payload["staleness"]["stale"])
        self.assertEqual(payload["next_action"], "re-verify")

    def test_status_without_ledger_points_to_issue(self) -> None:
        code, payload = devflow(self.repo, "status")
        self.assertEqual(code, 0)
        self.assertFalse(payload["exists"])
        self.assertEqual(payload["next_action"], "issue")

    def test_unknown_gate_confirmation_is_rejected(self) -> None:
        support.commit(
            self.repo,
            "feat",
            {"app.py": "print('i')\n", "tests/UT/datasets/test_app.py": "def test_ok():\n    assert True\n"},
            sign=True,
        )
        code, payload = devflow(
            self.repo, "develop", "--base", self.base, "--pr-type", "feature", "--confirm", "nope=1"
        )
        self.assertEqual(code, 2)
        self.assertIn("cannot confirm unknown gate", payload["error"])


if __name__ == "__main__":
    unittest.main()
