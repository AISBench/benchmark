"""Unit tests for scripts/gate_check.py (read-only local preflight)."""

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

import gate_check  # noqa: E402
import support  # noqa: E402


def run_gate(repo: Path, *args: str) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "gate_check.py"), "--repo-root", str(repo), *args],
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


class GateTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._work = support.new_workdir()
        self.repo = self._work / "repo"
        support.init_repo(self.repo)
        self.base = support.git(self.repo, "rev-parse", "HEAD").strip()

    def tearDown(self) -> None:
        support.cleanup(self._work)

    def body_file(self, text: str) -> str:
        path = self._work / "body.md"
        path.write_text(text, encoding="utf-8")
        return str(path)


class SizeGate(GateTestCase):
    def test_small_change_passes(self) -> None:
        support.commit(self.repo, "small", {"app.py": "print('x')\n"}, sign=True)
        code, payload = run_gate(self.repo, "--base", self.base)
        self.assertEqual(code, 0)
        self.assertEqual(payload["gates"]["size"]["status"], gate_check.PASS)
        self.assertEqual(payload["gates"]["size"]["value"]["additions"], 1)

    def test_oversized_change_fails_with_remedy(self) -> None:
        big = "\n".join(f"line {index}" for index in range(1200)) + "\n"
        support.commit(self.repo, "big", {"big.txt": big})
        code, payload = run_gate(self.repo, "--base", self.base)
        self.assertEqual(code, 1)
        row = payload["gates"]["size"]
        self.assertEqual(row["status"], gate_check.FAIL)
        self.assertGreater(row["value"]["additions"], 1000)
        self.assertIn("拆分", row["remedy"])


class IssueLinkGate(GateTestCase):
    def test_missing_body_is_unknown_not_pass(self) -> None:
        support.commit(self.repo, "change", {"app.py": "print('y')\n"})
        _, payload = run_gate(self.repo, "--base", self.base)
        self.assertEqual(payload["gates"]["issue_link"]["status"], gate_check.UNKNOWN)

    def test_body_with_expected_issue_passes(self) -> None:
        support.commit(self.repo, "change", {"app.py": "print('y')\n"})
        body = self.body_file("## Motivation\n\nFixes #651\n")
        _, payload = run_gate(self.repo, "--base", self.base, "--issue", "651", "--pr-body-file", body)
        self.assertEqual(payload["gates"]["issue_link"]["status"], gate_check.PASS)

    def test_body_with_wrong_issue_fails(self) -> None:
        support.commit(self.repo, "change", {"app.py": "print('y')\n"})
        body = self.body_file("Fixes #12\n")
        code, payload = run_gate(self.repo, "--base", self.base, "--issue", "651", "--pr-body-file", body)
        self.assertEqual(code, 1)
        self.assertEqual(payload["gates"]["issue_link"]["status"], gate_check.FAIL)


class UnitTestGate(GateTestCase):
    def test_feature_without_ut_fails(self) -> None:
        support.commit(self.repo, "feat", {"app.py": "print('z')\n"})
        _, payload = run_gate(self.repo, "--base", self.base, "--pr-type", "feature")
        row = payload["gates"]["ut_changed"]
        self.assertEqual(row["status"], gate_check.FAIL)
        self.assertIn("tests/UT/", row["remedy"])

    def test_feature_with_ut_passes(self) -> None:
        support.commit(
            self.repo,
            "feat with test",
            {"app.py": "print('z')\n", "tests/UT/datasets/test_app.py": "def test_ok():\n    assert True\n"},
        )
        _, payload = run_gate(self.repo, "--base", self.base, "--pr-type", "feature")
        self.assertEqual(payload["gates"]["ut_changed"]["status"], gate_check.PASS)

    def test_docs_type_does_not_require_ut(self) -> None:
        support.commit(self.repo, "docs", {"docs/source_zh_cn/x.md": "# x\n", "docs/source_en/x.md": "# x\n"})
        _, payload = run_gate(self.repo, "--base", self.base, "--pr-type", "docs")
        self.assertEqual(payload["gates"]["ut_changed"]["status"], gate_check.NOT_APPLICABLE)

    def test_bugfix_small_change_does_not_require_ut(self) -> None:
        support.commit(self.repo, "fix", {"app.py": "print('q')\n"})
        _, payload = run_gate(self.repo, "--base", self.base, "--pr-type", "bugfix")
        self.assertEqual(payload["gates"]["ut_changed"]["status"], gate_check.NOT_APPLICABLE)


class DcoGate(GateTestCase):
    def test_missing_signoff_fails(self) -> None:
        support.commit(self.repo, "no signoff", {"app.py": "print('a')\n"})
        _, payload = run_gate(self.repo, "--base", self.base)
        row = payload["gates"]["dco"]
        self.assertEqual(row["status"], gate_check.FAIL)
        self.assertEqual(len(row["value"]["missing"]), 1)

    def test_signed_commit_passes(self) -> None:
        support.commit(self.repo, "signed", {"app.py": "print('b')\n"}, sign=True)
        _, payload = run_gate(self.repo, "--base", self.base)
        self.assertEqual(payload["gates"]["dco"]["status"], gate_check.PASS)

    def test_no_commits_is_not_applicable(self) -> None:
        _, payload = run_gate(self.repo, "--base", self.base)
        self.assertEqual(payload["gates"]["dco"]["status"], gate_check.NOT_APPLICABLE)


class TitleGate(unittest.TestCase):
    def test_accepts_documented_formats(self) -> None:
        for title in ("[feat][datasets]支持 X", "[fix]修复 Y", "[docs][readme]更新"):
            self.assertEqual(gate_check.check_title(title)["status"], gate_check.PASS, title)

    def test_rejects_unprefixed_title(self) -> None:
        row = gate_check.check_title("add new dataset")
        self.assertEqual(row["status"], gate_check.FAIL)

    def test_absent_title_is_unknown(self) -> None:
        self.assertEqual(gate_check.check_title(None)["status"], gate_check.UNKNOWN)


class LabelGate(GateTestCase):
    def test_feature_option_maps_to_label(self) -> None:
        support.commit(self.repo, "feat", {"app.py": "print('c')\n"})
        body = self.body_file("**PR Type**\n- [x] Feature（功能新增）\n")
        _, payload = run_gate(
            self.repo, "--base", self.base, "--pr-type", "feature", "--pr-body-file", body
        )
        row = payload["gates"]["labels"]
        self.assertEqual(row["status"], gate_check.PASS)
        self.assertEqual(row["value"]["predicted_labels"], ["feature"])

    def test_ci_cd_option_hits_known_drift(self) -> None:
        support.commit(self.repo, "ci", {"app.py": "print('d')\n"})
        body = self.body_file("**PR Type**\n- [x] CI/CD（持续集成/持续部署）\n")
        code, payload = run_gate(
            self.repo, "--base", self.base, "--pr-type", "ci-cd", "--pr-body-file", body
        )
        row = payload["gates"]["labels"]
        self.assertEqual(code, 1)
        self.assertEqual(row["status"], gate_check.FAIL)
        self.assertTrue(row["drift"])
        self.assertIn("不要为迎合脚本改写模板", row["remedy"])

    def test_two_ticked_types_fail(self) -> None:
        support.commit(self.repo, "feat", {"app.py": "print('e')\n"})
        body = self.body_file("- [x] Feature（功能新增）\n- [x] Docs（文档更新）\n")
        _, payload = run_gate(
            self.repo, "--base", self.base, "--pr-type", "feature", "--pr-body-file", body
        )
        self.assertEqual(payload["gates"]["labels"]["status"], gate_check.FAIL)


class DocsParityGate(GateTestCase):
    def test_single_sided_docs_change_fails(self) -> None:
        support.commit(self.repo, "zh only", {"docs/source_zh_cn/guide.md": "# 指南\n"})
        _, payload = run_gate(self.repo, "--base", self.base)
        row = payload["gates"]["docs_parity"]
        self.assertEqual(row["status"], gate_check.FAIL)
        self.assertTrue(row["value"]["missing_counterparts"])

    def test_paired_docs_change_passes_but_content_stays_unknown(self) -> None:
        support.commit(
            self.repo,
            "both languages",
            {"docs/source_zh_cn/guide.md": "# 指南\n", "docs/source_en/guide.md": "# Guide\n"},
        )
        _, payload = run_gate(self.repo, "--base", self.base)
        self.assertEqual(payload["gates"]["docs_parity"]["status"], gate_check.PASS)
        self.assertEqual(payload["gates"]["docs_content"]["status"], gate_check.UNKNOWN)

    def test_readme_without_english_pair_fails(self) -> None:
        support.commit(self.repo, "readme", {"README.md": "# project\n"})
        _, payload = run_gate(self.repo, "--base", self.base)
        self.assertEqual(payload["gates"]["docs_parity"]["status"], gate_check.FAIL)
        self.assertTrue(payload["gates"]["docs_parity"]["value"]["readme_unpaired"])

    def test_nested_readme_pair_must_stay_synced(self) -> None:
        support.commit(
            self.repo,
            "add skill docs",
            {"skill/README.md": "# skill\n", "skill/README_en.md": "# skill\n"},
        )
        base = support.git(self.repo, "rev-parse", "HEAD").strip()
        support.commit(self.repo, "zh only", {"skill/README.md": "# skill v2\n"})
        _, payload = run_gate(self.repo, "--base", base)
        row = payload["gates"]["docs_parity"]
        self.assertEqual(row["status"], gate_check.FAIL)
        self.assertIn("skill/README_en.md", " ".join(row["value"]["readme_unpaired"]))

    def test_nested_readme_without_english_is_not_forced(self) -> None:
        support.commit(self.repo, "solo readme", {"solo/README.md": "# solo\n"})
        _, payload = run_gate(self.repo, "--base", self.base)
        self.assertEqual(payload["gates"]["docs_parity"]["status"], gate_check.PASS)

    def test_deleted_doc_requires_counterpart_deletion(self) -> None:
        support.commit(
            self.repo,
            "add pair",
            {"docs/source_zh_cn/old.md": "# 旧\n", "docs/source_en/old.md": "# old\n"},
        )
        base = support.git(self.repo, "rev-parse", "HEAD").strip()
        (self.repo / "docs" / "source_zh_cn" / "old.md").unlink()
        support.commit(self.repo, "delete zh only")
        _, payload = run_gate(self.repo, "--base", base)
        row = payload["gates"]["docs_parity"]
        self.assertEqual(row["status"], gate_check.FAIL)
        self.assertTrue(row["value"]["deleted_without_counterpart"])


class ReportContract(GateTestCase):
    def test_unverifiable_gates_never_report_pass(self) -> None:
        support.commit(self.repo, "change", {"app.py": "print('f')\n"}, sign=True)
        code, payload = run_gate(self.repo, "--base", self.base)
        for name in ("review_present", "threads_resolved", "rtd_and_smoke", "coverage_scope"):
            self.assertEqual(payload["gates"][name]["status"], gate_check.UNKNOWN, name)
        self.assertIn("review_present", payload["unverified"])
        self.assertTrue(payload["promotion_blockers"])
        self.assertEqual(code, 0)

    def test_exit_code_one_only_for_known_failure(self) -> None:
        support.commit(self.repo, "untitled", {"app.py": "print('g')\n"})
        code, payload = run_gate(self.repo, "--base", self.base, "--title", "bad title")
        self.assertEqual(code, 1)
        self.assertEqual(payload["gates"]["title_format"]["status"], gate_check.FAIL)

    def test_missing_repo_returns_exit_two(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "gate_check.py"), "--repo-root", str(self._work / "nope")],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("error", json.loads(proc.stdout))


class ProfileParsing(GateTestCase):
    def test_profile_reads_thresholds_from_workflows(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "gate_check.py"), "--repo-root", str(self.repo), "--profile"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["thresholds"]["size_limit"], 1000)
        self.assertEqual(payload["thresholds"]["ut_prefix"], "tests/UT/")
        self.assertEqual(payload["thresholds"]["bugfix_ut_threshold"], 50)
        self.assertEqual(payload["thresholds"]["coverage"], {"total": 80, "line": 80, "branch": 60, "fail_under": 80})

    def test_profile_lists_template_options_without_checklist_items(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "gate_check.py"), "--repo-root", str(self.repo), "--profile"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(len(payload["template_options"]), 9)
        drifted = {row["template_text"] for row in payload["template_label_drift"]}
        self.assertEqual(drifted, {"CI/CD（持续集成/持续部署）", "Dependency（依赖项更新）"})


if __name__ == "__main__":
    unittest.main()
