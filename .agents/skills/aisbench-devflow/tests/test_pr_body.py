"""Unit tests for scripts/pr_body.py (template rendering and self-check)."""

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

import pr_body  # noqa: E402
import support  # noqa: E402

TRAILER = "<!-- aisbench-devflow -->"


def run_cli(*args: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "pr_body.py"), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout


class BodyCase(unittest.TestCase):
    def setUp(self) -> None:
        self.work = support.new_workdir()
        self.repo = support.init_repo(self.work / "repo")
        self.template = self.repo / ".github" / "pull_request_template.md"
        self.workflow = self.repo / ".github" / "workflows" / "auto_label_pr.yml"

    def tearDown(self) -> None:
        support.cleanup(self.work)

    def sections(self, payload: dict) -> str:
        path = self.work / "sections.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return str(path)


class Render(BodyCase):
    def test_ticks_exactly_one_type_and_keeps_template_wording(self) -> None:
        out = self.work / "body.md"
        code, stdout = run_cli(
            "render",
            "--template",
            str(self.template),
            "--workflow",
            str(self.workflow),
            "--pr-type",
            "feature",
            "--out",
            str(out),
        )
        self.assertEqual(code, 0)
        report = json.loads(stdout)
        self.assertEqual(report["template_option"], "Feature（功能新增）")
        self.assertEqual(report["predicted_label"], "feature")
        self.assertTrue(report["label_mapping_ok"])

        body = out.read_text(encoding="utf-8")
        self.assertEqual(body.count("- [x]"), 1)
        self.assertIn("- [x] Feature（功能新增）", body)
        self.assertEqual(body.count(TRAILER), 1)

    def test_ci_cd_render_keeps_template_text_and_warns(self) -> None:
        out = self.work / "body.md"
        code, stdout = run_cli(
            "render",
            "--template",
            str(self.template),
            "--workflow",
            str(self.workflow),
            "--pr-type",
            "ci-cd",
            "--out",
            str(out),
        )
        self.assertEqual(code, 0)
        report = json.loads(stdout)
        self.assertIsNone(report["predicted_label"])
        self.assertFalse(report["label_mapping_ok"])
        self.assertIn("pr_template_label_drift", report["warning"])

        body = out.read_text(encoding="utf-8")
        # The template wording must survive untouched: no silent rewrite to match the workflow.
        self.assertIn("- [x] CI/CD（持续集成/持续部署）", body)
        self.assertNotIn("流水线工程", body)

    def test_issue_line_and_sections_are_filled(self) -> None:
        out = self.work / "body.md"
        sections = self.sections({"Motivation": "补充压测档位对比", "Modification": "新增 combo 展开"})
        run_cli(
            "render",
            "--template",
            str(self.template),
            "--workflow",
            str(self.workflow),
            "--pr-type",
            "feature",
            "--issue",
            "651",
            "--sections",
            sections,
            "--out",
            str(out),
        )
        body = out.read_text(encoding="utf-8")
        self.assertIn("Fixes #651", body)
        self.assertNotIn("Fixes #(issue ID", body)
        self.assertIn("补充压测档位对比", body)
        self.assertNotIn("Please describe the motivation.", body)
        self.assertNotIn("unfilled-sections", body)

    def test_missing_sections_are_reported_not_faked(self) -> None:
        out = self.work / "body.md"
        sections = self.sections({"Motivation": "只有一节"})
        run_cli(
            "render",
            "--template",
            str(self.template),
            "--workflow",
            str(self.workflow),
            "--pr-type",
            "feature",
            "--sections",
            sections,
            "--out",
            str(out),
        )
        body = out.read_text(encoding="utf-8")
        self.assertIn("unfilled-sections", body)

    def test_unknown_pr_type_is_rejected(self) -> None:
        code, _ = run_cli(
            "render",
            "--template",
            str(self.template),
            "--workflow",
            str(self.workflow),
            "--pr-type",
            "nonsense",
        )
        self.assertEqual(code, 2)


class Trailer(unittest.TestCase):
    def test_trailer_is_idempotent(self) -> None:
        once = pr_body.ensure_trailer("body")
        twice = pr_body.ensure_trailer(once)
        self.assertEqual(once, twice)
        self.assertEqual(twice.count(TRAILER), 1)

    def test_duplicate_trailers_collapse(self) -> None:
        text = f"body\n\n{TRAILER}\n\n{TRAILER}\n"
        self.assertEqual(pr_body.ensure_trailer(text).count(TRAILER), 1)


class Check(BodyCase):
    def write_body(self, text: str) -> str:
        path = self.work / "candidate.md"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def check(self, body: str, *extra: str) -> tuple[int, dict]:
        code, stdout = run_cli(
            "check", "--body-file", self.write_body(body), "--workflow", str(self.workflow), *extra
        )
        return code, json.loads(stdout)

    def test_complete_body_passes(self) -> None:
        body = "- [x] Feature（功能新增）\n\nFixes #651\n\n" + TRAILER + "\n"
        code, report = self.check(body, "--issue", "651", "--title", "[feat][datasets]支持 X")
        self.assertEqual(code, 0)
        self.assertEqual(report["checks"]["trailer"]["status"], "PASS")
        self.assertEqual(report["checks"]["pr_type_selection"]["status"], "PASS")
        self.assertEqual(report["checks"]["label_prediction"]["status"], "PASS")

    def test_missing_trailer_fails(self) -> None:
        code, report = self.check("- [x] Feature（功能新增）\nFixes #651\n", "--issue", "651")
        self.assertEqual(code, 1)
        self.assertEqual(report["checks"]["trailer"]["status"], "FAIL")

    def test_two_types_fail(self) -> None:
        body = f"- [x] Feature（功能新增）\n- [x] Docs（文档更新）\n\nFixes #651\n\n{TRAILER}\n"
        code, report = self.check(body, "--issue", "651")
        self.assertEqual(code, 1)
        self.assertEqual(report["checks"]["pr_type_selection"]["status"], "FAIL")

    def test_unmapped_option_fails_with_drift_hint(self) -> None:
        body = f"- [x] Dependency（依赖项更新）\n\nFixes #1\n\n{TRAILER}\n"
        code, report = self.check(body, "--issue", "1")
        self.assertEqual(code, 1)
        row = report["checks"]["label_prediction"]
        self.assertEqual(row["status"], "FAIL")
        self.assertTrue(row["value"]["unmapped"])
        self.assertIn("不要为迎合脚本改写模板", row["remedy"])

    def test_wrong_issue_fails(self) -> None:
        body = f"- [x] Feature（功能新增）\n\nFixes #9\n\n{TRAILER}\n"
        code, report = self.check(body, "--issue", "651")
        self.assertEqual(code, 1)
        self.assertEqual(report["checks"]["issue_link"]["status"], "FAIL")

    def test_bad_title_fails(self) -> None:
        body = f"- [x] Feature（功能新增）\n\nFixes #651\n\n{TRAILER}\n"
        code, report = self.check(body, "--issue", "651", "--title", "add feature")
        self.assertEqual(code, 1)
        self.assertEqual(report["checks"]["title_format"]["status"], "FAIL")


class Parsing(unittest.TestCase):
    def test_template_options_ignore_checklist_section(self) -> None:
        options = pr_body.template_options(support.PR_TEMPLATE)
        self.assertEqual(len(options), 9)
        self.assertIn("CI/CD（持续集成/持续部署）", options)

    def test_option_for_type_matches_prefix(self) -> None:
        options = pr_body.template_options(support.PR_TEMPLATE)
        self.assertEqual(pr_body.option_for_type(options, "test-cases"), "Test-Cases（测试用例更新）")

    def test_option_for_unknown_type_raises(self) -> None:
        options = pr_body.template_options(support.PR_TEMPLATE)
        with self.assertRaises(pr_body.BodyError):
            pr_body.option_for_type(options, "unknown")


if __name__ == "__main__":
    unittest.main()
