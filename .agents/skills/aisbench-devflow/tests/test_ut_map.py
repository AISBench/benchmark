"""Unit tests for scripts/ut_map.py (source/test mapping and scoped coverage)."""

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
import ut_map  # noqa: E402


class Suggest(unittest.TestCase):
    def setUp(self) -> None:
        self.work = support.new_workdir()
        self.repo = support.init_repo(self.work / "repo", with_profile=False)
        (self.repo / "ais_bench" / "benchmark" / "datasets").mkdir(parents=True)
        (self.repo / "ais_bench" / "benchmark" / "cli").mkdir(parents=True)
        (self.repo / "tests" / "UT" / "datasets").mkdir(parents=True)
        (self.repo / "tests" / "UT" / "cli").mkdir(parents=True)
        (self.repo / "ais_bench" / "benchmark" / "datasets" / "hle.py").write_text("x = 1\n", encoding="utf-8")
        (self.repo / "tests" / "UT" / "datasets" / "test_hle.py").write_text("def test(): pass\n", encoding="utf-8")

    def tearDown(self) -> None:
        support.cleanup(self.work)

    def test_source_maps_to_test_dir_and_coverage_scope(self) -> None:
        payload = ut_map.suggest(["ais_bench/benchmark/datasets/hle.py"], self.repo)
        row = payload["rows"][0]
        self.assertTrue(row["supported"])
        self.assertEqual(row["test_dir"], "tests/UT/datasets")
        self.assertEqual(row["suggested_test_file"], "tests/UT/datasets/test_hle.py")
        self.assertEqual(row["coverage_source_dir"], "ais_bench/benchmark/datasets")
        self.assertTrue(row["suggested_test_file_exists"])
        self.assertEqual(payload["sources_without_suggested_test"], [])

    def test_missing_test_file_is_reported(self) -> None:
        payload = ut_map.suggest(["ais_bench/benchmark/cli/main.py"], self.repo)
        row = payload["rows"][0]
        self.assertEqual(row["suggested_test_file"], "tests/UT/cli/test_main.py")
        self.assertFalse(row["suggested_test_file_exists"])
        self.assertEqual(payload["sources_without_suggested_test"], ["ais_bench/benchmark/cli/main.py"])

    def test_plugin_source_is_explicitly_unsupported(self) -> None:
        payload = ut_map.suggest(["plugins/prefix_cache/ais_bench_prefix_cache/x.py"], self.repo)
        row = payload["rows"][0]
        self.assertFalse(row["supported"])
        self.assertIn("人工指定", row["reason"])
        self.assertEqual(payload["unsupported"], ["plugins/prefix_cache/ais_bench_prefix_cache/x.py"])

    def test_paths_use_forward_slashes(self) -> None:
        payload = ut_map.suggest(["ais_bench/benchmark/datasets/hle.py"], self.repo)
        self.assertNotIn("\\", json.dumps(payload))


class MapTestDir(unittest.TestCase):
    def setUp(self) -> None:
        self.work = support.new_workdir()
        self.repo = support.init_repo(self.work / "repo", with_profile=False)
        (self.repo / "ais_bench" / "benchmark" / "datasets").mkdir(parents=True)
        (self.repo / "tests" / "UT" / "datasets").mkdir(parents=True)

    def tearDown(self) -> None:
        support.cleanup(self.work)

    def test_subdir_maps_to_matching_package_dir(self) -> None:
        payload = ut_map.map_test_dir("tests/UT/datasets", self.repo)
        self.assertEqual(payload["coverage_source_dirs"], ["ais_bench/benchmark/datasets"])

    def test_bare_ut_maps_to_whole_package(self) -> None:
        payload = ut_map.map_test_dir("tests/UT", self.repo)
        self.assertEqual(payload["coverage_source_dirs"], ["ais_bench/benchmark"])

    def test_foreign_dir_raises(self) -> None:
        with self.assertRaises(ut_map.MapError):
            ut_map.map_test_dir("plugins/prefix_cache/tests", self.repo)


class CoverageCli(unittest.TestCase):
    def setUp(self) -> None:
        self.work = support.new_workdir()
        self.repo = support.init_repo(self.work / "repo", with_profile=False)
        (self.repo / "ais_bench" / "benchmark" / "datasets").mkdir(parents=True)
        (self.repo / "tests" / "UT" / "datasets").mkdir(parents=True)

    def tearDown(self) -> None:
        support.cleanup(self.work)

    def run_cli(self, *args: str) -> tuple[int, dict]:
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "ut_map.py"), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return proc.returncode, json.loads(proc.stdout)

    def test_dry_run_never_executes_tests(self) -> None:
        code, payload = self.run_cli(
            "coverage",
            "--test-dir",
            "tests/UT/datasets",
            "--repo-root",
            str(self.repo),
            "--dry-run",
        )
        self.assertEqual(code, 0)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["scope"], "subset")
        self.assertIn("--source-dirs", payload["command"])
        self.assertFalse((self.repo / "test_reports").exists())

    def test_execute_without_runner_reports_error(self) -> None:
        code, payload = self.run_cli(
            "coverage",
            "--test-dir",
            "tests/UT/datasets",
            "--repo-root",
            str(self.repo),
        )
        self.assertEqual(code, 2)
        self.assertIn("tests/run_tests.py", payload["error"])

    def test_suggest_cli_contract(self) -> None:
        code, payload = self.run_cli(
            "suggest", "--changed", "ais_bench/benchmark/datasets/hle.py", "--repo-root", str(self.repo)
        )
        self.assertEqual(code, 0)
        self.assertEqual(payload["schema"], 1)
        self.assertEqual(payload["rows"][0]["subdir"], "datasets")


if __name__ == "__main__":
    unittest.main()
