import io
import json
import logging
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ais_bench_prefix_cache.artifacts import ArtifactPaths
from ais_bench_prefix_cache.cli import (
    PromptProgress,
    _install_logger,
    _persist_inspect_pointer,
    _reusable_inspect_timestamp,
    _resolve_log_file,
    console_main,
    main,
)
from ais_bench_prefix_cache.errors import PrefixCacheError
from ais_bench_prefix_cache.scenario import load_scenario
from tests.test_pipeline import write_case


class PromptProgressTest(unittest.TestCase):
    def test_update_ignores_non_positive_total(self):
        stream = io.StringIO()
        progress = PromptProgress(stream=stream, width=10)
        progress.update(1, 0)
        self.assertEqual(stream.getvalue(), "")

    def test_close_terminates_unfinished_line(self):
        stream = io.StringIO()
        progress = PromptProgress(stream=stream, width=10)
        progress.update(1, 4)
        progress.close()
        self.assertTrue(stream.getvalue().endswith("\n"))


class LogResolverTest(unittest.TestCase):
    def test_validate_resolves_log_from_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out_dir = root / "out"
            manifest_path = root / "m.json"
            manifest_path.write_text(
                json.dumps({"run_id": "pc-test", "effective_config": {"run": {"output_dir": str(out_dir)}}}),
                encoding="utf-8",
            )
            log_file = _resolve_log_file("validate", manifest_path=manifest_path)
            self.assertEqual(log_file, out_dir / "log" / "pc-test.validate.log")
            self.assertTrue(log_file.parent.is_dir())

    def test_validate_returns_none_for_missing_or_bad_manifest(self):
        self.assertIsNone(_resolve_log_file("validate"))
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = Path(folder) / "m.json"
            manifest_path.write_text("not json", encoding="utf-8")
            self.assertIsNone(_resolve_log_file("validate", manifest_path=manifest_path))

    def test_returns_none_without_scenario(self):
        self.assertIsNone(_resolve_log_file("prepare"))

    def test_returns_none_for_invalid_scenario(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(_resolve_log_file("prepare", scenario_path=Path(folder) / "missing.json"))


class ReusableTimestampTest(unittest.TestCase):
    def _pointer(self, root, **overrides):
        record = {
            "schema_version": "1.0",
            "timestamp": "20260825_123456",
            "run_id": "pc-test",
            "output_dir": str(root / "out"),
        }
        record.update(overrides)
        pointer = root / "out.inspect.json"
        pointer.write_text(json.dumps(record), encoding="utf-8")
        return pointer

    def test_none_without_pointer(self):
        with tempfile.TemporaryDirectory() as folder:
            scenario = load_scenario(write_case(Path(folder)))
            self.assertIsNone(_reusable_inspect_timestamp(scenario))

    def test_none_when_schema_version_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = load_scenario(write_case(root))
            self._pointer(root, schema_version="2.0")
            self.assertIsNone(_reusable_inspect_timestamp(scenario))

    def test_none_when_run_id_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = load_scenario(write_case(root))
            self._pointer(root, run_id="other")
            self.assertIsNone(_reusable_inspect_timestamp(scenario))

    def test_none_when_timestamp_malformed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = load_scenario(write_case(root))
            self._pointer(root, timestamp="bad")
            self.assertIsNone(_reusable_inspect_timestamp(scenario))

    def test_none_when_timestamp_not_a_string(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = load_scenario(write_case(root))
            self._pointer(root, timestamp=123)
            self.assertIsNone(_reusable_inspect_timestamp(scenario))

    def test_none_when_stamped_dir_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = load_scenario(write_case(root))
            self._pointer(root)
            self.assertIsNone(_reusable_inspect_timestamp(scenario))

    def test_reuses_valid_pointer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = load_scenario(write_case(root))
            self._pointer(root)
            (root / "out_20260825_123456").mkdir(parents=True)
            self.assertEqual(_reusable_inspect_timestamp(scenario), "20260825_123456")


class PersistPointerTest(unittest.TestCase):
    def test_load_failure_is_best_effort(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _persist_inspect_pointer(root / "missing.json", root / "x.log", "20260825_123456")
            self.assertFalse((root / "out.inspect.json").exists())

    def test_write_failure_is_best_effort(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = write_case(root)
            with patch("pathlib.Path.write_text", side_effect=OSError("boom")):
                _persist_inspect_pointer(scenario, root / "out_ts" / "log" / "x.log", "20260825_123456")


class InstallLoggerTest(unittest.TestCase):
    def test_console_fallback_when_no_log_file(self):
        _install_logger(None)
        plugin_logger = logging.getLogger("ais_bench_prefix_cache")
        self.assertIsInstance(plugin_logger.handlers[0], logging.StreamHandler)
        self.assertFalse(plugin_logger.propagate)

    def test_file_handler_when_log_file_given(self):
        with tempfile.TemporaryDirectory() as folder:
            log_path = Path(folder) / "x.log"
            _install_logger(log_path)
            plugin_logger = logging.getLogger("ais_bench_prefix_cache")
            self.assertIsInstance(plugin_logger.handlers[0], logging.FileHandler)
        # 清理 FileHandler，避免句柄泄漏影响后续用例。
        _install_logger(None)


class MainFlowTest(unittest.TestCase):
    def _fake_paths(self, scenario: Path) -> ArtifactPaths:
        result_dir = scenario.parent / "result"
        return ArtifactPaths(
            result_dir / "x.full.jsonl",
            result_dir / "x.requests.jsonl",
            result_dir / "x.manifest.json",
            result_dir / "x.analysis.json",
        )

    def test_validate_prints_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest_path = root / "m.json"
            manifest_path.write_text(
                json.dumps({"run_id": "pc-test", "effective_config": {"run": {"output_dir": str(root / "out")}}}),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with (
                patch("ais_bench_prefix_cache.cli.validate_artifacts", return_value={"ok": True, "rows": 0, "run_id": "pc-test"}),
                redirect_stdout(stdout),
            ):
                self.assertEqual(main(["validate", "--manifest", str(manifest_path)]), 0)
            output = json.loads(stdout.getvalue())
            self.assertTrue(output["ok"])

    def test_validate_error_returns_two(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest_path = root / "m.json"
            manifest_path.write_text(
                json.dumps({"run_id": "pc-test", "effective_config": {"run": {"output_dir": str(root / "out")}}}),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with (
                patch("ais_bench_prefix_cache.cli.validate_artifacts", side_effect=PrefixCacheError("bad manifest")),
                redirect_stderr(stderr),
            ):
                self.assertEqual(main(["validate", "--manifest", str(manifest_path)]), 2)
            self.assertIn("ERROR: bad manifest", stderr.getvalue())

    def test_prepare_error_closes_progress_and_returns_two(self):
        with tempfile.TemporaryDirectory() as folder:
            scenario = write_case(Path(folder))
            stderr = io.StringIO()
            with (
                patch("ais_bench_prefix_cache.cli.new_execution_timestamp", return_value="20260825_123456"),
                patch("ais_bench_prefix_cache.cli.prepare_scenario", side_effect=PrefixCacheError("boom")),
                redirect_stderr(stderr),
            ):
                self.assertEqual(main(["prepare", "--scenario", str(scenario)]), 2)
            self.assertIn("ERROR: boom", stderr.getvalue())

    def test_prepare_reuses_inspect_timestamp(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = write_case(root)
            (root / "out_20260825_123456").mkdir(parents=True)
            (root / "out.inspect.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "timestamp": "20260825_123456",
                        "run_id": "pc-test",
                        "output_dir": str(root / "out"),
                    }
                ),
                encoding="utf-8",
            )
            calls = []

            def fake_prepare(path, overwrite, progress, execution_timestamp):
                calls.append((path, overwrite, execution_timestamp))
                return self._fake_paths(path)

            with (
                patch("ais_bench_prefix_cache.cli.new_execution_timestamp") as new_ts,
                patch("ais_bench_prefix_cache.cli.prepare_scenario", side_effect=fake_prepare),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(["prepare", "--scenario", str(scenario)]), 0)
            new_ts.assert_not_called()
            self.assertEqual(calls[0][2], "20260825_123456")

    def test_prepare_falls_back_to_new_timestamp_on_bad_scenario(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bad_scenario = root / "bad.json"
            bad_scenario.write_text("not json", encoding="utf-8")
            stdout = io.StringIO()
            with (
                patch("ais_bench_prefix_cache.cli.new_execution_timestamp", return_value="20260825_123456") as new_ts,
                patch(
                    "ais_bench_prefix_cache.cli.prepare_scenario",
                    side_effect=lambda path, overwrite, progress, execution_timestamp: self._fake_paths(path),
                ),
                redirect_stdout(stdout),
            ):
                self.assertEqual(main(["prepare", "--scenario", str(bad_scenario)]), 0)
            new_ts.assert_called_once()
            output = json.loads(stdout.getvalue())
            self.assertNotIn("log", output)

    def test_inspect_without_log_file_omits_log_key(self):
        with tempfile.TemporaryDirectory() as folder:
            scenario = write_case(Path(folder))
            stdout = io.StringIO()
            with (
                patch("ais_bench_prefix_cache.cli.new_execution_timestamp", return_value="20260825_123456"),
                patch("ais_bench_prefix_cache.cli._resolve_log_file", return_value=None),
                patch("ais_bench_prefix_cache.cli.inspect_scenario", return_value={"run_id": "pc-test", "sends_requests": False}),
                redirect_stdout(stdout),
            ):
                self.assertEqual(main(["inspect", "--scenario", str(scenario)]), 0)
            output = json.loads(stdout.getvalue())
            self.assertNotIn("log", output)

    def test_console_main_raises_system_exit(self):
        with patch("ais_bench_prefix_cache.cli.main", return_value=3):
            with self.assertRaises(SystemExit) as context:
                console_main()
        self.assertEqual(context.exception.code, 3)


if __name__ == "__main__":
    unittest.main()
