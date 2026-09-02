import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ais_bench_prefix_cache.artifacts import ArtifactPaths
from ais_bench_prefix_cache.cli import PromptProgress, _resolve_log_file, main
from tests.test_pipeline import write_case


class CLITest(unittest.TestCase):
    def test_log_file_is_nested_under_log_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = write_case(root)
            for command in ("prepare", "inspect"):
                with self.subTest(command=command):
                    log_file = _resolve_log_file(command, scenario, execution_timestamp="20260825_123456")
                    self.assertEqual(
                        log_file,
                        root / "out_20260825_123456" / "log" / f"pc-test_20260825_123456.{command}.log",
                    )
                    self.assertTrue(log_file.parent.is_dir())

    def test_prompt_progress_renders_zero_updates_and_completion(self):
        stream = io.StringIO()
        progress = PromptProgress(stream=stream, width=10)
        progress.update(0, 4)
        progress.update(1, 4)
        progress.update(4, 4)
        output = stream.getvalue()
        self.assertIn("Generate prompts", output)
        self.assertIn("0/4", output)
        self.assertIn("1/4", output)
        self.assertIn("4/4", output)
        self.assertIn("100%", output)
        self.assertTrue(output.endswith("\n"))

    def test_prepare_cli_keeps_progress_on_stderr_and_result_on_stdout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = write_case(root)
            timestamp = "20260825_123456"
            result_dir = root / f"out_{timestamp}" / "result"
            paths = ArtifactPaths(
                result_dir / f"pc-test_{timestamp}.full.jsonl",
                result_dir / f"pc-test_{timestamp}.requests.jsonl",
                result_dir / f"pc-test_{timestamp}.manifest.json",
                result_dir / f"pc-test_{timestamp}.analysis.json",
            )

            def fake_prepare(path, overwrite, progress, execution_timestamp):
                self.assertEqual(path, scenario)
                self.assertEqual(execution_timestamp, timestamp)
                for completed in range(3):
                    progress(completed, 2)
                return paths

            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch("ais_bench_prefix_cache.cli.new_execution_timestamp", return_value=timestamp),
                patch("ais_bench_prefix_cache.cli.prepare_scenario", side_effect=fake_prepare),
                patch("ais_bench_prefix_cache.cli._install_logger"),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                self.assertEqual(main(["prepare", "--scenario", str(scenario)]), 0)

            output = json.loads(stdout.getvalue())
            self.assertEqual(output["manifest"], str(paths.manifest))
            self.assertEqual(
                output["log"],
                str(root / f"out_{timestamp}" / "log" / f"pc-test_{timestamp}.prepare.log"),
            )
            self.assertIn("Generate prompts", stderr.getvalue())
            self.assertIn("2/2", stderr.getvalue())

    def test_inspect_cli_returns_timestamped_log_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = write_case(root)
            timestamp = "20260825_123456"
            summary = {"run_id": "pc-test", "sends_requests": False}
            stdout = io.StringIO()
            with (
                patch("ais_bench_prefix_cache.cli.new_execution_timestamp", return_value=timestamp),
                patch("ais_bench_prefix_cache.cli.inspect_scenario", return_value=summary),
                patch("ais_bench_prefix_cache.cli._install_logger"),
                redirect_stdout(stdout),
            ):
                self.assertEqual(main(["inspect", "--scenario", str(scenario)]), 0)

            output = json.loads(stdout.getvalue())
            self.assertEqual(output["run_id"], "pc-test")
            self.assertEqual(
                output["log"],
                str(root / f"out_{timestamp}" / "log" / f"pc-test_{timestamp}.inspect.log"),
            )


if __name__ == "__main__":
    unittest.main()
