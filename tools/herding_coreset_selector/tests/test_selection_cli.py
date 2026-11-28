"""Regression tests for the herding CLI without model or dataset dependencies."""

import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, call, patch

from herding import __main__ as cli
from herding.selection import run_selection


REQUIRED_ARGS = [
    "--eval-dataset", "example",
    "--dataset-path", "source",
    "--model-path", "models/example-model/",
]


class CliTests(unittest.TestCase):
    def test_defaults(self):
        args = cli.parse_args(REQUIRED_ARGS)
        self.assertEqual(args.coreset_ratio, 0.2)
        self.assertEqual(args.output_dir, "./datasets")

    def test_invalid_ratio_fails_before_selection(self):
        for ratio, error in [
            ("invalid", "must be a floating-point number"),
            ("0", "must be in the range (0, 1]"),
            ("-1", "must be in the range (0, 1]"),
            ("1.01", "must be in the range (0, 1]"),
            ("NaN", "must be in the range (0, 1]"),
        ]:
            with self.subTest(ratio=ratio):
                stderr = io.StringIO()
                with patch.dict(sys.modules, {"herding.selection": None}):
                    with redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
                        cli.main([*REQUIRED_ARGS, f"--coreset-ratio={ratio}"])
                self.assertEqual(exc.exception.code, 2)
                self.assertIn(error, stderr.getvalue())

    def test_help_does_not_import_selection(self):
        stdout = io.StringIO()
        with patch.dict(sys.modules, {"herding.selection": None}):
            with redirect_stdout(stdout), self.assertRaises(SystemExit) as exc:
                cli.main(["--help"])
        self.assertEqual(exc.exception.code, 0)
        self.assertIn("--coreset-ratio", stdout.getvalue())
        self.assertIn("--output-dir", stdout.getvalue())

    def test_main_forwards_arguments_and_prints_output_path(self):
        stdout = io.StringIO()
        with patch("herding.selection.run_selection", return_value="saved.json") as run:
            with redirect_stdout(stdout):
                cli.main([*REQUIRED_ARGS, "--coreset-ratio", "1", "--output-dir", "out"])
        run.assert_called_once_with(
            dataset_name="example",
            dataset_path="source",
            model_path="models/example-model/",
            coreset_ratio=1.0,
            output_dir="out",
        )
        self.assertEqual(stdout.getvalue(), "output: saved.json\n")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.events = Mock()
        self.dataset = self.events.dataset
        self.dataset.load_indices.return_value = [10, 17, 3, 9, 2]
        self.dataset.save_data_by_indices.side_effect = ["origin.json", "coreset.json"]
        self.events.get_eval_dataset.return_value = self.dataset
        self.events.generate_coreset.return_value = [17, 2]

        adapters = ModuleType("herding.eval_datasets")
        adapters.get_eval_dataset = self.events.get_eval_dataset
        modules = patch.dict(sys.modules, {adapters.__name__: adapters})
        generator = patch("herding.generate_coreset", self.events.generate_coreset)
        for patcher in (modules, generator):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_workflow(self, ratio=0.5, model_path="models/example-model/\\"):
        return run_selection(
            dataset_name="example",
            dataset_path="source",
            model_path=model_path,
            coreset_ratio=ratio,
            output_dir="results",
        )

    def test_export_order_paths_rounding_and_selected_indices(self):
        # Five rows at 0.5 round to two, preserving Python's ties-to-even rule.
        self.assertEqual(self.run_workflow(), "coreset.json")
        output = Path("results") / "example" / "herding" / "example-model"
        self.assertEqual(self.events.mock_calls, [
            call.get_eval_dataset("example", dataset_path="source", output_dir=str(output)),
            call.dataset.load_indices(),
            call.dataset.save_data_by_indices([10, 17, 3, 9, 2], "origin"),
            call.generate_coreset(2, eval_dataset=self.dataset, model_path="models/example-model/\\"),
            call.dataset.save_data_by_indices([17, 2], "coreset"),
        ])

    def test_selection_size_is_at_least_one(self):
        self.events.generate_coreset.return_value = [17]
        self.run_workflow(ratio=0.01)
        self.events.generate_coreset.assert_called_once_with(
            1, eval_dataset=self.dataset, model_path="models/example-model/\\"
        )
        self.dataset.save_data_by_indices.assert_called_with([17], "coreset")

    def test_empty_dataset_fails_before_export_and_generation(self):
        self.dataset.load_indices.return_value = []
        with self.assertRaisesRegex(ValueError, "evaluation dataset is empty"):
            self.run_workflow()
        self.dataset.save_data_by_indices.assert_not_called()
        self.events.generate_coreset.assert_not_called()

    def test_missing_model_name_fails_before_adapter_creation(self):
        with self.assertRaisesRegex(ValueError, "Unable to infer model name"):
            self.run_workflow(model_path="///")
        self.events.get_eval_dataset.assert_not_called()
        self.events.generate_coreset.assert_not_called()
