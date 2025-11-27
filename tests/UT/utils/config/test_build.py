import unittest
from unittest.mock import patch, MagicMock

from mmengine.config import ConfigDict

from ais_bench.benchmark.utils.config import (
    build_dataset_from_cfg,
    build_model_from_cfg,
    build_perf_metric_calculator_from_cfg,
)
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError


class MockDataset:
    def __init__(self, **kwargs):
        self.abbr = kwargs.get("abbr", "test_dataset")
        self.path = kwargs.get("path", "")
        self.reader_cfg = kwargs.get("reader_cfg", {})
        self.type = kwargs.get("type", "MockDataset")


class TestConfigBuildUtilities(unittest.TestCase):
    """Test suite for public build utilities from ais_bench.benchmark.utils.config"""

    @patch("ais_bench.benchmark.utils.config.build.logger")
    @patch("os.path.exists", return_value=True)
    @patch("ais_bench.benchmark.utils.config.build.MODELS")
    def test_build_model_from_cfg_success(self, mock_models, _mock_exists, mock_logger):
        """Test successful model building with valid configuration"""
        mock_model = MagicMock()
        mock_models.build.return_value = mock_model

        valid_cfg = ConfigDict(
            {
                "attr": "local",
                "abbr": "test-model",
                "path": "/some/valid/path",
                "model": "gpt-4",
                "request_rate": 100,
                "retry": 5,
                "host_ip": "127.0.0.1",
                "host_port": 8080,
                "max_out_len": 512,
                "batch_size": 32,
                "generation_kwargs": {},
                "type": "some.model.Type",
                "run_cfg": {},
                "summarizer_abbr": "summ",
                "pred_postprocessor": None,
                "min_out_len": 10,
            }
        )

        result = build_model_from_cfg(valid_cfg)

        self.assertEqual(result, mock_model)
        mock_models.build.assert_called_once()

        # Verify that internal keys are removed before building
        passed_cfg = mock_models.build.call_args[0][0]
        self.assertNotIn("run_cfg", passed_cfg)
        self.assertNotIn("request_rate", passed_cfg)
        self.assertNotIn("batch_size", passed_cfg)
        self.assertNotIn("abbr", passed_cfg)
        self.assertNotIn("attr", passed_cfg)
        self.assertNotIn("summarizer_abbr", passed_cfg)
        self.assertNotIn("pred_postprocessor", passed_cfg)
        self.assertNotIn("min_out_len", passed_cfg)

        # Verify logging was called
        mock_logger.debug.assert_called()

    @patch("ais_bench.benchmark.utils.config.build.logger")
    @patch("ais_bench.benchmark.utils.config.build.MODELS")
    def test_build_model_from_cfg_validation_errors(self, _mock_models, mock_logger):
        """Test model building with various validation errors"""
        # Invalid attr
        with self.assertRaises(AISBenchConfigError):
            build_model_from_cfg(ConfigDict({"attr": "invalid", "type": "model.Type"}))
        mock_logger.warning.assert_called()

        # Invalid request_rate
        with self.assertRaises(AISBenchConfigError):
            build_model_from_cfg(ConfigDict({"request_rate": 100000, "type": "model.Type"}))

        # Invalid retry
        with self.assertRaises(AISBenchConfigError):
            build_model_from_cfg(ConfigDict({"retry": 2000, "type": "model.Type"}))

        # Invalid host_port
        with self.assertRaises(AISBenchConfigError):
            build_model_from_cfg(ConfigDict({"host_port": 70000, "type": "model.Type"}))

        # Invalid max_out_len
        with self.assertRaises(AISBenchConfigError):
            build_model_from_cfg(ConfigDict({"max_out_len": 200000, "type": "model.Type"}))

        # Invalid batch_size
        with self.assertRaises(AISBenchConfigError):
            build_model_from_cfg(ConfigDict({"batch_size": 200000, "type": "model.Type"}))

        # Invalid generation_kwargs
        with self.assertRaises(AISBenchConfigError):
            build_model_from_cfg(ConfigDict({"generation_kwargs": "not a dict", "type": "model.Type"}))

    @patch("ais_bench.benchmark.utils.config.build.logger")
    @patch("os.path.exists", return_value=True)
    @patch("ais_bench.benchmark.utils.config.build.MODELS")
    def test_build_model_from_cfg_with_traffic_cfg(self, mock_models, _mock_exists, _mock_logger):
        """Test model building with traffic configuration"""
        mock_models.build.return_value = MagicMock()

        cfg_with_traffic = ConfigDict(
            {
                "type": "model.Type",
                "attr": "service",
                "abbr": "test",
                "traffic_cfg": {
                    "burstiness": 1.5,
                    "ramp_up_strategy": "linear",
                    "ramp_up_start_rps": 10,
                    "ramp_up_end_rps": 100,
                },
            }
        )

        result = build_model_from_cfg(cfg_with_traffic)
        self.assertIsNotNone(result)

        # Test invalid traffic_cfg
        with self.assertRaises(AISBenchConfigError):
            build_model_from_cfg(
                ConfigDict(
                    {
                        "type": "model.Type",
                        "traffic_cfg": "not a dict",
                    }
                )
            )

        # Test invalid burstiness
        with self.assertRaises(AISBenchConfigError):
            build_model_from_cfg(
                ConfigDict(
                    {
                        "type": "model.Type",
                        "traffic_cfg": {"burstiness": -1},
                    }
                )
            )

        # Test invalid ramp_up_strategy
        with self.assertRaises(AISBenchConfigError):
            build_model_from_cfg(
                ConfigDict(
                    {
                        "type": "model.Type",
                        "traffic_cfg": {"ramp_up_strategy": "invalid"},
                    }
                )
            )

    @patch("ais_bench.benchmark.utils.config.build.logger")
    @patch("ais_bench.benchmark.utils.config.build.LOAD_DATASET")
    def test_build_dataset_from_cfg(self, mock_load_dataset, mock_logger):
        """Test dataset building with config sanitization"""
        mock_load_dataset.build.return_value = MockDataset(abbr="gsm8k")

        cfg = ConfigDict(
            {
                "type": "GSM8KDataset",
                "abbr": "gsm8k",
                "path": "ais_bench/datasets/gsm8k",
                "reader_cfg": {
                    "input_columns": ["question"],
                    "output_column": "answer",
                },
                "infer_cfg": {"prompt_template": {"type": "PromptTemplate"}},
                "eval_cfg": {"evaluator": {"type": "Gsm8kEvaluator"}},
            }
        )
        original_cfg = cfg.copy()

        result = build_dataset_from_cfg(cfg)

        # Verify infer_cfg and eval_cfg are removed
        call_cfg = mock_load_dataset.build.call_args[0][0]
        self.assertEqual(result.abbr, "gsm8k")
        self.assertNotIn("infer_cfg", call_cfg)
        self.assertNotIn("eval_cfg", call_cfg)
        self.assertEqual(call_cfg["type"], "GSM8KDataset")

        # Verify original config is not modified (deep copy)
        self.assertEqual(cfg, original_cfg)

        # Verify logging
        mock_logger.debug.assert_called()

    @patch("ais_bench.benchmark.utils.config.build.logger")
    @patch("ais_bench.benchmark.utils.config.build.LOAD_DATASET")
    def test_build_dataset_from_cfg_without_optional_keys(self, mock_load_dataset, _mock_logger):
        """Test dataset building without optional infer_cfg and eval_cfg"""
        mock_load_dataset.build.return_value = MockDataset(abbr="simple")

        cfg = ConfigDict(
            {
                "type": "SimpleDataset",
                "abbr": "simple",
                "path": "ais_bench/datasets/simple",
            }
        )

        result = build_dataset_from_cfg(cfg)

        self.assertEqual(result.abbr, "simple")
        mock_load_dataset.build.assert_called_once()

    @patch("ais_bench.benchmark.utils.config.build.logger")
    @patch("ais_bench.benchmark.utils.config.build.PERF_METRIC_CALCULATORS")
    def test_build_perf_metric_calculator_from_cfg(self, mock_calculators, mock_logger):
        """Test performance metric calculator building with deep copy"""
        mock_calc = MagicMock()
        mock_calculators.build.return_value = mock_calc

        cfg = ConfigDict(
            {
                "type": "DefaultPerfMetricCalculator",
                "metrics": ["latency", "throughput"],
                "percentiles": [50, 90, 95, 99],
            }
        )

        result = build_perf_metric_calculator_from_cfg(cfg)

        # Verify deep copy (passed config is not the same object)
        passed_cfg = mock_calculators.build.call_args[0][0]
        self.assertIsNot(passed_cfg, cfg)
        self.assertEqual(passed_cfg["metrics"], ["latency", "throughput"])
        self.assertEqual(passed_cfg["percentiles"], [50, 90, 95, 99])
        self.assertEqual(result, mock_calc)
        mock_logger.debug.assert_called()


if __name__ == "__main__":
    unittest.main()
