"""Unit tests for ais_bench.benchmark.tasks.swebench.swebench_infer"""

import unittest
import os
import sys
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ais_bench.benchmark.tasks.swebench import swebench_infer as infer_module


class TestGetMinisweagentConfig(unittest.TestCase):
    """Test _get_minisweagent_config function for SWE-bench (non-Pro)."""

    @classmethod
    def setUpClass(cls):
        cls.infer_module = infer_module

    def test_returns_dict_with_model_key(self):
        cfg = {"model": "test-model", "generation_kwargs": {"temperature": 0.7}}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertIn("model", result)
        self.assertIn("model_name", result["model"])
        self.assertIn("model_class", result["model"])
        self.assertIn("model_kwargs", result["model"])

    def test_sets_default_timeout(self):
        """Default timeout should match DEFAULT_LITELLM_TIMEOUT (not LiteLLM default of 600s)."""
        cfg = {}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(
            result["model"]["model_kwargs"]["timeout"],
            self.infer_module.DEFAULT_LITELLM_TIMEOUT,
        )

    def test_user_timeout_overrides_default(self):
        """User-specified timeout in generation_kwargs should override default of 200."""
        cfg = {"generation_kwargs": {"timeout": 120}}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["model_kwargs"]["timeout"], 120)

    def test_handles_empty_config(self):
        result = self.infer_module._get_minisweagent_config({})
        self.assertIsInstance(result, dict)
        self.assertIn("model", result)

    def test_sets_model_name_from_model_key(self):
        cfg = {"model": "gpt-4"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["model_name"], "gpt-4")

    def test_sets_model_name_from_model_name_key(self):
        cfg = {"model_name": "claude-3"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["model_name"], "claude-3")

    def test_includes_api_key_in_model_kwargs(self):
        cfg = {"api_key": "sk-test-123"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["model_kwargs"]["api_key"], "sk-test-123")

    def test_includes_url_as_api_base(self):
        cfg = {"url": "https://api.example.com/v1"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(
            result["model"]["model_kwargs"]["api_base"], "https://api.example.com/v1"
        )

    def test_includes_cost_tracking(self):
        cfg = {"model": "test-model"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["cost_tracking"], "ignore_errors")

    def test_sets_litellm_class_by_default(self):
        cfg = {"model": "test-model"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["model_class"], "litellm")

    def test_sets_hosted_vllm_prefix(self):
        cfg = {"model": "qwen3", "url": "http://127.0.0.1:8000/v1"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["model_name"], "hosted_vllm/qwen3")


if __name__ == "__main__":
    unittest.main()
