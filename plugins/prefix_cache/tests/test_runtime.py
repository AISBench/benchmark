import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from ais_bench_prefix_cache.artifacts import artifact_paths
from ais_bench_prefix_cache.config import _manifest, build_dataset_config, build_model_config
from ais_bench_prefix_cache.metrics import MetricSnapshot, RankMetrics
from ais_bench_prefix_cache.runtime import render_aisbench_config, run_aisbench_with_polling, run_scenario
from ais_bench_prefix_cache.scenario import load_scenario, with_execution_timestamp
from tests.test_pipeline import write_case


class RuntimeIntegrationTest(unittest.TestCase):
    def test_dataset_config_reads_user_settings_only_from_scenario(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = write_case(root)
            raw = json.loads(source.read_text(encoding="utf-8"))
            raw["aisbench"] = {
                "dataset": {
                    "abbr": "custom-dataset",
                    "input_columns": ["question", "max_out_len"],
                    "output_column": "answer",
                    "prompt_template": "{question}",
                    "pred_role": "ASSISTANT",
                }
            }
            source.write_text(json.dumps(raw), encoding="utf-8")
            scenario = load_scenario(source)
            paths = artifact_paths(scenario.output_dir, scenario.run_id)
            paths.manifest.parent.mkdir(parents=True, exist_ok=True)
            paths.manifest.write_text(
                json.dumps({
                    "run_id": scenario.run_id,
                    "artifacts": {
                        "requests": {"path": str(paths.requests)},
                        "full": {"path": str(paths.full)},
                    },
                }),
                encoding="utf-8",
            )

            class FakeAccEvaluator:
                pass

            class FakePromptTemplate:
                pass

            class FakeZeroRetriever:
                pass

            class FakePrefixCacheDataset:
                pass

            class FakePrefixCacheGenInferencer:
                pass

            module_values = {
                "ais_bench.benchmark.openicl.icl_evaluator": ("AccEvaluator", FakeAccEvaluator),
                "ais_bench.benchmark.openicl.icl_prompt_template": ("PromptTemplate", FakePromptTemplate),
                "ais_bench.benchmark.openicl.icl_retriever": ("ZeroRetriever", FakeZeroRetriever),
                "ais_bench_prefix_cache.datasets": ("PrefixCacheDataset", FakePrefixCacheDataset),
                "ais_bench_prefix_cache.openicl.icl_inferencer": ("PrefixCacheGenInferencer", FakePrefixCacheGenInferencer),
            }
            modules = {}
            for name, (attribute, value) in module_values.items():
                module = ModuleType(name)
                setattr(module, attribute, value)
                modules[name] = module
            with (
                patch.dict(sys.modules, modules),
                patch.dict(os.environ, {"AISBENCH_PREFIX_CACHE_MANIFEST": str(paths.manifest)}),
            ):
                dataset_config = build_dataset_config(source)

            self.assertEqual(dataset_config["abbr"], "custom-dataset")
            self.assertIs(dataset_config["type"], FakePrefixCacheDataset)
            self.assertEqual(dataset_config["reader_cfg"], {
                "input_columns": ["question", "max_out_len"],
                "output_column": "answer",
            })
            self.assertEqual(dataset_config["infer_cfg"]["prompt_template"]["template"], "{question}")
            self.assertEqual(dataset_config["eval_cfg"]["pred_role"], "ASSISTANT")

    def test_model_config_enables_streaming_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = write_case(root)
            scenario = load_scenario(source)
            paths = artifact_paths(scenario.output_dir, scenario.run_id)
            paths.manifest.parent.mkdir(parents=True, exist_ok=True)
            paths.manifest.write_text(
                json.dumps({"run_id": scenario.run_id}),
                encoding="utf-8",
            )

            class FakeVLLMPrefixCacheAPI:
                pass

            models_module = ModuleType("ais_bench_prefix_cache.models")
            models_module.VLLMPrefixCacheAPI = FakeVLLMPrefixCacheAPI
            with (
                patch.dict(sys.modules, {models_module.__name__: models_module}),
                patch.dict(
                    os.environ,
                    {"AISBENCH_PREFIX_CACHE_MANIFEST": str(paths.manifest)},
                ),
            ):
                model_config = build_model_config(source)

            self.assertIs(model_config["type"], FakeVLLMPrefixCacheAPI)
            self.assertIs(model_config["stream"], True)
            self.assertEqual(model_config["max_out_len"], 1)
            self.assertEqual(model_config["retry"], 2)
            self.assertEqual(model_config["batch_size"], 1)
            self.assertEqual(model_config["generation_kwargs"], {"temperature": 0, "ignore_eos": True})

    def test_model_config_reads_user_settings_only_from_scenario(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = write_case(root)
            raw = json.loads(source.read_text(encoding="utf-8"))
            raw["aisbench"] = {
                "model": {
                    "abbr": "custom-model",
                    "stream": False,
                    "max_out_len": 7,
                    "retry": 4,
                    "batch_size": 3,
                    "generation_kwargs": {
                        "temperature": 0.25,
                        "ignore_eos": False,
                        "top_p": 0.9,
                    },
                }
            }
            source.write_text(json.dumps(raw), encoding="utf-8")
            scenario = load_scenario(source)
            paths = artifact_paths(scenario.output_dir, scenario.run_id)
            paths.manifest.parent.mkdir(parents=True, exist_ok=True)
            paths.manifest.write_text(json.dumps({"run_id": scenario.run_id}), encoding="utf-8")

            class FakeVLLMPrefixCacheAPI:
                pass

            models_module = ModuleType("ais_bench_prefix_cache.models")
            models_module.VLLMPrefixCacheAPI = FakeVLLMPrefixCacheAPI
            with (
                patch.dict(sys.modules, {models_module.__name__: models_module}),
                patch.dict(os.environ, {"AISBENCH_PREFIX_CACHE_MANIFEST": str(paths.manifest)}),
            ):
                model_config = build_model_config(source)

            self.assertEqual(model_config["abbr"], "custom-model")
            self.assertIs(model_config["stream"], False)
            self.assertEqual(model_config["max_out_len"], 7)
            self.assertEqual(model_config["retry"], 4)
            self.assertEqual(model_config["batch_size"], 3)
            self.assertEqual(
                model_config["generation_kwargs"],
                {"temperature": 0.25, "ignore_eos": False, "top_p": 0.9},
            )

    def test_config_falls_back_to_latest_prepared_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = write_case(root)
            scenario = load_scenario(source)
            stamped = with_execution_timestamp(scenario, "20260827_123456")
            paths = artifact_paths(stamped.output_dir, stamped.run_id)
            paths.manifest.parent.mkdir(parents=True)
            paths.manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "status": "prepared",
                        "run_id": stamped.run_id,
                        "scenario_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "effective_config": stamped.to_effective_dict(),
                        "artifacts": {},
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ):
                os.environ.pop("AISBENCH_PREFIX_CACHE_MANIFEST", None)
                manifest_path, manifest = _manifest(scenario)

            self.assertEqual(manifest_path, paths.manifest)
            self.assertEqual(manifest["run_id"], stamped.run_id)

    def test_render_config_receives_exact_timestamped_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = write_case(root)
            scenario = with_execution_timestamp(load_scenario(source), "20260827_123456")
            expected = artifact_paths(scenario.output_dir, scenario.run_id).manifest
            config = root / "perf.py"
            config.write_text(
                "import os\n"
                f"assert os.environ['AISBENCH_PREFIX_CACHE_MANIFEST'] == {str(expected)!r}\n"
                "datasets = []\nmodels = []\ninfer = {}\n",
                encoding="utf-8",
            )

            generated = render_aisbench_config(config, scenario)

            self.assertTrue(generated.is_file())
            self.assertNotIn("AISBENCH_PREFIX_CACHE_MANIFEST", os.environ)

    def test_run_uses_timestamped_result_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = write_case(root, mode="cold")
            timestamp = "20260827_123456"
            stamped = with_execution_timestamp(load_scenario(source), timestamp)
            paths = artifact_paths(stamped.output_dir, stamped.run_id)

            def fake_prepare(path, progress, execution_timestamp):
                self.assertEqual(Path(path), source.resolve())
                self.assertEqual(execution_timestamp, timestamp)
                paths.manifest.parent.mkdir(parents=True, exist_ok=True)
                paths.manifest.write_text(
                    json.dumps(
                        {
                            "scenario_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "warmup": {"plan": []},
                        }
                    ),
                    encoding="utf-8",
                )
                paths.analysis.write_text(
                    json.dumps(
                        {
                            "theoretical_hit_rate": 0.5,
                            "warnings": [],
                            "validation": {"status": "PASS"},
                        }
                    ),
                    encoding="utf-8",
                )
                return paths

            before = MetricSnapshot(
                {0: RankMetrics(10, 5), 1: RankMetrics(20, 10)},
                {"queries": "q", "hits": "h"},
                "before",
            )
            after = MetricSnapshot(
                {0: RankMetrics(20, 10), 1: RankMetrics(40, 20)},
                {"queries": "q", "hits": "h"},
                "after",
            )

            class FakeClient:
                def __init__(self, scenario):
                    self.snapshots = iter((before, after))

                def precheck(self):
                    return {"ok": True}

                def reset(self):
                    return []

                def snapshot(self):
                    return next(self.snapshots)

            class FakePopen:
                def __init__(self, command, env=None):
                    pass

                def poll(self):
                    return 0

                def wait(self):
                    return 0

            with (
                patch("ais_bench_prefix_cache.runtime.prepare_scenario", side_effect=fake_prepare) as prepare,
                patch("ais_bench_prefix_cache.runtime.validate_artifacts"),
                patch("ais_bench_prefix_cache.runtime.VLLMClient", FakeClient),
                patch("ais_bench_prefix_cache.runtime.render_aisbench_config", return_value=root / "generated.py"),
                patch("ais_bench_prefix_cache.runtime.subprocess.Popen", FakePopen),
            ):
                result = run_scenario(source, execution_timestamp=timestamp)

            prepare.assert_called_once()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["actual"]["global_hit_rate"], 0.5)
            self.assertEqual(result["runtime"]["kv_cache_polling"]["count"], 0)
            self.assertIn("global_kv_cache_usage_peak", result["actual"])
            self.assertTrue(paths.analysis.is_file())

    def test_run_aisbench_with_polling_samples_kv_until_exit(self):
        class FakeProcess:
            def __init__(self, command, env=None):
                self.calls = 0

            def poll(self):
                self.calls += 1
                return None if self.calls < 4 else 0

            def wait(self):
                return 0

        snapshots = iter([
            MetricSnapshot({0: RankMetrics(1, 0, 0.5)}, {"queries": "q", "hits": "h"}, ""),
            MetricSnapshot({0: RankMetrics(2, 0, 0.8)}, {"queries": "q", "hits": "h"}, ""),
            MetricSnapshot({0: RankMetrics(3, 0, 0.4)}, {"queries": "q", "hits": "h"}, ""),
        ])

        class FakeClient:
            def snapshot(self):
                return next(snapshots)

        with (
            patch("ais_bench_prefix_cache.runtime.subprocess.Popen", FakeProcess),
            patch("ais_bench_prefix_cache.runtime.time.sleep"),
        ):
            returncode, samples = run_aisbench_with_polling(["cmd"], {}, FakeClient(), 0.1)

        self.assertEqual(returncode, 0)
        self.assertEqual([list(row.values())[0] for _, row in samples], [0.5, 0.8, 0.4])


if __name__ == "__main__":
    unittest.main()
