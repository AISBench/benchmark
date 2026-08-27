import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ais_bench_prefix_cache.artifacts import artifact_paths
from ais_bench_prefix_cache.config import _manifest
from ais_bench_prefix_cache.metrics import MetricSnapshot, RankMetrics
from ais_bench_prefix_cache.runtime import render_aisbench_config, run_scenario
from ais_bench_prefix_cache.scenario import load_scenario, with_execution_timestamp
from tests.test_pipeline import write_case


class RuntimeIntegrationTest(unittest.TestCase):
    def test_config_falls_back_to_timestamp_pointer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = write_case(root)
            scenario = load_scenario(source)
            stamped = with_execution_timestamp(scenario, "20260827_123456")
            paths = artifact_paths(stamped.output_dir, stamped.run_id)
            paths.manifest.parent.mkdir(parents=True)
            paths.manifest.write_text(json.dumps({"run_id": stamped.run_id}), encoding="utf-8")
            (root / "out.inspect.json").write_text(
                json.dumps(
                    {
                        "timestamp": "20260827_123456",
                        "run_id": scenario.run_id,
                        "output_dir": str(scenario.output_dir),
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

            with (
                patch("ais_bench_prefix_cache.runtime.prepare_scenario", side_effect=fake_prepare) as prepare,
                patch("ais_bench_prefix_cache.runtime.validate_artifacts"),
                patch("ais_bench_prefix_cache.runtime.VLLMClient", FakeClient),
                patch("ais_bench_prefix_cache.runtime.render_aisbench_config", return_value=root / "generated.py"),
                patch("ais_bench_prefix_cache.runtime.subprocess.run", return_value=SimpleNamespace(returncode=0)),
            ):
                result = run_scenario(source, execution_timestamp=timestamp)

            prepare.assert_called_once()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["actual"]["global_hit_rate"], 0.5)
            self.assertTrue(paths.analysis.is_file())


if __name__ == "__main__":
    unittest.main()
