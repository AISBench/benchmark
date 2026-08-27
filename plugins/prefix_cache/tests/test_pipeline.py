import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ais_bench_prefix_cache.artifacts import read_jsonl, sha256_file, validate_artifacts
from ais_bench_prefix_cache.pipeline import _length_summary, inspect_scenario, prepare_scenario
from tests.test_core import scenario_dict


class FakeTokenizer:
    all_special_ids = list(range(32))

    def __len__(self):
        return 128

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)


def write_case(root: Path, mode: str = "cold") -> Path:
    questions = ["alpha arithmetic question", "beta arithmetic question", "gamma arithmetic question", "delta arithmetic question"]
    (root / "gsm.jsonl").write_text("".join(json.dumps({"question": value}) + "\n" for value in questions), encoding="utf-8")
    data = scenario_dict(root, mode=mode)
    path = root / "scenario.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class PipelineTest(unittest.TestCase):
    def test_four_artifacts_and_minimal_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            scenario = write_case(Path(folder))
            paths = prepare_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
            self.assertTrue(all(path.exists() for path in paths.__dict__.values()))
            self.assertTrue(all(path.parent.name == "result" for path in paths.__dict__.values()))
            requests = read_jsonl(paths.requests)
            self.assertEqual(set(requests[0]), {"question", "answer", "max_tokens"})
            first_line = paths.requests.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(list(json.loads(first_line)), ["question", "answer", "max_tokens"])
            self.assertTrue(validate_artifacts(paths.manifest)["ok"])

    def test_prepare_reports_each_generated_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            scenario = write_case(Path(folder))
            events = []
            prepare_scenario(
                scenario,
                tokenizer_loader=lambda _: FakeTokenizer(),
                progress=lambda completed, total: events.append((completed, total)),
            )
            self.assertEqual(events, [(completed, 8) for completed in range(9)])

    def test_prepare_appends_one_timestamp_to_run_id_and_output_dir(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = write_case(root)
            timestamp = "20260825_123456"
            paths = prepare_scenario(
                scenario,
                tokenizer_loader=lambda _: FakeTokenizer(),
                execution_timestamp=timestamp,
            )
            expected_root = root / f"out_{timestamp}"
            self.assertEqual(paths.manifest.parent, expected_root / "result")
            self.assertEqual(paths.manifest.name, f"pc-test_{timestamp}.manifest.json")
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_id"], f"pc-test_{timestamp}")
            self.assertEqual(manifest["effective_config"]["run"]["output_dir"], str(expected_root))

    def test_inspect_reports_reachability_without_persisting_run_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = write_case(root)
            summary = inspect_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
            self.assertIn("reachable_min", summary)
            self.assertIn("reachable_max", summary)
            self.assertEqual(sum(summary["groups"].values()), 8)
            self.assertFalse(summary["sends_requests"])
            self.assertFalse((root / "out").exists())

    def test_deterministic_content_hashes(self):
        hashes = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as folder:
                scenario = write_case(Path(folder))
                paths = prepare_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
                hashes.append((sha256_file(paths.full), sha256_file(paths.requests)))
        self.assertEqual(hashes[0], hashes[1])

    def test_manifest_does_not_persist_api_key(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = write_case(root)
            data = json.loads(scenario.read_text(encoding="utf-8"))
            data["service"]["api_key"] = "do-not-write-this-secret"
            scenario.write_text(json.dumps(data), encoding="utf-8")
            paths = prepare_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
            manifest_text = paths.manifest.read_text(encoding="utf-8")
            self.assertNotIn("do-not-write-this-secret", manifest_text)
            manifest = json.loads(manifest_text)
            self.assertTrue(manifest["effective_config"]["service"]["api_key_configured"])

    def test_manifest_contains_detailed_audit_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            scenario = write_case(Path(folder))
            paths = prepare_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
            analysis = json.loads(paths.analysis.read_text(encoding="utf-8"))
            rows = read_jsonl(paths.full)
            self.assertIn("p99", manifest["requests"]["input_length_summary"])
            self.assertIn("bins", manifest["requests"]["input_length_summary"])
            self.assertIn("target_reachable", manifest["prefix_cache"])
            self.assertTrue(all("reachable_max" in group for group in manifest["groups"].values()))
            self.assertEqual(manifest["divergence"]["collision_status"], "pass")
            self.assertEqual(len({row["request_random_seed"] for row in rows}), len(rows))
            self.assertTrue(all(row["divergence_unique"] for row in rows))
            self.assertIn(analysis["validation"]["status"], {"PASS", "PASS_WITH_WARNING"})
            self.assertFalse(analysis["validation"]["affects_exit_code"])
            self.assertIn("target_signed_difference_pp", analysis)

    def test_length_summary_bins_report_actual_value_min_max(self):
        # bins 中每项的 min/max 必须是桶内实际取值的最小/最大值，
        # 而非桶边界：count == 1 时二者必须相等。
        values = [1024, 2048, 512, 768, 1024]
        summary = _length_summary(values)
        self.assertEqual(
            summary["bins"],
            [
                {"min": 512, "max": 512, "count": 1},
                {"min": 768, "max": 768, "count": 1},
                {"min": 1024, "max": 1024, "count": 2},
                {"min": 2048, "max": 2048, "count": 1},
            ],
        )
        for entry in summary["bins"]:
            if entry["count"] == 1:
                self.assertEqual(entry["min"], entry["max"])
        self.assertEqual(sum(entry["count"] for entry in summary["bins"]), len(values))
        self.assertEqual(summary["min"], 512)
        self.assertEqual(summary["max"], 2048)

    def test_warmup_manifest_has_every_group_rank(self):
        with tempfile.TemporaryDirectory() as folder:
            scenario = write_case(Path(folder), mode="warmup")
            paths = prepare_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
            pairs = {(row["group_id"], row["dp_rank"]) for row in manifest["warmup"]["plan"]}
            self.assertEqual(pairs, {(f"group-{group}", rank) for group in range(2) for rank in range(2)})
            self.assertTrue(all(not row["included_in_formal_statistics"] for row in manifest["warmup"]["plan"]))

    def test_group_override_and_multi_sample_suffix(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = write_case(root)
            data = json.loads(scenario.read_text(encoding="utf-8"))
            data["prefix_cache"]["groups"]["overrides"] = {
                "group-0": {
                    "input_length": {"mode": "fixed", "value": 80},
                    "corpus_selection": {"mode": "indices", "values": [0, 1]},
                }
            }
            data["prefix_cache"]["target_hit_rate"] = 0.0
            scenario.write_text(json.dumps(data), encoding="utf-8")
            paths = prepare_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
            rows = [row for row in read_jsonl(paths.full) if row["group_id"] == "group-0"]
            self.assertTrue(all(row["actual_input_tokens"] == 80 for row in rows))
            self.assertTrue(any(len(set(row["gsm_indices"])) >= 2 for row in rows))

if __name__ == "__main__":
    unittest.main()
