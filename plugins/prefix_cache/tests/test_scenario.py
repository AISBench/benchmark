import json
import tempfile
import unittest
from pathlib import Path

from ais_bench_prefix_cache.errors import ScenarioValidationError
from ais_bench_prefix_cache.scenario import (
    Scenario,
    _minimum_input_tokens,
    _mode,
    _positive,
    _require_dict,
    _strict_keys,
    _validate_input_config,
    _validate_output_config,
    load_scenario,
    with_execution_timestamp,
)
from tests.test_core import scenario_dict


class ValidationHelpersTest(unittest.TestCase):
    def test_require_dict_rejects_non_object(self):
        with self.assertRaisesRegex(ScenarioValidationError, "must be an object"):
            _require_dict([], "run")

    def test_strict_keys_rejects_unknown_top_level_and_nested(self):
        with self.assertRaisesRegex(ScenarioValidationError, "unknown field: bogus"):
            _strict_keys({"bogus": 1}, "")
        with self.assertRaisesRegex(ScenarioValidationError, "unknown field: run.bogus"):
            _strict_keys({"run": {"bogus": 1}}, "")
        # 合法嵌套结构应静默通过。
        _strict_keys({"run": {"run_id": "x"}, "service": {"model": "m"}}, "")

    def test_positive_rejects_bool_zero_negative_and_non_int(self):
        for value in (True, 0, -1, "3", 3.0):
            with self.assertRaisesRegex(ScenarioValidationError, "positive integer"):
                _positive(value, "x")
        self.assertEqual(_positive(3, "x"), 3)

    def test_mode_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ScenarioValidationError, "mode must be one of"):
            _mode({"mode": "bogus"}, {"fixed", "csv"}, "requests.input_length")
        self.assertEqual(_mode({"mode": "csv"}, {"fixed", "csv"}, "p"), "csv")


class InputConfigTest(unittest.TestCase):
    def test_explicit_mode_valid(self):
        config = {"mode": "explicit", "values": [32, 32, 32, 32]}
        _validate_input_config(config, "requests.input_length", Path("."), 4)

    def test_explicit_mode_rejects_bad_values(self):
        with self.assertRaisesRegex(ScenarioValidationError, "non-empty list"):
            _validate_input_config({"mode": "explicit", "values": []}, "p", Path("."), None)
        with self.assertRaisesRegex(ScenarioValidationError, "non-empty list"):
            _validate_input_config({"mode": "explicit", "values": "32"}, "p", Path("."), None)
        with self.assertRaisesRegex(ScenarioValidationError, "positive integer"):
            _validate_input_config({"mode": "explicit", "values": [32, -1]}, "p", Path("."), 2)
        with self.assertRaisesRegex(ScenarioValidationError, "must equal expected request count"):
            _validate_input_config({"mode": "explicit", "values": [32, 32]}, "p", Path("."), 3)

    def test_range_mode_valid(self):
        config = {
            "mode": "range",
            "ranges": [
                {"min": 16, "max": 32, "count": 2},
                {"min": 64, "max": 64, "count": 2},
            ],
        }
        _validate_input_config(config, "p", Path("."), 4)

    def test_range_mode_rejects_bad_ranges(self):
        with self.assertRaisesRegex(ScenarioValidationError, "non-empty list"):
            _validate_input_config({"mode": "range", "ranges": []}, "p", Path("."), None)
        with self.assertRaisesRegex(ScenarioValidationError, "invalid fields"):
            _validate_input_config({"mode": "range", "ranges": ["x"]}, "p", Path("."), None)
        with self.assertRaisesRegex(ScenarioValidationError, "invalid fields"):
            _validate_input_config({"mode": "range", "ranges": [{"min": 1, "max": 2, "count": 1, "bogus": 1}]}, "p", Path("."), None)
        with self.assertRaisesRegex(ScenarioValidationError, "must be >= min"):
            _validate_input_config({"mode": "range", "ranges": [{"min": 8, "max": 4, "count": 1}]}, "p", Path("."), None)
        with self.assertRaisesRegex(ScenarioValidationError, "must equal expected request count"):
            _validate_input_config({"mode": "range", "ranges": [{"min": 8, "max": 16, "count": 2}]}, "p", Path("."), 3)

    def test_truncated_normal_valid(self):
        _validate_input_config({"mode": "truncated_normal", "min": 16, "max": 64, "std": 8}, "p", Path("."), None)
        # std 缺省时不校验，也合法。
        _validate_input_config({"mode": "truncated_normal", "min": 16, "max": 64}, "p", Path("."), None)

    def test_truncated_normal_rejects_bad_bounds(self):
        with self.assertRaisesRegex(ScenarioValidationError, "must be >= min"):
            _validate_input_config({"mode": "truncated_normal", "min": 64, "max": 16}, "p", Path("."), None)
        with self.assertRaisesRegex(ScenarioValidationError, "std must be positive"):
            _validate_input_config({"mode": "truncated_normal", "min": 16, "max": 64, "std": 0}, "p", Path("."), None)

    def test_csv_mode_resolves_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {"mode": "csv", "path": "lens.csv"}
            _validate_input_config(config, "p", root, None)
            self.assertEqual(config["path"], str((root / "lens.csv").resolve()))

    def test_csv_mode_rejects_missing_path(self):
        for path in (None, ""):
            with self.assertRaisesRegex(ScenarioValidationError, "non-empty string"):
                _validate_input_config({"mode": "csv", "path": path}, "p", Path("."), None)

    def test_unknown_field_rejected(self):
        with self.assertRaisesRegex(ScenarioValidationError, "unknown field: p.bogus"):
            _validate_input_config({"mode": "fixed", "value": 32, "bogus": 1}, "p", Path("."), None)


class OutputConfigTest(unittest.TestCase):
    def test_uniform_valid(self):
        _validate_output_config({"mode": "uniform", "min": 1, "max": 8}, "p", Path("."))

    def test_uniform_rejects_max_below_min(self):
        with self.assertRaisesRegex(ScenarioValidationError, "must be >= min"):
            _validate_output_config({"mode": "uniform", "min": 8, "max": 1}, "p", Path("."))

    def test_truncated_normal_rejects_non_positive_std(self):
        with self.assertRaisesRegex(ScenarioValidationError, "std must be positive"):
            _validate_output_config({"mode": "truncated_normal", "min": 1, "max": 8, "std": 0}, "p", Path("."))

    def test_csv_mode_resolves_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {"mode": "csv", "path": "lens.csv"}
            _validate_output_config(config, "p", root)
            self.assertEqual(config["path"], str((root / "lens.csv").resolve()))

    def test_csv_mode_rejects_missing_path(self):
        with self.assertRaisesRegex(ScenarioValidationError, "non-empty string"):
            _validate_output_config({"mode": "csv", "path": ""}, "p", Path("."))

    def test_unknown_field_rejected(self):
        with self.assertRaisesRegex(ScenarioValidationError, "unknown field: p.bogus"):
            _validate_output_config({"mode": "uniform", "min": 1, "max": 8, "bogus": 1}, "p", Path("."))


class MinimumInputTokensTest(unittest.TestCase):
    def test_mode_dispatch(self):
        self.assertEqual(_minimum_input_tokens({"mode": "fixed", "value": 32}, "p"), 32)
        self.assertEqual(_minimum_input_tokens({"mode": "explicit", "values": [64, 32]}, "p"), 32)
        self.assertEqual(_minimum_input_tokens({"mode": "range", "ranges": [{"min": 48}]}, "p"), 48)
        self.assertEqual(_minimum_input_tokens({"mode": "truncated_normal", "min": 24}, "p"), 24)

    def test_csv_reads_min_of_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for alias in ("input_prompt_tokens", "content_tokens", "input_tokens"):
                csv_path = root / f"{alias}.csv"
                csv_path.write_text(f"{alias}\n32\n48\n", encoding="utf-8")
                self.assertEqual(_minimum_input_tokens({"mode": "csv", "path": str(csv_path)}, "p"), 32)

    def test_csv_handles_bom_header(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            csv_path = root / "lens.csv"
            csv_path.write_text("﻿input_tokens\n40\n", encoding="utf-8")
            self.assertEqual(_minimum_input_tokens({"mode": "csv", "path": str(csv_path)}, "p"), 40)

    def test_csv_empty_rows_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            csv_path = root / "lens.csv"
            csv_path.write_text("input_tokens\n", encoding="utf-8")
            with self.assertRaisesRegex(ScenarioValidationError, "at least one data row"):
                _minimum_input_tokens({"mode": "csv", "path": str(csv_path)}, "p")

    def test_csv_missing_column_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            csv_path = root / "lens.csv"
            csv_path.write_text("bogus\n32\n", encoding="utf-8")
            with self.assertRaisesRegex(ScenarioValidationError, "requires one of columns"):
                _minimum_input_tokens({"mode": "csv", "path": str(csv_path)}, "p")

    def test_csv_invalid_value_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            csv_path = root / "lens.csv"
            csv_path.write_text("input_tokens\nabc\n", encoding="utf-8")
            with self.assertRaisesRegex(ScenarioValidationError, "invalid input length"):
                _minimum_input_tokens({"mode": "csv", "path": str(csv_path)}, "p")

    def test_csv_unreadable_rejected(self):
        with self.assertRaisesRegex(ScenarioValidationError, "cannot be read"):
            _minimum_input_tokens({"mode": "csv", "path": "/nonexistent/lens.csv"}, "p")


class ScenarioObjectTest(unittest.TestCase):
    def test_section_and_to_effective_dict_are_independent(self):
        scenario = Scenario(Path("s.json"), {"run": {"run_id": "r"}})
        self.assertEqual(scenario.section("run")["run_id"], "r")
        effective = scenario.to_effective_dict()
        effective["run"]["run_id"] = "changed"
        self.assertEqual(scenario.run_id, "r")

    def test_with_execution_timestamp_rejects_malformed_timestamps(self):
        scenario = Scenario(Path("s.json"), {"run": {"run_id": "r", "output_dir": "/tmp/out"}})
        for timestamp in ("2026-08-25 12:00:00", "20260825123456", "abcd1234_5678"):
            with self.assertRaisesRegex(ScenarioValidationError, "YYYYMMDD_HHMMSS"):
                with_execution_timestamp(scenario, timestamp)

    def test_with_execution_timestamp_rejects_empty_output_dir_name(self):
        scenario = Scenario(Path("s.json"), {"run": {"run_id": "r", "output_dir": "/"}})
        with self.assertRaisesRegex(ScenarioValidationError, "final directory name"):
            with_execution_timestamp(scenario, "20260825_123456")


class LoadScenarioErrorTest(unittest.TestCase):
    def _write(self, folder, content) -> Path:
        path = Path(folder) / "scenario.json"
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        return path

    def _expect(self, content, pattern):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ScenarioValidationError, pattern):
                load_scenario(self._write(folder, content))

    def test_non_object_scenario_rejected(self):
        self._expect("[1, 2]", "must be an object")

    def test_unknown_field_rejected(self):
        self._expect({"run": {}, "bogus": 1}, "unknown field: bogus")
        self._expect({"run": {"bogus": 1}}, "unknown field: run.bogus")

    def test_schema_version_rejected(self):
        self._expect({"schema_version": "2.0"}, "schema_version")

    def test_run_id_rejected(self):
        self._expect({"run": {"run_id": ""}}, "run.run_id")
        self._expect({"run": {"run_id": 123}}, "run.run_id")

    def test_random_seed_rejected(self):
        self._expect({"run": {"random_seed": True}}, "random_seed")
        self._expect({"run": {"random_seed": "42"}}, "random_seed")

    def test_block_size_rejected(self):
        self._expect({"tokenizer": {"block_size": 0}}, "block_size")

    def test_input_length_must_be_object(self):
        self._expect({"requests": {"input_length": "x"}}, "input_length must be an object")

    def test_target_hit_rate_rejected(self):
        self._expect({"prefix_cache": {"target_hit_rate": 1.5}}, "target_hit_rate")
        self._expect({"prefix_cache": {"target_hit_rate": True}}, "target_hit_rate")

    def test_seed_blocks_rejected(self):
        self._expect({"prefix_cache": {"seed_blocks": 0}}, "seed_blocks")

    def test_minimum_non_shared_below_seed_rejected(self):
        self._expect({"prefix_cache": {"minimum_non_shared_length": 8}}, "non_shared_length")

    def test_input_too_small_for_reserved_region_rejected(self):
        self._expect(
            {"requests": {"input_length": {"mode": "fixed", "value": 8}}},
            "at least 16 tokens",
        )

    def test_groups_must_be_object(self):
        self._expect({"prefix_cache": {"groups": "x"}}, "groups must be an object")

    def test_groups_count_rejected(self):
        self._expect({"prefix_cache": {"groups": {"count": 0}}}, "groups.count")

    def test_assignment_mode_rejected(self):
        self._expect(
            {"prefix_cache": {"groups": {"assignment": {"mode": "bogus"}}}},
            "assignment.mode must be one of",
        )

    def test_overrides_must_be_object(self):
        self._expect({"prefix_cache": {"groups": {"overrides": []}}}, "overrides must be an object")

    def test_override_invalid_group_id_rejected(self):
        self._expect(
            {"prefix_cache": {"groups": {"overrides": {"bad": {}}}}},
            "invalid Prefix Group override id",
        )
        self._expect(
            {"prefix_cache": {"groups": {"count": 2, "overrides": {"group-9": {}}}}},
            "invalid Prefix Group override id",
        )

    def test_override_must_be_object(self):
        self._expect(
            {"prefix_cache": {"groups": {"overrides": {"group-0": []}}}},
            "must be an object",
        )

    def test_override_unknown_field_rejected(self):
        self._expect(
            {"prefix_cache": {"groups": {"overrides": {"group-0": {"bogus": 1}}}}},
            "unknown field: prefix_cache.groups.overrides.group-0.bogus",
        )

    def test_override_input_length_too_small_rejected(self):
        self._expect(
            {"prefix_cache": {"groups": {"overrides": {"group-0": {"input_length": {"mode": "fixed", "value": 8}}}}}},
            "at least 16 tokens",
        )

    def test_override_corpus_selection_mode_rejected(self):
        self._expect(
            {"prefix_cache": {"groups": {"overrides": {"group-0": {"corpus_selection": {"mode": "bogus"}}}}}},
            "corpus_selection.mode must be one of",
        )

    def test_order_strategy_rejected(self):
        self._expect({"prefix_cache": {"order": {"strategy": "bogus"}}}, "order.strategy")

    def test_service_fields_rejected(self):
        for field in ("inference_url", "metrics_url", "model"):
            self._expect({"service": {field: ""}}, f"service.{field}")

    def test_unreadable_file_rejected(self):
        with self.assertRaisesRegex(ScenarioValidationError, "cannot read scenario"):
            load_scenario(Path("/nonexistent/scenario.json"))

    def test_invalid_json_rejected(self):
        self._expect("not json", "cannot read scenario")


class LoadScenarioMultimodeTest(unittest.TestCase):
    def test_valid_explicit_zipf_scenario_with_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = scenario_dict(root)
            data["requests"] = {
                "count": 4,
                "input_length": {"mode": "explicit", "values": [32, 32, 32, 32]},
                "output_length": {"mode": "uniform", "min": 1, "max": 8},
            }
            data["corpus"]["selection"] = {"mode": "indices", "indices": [0, 1]}
            data["prefix_cache"] = {
                "mode": "cold",
                "target_hit_rate": 0.5,
                "seed_blocks": 1,
                "groups": {
                    "count": 2,
                    "assignment": {"mode": "zipf", "exponent": 1.2},
                    "overrides": {
                        "group-0": {
                            "input_length": {"mode": "fixed", "value": 32},
                            "output_length": {"mode": "uniform", "min": 1, "max": 4},
                            "corpus_selection": {"mode": "indices"},
                        }
                    },
                },
                "order": {"strategy": "input_len_asc"},
            }
            path = root / "scenario.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            scenario = load_scenario(path)
            # 多态配置不应被注入 fixed 模式的默认字段。
            self.assertNotIn("value", scenario.data["requests"]["input_length"])
            self.assertNotIn("value", scenario.data["requests"]["output_length"])
            self.assertEqual(scenario.data["prefix_cache"]["groups"]["assignment"]["mode"], "zipf")
            self.assertEqual(scenario.data["prefix_cache"]["order"]["strategy"], "input_len_asc")

    def test_valid_csv_input_scenario(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            csv_path = root / "lens.csv"
            csv_path.write_text("input_tokens\n32\n40\n", encoding="utf-8")
            data = scenario_dict(root)
            data["requests"] = {
                "count": 8,
                "input_length": {"mode": "csv", "path": str(csv_path)},
                "output_length": {"mode": "fixed", "value": 2},
            }
            path = root / "scenario.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            scenario = load_scenario(path)
            self.assertEqual(scenario.data["requests"]["input_length"]["path"], str(csv_path.resolve()))


if __name__ == "__main__":
    unittest.main()
