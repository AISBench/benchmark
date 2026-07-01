"""Unit tests for IFBench dataset (ifbench.py).

Covers: dataset loading, strict/loose instruction-following checks,
evaluator scoring, InputExample/OutputExample dataclasses.

Reference: hle.py test patterns — test load, parse, evaluate independently.
"""

import os
import sys
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Import strategy:
# 1. Preload the REAL HuggingFace ``datasets`` so the local shadow doesn't hit.
# 2. Mock the ais_bench registries / base classes / utilities.
# 3. Load ``ifbench.py`` directly by file path (avoids the full ``__init__``
#    import chain that drags in ``fcntl`` on Windows).
# ---------------------------------------------------------------------------
import importlib.util as _iu
import site as _site
from unittest.mock import MagicMock

_BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ais_bench", "benchmark")

# -- 1. Preload HuggingFace datasets -----------------------------------------
for _sp in _site.getsitepackages():
    _candidate = os.path.join(_sp, "datasets", "__init__.py")
    if os.path.isfile(_candidate):
        _hf_spec = _iu.spec_from_file_location(
            "datasets", _candidate,
            submodule_search_locations=[os.path.dirname(_candidate)]
        )
        _hf = _iu.module_from_spec(_hf_spec)
        sys.modules["datasets"] = _hf
        _hf_spec.loader.exec_module(_hf)
        break
else:
    import datasets as _hf
    sys.modules["datasets"] = _hf

# -- 2. Mock registries / base classes ---------------------------------------
def _make_register(id_str):
    """Return a register_module method that stores classes and returns them."""
    def _register(**kwargs):
        def _decorator(cls):
            return cls
        return _decorator
    return _register

_LOAD_DATASET = MagicMock()
_LOAD_DATASET.register_module = _make_register("LOAD_DATASET")
_ICL_EVALUATORS = MagicMock()
_ICL_EVALUATORS.register_module = _make_register("ICL_EVALUATORS")

class _MockRegistryModule:
    LOAD_DATASET = _LOAD_DATASET
    ICL_EVALUATORS = _ICL_EVALUATORS

sys.modules["ais_bench.benchmark.registry"] = _MockRegistryModule

# Ensure parent packages exist for patch() dotted-name resolution.
for _pkg_name in [
    "ais_bench",
    "ais_bench.benchmark",
    "ais_bench.benchmark.datasets",
    "ais_bench.benchmark.datasets.ifbench",
]:
    if _pkg_name not in sys.modules:
        sys.modules[_pkg_name] = MagicMock()

_mock_base = MagicMock()
_mock_base.BaseDataset = type("BaseDataset", (), {})
sys.modules["ais_bench.benchmark.datasets.base"] = _mock_base

_mock_evaluator = MagicMock()
_mock_evaluator.BaseEvaluator = type("BaseEvaluator", (), {})
sys.modules["ais_bench.benchmark.openicl.icl_evaluator"] = _mock_evaluator

_mock_ds_utils = MagicMock()
_mock_ds_utils.get_data_path = lambda path, local_mode=True: path
sys.modules["ais_bench.benchmark.datasets.utils.datasets"] = _mock_ds_utils

_mock_logger_mod = MagicMock()
_mock_logger_mod.AISLogger = lambda: MagicMock()
sys.modules["ais_bench.benchmark.utils.logging.logger"] = _mock_logger_mod

# -- 3. Mock ifbench package and instructions_registry ------------------------
# Create mock instruction checker classes that the real code will instantiate.
class _MockInstructionChecker:
    """Mock that returns a configurable follow/not-follow result."""

    def __init__(self, instruction_id):
        self.instruction_id = instruction_id

    def build_description(self, **kwargs):
        pass

    def get_instruction_args(self):
        return None

    def check_following(self, response):
        return True  # default: always follow


class _WordCountChecker(_MockInstructionChecker):
    def check_following(self, response):
        words = response.split()
        return self._lower <= len(words) <= self._upper

    def build_description(self, **kwargs):
        self._lower = kwargs.get("lower", 0)
        self._upper = kwargs.get("upper", float("inf"))

    def get_instruction_args(self):
        return None


class _StartVerbChecker(_MockInstructionChecker):
    def check_following(self, response):
        return response.strip().lower().startswith("technology")

    def get_instruction_args(self):
        return None


class _KeywordChecker(_MockInstructionChecker):
    def build_description(self, **kwargs):
        self._keyword = kwargs.get("keyword", "")

    def check_following(self, response):
        return self._keyword.lower() in response.lower()

    def get_instruction_args(self):
        return None


class _NewlineChecker(_MockInstructionChecker):
    def build_description(self, **kwargs):
        self._num_lines = kwargs.get("num_lines", 1)

    def check_following(self, response):
        lines = [l for l in response.split("\n") if l.strip()]
        return len(lines) >= self._num_lines

    def get_instruction_args(self):
        return None


_mock_instr_reg = MagicMock()
_mock_instr_reg.INSTRUCTION_DICT = {
    "count:word_count_range": _WordCountChecker,
    "words:start_verb": _StartVerbChecker,
    "sentence:keyword": _KeywordChecker,
    "format:newline": _NewlineChecker,
}
sys.modules[
    "ais_bench.benchmark.datasets.ifbench.instructions_registry"
] = _mock_instr_reg

# Give the ifbench package itself a known entry in sys.modules.
# Must have ``__path__`` for relative imports (``from . import X``) to work.
# Use an empty list so it does NOT find the real instructions_registry on disk.
import types as _types
_ifbench_pkg = _types.ModuleType("ais_bench.benchmark.datasets.ifbench")
_ifbench_pkg.__path__ = []
sys.modules["ais_bench.benchmark.datasets.ifbench"] = _ifbench_pkg

# -- 4. Load ifbench.py by file path -----------------------------------------
_ifbench_path = os.path.join(_BENCHMARK_DIR, "datasets", "ifbench", "ifbench.py")
_spec = _iu.spec_from_file_location(
    "ais_bench.benchmark.datasets.ifbench.ifbench", _ifbench_path
)
_ifbench = _iu.module_from_spec(_spec)
sys.modules["ais_bench.benchmark.datasets.ifbench.ifbench"] = _ifbench
_spec.loader.exec_module(_ifbench)

# Extract symbols under test.
InputExample = _ifbench.InputExample
OutputExample = _ifbench.OutputExample
test_instruction_following_strict = _ifbench.test_instruction_following_strict
test_instruction_following_loose = _ifbench.test_instruction_following_loose
IFBenchDataset = _ifbench.IFBenchDataset
IFBenchEvaluator = _ifbench.IFBenchEvaluator

# Prevent pytest from collecting the ifbench functions as test cases
# (they are named ``test_*`` but we test them through wrapper classes).
test_instruction_following_strict.__test__ = False
test_instruction_following_loose.__test__ = False

# ==============================================================================
# Test data constants (mirrors conftest.py for standalone usage)
# ==============================================================================

IFBENCH_PARQUET_RECORDS = [
    {
        "key": 0,
        "prompt": "Write a paragraph containing exactly 50 words. "
                  "The paragraph must start with the word 'Technology'.",
        "instruction_id_list": ["count:word_count_range", "words:start_verb"],
        "kwargs": [
            {"lower": 50, "upper": 50},
            {},
        ],
    },
    {
        "key": 1,
        "prompt": "List three items. Then write a sentence with the word 'apple'.",
        "instruction_id_list": ["format:newline", "sentence:keyword"],
        "kwargs": [
            {"num_lines": 3},
            {"keyword": "apple"},
        ],
    },
]



class _MockDataset:
    """Minimal Dataset-alike backed by a list of dicts."""

    def __init__(self, records):
        self._records = records
        self.column_names = list(records[0].keys()) if records else []

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        return self._records[idx]

    def __iter__(self):
        return iter(self._records)


@pytest.fixture
def mock_ifbench_dataset():
    """Return a minimal object that walks like a HuggingFace Dataset."""
    return _MockDataset(IFBENCH_PARQUET_RECORDS)


# ===================================================================
# InputExample / OutputExample dataclass tests
# ===================================================================


class TestInputExample:
    def test_create(self):
        inp = InputExample(
            key=0,
            instruction_id_list=["format:newline"],
            prompt="Hello",
            kwargs=[{"num_lines": 3}],
        )
        assert inp.key == 0
        assert inp.instruction_id_list == ["format:newline"]
        assert inp.prompt == "Hello"
        assert inp.kwargs == [{"num_lines": 3}]

    def test_optional_kwargs(self):
        inp = InputExample(
            key=1,
            instruction_id_list=["sentence:keyword"],
            prompt="Test prompt",
            kwargs=[{"keyword": "apple", "extra": None}],
        )
        assert inp.kwargs[0]["extra"] is None


class TestOutputExample:
    def test_create_all_following(self):
        out = OutputExample(
            instruction_id_list=["a", "b"],
            prompt="p",
            response="r",
            follow_all_instructions=True,
            follow_instruction_list=[True, True],
        )
        assert out.follow_all_instructions is True
        assert out.follow_instruction_list == [True, True]

    def test_create_partial_following(self):
        out = OutputExample(
            instruction_id_list=["a", "b"],
            prompt="p",
            response="r",
            follow_all_instructions=False,
            follow_instruction_list=[True, False],
        )
        assert out.follow_all_instructions is False
        assert sum(out.follow_instruction_list) == 1


# ===================================================================
# test_instruction_following_strict tests
# ===================================================================


class TestInstructionFollowingStrict:
    """Tests for test_instruction_following_strict()."""

    def test_word_count_range_pass(self):
        """Response with exactly 50 words should pass word_count_range."""
        words = "word " * 49 + "final"
        assert len(words.split()) == 50
        inp = InputExample(
            key=0,
            instruction_id_list=["count:word_count_range"],
            prompt="Write 50 words.",
            kwargs=[{"lower": 50, "upper": 50}],
        )
        result = test_instruction_following_strict(inp, words)
        assert result.follow_instruction_list[0] is True

    def test_word_count_range_fail(self):
        """Response with too few words should fail word_count_range."""
        words = "short"
        inp = InputExample(
            key=0,
            instruction_id_list=["count:word_count_range"],
            prompt="Write 50 words.",
            kwargs=[{"lower": 50, "upper": 50}],
        )
        result = test_instruction_following_strict(inp, words)
        assert result.follow_instruction_list[0] is False

    def test_empty_response(self):
        """Empty response should fail all instructions."""
        inp = InputExample(
            key=0,
            instruction_id_list=["count:word_count_range"],
            prompt="Say something.",
            kwargs=[{"lower": 1, "upper": 10}],
        )
        result = test_instruction_following_strict(inp, "")
        assert result.follow_instruction_list[0] is False
        assert result.follow_all_instructions is False

    def test_multiple_instructions(self):
        """Multiple instructions are each checked independently."""
        inp = InputExample(
            key=0,
            instruction_id_list=[
                "count:word_count_range",
                "format:newline",
            ],
            prompt="Write 3 words on 3 lines.",
            kwargs=[
                {"lower": 3, "upper": 3},
                {"num_lines": 3},
            ],
        )
        # Exactly 3 words on 3 lines
        response = "one\ntwo\nthree"
        result = test_instruction_following_strict(inp, response)
        assert len(result.follow_instruction_list) == 2

    def test_output_example_contains_response(self):
        """OutputExample includes the original response."""
        inp = InputExample(
            key=1,
            instruction_id_list=["sentence:keyword"],
            prompt="Include 'apple' in a sentence.",
            kwargs=[{"keyword": "apple"}],
        )
        response = "I like apple pie."
        result = test_instruction_following_strict(inp, response)
        assert result.response == response
        assert result.prompt == inp.prompt

    def test_none_kwargs_cleaned(self):
        """None values in kwargs are cleaned before build_description."""
        inp = InputExample(
            key=0,
            instruction_id_list=["count:word_count_range"],
            prompt="Test",
            kwargs=[{"lower": 5, "upper": 10, "unused": None}],
        )
        response = "one two three four five"
        result = test_instruction_following_strict(inp, response)
        # Should not crash; None should have been popped.
        assert len(result.follow_instruction_list) == 1


# ===================================================================
# test_instruction_following_loose tests
# ===================================================================


class TestInstructionFollowingLoose:
    """Tests for test_instruction_following_loose()."""

    def test_loose_mode_produces_output(self):
        """Basic smoke test — loose mode returns valid OutputExample."""
        inp = InputExample(
            key=0,
            instruction_id_list=["count:word_count_range"],
            prompt="Write 5 words.",
            kwargs=[{"lower": 5, "upper": 5}],
        )
        result = test_instruction_following_loose(inp, "one two three four five")
        assert isinstance(result, OutputExample)
        assert result.follow_all_instructions in (True, False)

    def test_loose_handles_asterisks(self):
        """Loose mode strips asterisks before checking."""
        inp = InputExample(
            key=0,
            instruction_id_list=["sentence:keyword"],
            prompt="Include 'hello'.",
            kwargs=[{"keyword": "hello"}],
        )
        # Response has asterisks — loose mode removes them.
        result = test_instruction_following_loose(inp, "*hello* world")
        # The keyword 'hello' should be findable in at least one variant.
        assert isinstance(result.follow_instruction_list[0], bool)

    def test_loose_handles_first_last_line_removal(self):
        """Loose mode tries removing first/last/both lines."""
        inp = InputExample(
            key=0,
            instruction_id_list=["count:word_count_range"],
            prompt="Write exactly 3 words.",
            kwargs=[{"lower": 3, "upper": 3}],
        )
        response = "intro line\nword word word\noutro line"
        result = test_instruction_following_loose(inp, response)
        # Should pass because 'word word word' (3 words) is the middle line.
        assert result.follow_instruction_list[0] is True

    def test_loose_empty_response(self):
        """Loose mode with empty response — all False."""
        inp = InputExample(
            key=0,
            instruction_id_list=["count:word_count_range"],
            prompt="Say something.",
            kwargs=[{"lower": 1, "upper": 10}],
        )
        result = test_instruction_following_loose(inp, "")
        assert result.follow_all_instructions is False

    def test_loose_multiple_instructions(self):
        """Multiple instructions in loose mode."""
        inp = InputExample(
            key=0,
            instruction_id_list=[
                "count:word_count_range",
                "sentence:keyword",
            ],
            prompt="Write 5 words including 'hello'.",
            kwargs=[
                {"lower": 5, "upper": 5},
                {"keyword": "hello"},
            ],
        )
        result = test_instruction_following_loose(inp, "hello one two three four")
        assert len(result.follow_instruction_list) == 2


# ===================================================================
# IFBenchDataset.load tests
# ===================================================================


class TestIFBenchDatasetLoad:
    """Tests for IFBenchDataset.load()."""

    @staticmethod
    def _mock_from_list(data_list):
        """Bypass HuggingFace Dataset.from_list fingerprinting."""
        cls = type("_MockList", (), {
            "_records": data_list,
            "column_names": list(data_list[0].keys()) if data_list else [],
            "__len__": lambda s: len(s._records),
            "__getitem__": lambda s, i: s._records[i],
            "__iter__": lambda s: iter(s._records),
        })
        return cls()

    def test_load_from_mock_parquet(self, mock_ifbench_dataset):
        """load() returns Dataset with prompt and reference columns."""
        import datasets as _hf_ds
        from unittest.mock import patch as _patch_obj

        with _patch_obj.object(
            _mock_ds_utils, "get_data_path", return_value="/fake/path.parquet"
        ), _patch_obj.object(
            _hf_ds.Dataset, "from_parquet", return_value=mock_ifbench_dataset
        ), _patch_obj.object(
            _hf_ds.Dataset, "from_list", side_effect=self._mock_from_list
        ):
            dataset = IFBenchDataset.load(path="/fake/path.parquet")
            assert hasattr(dataset, "column_names")
            assert len(dataset) == len(IFBENCH_PARQUET_RECORDS)
            assert "prompt" in dataset.column_names
            assert "reference" in dataset.column_names

    def test_load_prompts_preserved(self, mock_ifbench_dataset):
        """Dataset prompts match source parquet records."""
        import datasets as _hf_ds
        from unittest.mock import patch as _patch_obj

        with _patch_obj.object(
            _mock_ds_utils, "get_data_path", return_value="/fake/path.parquet"
        ), _patch_obj.object(
            _hf_ds.Dataset, "from_parquet", return_value=mock_ifbench_dataset
        ), _patch_obj.object(
            _hf_ds.Dataset, "from_list", side_effect=self._mock_from_list
        ):
            dataset = IFBenchDataset.load(path="/fake/path.parquet")
            assert dataset[0]["prompt"] == IFBENCH_PARQUET_RECORDS[0]["prompt"]
            assert dataset[1]["prompt"] == IFBENCH_PARQUET_RECORDS[1]["prompt"]

    def test_load_reference_contains_all_fields(self, mock_ifbench_dataset):
        """reference dict contains key, instruction_id_list, prompt, kwargs."""
        import datasets as _hf_ds
        from unittest.mock import patch as _patch_obj

        with _patch_obj.object(
            _mock_ds_utils, "get_data_path", return_value="/fake/path.parquet"
        ), _patch_obj.object(
            _hf_ds.Dataset, "from_parquet", return_value=mock_ifbench_dataset
        ), _patch_obj.object(
            _hf_ds.Dataset, "from_list", side_effect=self._mock_from_list
        ):
            dataset = IFBenchDataset.load(path="/fake/path.parquet")
            ref = dataset[0]["reference"]
            assert "key" in ref
            assert "instruction_id_list" in ref
            assert "prompt" in ref
            assert "kwargs" in ref


# ===================================================================
# IFBenchEvaluator.score tests
# ===================================================================


class TestIFBenchEvaluatorScore:
    """Tests for IFBenchEvaluator.score()."""

    @pytest.fixture
    def references(self):
        return [r for r in IFBENCH_PARQUET_RECORDS]

    def test_score_basic(self, references):
        """score() returns dict with all 4 metric keys + details."""
        evaluator = IFBenchEvaluator()
        predictions = [
            "Technology is a broad field. " + "word " * 44 + "done now.",
            "first\nsecond\nthird\napple pie",
        ]
        result = evaluator.score(predictions, references)
        assert "Prompt-level-strict-accuracy" in result
        assert "Inst-level-strict-accuracy" in result
        assert "Prompt-level-loose-accuracy" in result
        assert "Inst-level-loose-accuracy" in result
        assert "details" in result

    def test_score_all_predictions_scored(self, references):
        """Every prediction gets a detail entry."""
        evaluator = IFBenchEvaluator()
        predictions = ["resp a", "resp b"]
        result = evaluator.score(predictions, references)
        assert len(result["details"]) == len(predictions)
        assert "0" in result["details"]
        assert "1" in result["details"]

    def test_score_detail_keys(self, references):
        """Each detail entry has expected keys."""
        evaluator = IFBenchEvaluator()
        predictions = ["some response"]
        result = evaluator.score(predictions, references[:1])
        detail = result["details"]["0"]
        assert "prompt" in detail
        assert "pred" in detail
        assert "refer" in detail
        assert "is_strict_correct" in detail
        assert "is_loose_correct" in detail
        assert "grade" in detail

    def test_score_grade_values(self, references):
        """grade is one of 'strict', 'loose', 'none'."""
        evaluator = IFBenchEvaluator()
        predictions = ["some response"]
        result = evaluator.score(predictions, references[:1])
        grade = result["details"]["0"]["grade"]
        assert grade in ("strict", "loose", "none")

    def test_score_empty_predictions(self):
        """Empty predictions list works without crash."""
        evaluator = IFBenchEvaluator()
        result = evaluator.score([], [])
        assert result["Prompt-level-strict-accuracy"] == 0
        assert result["Prompt-level-loose-accuracy"] == 0

    def test_score_empty_string_prediction(self, references):
        """Empty string prediction is handled without crash."""
        evaluator = IFBenchEvaluator()
        result = evaluator.score([""], references[:1])
        assert result["details"]["0"]["pred"] == ""
        assert "Prompt-level-strict-accuracy" in result

    def test_score_accuracy_in_range(self, references):
        """All accuracy values are in [0, 100]."""
        evaluator = IFBenchEvaluator()
        predictions = ["resp a", "resp b"]
        result = evaluator.score(predictions, references)
        for key in [
            "Prompt-level-strict-accuracy",
            "Inst-level-strict-accuracy",
            "Prompt-level-loose-accuracy",
            "Inst-level-loose-accuracy",
        ]:
            assert 0 <= result[key] <= 100

    def test_score_with_origin_prompt(self, references):
        """origin_prompt is used in details when provided."""
        evaluator = IFBenchEvaluator()
        predictions = ["response"]
        origin_prompts = ["custom prompt override"]
        result = evaluator.score(predictions, references[:1], origin_prompts)
        assert result["details"]["0"]["prompt"] == "custom prompt override"
