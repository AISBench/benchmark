"""Unit tests for AA-LCR dataset (aa_lcr.py).

Covers: dataset loading, prompt construction, document context retrieval,
judge evaluator scoring, and judge prompt formatting.

Reference: hle.py test patterns — test load, parse, evaluate independently.
"""

import os
import re
import sys
from unittest.mock import patch

import pytest
from pathlib import Path

# ==============================================================================
# AA-LCR test data
# ==============================================================================

AA_LCR_CSV_CONTENT = (
    "question_id,question,answer,document_category,document_set_id,"
    "data_source_filenames,data_source_urls\n"
    "q1,What is the revenue growth?,"
    "10%,Company_Documents,co_dc_2Q23,"
    '"Copy of Equinix Q2 2023 Press Release and Financials.txt"' + "\n"
    # ---
    "q2,Summarize the paper.,"
    "A summary here.,Academia,ac_hack,"
    '"2401.07612v1.txt;2402.13457v2.txt"' + "\n"
)

DOC_TEXT_1 = "Revenue grew 10% year-over-year."
DOC_TEXT_2 = "Paper discusses novel approaches to summarization."


@pytest.fixture
def aa_lcr_data_dir(tmp_path: Path) -> Path:
    """Create a temporary AA-LCR data directory with CSV and extracted documents."""
    data_root = tmp_path / "aa_lcr"
    data_root.mkdir()

    csv_path = data_root / "AA-LCR_Dataset.csv"
    csv_path.write_text(AA_LCR_CSV_CONTENT, encoding="utf-8")

    extracted_dir = data_root / "extracted_text" / "AA-LCR_extracted-text" / "lcr"

    doc_dir1 = extracted_dir / "Company_Documents" / "co_dc_2Q23"
    doc_dir1.mkdir(parents=True)
    (doc_dir1 / "Copy of Equinix Q2 2023 Press Release and Financials.txt").write_text(
        DOC_TEXT_1, encoding="utf-8"
    )

    doc_dir2 = extracted_dir / "Academia" / "ac_hack"
    doc_dir2.mkdir(parents=True)
    (doc_dir2 / "2401.07612v1.txt").write_text(DOC_TEXT_2, encoding="utf-8")
    (doc_dir2 / "2402.13457v2.txt").write_text(
        "Additional findings support the conclusion.", encoding="utf-8"
    )

    import zipfile
    zip_path = data_root / "extracted_text" / "AA-LCR_extracted-text.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(str(extracted_dir)):
            for fn in files:
                fp = os.path.join(root, fn)
                arcname = os.path.relpath(fp, str(extracted_dir.parent))
                zf.write(fp, arcname)

    return data_root

# ---------------------------------------------------------------------------
# Import strategy:
# 1. Preload the REAL HuggingFace ``datasets`` so the local shadow doesn't hit.
# 2. Mock the ais_bench registries / base classes / utilities.
# 3. Load ``aa_lcr.py`` directly by file path (avoids the full ``__init__``
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

# -- mock registries so the module-level decorators are no-ops ----------------
def _make_register(id_str):
    """Return a register_module method that stores classes in a dict."""
    _registry = {}

    def _register(**kwargs):
        def _decorator(cls):
            _registry[cls.__name__] = cls
            return cls
        return _decorator

    _register._registry = _registry
    return _register


_LOAD_DATASET = MagicMock()
_LOAD_DATASET.register_module = _make_register("LOAD_DATASET")
_ICL_EVALUATORS = MagicMock()
_ICL_EVALUATORS.register_module = _make_register("ICL_EVALUATORS")

class _MockRegistryModule:
    LOAD_DATASET = _LOAD_DATASET
    ICL_EVALUATORS = _ICL_EVALUATORS

sys.modules["ais_bench.benchmark.registry"] = _MockRegistryModule

# -- mock base classes --------------------------------------------------------
_mock_base = MagicMock()
_mock_base.BaseDataset = type("BaseDataset", (), {})
_mock_base.BaseDataset.load = staticmethod(lambda **kw: None)
sys.modules["ais_bench.benchmark.datasets.base"] = _mock_base

# -- mock LLM judge base ------------------------------------------------------
_mock_llm_judge = MagicMock()
_mock_llm_judge.LLMJudgeDataset = type("LLMJudgeDataset", (), {})
_mock_llm_judge.LLMJudgeCorrectEvaluator = type("LLMJudgeCorrectEvaluator", (), {})
sys.modules["ais_bench.benchmark.datasets.utils.llm_judge"] = _mock_llm_judge

# -- mock evaluator base ------------------------------------------------------
_mock_evaluator = MagicMock()
_mock_evaluator.BaseEvaluator = type("BaseEvaluator", (), {})
sys.modules["ais_bench.benchmark.openicl.icl_evaluator"] = _mock_evaluator

# -- mock utilities -----------------------------------------------------------
_mock_ds_utils = MagicMock()
_mock_ds_utils.get_content_str = lambda msgs: "".join(
    m.get("text", "") for m in msgs if m.get("type") == "text"
)
_mock_ds_utils.get_data_path = lambda path, local_mode=True: path
_mock_ds_utils.get_cache_dir = lambda d: d
sys.modules["ais_bench.benchmark.datasets.utils.datasets"] = _mock_ds_utils

# -- mock logger --------------------------------------------------------------
_mock_logger_mod = MagicMock()
_mock_logger_mod.AISLogger = lambda: MagicMock()
sys.modules["ais_bench.benchmark.utils.logging.logger"] = _mock_logger_mod
sys.modules["ais_bench.benchmark.utils.logging.error_codes"] = MagicMock()

# Ensure parent packages exist .
for _pkg_name in [
    "ais_bench",
    "ais_bench.benchmark",
    "ais_bench.benchmark.datasets",
]:
    if _pkg_name not in sys.modules:
        sys.modules[_pkg_name] = MagicMock()

# -- now load aa_lcr.py by file path -------------------------------------------
_aa_lcr_path = os.path.join(_BENCHMARK_DIR, "datasets", "aa_lcr.py")
_spec = _iu.spec_from_file_location("aa_lcr", _aa_lcr_path)
_aa_lcr = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_aa_lcr)

# Extract the symbols under test.
AALCRDataset = _aa_lcr.AALCRDataset
AALCRJGDataset = _aa_lcr.AALCRJGDataset
AALCRJudgeEvaluator = _aa_lcr.AALCRJudgeEvaluator
JUDGE_PROMPT = _aa_lcr.JUDGE_PROMPT
PROMPT_TEMPLATE = _aa_lcr.PROMPT_TEMPLATE
_get_context = _aa_lcr._get_context
_ensure_text_dir_downloaded = _aa_lcr._ensure_text_dir_downloaded


# ===================================================================
# Constants / helpers
# ===================================================================


def test_judge_prompt_format():
    """JUDGE_PROMPT formats correctly with question/answers/model_answer."""
    result = JUDGE_PROMPT.format(
        question="What is 2+2?",
        answers="4",
        model_answer="The answer is 4.",
    )
    assert "What is 2+2?" in result
    assert "The OFFICIAL ANSWER: 4" in result
    assert "CANDIDATE ANSWER TO ASSESS: The answer is 4." in result
    assert result.strip().endswith("CORRECT or INCORRECT.")


def test_prompt_template_format():
    """PROMPT_TEMPLATE wraps documents and question correctly."""
    result = PROMPT_TEMPLATE.format(
        documents_text="DOC1 content\n\nDOC2 content",
        question="What is the answer?",
    )
    assert "BEGIN INPUT DOCUMENTS" in result
    assert "END INPUT DOCUMENTS" in result
    assert "DOC1 content" in result
    assert "DOC2 content" in result
    assert "START QUESTION" in result
    assert "END QUESTION" in result
    assert "What is the answer?" in result


# ===================================================================
# _get_context tests
# ===================================================================


def test_get_context_with_filenames_list(aa_lcr_data_dir):
    """_get_context loads documents in data_source_filenames order (list)."""
    text_dir = (
        aa_lcr_data_dir / "extracted_text" / "AA-LCR_extracted-text" / "lcr"
    )
    record = {
        "document_category": "Academia",
        "document_set_id": "ac_hack",
        "data_source_filenames": ["2401.07612v1.txt"],
    }
    context = _get_context(text_dir, record)
    assert "BEGIN DOCUMENT 1" in context
    assert "END DOCUMENT 1" in context
    assert "Paper discusses novel approaches to summarization." in context


def test_get_context_with_semicolon_separated_string(aa_lcr_data_dir):
    """_get_context handles semicolon-separated filenames (CSV format)."""
    text_dir = (
        aa_lcr_data_dir / "extracted_text" / "AA-LCR_extracted-text" / "lcr"
    )
    record = {
        "document_category": "Academia",
        "document_set_id": "ac_hack",
        "data_source_filenames": "2401.07612v1.txt;2402.13457v2.txt",
    }
    context = _get_context(text_dir, record)
    assert "BEGIN DOCUMENT 1" in context
    assert "BEGIN DOCUMENT 2" in context
    assert "novel approaches" in context
    assert "Additional findings" in context


def test_get_context_missing_folder_returns_empty(aa_lcr_data_dir):
    """_get_context returns empty string for nonexistent folder."""
    text_dir = (
        aa_lcr_data_dir / "extracted_text" / "AA-LCR_extracted-text" / "lcr"
    )
    record = {
        "document_category": "Nope",
        "document_set_id": "does_not_exist",
        "data_source_filenames": [],
    }
    context = _get_context(text_dir, record)
    assert context == ""


def test_get_context_missing_file_skipped(aa_lcr_data_dir):
    """_get_context skips missing files gracefully."""
    text_dir = (
        aa_lcr_data_dir / "extracted_text" / "AA-LCR_extracted-text" / "lcr"
    )
    record = {
        "document_category": "Academia",
        "document_set_id": "ac_hack",
        "data_source_filenames": ["nonexistent.txt"],
    }
    context = _get_context(text_dir, record)
    assert context == ""  # no valid docs


def test_get_context_fallback_directory_iteration(aa_lcr_data_dir):
    """_get_context falls back to iterating directory when no filenames given."""
    text_dir = (
        aa_lcr_data_dir / "extracted_text" / "AA-LCR_extracted-text" / "lcr"
    )
    record = {
        "document_category": "Company_Documents",
        "document_set_id": "co_dc_2Q23",
        "data_source_filenames": [],
    }
    context = _get_context(text_dir, record)
    assert "BEGIN DOCUMENT 1" in context
    assert "Revenue grew 10% year-over-year." in context


# ===================================================================
# AALCRJudgeEvaluator tests
# ===================================================================


class TestAALCRJudgeEvaluator:
    """Tests for AALCRJudgeEvaluator.score()."""

    def test_all_correct(self):
        evaluator = AALCRJudgeEvaluator()
        predictions = ["CORRECT", "CORRECT"]
        references = ["ref_a", "ref_b"]
        result = evaluator.score(predictions, references)
        assert result["accuracy"] == 100.0
        assert result["details"]["0"]["correct"] is True
        assert result["details"]["1"]["correct"] is True

    def test_all_incorrect(self):
        evaluator = AALCRJudgeEvaluator()
        predictions = ["INCORRECT", "INCORRECT"]
        references = ["ref_a", "ref_b"]
        result = evaluator.score(predictions, references)
        assert result["accuracy"] == 0.0

    def test_mixed_results(self):
        evaluator = AALCRJudgeEvaluator()
        predictions = ["CORRECT", "INCORRECT", "CORRECT"]
        references = ["a", "b", "c"]
        result = evaluator.score(predictions, references)
        assert result["accuracy"] == pytest.approx(200.0 / 3.0)

    def test_word_boundary_matching(self):
        """'INCORRECT' should NOT be falsely matched as 'CORRECT'."""
        evaluator = AALCRJudgeEvaluator()
        predictions = ["INCORRECT", "INCORRECTLY WORDED"]
        references = ["a", "b"]
        result = evaluator.score(predictions, references)
        assert result["accuracy"] == 0.0

    def test_case_insensitive(self):
        evaluator = AALCRJudgeEvaluator()
        predictions = ["correct", "Correct", "CORRECT"]
        references = ["a", "b", "c"]
        result = evaluator.score(predictions, references)
        assert result["accuracy"] == 100.0

    def test_newline_embedded_correct(self):
        """CORRECT embedded in text with newlines."""
        evaluator = AALCRJudgeEvaluator()
        predictions = ["Some text.\nCORRECT\nMore text."]
        references = ["ref"]
        result = evaluator.score(predictions, references)
        assert result["accuracy"] == 100.0

    def test_length_mismatch_returns_error(self):
        evaluator = AALCRJudgeEvaluator()
        result = evaluator.score(["CORRECT"], ["a", "b"])
        assert "error" in result

    def test_empty_predictions(self):
        evaluator = AALCRJudgeEvaluator()
        result = evaluator.score([], [])
        assert result["accuracy"] == 0.0


# ===================================================================
# AALCRDataset.load tests (integration-light)
# ===================================================================


class TestAALCRDatasetLoad:
    """Tests for AALCRDataset.load() using the mock data directory."""

    @staticmethod
    def _mock_from_list(data_list):
        """Bypass HuggingFace Dataset.from_list fingerprinting (avoids recursion)."""
        cls = type("_MockList", (), {
            "_records": data_list,
            "column_names": list(data_list[0].keys()) if data_list else [],
            "__len__": lambda s: len(s._records),
            "__getitem__": lambda s, i: s._records[i],
            "__iter__": lambda s: iter(s._records),
        })
        return cls()

    def test_load_returns_dataset(self, aa_lcr_data_dir):
        """load() returns a HuggingFace Dataset with expected columns."""
        import datasets as _hf_ds

        text_dir = (
            aa_lcr_data_dir / "extracted_text" / "AA-LCR_extracted-text" / "lcr"
        )
        with patch.object(
            _aa_lcr, "_CSV_PATH",
            str(aa_lcr_data_dir / "AA-LCR_Dataset.csv"),
        ), patch.object(
            _aa_lcr, "_ensure_text_dir_downloaded",
            return_value=text_dir,
        ), patch.object(
            _hf_ds.Dataset, "from_list", side_effect=self._mock_from_list
        ):
            dataset = AALCRDataset.load(
                path=str(aa_lcr_data_dir), name="default"
            )
            assert hasattr(dataset, "column_names")
            assert len(dataset) == 2
            assert "input" in dataset.column_names
            assert "answers" in dataset.column_names
            assert "question" in dataset.column_names
            assert "document_category" in dataset.column_names
            assert "document_set_id" in dataset.column_names

    def test_load_prompt_contains_documents(self, aa_lcr_data_dir):
        """The constructed prompt includes BEGIN/END INPUT DOCUMENTS."""
        import datasets as _hf_ds

        text_dir = (
            aa_lcr_data_dir / "extracted_text" / "AA-LCR_extracted-text" / "lcr"
        )
        with patch.object(
            _aa_lcr, "_CSV_PATH",
            str(aa_lcr_data_dir / "AA-LCR_Dataset.csv"),
        ), patch.object(
            _aa_lcr, "_ensure_text_dir_downloaded",
            return_value=text_dir,
        ), patch.object(
            _hf_ds.Dataset, "from_list", side_effect=self._mock_from_list
        ):
            dataset = AALCRDataset.load(
                path=str(aa_lcr_data_dir), name="default"
            )
            first_input = dataset[0]["input"]
            assert "BEGIN INPUT DOCUMENTS" in first_input
            assert "END INPUT DOCUMENTS" in first_input
            assert "Revenue grew 10%" in first_input
            assert "START QUESTION" in first_input
            assert "END QUESTION" in first_input

    def test_load_answers_match_csv(self, aa_lcr_data_dir):
        """Answers are loaded correctly from the CSV."""
        import datasets as _hf_ds

        text_dir = (
            aa_lcr_data_dir / "extracted_text" / "AA-LCR_extracted-text" / "lcr"
        )
        with patch.object(
            _aa_lcr, "_CSV_PATH",
            str(aa_lcr_data_dir / "AA-LCR_Dataset.csv"),
        ), patch.object(
            _aa_lcr, "_ensure_text_dir_downloaded",
            return_value=text_dir,
        ), patch.object(
            _hf_ds.Dataset, "from_list", side_effect=self._mock_from_list
        ):
            dataset = AALCRDataset.load(
                path=str(aa_lcr_data_dir), name="default"
            )
            assert dataset[0]["answers"] == "10%"
            assert dataset[1]["answers"] == "A summary here."


# ===================================================================
# AALCRJGDataset
# ===================================================================


def test_aalcr_jg_dataset_get_class():
    """AALCRJGDataset._get_dataset_class returns AALCRDataset."""
    jg = AALCRJGDataset.__new__(AALCRJGDataset)
    assert jg._get_dataset_class() is AALCRDataset
