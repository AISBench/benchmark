import pytest

from ais_bench.benchmark.datasets.dapo_math import (
    last_boxed_only_string, remove_boxed, normalize_final_answer,
    extract_pred_by_minerva, extract_pred_by_strict_box,
    dapo_math_postprocess, dapo_math_postprocess_v2,
    DAPOMathEvaluator, DAPOMathEvaluatorV2,
)
from ais_bench.benchmark.utils.logging.exceptions import AISBenchDataContentError


class TestBoxedFunctions:
    def test_last_boxed_found(self):
        result = last_boxed_only_string(r"The answer is \boxed{42}.")
        assert result == r"\boxed{42}"

    def test_last_boxed_not_found(self):
        result = last_boxed_only_string("No boxed expression here.")
        assert result is None

    def test_last_boxed_unclosed(self):
        result = last_boxed_only_string(r"\boxed{42")
        assert result is None

    def test_last_boxed_nested(self):
        result = last_boxed_only_string(r"First \boxed{a} then \boxed{\frac{1}{2}} end.")
        assert result == r"\boxed{\frac{1}{2}}"

    def test_remove_boxed_success(self):
        result = remove_boxed(r"\boxed{42}")
        assert result == "42"

    def test_remove_boxed_invalid_prefix(self):
        with pytest.raises(AISBenchDataContentError):
            remove_boxed(r"\notboxed{42}")

    def test_remove_boxed_invalid_suffix(self):
        with pytest.raises(AISBenchDataContentError):
            remove_boxed(r"\boxed{42")


class TestNormalizeFinalAnswer:
    def test_basic(self):
        assert normalize_final_answer("42") == "42"

    def test_with_substitutions(self):
        result = normalize_final_answer("an apple")
        assert result == "apple"

    def test_with_removed_expressions(self):
        result = normalize_final_answer("5 degrees")
        assert result == "5"

    def test_with_comma_numbers(self):
        result = normalize_final_answer("1,234")
        assert result == "1234"

    def test_with_textbf(self):
        result = normalize_final_answer(r"\textbf{hello}")
        assert result == "hello"

    def test_with_overline(self):
        result = normalize_final_answer(r"\overline{AB}")
        assert result == "AB"

    def test_with_boxed(self):
        result = normalize_final_answer(r"\boxed{99}")
        assert result == "99"

    def test_with_dollar_signs(self):
        result = normalize_final_answer("$42$")
        assert result == "42"

    def test_with_frac_shorthand(self):
        result = normalize_final_answer(r"frac12")
        assert result == r"frac{1}{2}"

    def test_with_sqrt_shorthand(self):
        result = normalize_final_answer(r"sqrt3")
        assert result == r"sqrt{3}"


class TestExtractPredByMinerva:
    def test_with_valid_answer(self):
        result = extract_pred_by_minerva("The answer is Answer: 42\nmore text")
        assert result == "42"

    def test_with_no_match_returns_invalid(self):
        result = extract_pred_by_minerva("No answer pattern here.")
        assert result == "[INVALID]"

    def test_case_insensitive(self):
        result = extract_pred_by_minerva("The answer is answer: 100\n")
        assert result == "100"


class TestExtractPredByStrictBox:
    def test_with_boxed_answer(self):
        result = extract_pred_by_strict_box(r"some text \boxed{42} end")
        assert result == "42"

    def test_without_boxed_returns_empty(self):
        result = extract_pred_by_strict_box("no boxed expression here")
        assert result == ""


class TestDapoMathPostprocessors:
    def test_dapo_math_postprocess(self):
        result = dapo_math_postprocess("blah Answer: 42\n")
        assert result == "42"

    def test_dapo_math_postprocess_v2(self):
        result = dapo_math_postprocess_v2(r"blah \boxed{42}")
        assert result == "42"


class TestDAPOMathEvaluator:
    def test_score_basic(self):
        evaluator = DAPOMathEvaluator()
        result = evaluator.score(["42", "100"], ["42", "200"])
        assert result["accuracy"] == 50.0
        assert len(result["details"]) == 2
        assert result["details"][0]["correct"] is True
        assert result["details"][1]["correct"] is False

    def test_score_all_correct(self):
        evaluator = DAPOMathEvaluator()
        result = evaluator.score(["1", "2", "3"], ["1", "2", "3"])
        assert result["accuracy"] == 100.0
        assert all(d["correct"] for d in result["details"])

    def test_score_all_wrong(self):
        evaluator = DAPOMathEvaluator()
        result = evaluator.score(["1", "2", "3"], ["4", "5", "6"])
        assert result["accuracy"] == 0.0
        assert not any(d["correct"] for d in result["details"])

    def test_score_length_mismatch(self):
        evaluator = DAPOMathEvaluator()
        result = evaluator.score(["1", "2"], ["1"])
        assert "error" in result
        assert "different length" in result["error"]

    def test_score_normalizes_references(self):
        evaluator = DAPOMathEvaluator()
        result = evaluator.score(["1234"], ["1,234"])
        assert result["accuracy"] == 100.0
        assert result["details"][0]["correct"] is True


class TestDAPOMathEvaluatorV2:
    def test_score_basic(self):
        evaluator = DAPOMathEvaluatorV2()
        result = evaluator.score(["42", "100"], ["42", "200"])
        assert result["accuracy"] == 50.0
        assert len(result["details"]) == 2
        assert result["details"][0]["correct"] is True
        assert result["details"][1]["correct"] is False

    def test_score_length_mismatch(self):
        evaluator = DAPOMathEvaluatorV2()
        result = evaluator.score(["1", "2"], ["1"])
        assert "error" in result
        assert "different length" in result["error"]
