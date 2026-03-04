import sys
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from ais_bench.benchmark.datasets.utils.llm_judge import (
    get_a_or_b,
    LLMJudgeDataset,
    LLMJudgeCorrectEvaluator
)


class TestGetAOrB:
    def test_get_a_or_b_with_a(self):
        """测试提取A"""
        result = get_a_or_b("The answer is A")
        assert result == "A"

    def test_get_a_or_b_with_b(self):
        """测试提取B"""
        result = get_a_or_b("The answer is B")
        assert result == "B"

    def test_get_a_or_b_no_match(self):
        """测试没有匹配时返回B"""
        result = get_a_or_b("The answer is C")
        assert result == "B"

    def test_get_a_or_b_empty(self):
        """测试空字符串"""
        result = get_a_or_b("")
        assert result == "B"

    def test_get_a_or_b_at_end(self):
        """测试A在末尾"""
        result = get_a_or_b("something A")
        assert result == "A"


class TestLLMJudgeDataset:
    def test_load_from_predictions_file_not_exists(self):
        """测试文件不存在的情况"""
        ds = LLMJudgeDataset.__new__(LLMJudgeDataset)
        ds.logger = MagicMock()

        with patch('os.path.exists', return_value=False):
            result = ds._load_from_predictions('/test/nonexistent.jsonl')
            assert result == []

    def test_load_from_predictions_success(self):
        """测试成功加载predictions"""
        mock_preds = [
            {"id": 1, "prediction": "pred1"},
            {"id": 0, "prediction": "pred2"}
        ]

        ds = LLMJudgeDataset.__new__(LLMJudgeDataset)
        ds.logger = MagicMock()
        ds.task_state_manager = None

        with patch('os.path.exists', return_value=True):
            # patch load_jsonl in the module where it's used
            import ais_bench.benchmark.datasets.utils.llm_judge as llm_judge_module
            with patch.object(llm_judge_module, 'load_jsonl', return_value=mock_preds):
                result = ds._load_from_predictions('/test/predictions.jsonl')

                assert len(result) == 2
                assert result[0]["id"] == 0
                assert result[1]["id"] == 1


class TestLLMJudgeCorrectEvaluator:
    def test_score_all_correct(self):
        """测试全部正确的情况"""
        evaluator = LLMJudgeCorrectEvaluator()
        predictions = ["A", "A", "A"]
        references = ["correct", "correct", "correct"]

        result = evaluator.score(predictions, references)

        assert result["accuracy"] == 100.0
        assert len(result["details"]) == 3
        for detail in result["details"]:
            assert detail["correct"] is True

    def test_score_all_wrong(self):
        """测试全部错误的情况"""
        evaluator = LLMJudgeCorrectEvaluator()
        predictions = ["B", "B", "B"]
        references = ["correct", "correct", "correct"]

        result = evaluator.score(predictions, references)

        assert result["accuracy"] == 0.0
        for detail in result["details"]:
            assert detail["correct"] is False

    def test_score_mixed(self):
        """测试混合情况"""
        evaluator = LLMJudgeCorrectEvaluator()
        predictions = ["A", "B", "A"]
        references = ["correct1", "correct2", "correct3"]

        result = evaluator.score(predictions, references)

        assert result["accuracy"] == pytest.approx(100 * 2 / 3, rel=1e-2)
        assert result["details"][0]["correct"] is True
        assert result["details"][1]["correct"] is False
        assert result["details"][2]["correct"] is True

    def test_score_length_mismatch(self):
        """测试长度不匹配的情况"""
        evaluator = LLMJudgeCorrectEvaluator()
        predictions = ["A", "A"]
        references = ["correct", "correct", "correct"]

        result = evaluator.score(predictions, references)

        assert "error" in result

    def test_score_empty_predictions(self):
        """测试空predictions"""
        evaluator = LLMJudgeCorrectEvaluator()
        predictions = []
        references = ["correct"]

        result = evaluator.score(predictions, references)

        assert "error" in result

    def test_score_empty_references(self):
        """测试空references"""
        evaluator = LLMJudgeCorrectEvaluator()
        predictions = ["A"]
        references = []

        result = evaluator.score(predictions, references)

        assert "error" in result
