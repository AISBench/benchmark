# -*- coding: utf-8 -*-
"""Unit tests for the CorpusQA dataset adaptation.

Covers:
- ``CorpusQADataset.load``: streaming JSONL loading plus the answer-field
  type normalisation that fixes Arrow build errors (str / int / float /
  empty-list answers must all become str).
- ``CorpusQAPromptTemplate.generate_item``: raw multi-turn prompt ->
  framework PromptList with ``begin``/``round`` section markers and
  OpenAI-style role mapping (system -> SYSTEM, user -> HUMAN,
  assistant -> BOT).
- ``CorpusQAJGDataset._extract_answer`` / ``_modify_dataset_item``: the
  official ``The answer is: ...`` extraction used before the ORM judge.
- ``CorpusQAEvaluator``: ``[[YES]]``/``[[NO]]`` parsing and accuracy
  computation, mirroring the official ORM evaluation.
"""

import sys
import os
import json
from unittest.mock import patch, mock_open

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from ais_bench.benchmark.datasets.corpusqa import (
    CORPUSQA_JUDGE_SYSTEM_PROMPT,
    CORPUSQA_JUDGE_USER_TEMPLATE,
    CorpusQADataset,
    CorpusQAEvaluator,
    CorpusQAJGDataset,
    CorpusQAPromptTemplate,
)


def _make_item(sample_id=0, answer="ans"):
    return {
        "id": sample_id,
        "prompt": [{"role": "user", "content": f"question {sample_id}"}],
        "question": f"question {sample_id}",
        "answer": answer,
    }


class TestCorpusQADataset:
    def _load(self, lines, path="/fake/corpusqa.jsonl"):
        read_data = "\n".join(lines) + "\n" if lines else ""
        m = mock_open(read_data=read_data)
        with patch("ais_bench.benchmark.datasets.corpusqa.get_data_path", return_value=path):
            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", m):
                    return CorpusQADataset.load(path)

    def test_load_normalises_mixed_answer_types(self):
        """混合类型 answer（str/int/float/空列表）必须被统一为 str，否则 Arrow 建表报错"""
        lines = [
            json.dumps(_make_item(0, "string answer")),
            json.dumps(_make_item(1, 42)),
            json.dumps(_make_item(2, 3.14)),
            json.dumps(_make_item(3, [])),
        ]
        ds = self._load(lines)
        assert len(ds) == 4
        answers = list(ds["answer"])
        assert answers == ["string answer", "42", "3.14", "[]"]
        # 所有 answer 都是 str，与官方 ``Answer 2: {}`` 的 str.format 输出保持一致
        assert all(isinstance(a, str) for a in answers)

    def test_load_keeps_raw_prompt_and_fields(self):
        """原始多轮 prompt 原样保留，id/question/answer 字段完整"""
        item = _make_item(7, "ans7")
        ds = self._load([json.dumps(item)])
        assert ds[0]["id"] == 7
        assert ds[0]["question"] == "question 7"
        assert ds[0]["prompt"] == [{"role": "user", "content": "question 7"}]
        assert ds[0]["answer"] == "ans7"

    def test_load_skips_invalid_json_line(self):
        """无法解析的行应被跳过"""
        lines = [
            json.dumps(_make_item(0, "a")),
            "not a json line",
            json.dumps(_make_item(1, "b")),
        ]
        ds = self._load(lines)
        assert len(ds) == 2

    def test_load_skips_line_missing_prompt_or_answer(self):
        """缺少 prompt 或 answer 字段的行应被跳过"""
        lines = [
            json.dumps({"id": 0, "question": "no prompt"}),
            json.dumps({"id": 1, "prompt": [{"role": "user", "content": "q"}]}),
            json.dumps(_make_item(2, "ok")),
        ]
        ds = self._load(lines)
        assert len(ds) == 1
        assert ds[0]["id"] == 2

    def test_load_default_id(self):
        """缺少 id 时使用当前行号"""
        lines = [
            json.dumps({"prompt": [{"role": "user", "content": "q"}], "answer": "a"}),
        ]
        ds = self._load(lines)
        assert ds[0]["id"] == 0

    def test_load_file_not_found(self):
        """文件不存在时抛出 FileNotFoundError"""
        with patch("ais_bench.benchmark.datasets.corpusqa.get_data_path", return_value="/nope.jsonl"):
            with patch("os.path.exists", return_value=False):
                with pytest.raises(FileNotFoundError):
                    CorpusQADataset.load("/nope.jsonl")


class TestCorpusQAPromptTemplate:
    def _prompt(self, messages):
        return CorpusQAPromptTemplate(template="").generate_item({"prompt": messages})

    def test_str_prompt_passthrough(self):
        """字符串 prompt 原样返回"""
        out = self._prompt("plain text prompt")
        assert out == "plain text prompt"

    def test_dict_prompt_wrapped_in_list(self):
        """单个 dict prompt 被包装成列表并生成对话模板"""
        out = self._prompt({"role": "user", "content": "hi"})
        assert isinstance(out, list)
        # 只有 user 消息：无 system 段，直接 round 段
        assert out == [
            {"section": "round", "pos": "begin"},
            {"role": "HUMAN", "prompt": "hi"},
            {"section": "round", "pos": "end"},
        ]

    def test_multiturn_role_mapping_and_sections(self):
        """system/user/assistant 映射为 SYSTEM/HUMAN/BOT，system 进 begin 段，其余进 round 段"""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        out = self._prompt(messages)
        assert out == [
            {"section": "begin", "pos": "begin"},
            {"role": "SYSTEM", "prompt": "sys"},
            {"section": "begin", "pos": "end"},
            {"section": "round", "pos": "begin"},
            {"role": "HUMAN", "prompt": "u1"},
            {"role": "BOT", "prompt": "a1"},
            {"role": "HUMAN", "prompt": "u2"},
            {"section": "round", "pos": "end"},
        ]

    def test_message_without_role_defaults_to_user(self):
        """缺少 role 的消息默认按 user 处理"""
        out = self._prompt([{"content": "no role"}])
        assert {"role": "HUMAN", "prompt": "no role"} in out

    def test_non_dict_message_converted(self):
        """列表中的非 dict 消息被转为 user 消息"""
        out = self._prompt(["just a string"])
        assert {"role": "HUMAN", "prompt": "just a string"} in out

    def test_extra_keys_preserved(self):
        """消息中的额外字段被保留"""
        out = self._prompt([{"role": "user", "content": "q", "extra": 1}])
        assert {"role": "HUMAN", "prompt": "q", "extra": 1} in out


class TestCorpusQAJGDataset:
    def test_extract_answer_found(self):
        """提取 ``The answer is: X`` 中的 X"""
        assert CorpusQAJGDataset._extract_answer("some text\nThe answer is: 42") == "42"

    def test_extract_answer_last_match_wins(self):
        """多次出现时取最后一次（推理内容可能包含草稿答案）"""
        response = "The answer is: draft\nreasoning\n\nThe answer is: 42"
        assert CorpusQAJGDataset._extract_answer(response) == "42"

    def test_extract_answer_no_match(self):
        """无匹配时返回去除首尾空白的完整内容"""
        assert CorpusQAJGDataset._extract_answer("  no marker here  ") == "no marker here"

    def test_extract_answer_non_str(self):
        """非字符串输入转为字符串处理"""
        assert CorpusQAJGDataset._extract_answer(123) == "123"

    def test_modify_dataset_item_prefers_content(self):
        """存在 content 字段时，基于它做答案提取（推理+最终答案场景）"""
        ds = CorpusQAJGDataset.__new__(CorpusQAJGDataset)
        dataset_item = {}
        pred_item = {
            "prediction": "reasoning\nThe answer is: gold",
            "content": "reasoning\nThe answer is: gold",
        }
        ds._modify_dataset_item(dataset_item, pred_item)
        assert dataset_item["model_answer"] == "gold"

    def test_modify_dataset_item_fallback_to_model_answer(self):
        """无 content 字段时回退到已有 model_answer 并重新提取"""
        ds = CorpusQAJGDataset.__new__(CorpusQAJGDataset)
        dataset_item = {"model_answer": "xxx The answer is: fallback"}
        pred_item = {"prediction": "xxx The answer is: fallback"}
        ds._modify_dataset_item(dataset_item, pred_item)
        assert dataset_item["model_answer"] == "fallback"


class TestCorpusQAEvaluator:
    def test_score_all_correct(self):
        """全部 [[YES]] -> accuracy 100"""
        eva = CorpusQAEvaluator()
        out = eva.score(["[[YES]]", "[[YES]]"], ["a", "b"])
        assert out["accuracy"] == 100.0
        assert out["num_correct"] == 2
        assert out["num_total"] == 2

    def test_score_mixed(self):
        """混合 YES/NO -> 正确计算 accuracy"""
        eva = CorpusQAEvaluator()
        out = eva.score(["[[YES]]", "[[NO]]", "[[YES]]"], ["a", "b", "c"])
        assert out["accuracy"] == pytest.approx(100 * 2 / 3)
        assert out["details"][0]["correct"] is True
        assert out["details"][1]["correct"] is False
        assert out["details"][2]["correct"] is True

    def test_score_no_marker_treated_incorrect(self):
        """judge 输出不含标记时视为错误"""
        eva = CorpusQAEvaluator()
        out = eva.score(["no marker here"], ["a"])
        assert out["num_correct"] == 0
        assert out["details"][0]["correct"] is False

    def test_score_length_mismatch(self):
        """预测与参考答案数量不一致时返回 error"""
        eva = CorpusQAEvaluator()
        out = eva.score(["[[YES]]"], ["a", "b"])
        assert "error" in out
        assert "different length" in out["error"]

    def test_score_empty(self):
        """空输入 -> accuracy 0"""
        eva = CorpusQAEvaluator()
        out = eva.score([], [])
        assert out["accuracy"] == 0.0

    def test_is_correct_yes(self):
        assert CorpusQAEvaluator._is_correct("explanation\n[[YES]]") is True

    def test_is_correct_no(self):
        assert CorpusQAEvaluator._is_correct("explanation\n[[NO]]") is False

    def test_is_correct_last_marker_wins(self):
        """推理中同时出现两种标记时，以最后一个为准（官方取最终结论）"""
        assert CorpusQAEvaluator._is_correct("[[NO]] ... conclusion [[YES]]") is True
        assert CorpusQAEvaluator._is_correct("[[YES]] ... conclusion [[NO]]") is False

    def test_is_correct_empty_or_no_marker(self):
        assert CorpusQAEvaluator._is_correct("") is None
        assert CorpusQAEvaluator._is_correct(None) is None
        assert CorpusQAEvaluator._is_correct("no markers") is None


class TestCorpusQAJudgePrompts:
    def test_system_prompt_non_empty(self):
        """官方 ORM system prompt 非空且要求输出 [[YES]]/[[NO]]"""
        assert "[[YES]]" in CORPUSQA_JUDGE_SYSTEM_PROMPT
        assert "[[NO]]" in CORPUSQA_JUDGE_SYSTEM_PROMPT

    def test_user_template_format(self):
        """官方 user template 包含 question/model_answer/answer 占位符"""
        rendered = CORPUSQA_JUDGE_USER_TEMPLATE.format(
            question="q", model_answer="ma", answer="a"
        )
        assert "Problem: q" in rendered
        assert "Answer 1: ma" in rendered
        assert "Answer 2: a" in rendered


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
