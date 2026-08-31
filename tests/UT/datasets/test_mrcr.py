# -*- coding: utf-8 -*-
"""Unit tests for the MRCR dataset integration.

Covers:
- ``MRCRPromptTemplate.generate_item``: raw multi-turn prompt ->
  framework PromptList with ``begin``/``round`` section markers and
  OpenAI-style role mapping (system -> SYSTEM, user -> HUMAN,
  assistant -> BOT), plus plain-str / single-dict prompt handling.
- ``MRCRDataset.load``: parquet shard streaming, JSON-encoded prompt
  parsing, token-bin filtering via the precomputed ``num_tokens``
  column, and config-error handling.
- ``MRCREvaluator``: official ``grade`` semantics -- prefix gate +
  SequenceMatcher ratio, ``content`` over ``prediction`` preference for
  thinking models, prefixes taken from the dataset column, metrics and
  details fields.
- ``mrcr_postprocess``: identity postprocessing.
"""

import json
import sys
import os
from difflib import SequenceMatcher
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from ais_bench.benchmark.datasets.mrcr import (
    MRCR_BIN_BOUNDARIES,
    MRCRDataset,
    MRCREvaluator,
    MRCRPromptTemplate,
    mrcr_postprocess,
)

PREFIX = "a1b2c3"
NEEDLE = "Once upon a time, a tapir wandered through the misty forest."
ANSWER = PREFIX + NEEDLE


def _make_test_set(n=1, prefix=PREFIX, answer=ANSWER):
    """Dataset mimicking the loaded MRCR test set (prefixes column)."""
    return Dataset.from_list(
        [
            {
                "id": i,
                "prompt": [{"role": "user", "content": f"q{i}"}],
                "answer": answer,
                "random_string_to_prepend": prefix,
            }
            for i in range(n)
        ]
    )


def _write_shard(dir_path, rows, name="part-0.parquet"):
    """Write one parquet shard from a dict-of-columns ``rows``."""
    path = os.path.join(dir_path, name)
    pq.write_table(pa.table(rows), path)
    return path


# ---------------------------------------------------------------------------
# MRCRPromptTemplate
# ---------------------------------------------------------------------------


def _messages(result):
    """Extract message dicts (drop begin/round section markers)."""
    return [d for d in result if isinstance(d, dict) and "role" in d]


class TestMRCRPromptTemplate:
    def _prompt(self, messages):
        return MRCRPromptTemplate(template="").generate_item({"prompt": messages})

    def test_multiturn_role_mapping(self):
        """system -> begin/SYSTEM，user/assistant -> round/HUMAN/BOT"""
        messages = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        msgs = _messages(self._prompt(messages))
        assert [m["role"] for m in msgs] == ["SYSTEM", "HUMAN", "BOT", "HUMAN"]
        assert [m["prompt"] for m in msgs] == ["sys prompt", "q1", "a1", "q2"]
        # 消息顺序原样保留（多轮共指消解依赖完整对话顺序）
        assert [m["prompt"] for m in msgs] == [m["content"] for m in messages]

    def test_no_system_message_omits_begin_section(self):
        """没有 system 消息时不产生 begin 段"""
        result = self._prompt([{"role": "user", "content": "q"}])
        assert not any(
            d.get("section") == "begin" for d in result if isinstance(d, dict)
        )
        assert [m["role"] for m in _messages(result)] == ["HUMAN"]

    def test_str_prompt_passthrough(self):
        """纯文本 prompt 原样返回"""
        assert self._prompt("plain text prompt") == "plain text prompt"

    def test_dict_prompt_wrapped_to_list(self):
        """单条 dict 消息应被包装为列表处理"""
        msgs = _messages(self._prompt({"role": "user", "content": "q"}))
        assert [m["role"] for m in msgs] == ["HUMAN"]
        assert [m["prompt"] for m in msgs] == ["q"]

    def test_unknown_role_defaults_to_human(self):
        """未知 role 按用户轮次处理（与官方 user 默认一致）"""
        msgs = _messages(self._prompt([{"content": "q"}]))
        assert [m["role"] for m in msgs] == ["HUMAN"]

    def test_extra_message_fields_preserved(self):
        """role/content 以外的消息字段应被保留转发"""
        msgs = _messages(
            self._prompt([{"role": "user", "content": "q", "foo": "bar"}])
        )
        assert msgs[0].get("foo") == "bar"


# ---------------------------------------------------------------------------
# MRCRDataset
# ---------------------------------------------------------------------------


class TestMRCRDataset:
    def _load(self, tmp_path, rows, subset="2needle", **kwargs):
        subset_dir = os.path.join(str(tmp_path), subset)
        os.makedirs(subset_dir, exist_ok=True)
        if rows:
            _write_shard(subset_dir, rows)
        kwargs.setdefault("length_bin", None)
        with patch(
            "ais_bench.benchmark.datasets.mrcr.get_data_path",
            return_value=str(tmp_path),
        ):
            return MRCRDataset.load(
                str(tmp_path), subset=subset, **kwargs
            )

    def _rows(self, n=2, num_tokens=None):
        rows = {
            "prompt": [
                json.dumps(
                    [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": f"q{i}"},
                    ]
                )
                for i in range(n)
            ],
            "answer": [ANSWER] * n,
            "random_string_to_prepend": [PREFIX] * n,
        }
        if num_tokens is not None:
            rows["num_tokens"] = num_tokens
        return rows

    def test_load_parses_rows_and_json_prompt(self):
        """JSON 编码的 prompt 被解析为消息列表，字段完整"""
        ds = self._load_rows_only()
        assert len(ds) == 2
        assert ds[0]["id"] == 0
        assert ds[0]["prompt"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q0"},
        ]
        assert ds[0]["answer"] == ANSWER
        assert ds[0]["random_string_to_prepend"] == PREFIX

    def _load_rows_only(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            return self._load(tmp, self._rows(2))

    def test_load_filters_by_token_bin(self):
        """按官方 (bin_lo, bin_hi] 区间过滤 num_tokens 列"""
        import tempfile

        # 1m bin = (524288, 1048576]
        rows = self._rows(n=3, num_tokens=[600000, 524288, 1048576])
        with tempfile.TemporaryDirectory() as tmp:
            ds = self._load(tmp, rows, length_bin="1m")
        # 524288 is out (exclusive lower bound), 1048576 is in
        assert len(ds) == 2
        assert list(ds["id"]) == [0, 2]

    def test_load_invalid_length_bin_raises(self):
        """未知 length_bin 应快速失败（配置拼写错误）"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="Unknown length_bin"):
                self._load(tmp, self._rows(1), length_bin="2m")

    def test_load_missing_subset_dir_raises(self):
        """subset 子目录缺失时抛 FileNotFoundError"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError, match="subset directory"):
                self._load(tmp, None, subset="4needle")

    def test_load_no_matching_samples_raises(self):
        """bin 过滤后无样本时抛 ValueError（防止静默空跑）"""
        import tempfile

        rows = self._rows(n=1, num_tokens=[10])
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="No MRCR samples"):
                self._load(tmp, rows, length_bin="1m")

    def test_bin_boundaries_cover_official_bins(self):
        """官方 bin 边界完整注册（键 = 官方图表标签，即 bin 上界）"""
        assert set(MRCR_BIN_BOUNDARIES) == {
            "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1m"
        }
        # 官方 "8K" bin 实际是 [4096, 8192]
        assert MRCR_BIN_BOUNDARIES["8k"] == (4096, 8192)
        assert MRCR_BIN_BOUNDARIES["512k"] == (262144, 524288)
        assert MRCR_BIN_BOUNDARIES["1m"] == (524288, 1048576)


# ---------------------------------------------------------------------------
# MRCREvaluator
# ---------------------------------------------------------------------------


class TestMRCREvaluator:
    def _score(self, predictions, references, content=None, test_set=None):
        return MRCREvaluator().score(
            predictions=predictions,
            references=references,
            content=content,
            test_set=test_set,
        )

    def test_exact_match_scores_100(self):
        """完全复现答案：score/prefix_hit_rate/strict_acc 全为 100"""
        result = self._score(
            [ANSWER], [ANSWER], test_set=_make_test_set()
        )
        assert result["score"] == 100.0
        assert result["prefix_hit_rate"] == 100.0
        assert result["strict_acc"] == 100.0
        assert result["num_total"] == 1
        assert result["details"][0]["correct"] is True

    def test_missing_prefix_scores_0(self):
        """未加 random_string 前缀：官方 grade 返回 0"""
        result = self._score(
            [NEEDLE], [ANSWER], test_set=_make_test_set()
        )
        assert result["score"] == 0.0
        assert result["prefix_hit_rate"] == 0.0
        assert result["details"][0]["prefix_hit"] is False

    def test_partial_ratio_matches_official_grade(self):
        """部分匹配得分与官方 SequenceMatcher 公式逐字节一致"""
        response = PREFIX + "Once upon a time, a tapir wandered thru the forest."
        result = self._score(
            [response], [ANSWER], test_set=_make_test_set()
        )
        expected = SequenceMatcher(None, response[len(PREFIX):], NEEDLE).ratio()
        assert result["score"] == pytest.approx(100.0 * expected)
        assert result["prefix_hit_rate"] == 100.0
        assert result["strict_acc"] == 0.0

    def test_prefers_content_over_prediction(self):
        """思考模型：优先使用无 reasoning 的 content 字段评分"""
        # prediction 混入了 reasoning 草稿（不以 prefix 开头），content 是纯答案
        prediction = "Let me find the 2nd poem... final answer follows.\n\n" + ANSWER
        result = self._score(
            [prediction], [ANSWER], content=[ANSWER], test_set=_make_test_set()
        )
        assert result["score"] == 100.0

    def test_falls_back_to_prediction_without_content(self):
        """非思考模型 / 旧版结果文件：无 content 时回退 prediction"""
        result = self._score(
            [ANSWER], [ANSWER], content=[None], test_set=_make_test_set()
        )
        assert result["score"] == 100.0

    def test_prediction_fails_when_content_missing(self):
        """无 content 回退时，混入 reasoning 的 prediction 无法通过前缀门"""
        prediction = "draft... " + ANSWER
        result = self._score(
            [prediction], [ANSWER], content=[None], test_set=_make_test_set()
        )
        assert result["score"] == 0.0

    def test_list_payload_takes_first_element(self):
        """pass@k 列表载荷取第一个元素（与框架列表预测兼容）"""
        result = self._score(
            [[NEEDLE, ANSWER]], [ANSWER], content=[[ANSWER, ""]],
            test_set=_make_test_set(),
        )
        assert result["score"] == 100.0

    def test_length_mismatch_returns_error(self):
        """预测与参考长度不一致返回 error 字段"""
        result = self._score([ANSWER, ANSWER], [ANSWER], test_set=_make_test_set(2))
        assert "error" in result

    def test_prefixes_taken_from_test_set(self):
        """前缀来自数据集列，而非预测文件（预测文件不存该字段）"""
        # 同一 prediction，不同 prefix 的两个样本：第二个必须失败
        test_set = _make_test_set(2)
        test_set = test_set.map(
            lambda x, i: {"random_string_to_prepend": PREFIX if i == 0 else "zzz"},
            with_indices=True,
        )
        result = self._score(
            [ANSWER, ANSWER], [ANSWER, ANSWER], test_set=test_set
        )
        assert result["score"] == pytest.approx(50.0)
        assert result["prefix_hit_rate"] == 50.0

    def test_details_record_official_fields(self):
        """details 记录离线重评分所需的官方字段"""
        result = self._score(
            [ANSWER], [ANSWER], test_set=_make_test_set()
        )
        detail = result["details"][0]
        assert detail["pred"] == ANSWER
        assert detail["answer"] == ANSWER
        assert detail["random_string_to_prepend"] == PREFIX
        assert detail["ratio"] == 1.0
        assert detail["correct"] is True


# ---------------------------------------------------------------------------
# mrcr_postprocess
# ---------------------------------------------------------------------------


class TestMRCRPostprocess:
    def test_identity(self):
        """官方对原始回复评分：后处理必须为恒等"""
        assert mrcr_postprocess(ANSWER) == ANSWER
        assert mrcr_postprocess("") == ""
