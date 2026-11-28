"""CMATH helpers adapted from XiaoMi/cmath, MIT; see repository LICENSE.
Upstream revision: 9fa13ef0b2d03f5a18c1ace7001248d3981d65ea.
"""
import json
import math
import re
from pathlib import Path
from datasets import Dataset, DatasetDict
from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from .base import BaseDataset

__all__ = ['CMATHDataset', 'cmath_postprocess', 'CmathEvaluator']

REG_DIGITS_SINGLETON = re.compile(r"^\d+[./]?\d*%?$")
REG_DIGITS_BEGIN = re.compile(r"^(\d+[./]?\d*%?) ?(?=[\u4e00-\u9fa5,，。°℃$])")
REG_DIGITS_MIDDLE = re.compile(r"(?<=[\u4e00-\u9fa5$:：，＝≈=\->{]) ?(\d+[./]?\d*%?) ?(?:[\u4e00-\u9fa5,，。°℃$}（]|\([^\d\s]|\.$)")
REG_DIGITS_END = re.compile(r"(?<=[\u4e00-\u9fa5$:：＝≈=\->]) ?(\d+[./]?\d*%?)$")

REG_NORM = re.compile(r"(?<=\d)[, ](?=\d)")
REG_LATEX_FRAC = re.compile(r'\\frac{([^}]+)}{([^}]+)}')
REG_CN_FRAC = re.compile(r'(\d+)分之(\d+)')


def has_exception(answer: str) -> bool:
    """用于判断模型生成的答案是否为异常"""

    if not isinstance(answer, str):
        return True

    if len(answer.strip()) == 0:
        return True

    reg_timeout = re.compile("(请求.*超时)|(timeout)")
    if bool(re.search(reg_timeout, answer)):
        return True

    reg_error = re.compile("error|异常|失败|content_filter")

    # 请求出现错误或异常
    if "{" in answer and "}" in answer and (bool(re.search(reg_error, answer))):
        return True

    return False


def extract_cn_fractal(line):
    """提取输入字符串中的中文汉字分数表达，例如`五分之三`，并将其转换为基于阿拉伯数字的分数表达`3/5`"""

    res = re.findall(REG_CN_FRAC, line)
    if len(res) != 0:
        return ["{}/{}".format(b, a) for a, b in res]
    else:
        return res


def extract_digits_from_line(line):
    """提取输入字符串中的所有数字"""

    res1 = re.findall(REG_DIGITS_BEGIN, line)
    res2 = re.findall(REG_DIGITS_MIDDLE, line)
    res3 = re.findall(REG_DIGITS_END, line)
    res4 = re.findall(REG_DIGITS_SINGLETON, line)
    res_cn_frac = extract_cn_fractal(line)
    concat = res1 + res2 + res_cn_frac + res3 + res4
    candidates = [s.strip() for s in concat]
    return [s for s in candidates if not (s.startswith("/") or s.endswith("/"))]


def extract_digits_prediction(response, truncation="t"):
    """从模型回复中提取答案候选"""

    # 检测模型回复是否出现异常
    if has_exception(response):
        return ["ERROR"]

    response = REG_LATEX_FRAC.sub(r'\1/\2', response)
    response = re.sub(REG_NORM, "", response)

    candidates = []
    # 逐行提取数字
    for line in response.splitlines():
        candidates += extract_digits_from_line(line)

    if truncation is None:
        # 不对candidate做截断
        res = candidates

    elif truncation == "t":
        # 只考虑最末尾的两个数字作为模型可能的回答
        if len(candidates) <= 2:
            res = candidates
        else:
            res = candidates[-2:]

    elif truncation == "h":
        res = candidates[:2]

    elif truncation == "ht" or truncation == "th":
        if len(candidates) < 4:
            res = candidates
        else:
            res = candidates[:2] + candidates[-2:]
    else:
        raise ValueError("Illegal truncation argument... Only `h`, `t`, `ht` or None are supported.")
    return list(set(res))


def string2num(string: str):
    """尝试把一个string转化成一个浮点数或整数"""
    string = string.strip()

    if string.endswith("%"):
        string = string.replace("%", "")
        return float(string) / 100

    if "/" in string:
        parts = string.split("/")
        if len(parts) != 2 or float(parts[1]) == 0.0:
            raise ValueError(f"Invalid fraction: {string}")
        else:
            return float(parts[0]) / float(parts[1])

    if "." in string:
        return float(string)

    return int(string)


def match_digits(a,  b):
    """判断数字a和b是否近似相等"""

    if isinstance(a, int) and isinstance(b, int):
        return a == b
    else:
        relative_diff = abs(a - b) / (min(abs(a), abs(b)) + 1e-6)
        return relative_diff < 1e-2


def match_digit_response(golden, responses: list) -> bool:
    """判断标准答案是与提取的答案候选中的某一个匹配"""

    if "ERROR" in responses:
        return False

    if isinstance(golden, str):
        golden = string2num(golden)

    for r in responses:
        try:
            num = string2num(r)
            if match_digits(golden, num):
                return True
        except ValueError:
            pass
    return False


@LOAD_DATASET.register_module()
class CMATHDataset(BaseDataset):
    """Load official dev into the framework evaluation split."""
    @staticmethod
    def load(path):
        source = Path(get_data_path(path, local_mode=True))
        if source.is_dir():
            source = source / 'cmath_dev.jsonl'
        with source.open(encoding='utf-8') as handle:
            rows = json.load(handle) if source.suffix == '.json' else [json.loads(line) for line in handle if line.strip()]
        if not isinstance(rows, list) or not rows:
            raise ValueError('CMATH requires nonempty JSON array or JSONL')
        records = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get('input'), str) or not row['input'].strip():
                raise ValueError(f'Invalid input at row {index}')
            try:
                answer = row['golden']
                number = string2num(str(answer))
                if not math.isfinite(number):
                    raise ValueError('Invalid numeric reference')
            except (KeyError, ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
                raise ValueError(f'Invalid reference at row {index}') from exc
            records.append(dict(row, question=row['input'], answer=str(answer)))
        test = Dataset.from_list(records)
        return DatasetDict(train=test.select([]), test=test)


@TEXT_POSTPROCESSORS.register_module('cmath')
def cmath_postprocess(text):
    """Raw response; official multi-candidate extraction happens in evaluator."""
    return text


class CmathEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}
        if not references:
            return {'error': 'empty references'}
        details = []
        for pred, ref in zip(predictions, references):
            candidates = extract_digits_prediction(pred)
            correct = match_digit_response(ref, candidates)
            details.append(dict(pred=pred, answer=ref, candidates=candidates, correct=correct))
        return dict(accuracy=100 * sum(d['correct'] for d in details) / len(details), details=details)
