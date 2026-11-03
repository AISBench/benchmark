# CMATH
[中文](README.md) | English

## Dataset Introduction
CMATH is a Chinese elementary-school math benchmark. This integration evaluates the 600 publicly released dev examples, not the unreleased test set.

> https://github.com/XiaoMi/cmath

Revision: `9fa13ef0b2d03f5a18c1ace7001248d3981d65ea`. Code: MIT (repository LICENSE); data: CC BY 4.0. Attribution: Tianwen Wei et al., CMATH (2023).

## Dataset Deployment
```bash
# Run from the tool root directory
mkdir -p ais_bench/datasets/cmath
wget https://raw.githubusercontent.com/XiaoMi/cmath/9fa13ef0b2d03f5a18c1ace7001248d3981d65ea/datasets/cmath_dev.jsonl -O ais_bench/datasets/cmath/cmath_dev.jsonl
```

```text
cmath/
└── cmath_dev.jsonl
```

## Available Dataset Tasks
| Task | Description | Metric | Few-shot | Prompt | Import | Config |
|---|---|---|---|---|---|---|
| cmath_gen_0_shot_cot_chat_prompt | Chinese CoT generation | accuracy | 0-shot | chat_prompt | `from ais_bench.benchmark.configs.datasets.cmath.cmath_gen_0_shot_cot_chat_prompt import cmath_datasets as datasets` | [cmath_gen_0_shot_cot_chat_prompt.py](cmath_gen_0_shot_cot_chat_prompt.py) |
| cmath_gen_0_shot_cot_str | Chinese CoT generation | accuracy | 0-shot | str | `from ais_bench.benchmark.configs.datasets.cmath.cmath_gen_0_shot_cot_str import cmath_datasets as datasets` | [cmath_gen_0_shot_cot_str.py](cmath_gen_0_shot_cot_str.py) |

## Evaluation Protocol and Validation
Official candidate extraction and numeric matching are retained for valid inputs: integer equality, otherwise relative difference below 1%. Failed predictions remain in the denominator. Invalid fractions raise ValueError and non-string predictions are marked incorrect. The middle regex hyphen is explicitly escaped; the unused exception_regs parameter was removed. These are documented robustness fixes to upstream helpers. Chinese fraction conversion accepts Arabic digits such as 3分之1, not Chinese numeral words. Train is empty; chat/str abbreviations are distinct.

```bash
ais_bench --models vllm_api_general_chat --datasets cmath_gen_0_shot_cot_chat_prompt --num-prompts 10
```

Configure model service and credentials before running; omit --num-prompts for all 600 examples. Previous Linux ARM64 CLI smoke: 10/10 correct (deepseek-flash, temperature 0, max_tokens 1024, thinking disabled); not a full-set score reproduction. Regression tests cover malformed fractions, exception outputs, fraction conversion, truncation, directory loading and configuration contracts.
