# CMATH
中文 | [English](README_en.md)

## 数据集简介
CMATH 是中文小学数学基准。本接入评估公开的 600 条 dev，不将其称为官方 test 全集。

> https://github.com/XiaoMi/cmath

Revision: `9fa13ef0b2d03f5a18c1ace7001248d3981d65ea`. Code: MIT (repository LICENSE); data: CC BY 4.0. Attribution: Tianwen Wei et al., CMATH (2023).

## 数据集部署
```bash
# Run from the tool root directory
mkdir -p ais_bench/datasets/cmath
wget https://raw.githubusercontent.com/XiaoMi/cmath/9fa13ef0b2d03f5a18c1ace7001248d3981d65ea/datasets/cmath_dev.jsonl -O ais_bench/datasets/cmath/cmath_dev.jsonl
```

```text
cmath/
└── cmath_dev.jsonl
```

## 可用数据集任务
| Task | Description | Metric | Few-shot | Prompt | Import | Config |
|---|---|---|---|---|---|---|
| cmath_gen_0_shot_cot_chat_prompt | Chinese CoT generation | accuracy | 0-shot | chat_prompt | `from ais_bench.benchmark.configs.datasets.cmath.cmath_gen_0_shot_cot_chat_prompt import cmath_datasets as datasets` | [cmath_gen_0_shot_cot_chat_prompt.py](cmath_gen_0_shot_cot_chat_prompt.py) |
| cmath_gen_0_shot_cot_str | Chinese CoT generation | accuracy | 0-shot | str | `from ais_bench.benchmark.configs.datasets.cmath.cmath_gen_0_shot_cot_str import cmath_datasets as datasets` | [cmath_gen_0_shot_cot_str.py](cmath_gen_0_shot_cot_str.py) |

## 评测口径与验证
合法输入沿用官方候选抽取和数值比较：整数严格相等，其他相对差小于 1%。失败预测保留分母。非法分数抛出 ValueError，非字符串预测判错；修复中间匹配正则的连字符转义并删除未使用的 exception_regs 参数。以上属于对上游异常行为的健壮性修复。中文分数转换支持“3分之1”等阿拉伯数字形式，不支持“五分之三”等汉字数字。训练集为空，chat/str 使用不同 abbr。

```bash
ais_bench --models vllm_api_general_chat --datasets cmath_gen_0_shot_cot_chat_prompt --num-prompts 10
```

运行前配置模型服务及凭据；删除 --num-prompts 可运行全部 600 条。此前 Linux ARM64 CLI 冒烟为 10/10 正确（deepseek-flash，temperature 0，max_tokens 1024，thinking disabled），不代表全量成绩复现。回归测试覆盖非法分数、异常输出、分数转换、截断、目录加载与配置契约。
