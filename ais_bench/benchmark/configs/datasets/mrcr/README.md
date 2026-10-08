# MRCR (Multi-Round Co-reference Resolution)
中文 | [English](README_en.md)

## 数据集简介

MRCR（Multi-Round Co-reference Resolution）是 OpenAI 随 GPT-4.1 发布的长上下文多针检索（multiple needle in a haystack）基准数据集（MIT 协议），灵感来自 Gemini 团队的 MRCR 评测（arXiv:2409.12640）。

任务形式：模型读入一段最长约 1M token 的多轮合成对话，其中同一写作请求（如 "write a poem about tapirs"）以 2/4/8 次重复隐藏在同分布的干扰请求中（所有 assistant 回复均由 gpt4o 生成，针与干草堆不可区分），模型最终被要求返回第 N 次出现的实例，并在答案前拼接指定随机串：

```
User: Prepend aYooSG8CQg to the 2nd (1 indexed) poem about tapirs.
      Do not include any other text in your response.
Assistant: aYooSG8CQg（第 2 首关于 tapirs 的诗）
```

该任务的挑战在于：针与干扰项同分布生成、不可区分，模型必须区分多个相同请求的出现顺序；needle 数越多、上下文越长，难度越高。

数据按 needle 数分为 `2needle` / `4needle` / `8needle` 三个子集，每个子集按 prompt+answer 的 `o200k_base` token 数划分为 8 个 bin，每个 bin 含 100 条样本（单子集约 800 条，全量约 2400 条）。

官方评测协议为纯规则评分（无 LLM 裁判）：响应必须以 `random_string_to_prepend` 开头，否则记 0 分；通过后双方剥离前缀，计算 `SequenceMatcher(None, response, answer).ratio()`；最终得分为所有样本 ratio 的平均值。

> 🔗 数据集主页链接: [https://huggingface.co/datasets/openai/mrcr](https://huggingface.co/datasets/openai/mrcr)
>
> 🔗 GPT-4.1 博客（MRCR 结果）: [https://openai.com/index/gpt-4-1/](https://openai.com/index/gpt-4-1/)

## 数据集部署

- 从 HuggingFace 数据集链接 🔗 [https://huggingface.co/datasets/openai/mrcr](https://huggingface.co/datasets/openai/mrcr) 获取数据集。
- MRCR 数据集为 Parquet 格式，按 needle 子集分目录存放，建议部署在 `{tool_root_path}/ais_bench/datasets/MRCR/` 目录下：

```bash
huggingface-cli download openai/mrcr --repo-type dataset --local-dir {tool_root_path}/ais_bench/datasets/MRCR
```

- 在 `{tool_root_path}/ais_bench/datasets/MRCR/` 目录下执行 `ls` 检查目录结构。如果目录结构如下所示，则数据集部署成功（分片数量以实际发布为准）：

```
{tool_root_path}/ais_bench/datasets/MRCR/
├── 2needle/
│   ├── 2needle_0.parquet
│   └── 2needle_1.parquet
├── 4needle/
│   ├── 4needle_0.parquet
│   └── ...
└── 8needle/
    ├── 8needle_0.parquet
    └── ...
```

## 切片配置（subset / length_bin）

数据集配置支持两个切片参数（见 `datasets/mrcr.py` 与 `mrcr_1m_gen.py`）：

- `subset`：needle 子集目录名，可选 `2needle` / `4needle` / `8needle`。
- `length_bin`：按官方 bin 边界（prompt+answer 的 `o200k_base` token 数）过滤，可选 `8k` / `16k` / `32k` / `64k` / `128k` / `256k` / `512k` / `1m`；`None` 表示不过滤（全量，快速冒烟路径）。

| bin  | token 范围        | bin   | token 范围            |
| ---- | ----------------- | ----- | --------------------- |
| 8k   | [4096, 8192]      | 128k  | (65536, 131072]       |
| 16k  | (8192, 16384]     | 256k  | (131072, 262144]      |
| 32k  | (16384, 32768]    | 512k  | (262144, 524288]      |
| 64k  | (32768, 65536]    | 1m    | (524288, 1048576]     |

默认任务 `mrcr_1m_gen` 采用 `subset='8needle'`、`length_bin=None`（全部 8 个 bin，约 800 条），与技术报告（如 DeepSeek-V4 "MRCR 1M"）的公开口径一致：8needle 在 8K–1M 全部 bin 上的平均。各 bin 样本数相同（100 条），全样本均值等于逐 bin 宏平均。

## 推理参数对齐说明

| 参数        | 建议值  | 说明                                                                                                        |
| ----------- | ------- | ----------------------------------------------------------------------------------------------------------- |
| temperature | 1.0     | 官方不传采样参数（OpenAI API 默认 1.0）；本地 vLLM 服务的 generation_config 可能默认 0.0，需在模型配置 generation_kwargs 中显式设置，防止静默偏离 |
| max_out_len | ≥ 8192  | 官方不限输出；针答案约 1K token，但 thinking 模型的推理 token 计入输出预算，预算不足会截断最终答案、前缀门控直接判 0                              |

- thinking 模型：推理内容与最终答案分离返回时，评测自动使用不含推理内容的 `content` 字段评分（`prediction` 字段拼接了推理内容，会导致前缀门控失败）。
- `--num-prompts` 按顺序截断时可能只覆盖首个 bin 的样本，部分跑分不具代表性；对齐公开报告请全量运行（8needle 约 800 条，单条最高 1M token）。

## 可用数据集任务

|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|mrcr_1m_gen|MRCR 8needle 全量（8K–1M bin）多针检索任务|score（SequenceMatcher ratio 均值）、prefix_hit_rate（前缀命中率）、strict_acc（精确复现率）|0-shot|对话格式|`from ais_bench.benchmark.configs.datasets.mrcr.mrcr_1m_gen import mrcr_1m_datasets as datasets`|[mrcr_1m_gen.py](mrcr_1m_gen.py)|

## 运行示例

```bash
# 推理（temperature=1.0 对齐官方，建议 max_out_len ≥ 8192）
ais_bench -m infer --models vllm_api_general_chat --datasets mrcr_1m_gen
# 评测（纯规则评分，无需裁判模型）
ais_bench -m eval --models vllm_api_general_chat --datasets mrcr_1m_gen
```
