# 多模态理解数据构造与压测

该入口构造并压测两组固定规格数据：

| 场景 | 请求数 | 图片 | 文本 | 输出 |
|---|---:|---|---|---:|
| `single_1080p` | 1319 | 每请求 1 张相同的 MMMU 1920×1080 图片 | 1319 条不同 GSM8K 文本，每条精确 30 token | 256 token |
| `multi_720p_5` | 1319 | 每请求重复同一张 MMMU 1280×720 图片 5 次；所有请求一致 | 同上 | 256 token |

图片 content 固定放在变化文本之前，以形成可复用的多模态前缀。生成器直接扫描 MMMU Parquet 的 `image_1`～`image_7` 二进制字段，挑选原生尺寸严格等于 1920×1080 和 1280×720 的图片，不进行缩放。随后将图片编码为 Base64，并校验 GSM8K 行数与唯一性、文本 token 数、图片数量、分辨率和 SHA-256；不满足要求时直接失败。

## 1. 构造数据

先安装仓库和插件：

```bash
python -m pip install -e .
python -m pip install -e ./plugins/prefix_cache
```

使用与被测 vLLM 模型一致的 tokenizer：

```bash
ais-bench-prefix-cache-mm prepare \
  --tokenizer /path/to/vlm-tokenizer \
  --gsm8k 'C:\cjs\datasets\grade-school-math\grade_school_math\data\test.jsonl' \
  --mmmu-parquet-dir '../MMMU' \
  --output-dir ./outputs/prefix_cache_multimodal
```

`--mmmu-parquet-dir` 指向 MMMU 的 Parquet 数据集根目录，不需要也不接受单独图片目录。程序按稳定顺序扫描分片，并从图片 bytes 中选择首张符合目标原生尺寸的图片。默认值已经是 1319 / 30 / 256；验收数据不要覆盖这些值。

为避免在 1319 行中反复写入相同 Base64（多图场景会造成数 GB 的无意义重复），Manifest 对每个场景只保存一次完整 `data:image/...;base64,...`；JSONL 中的每条 `prompt` 保存相应的 `base64_ref`。AISBench 生成每个最终请求时会把引用展开为完整 Base64 URL，因此实际发送给服务端的 prompt 不依赖本地图片路径。

tokenizer 确实依赖自定义代码时才追加 `--trust-remote-code`。

输出：

```text
outputs/prefix_cache_multimodal/
├── single_1080p_1319.jsonl
├── multi_720p_5_1319.jsonl
└── multimodal_prefix_cache.manifest.json
```

再次独立校验：

```bash
ais-bench-prefix-cache-mm validate \
  --manifest ./outputs/prefix_cache_multimodal/multimodal_prefix_cache.manifest.json
```

注意：30 token 是每条 GSM8K 文本本身的 tokenizer 长度，不包含模型 chat template 和视觉 placeholder token。Manifest 会记录 tokenizer、语料和图片指纹。

## 2. 运行压测

vLLM 必须启用 Automatic Prefix Caching。多模态请求使用 `/v1/chat/completions`：

```bash
ais-bench-prefix-cache-mm run \
  --manifest ./outputs/prefix_cache_multimodal/multimodal_prefix_cache.manifest.json \
  --scenario all \
  --inference-url http://127.0.0.1:8000/v1/chat/completions \
  --model /path/to/vlm \
  --tokenizer /path/to/vlm-tokenizer \
  --batch-size 8 \
  --work-dir ./outputs/prefix_cache_multimodal_perf
```

`--batch-size` 是最大并发。AISBench 的其他参数可重复传入，例如：

```bash
--extra-arg=--request-rate --extra-arg=16
```

图片始终通过 Base64 data URL 发送，不需要 vLLM 服务访问 MMMU 本地路径。多图场景每个请求包含 5 个相同 Base64 图片内容，HTTP 请求体会相应增大。

## 3. 验收 TTFT / TPOT / ITL

`run` 完成后会直接打印以下指标的 Average、Min、Max、Median、P75、P90、P99 和样本数：

- `TTFT`：首 token 时延；
- `TPOT`：每输出 token 时延；
- `ITL`：相邻输出 token 间隔；
- `E2EL`、InputTokens、OutputTokens；
- Request Throughput、成功/失败请求数等公共指标。

也可对已有输出重新汇总：

```bash
ais-bench-prefix-cache-mm report \
  --work-dir ./outputs/prefix_cache_multimodal_perf/single_1080p \
  --dataset-abbr single_1080p
```

原始结果位于 AISBench 输出目录的：

```text
performances/prefix-cache-mm-vllm/<scenario>.csv
performances/prefix-cache-mm-vllm/<scenario>.json
```

必须保持 `stream=True`；非流式响应无法得到有效的 TTFT、TPOT 和 ITL。
