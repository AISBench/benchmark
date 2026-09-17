# 多模态理解数据构造与压测

多模态与纯文本现在共用 `ais-bench-prefix-cache` 入口和同一种 Scenario JSON：

```bash
ais-bench-prefix-cache prepare --mode text --scenario ./scenario.json
ais-bench-prefix-cache prepare --mode mm --scenario ./scenario.json
```

`prepare` 必须显式指定 `--mode text|mm`。`run`、`validate`、`report` 从 prepare 生成的 Manifest 中读取 `benchmark_mode`，不再接收 `--mode`。旧的 `ais-bench-prefix-cache-mm` 命令已删除。

## Scenario 配置

多模态模式复用 [scenario.example.json](config_examples/scenario.example.json) 的文本配置：

- `tokenizer.path`：被测 VLM 对应的 tokenizer；
- `corpus.path` / `corpus.field`：GSM8K JSONL 和问题字段；
- `requests.count`：请求数量；
- `requests.input_length`：必须为 `fixed`，其 `value` 是每条 GSM8K 文本的精确 token 数；
- `requests.output_length`：必须为 `fixed`，其 `value` 是每请求输出 token 数；
- `run.output_dir`：统一的时间戳产物根目录；
- `service`：OpenAI 兼容推理服务、模型和 API key；
- `aisbench`：工作目录、额外参数、batch size、stream、retry 和 generation kwargs。

只需追加图片特有配置：

```json
{
  "multimodal": {
    "mmmu_parquet_dir": "../../../../MMMU",
    "scenarios": ["single_1080p", "multi_720p_5"]
  }
}
```

`multimodal.mmmu_parquet_dir` 在 `--mode mm` 时必填，相对路径以 Scenario 文件目录为基准。`multimodal.scenarios` 可省略，默认只生成 `single_1080p`；支持：

- `single_1080p`：每请求1张相同的原生1920×1080 MMMU图片；
- `multi_720p_5`：每请求重复同一张原生1280×720 MMMU图片5次。

程序直接扫描 MMMU Parquet 的 `image_1`～`image_7` bytes 字段，严格按原生尺寸选择图片，不进行缩放。图片编码为 Base64；Manifest 每个场景只保存一次完整 data URL，JSONL prompt 保存 `base64_ref`，发送请求时再展开为完整 `data:image/...;base64,...`。

1319条、30-token文本、256-token输出的关键配置如下：

```json
{
  "requests": {
    "count": 1319,
    "input_length": {"mode": "fixed", "value": 30},
    "output_length": {"mode": "fixed", "value": 256}
  },
  "service": {
    "inference_url": "http://127.0.0.1:8000/v1/chat/completions",
    "model": "model-name"
  },
  "aisbench": {
    "extra_args": ["--num-warmups", "0"],
    "model": {
      "stream": true,
      "batch_size": 8,
      "retry": 2,
      "generation_kwargs": {"temperature": 0, "ignore_eos": true}
    }
  }
}
```

## 构造、运行与验收

```bash
ais-bench-prefix-cache prepare --mode mm --scenario ./scenario.json
```

产物使用与文本模式相同的时间戳布局：

```text
<output_dir>_<YYYYMMDD_HHMMSS>/
├── log/
└── result/
    ├── <run_id>_<timestamp>.manifest.json
    ├── <run_id>_<timestamp>.single_1080p.requests.jsonl
    └── <run_id>_<timestamp>.multi_720p_5.requests.jsonl
```

运行压测：

```bash
ais-bench-prefix-cache run --scenario ./scenario.json
```

校验指定 Manifest：

```bash
ais-bench-prefix-cache validate --manifest /path/to/run.manifest.json
```

重新汇总 TTFT、TPOT、ITL 等性能指标：

```bash
ais-bench-prefix-cache report --manifest /path/to/run.manifest.json
```

`stream=true` 是获得有效 TTFT、TPOT、ITL 的必要条件。图片始终通过 Base64 data URL 发送，服务端不需要访问本地 MMMU 路径。
