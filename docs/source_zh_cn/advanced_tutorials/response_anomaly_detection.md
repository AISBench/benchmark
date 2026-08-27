# 推理响应异常检测

## 概述

AISBench 集成 msProbe 的 `ILLDetector`，支持在推理评测的同时自动检测大模型响应中的生成异常。检测结果覆盖以下类型：

| `anomaly_type` | `anomaly_type_name` | 含义 |
| -------------- | ------------------- | ---- |
| 0 | `normal` | 正常 |
| 1 | `rare_character` | 生僻字 |
| 2 | `garbled` | 乱码 |
| 3 | `repetition` | 重复 |
| 4 | `nan_value` | NaN Value |

**异常检测结果不影响原有评测指标**：异常 Case 不会被改写为推理失败，精度/性能指标照常计算，异常信息是独立的审计结果。

检测通过 msProbe 的 `ILLDetector(config_path, mtype_path, tk2cat_path).run(...)` 完成，三个文件路径均可由 AISBench 配置（见[配置说明](#配置说明)）。

---

## 前置条件

1. **推理后端**：当前响应异常检测仅支持基于 vLLM Chat API 的 `vllm_api_general_chat`、`vllm_api_stream_chat` 和 `vllm_api_stream_chat_multiturn` 模型配置，其他模型后端暂不支持。
2. **评测模式**：仅支持 `all`、`infer`、`infer_judge` 普通生成链路；`perf` / `perf_viz` 性能评测模式以及 Agent / 函数调用等自定义链路不支持，启用时会在配置初始化阶段显式报错。
3. **服务端要求**：服务响应必须包含 `token_ids`（或 `tokens`）和 `topk_logprobs` 字段；缺少这些字段的 Case 会以 `skipped` 状态落盘。
4. **可选依赖**：需安装 `response_anomaly` extra（见下节）。

> 💡 检测串行绑定在推理阶段内：推理结束后启动检测并等待其完成（专属状态面板打印最终结果后）才进入后续 Judge / Eval / 汇总流程，保证推理阶段退出时检测结果与 payload 归档均已落盘。模式支持范围详见[模式说明](../base_tutorials/all_params/mode.md)。

---

## 安装依赖

响应异常检测依赖可选包 `mindstudio-probe`，通过 AISBench 的 extra 安装：

```bash
pip install 'ais-bench-benchmark[response_anomaly]'
```

安装过程中 pip 会从 GitCode 下载并构建已固定提交的 msProbe 源码，因此安装环境需要 Git 和网络访问。

未安装该依赖时评测流程仍可正常运行，但所有 Case 的检测结果会标记为 `unavailable` 状态（见[检测结果说明](#检测结果说明)）；安装后重跑即可恢复检测。

---

## 快速使用

**1. 在模型配置中添加 `response_anomaly` 字段**（最小配置只需提供 `model_name` 或 `model_path` 之一）：

```python
models = [
    dict(
        abbr='qwen3-30b',
        attr='service',
        response_anomaly=dict(
            model_name='Qwen3-30B-A3B',   # 或仅提供 model_path，自动取其中的模型名称
        ),
    ),
]
```

**2. 在命令行增加 `--response-anomaly` 运行评测**：

```bash
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --response-anomaly
```

> ⚠️ 异常检测的**功能开关仅支持命令行**：在命令中增加 `--response-anomaly` 开启（不增加默认不开启）；配置文件中的 `response_anomaly` 仅用于非开关类配置（`payload_retention`、`payload_storage` 等）。

**3. 查看检测结果**：推理与检测结束后，检测结果位于 `<work_dir>/response_anomaly/<模型 abbr>/<数据集 abbr>.jsonl`，每行一个 Case（完整落盘结构见[运行流程与落盘结构](#运行流程与落盘结构)）。

---

## 配置说明

### 全局配置

全局 `response_anomaly` 配置控制 payload 的保留策略与存储格式：

```python
response_anomaly = dict(
    payload_retention='anomalies',  # all | anomalies | none
    payload_storage=dict(
        format='jsonl',
        compression='zstd',
        compression_level=3,
        rows_per_shard=2000,
    ),
)
```

`payload_retention` 决定检测完成后 payload 的保留范围：

| 取值 | 行为 |
| ---- | ---- |
| `all` | 保存全部 payload，检测后直接将 staging 原子转为正式归档，不会二次压缩 |
| `anomalies`（默认） | 只保存已检出异常以及检测失败/不可用 Case |
| `none` | 不保存 payload |

三种模式都保留独立检测结果。命令行参数 `--response-anomaly-payload-retention` 可覆盖配置文件取值。

> 💡 运行期细节：检测结果按批写盘，状态最多每秒刷新一次；msProbe token 分类映射按模型和 EOS token 缓存，避免每个 Case 重复解析大 JSON 文件。

### 模型级配置（msProbe）

模型相关的 msProbe 配置放在模型配置中：

```python
models = [
    dict(
        abbr='qwen3-30b',
        attr='service',
        response_anomaly=dict(
            model_name="",   # 填写模型名称，如 Qwen3-30B-A3B
            model_path="",   # 填写本地模型目录，如 /home/Qwen3-30B-A3B；可选，用于自动生成配置
            msprobe_config_path="",  # 可选，msProbe 算法阈值配置 config.yaml 路径，用于手工调优检测阈值
            msprobe_mtype_path="",  # 可选，msProbe 模型名与 BOS/EOS token id 映射文件 mtype_config.json 路径
            msprobe_token2category_dir="",  # 可选，msProbe token2category 目录，存放各模型的 token id 到字符类别映射
        ),
    ),
]
```

**`model_name` 的取值规则**：未显式配置时自动取模型路径（`model_path` 或模型 `path` 字段）中的**模型名称**（如 `/home/Qwen3-30B-A3B` → `Qwen3-30B-A3B`），与配置生成工具的默认取值一致。当既未配置 `model_name`、也没有可用的模型路径（如仅显式配置 msProbe 三件套路径）时，任务启动即报错，要求显式配置 `model_name`，避免以错误的模型名静默运行导致检测失效。

`model_name` 需与 msProbe 的 `mtype_config.json` 配置以及 token 分类映射保持一致。

未提供 `msprobe_mtype_path` / `msprobe_token2category_dir` 时回退到 msProbe 包内默认文件。

### msProbe 配置的自动生成与手动生成

配置了 `model_path` 时，msProbe 配置会自动生成到 `<work_dir>/response_anomaly_config/<模型 abbr>/`（自动生成不会覆盖已存在的 `config.yaml`，便于保留手工调优的阈值）。也可手动生成：

```bash
ais_bench-gen-response-anomaly-config \
  --model-path /home/Qwen3-30B-A3B \
  --model-name Qwen3-30B-A3B \
  --output-dir ./msprobe_configs
```

---

## 运行流程与落盘结构

### 请求参数自动注入

启用后，AISBench 会在服务推理请求中补充 `logprobs=True` 与固定的 `top_logprobs=20`；该值由检测算法约束，不支持外部配置。对 vLLM 后端还会追加 `return_token_ids=True` 与 `return_tokens_as_token_ids=True` 以获取 token id，服务端版本过低不支持这些参数时请求可能失败，需升级 vLLM。

### 检测流程

1. **推理阶段**：完整 payload 直接写入 `response_anomaly/<模型>/payload_staging/<数据集>/*.jsonl.zst`，prediction 从一开始只保存轻量结果；
2. **检测阶段**：推理结束后，检测线程流式解压 staging 数据并调用 msProbe，检测结果写入 `response_anomaly/<模型>/<数据集>.jsonl`；
3. **归档收尾**：检测完成后按 `payload_retention` 保留或清理 staging。

状态面板会显示配置准备、检测器加载、流式检测和归档收尾阶段。

### 落盘结构

检测相关文件在 `<work_dir>` 下的落盘结构如下：

```text
<work_dir>/
├── predictions/<模型 abbr>/<数据集 abbr>.jsonl          # 推理结果（轻量，不含 token/logprobs payload）
├── response_anomaly/
│   └── <模型 abbr>/
│       ├── <数据集 abbr>.jsonl                          # 检测结果，每行一个 Case
│       ├── payload_staging/<数据集 abbr>/               # 推理期间的临时存放区，检测完成后自动清理
│       │   └── part-*.jsonl.zst
│       └── payload/<数据集 abbr>/                       # payload 归档；payload_retention 为 none 时不存在
│           ├── payload_manifest.json                    # 归档清单（分片行数、大小、sha256）
│           └── part-*.jsonl.zst                         # 压缩 payload 分片（每片最多 rows_per_shard 条 Case）
├── response_anomaly_config/                             # 仅配置 model_path 自动生成时存在
│   └── <模型 abbr>/
│       ├── configs/
│       │   ├── config.yaml                              # 检测算法阈值配置（已存在时不覆盖）
│       │   └── mtype_config.json                        # 模型名与 BOS/EOS token id 映射
│       └── token2category/
│           └── <模型名>_<词表大小>.json                  # token id 到字符类别映射
└── logs/
    └── response_anomaly/<模型 abbr>/<数据集 abbr>.out   # 检测专属日志
```

各路径说明：

- **检测结果** `response_anomaly/<模型 abbr>/<数据集 abbr>.jsonl`：每行一个 Case 的检测结果，字段含义见[检测结果说明](#检测结果说明)。
- **payload 归档** `response_anomaly/<模型 abbr>/payload/<数据集 abbr>/`：`all` 保留全部 Case，`anomalies` 只保留异常及检测失败/不可用 Case，`none` 不保留（目录不存在）。读取时用 zstandard 解压 `part-*.jsonl.zst` 分片后逐行解析 JSON；`payload_manifest.json` 记录每个分片的行数、大小与 sha256 校验值，可用于完整性校验。注意：`anomalies` 模式下即使无任何需保留的 Case，仍会发布一个仅含空 manifest 的归档目录，表示归档流程已成功完成，不是残留文件。
- **临时文件**：`payload_staging/` 在推理期间逐条接收 payload 写入，检测完成后自动清理；检测中断后残留的 `.<数据集>.payload-build-*` 构建目录会在下次检测启动时自动清理；`status_tmp/tmp_ResponseAnomaly.json` 为运行期状态文件（检测进度与类型统计），工作流结束后随状态目录一并清理。
- **自动生成的 msProbe 配置** `response_anomaly_config/<模型 abbr>/`：仅当配置了 `model_path` 且未显式提供 mtype/token2category 路径时生成。`config.yaml` 已存在时不会被覆盖（保留手工调优的阈值）；`mtype_config.json` 支持多模型合并，多次生成不互相覆盖。
- **检测日志** `logs/response_anomaly/<模型 abbr>/<数据集 abbr>.out`：记录对应模型/数据集组的检测过程，含检测器初始化失败与单 Case 失败的具体原因。

---

## 检测结果说明

检测结果 `response_anomaly/<模型 abbr>/<数据集 abbr>.jsonl` 每行一个 Case，包含 `id`、`uuid`、`is_anomaly`、`anomaly_type`（0：正常，1：生僻字，2：乱码，3：重复，4：NaN Value）、`anomaly_type_name`（类型名字符串，如 `normal`/`garbled`/`repetition`，统计时更常用）和 `detection_status`。

检测结果中的 `detection_status` 取值如下：

| 状态 | 含义 | 排查建议 |
| --- | --- | --- |
| `completed` | 已调用 msProbe 并得到检测结果 | 无需处理 |
| `skipped` | 推理响应未携带 token id 或 top-k logprobs | 检查服务端是否支持并返回 `logprobs` / `top_logprobs` / token id 字段 |
| `unavailable` | 未安装 `mindstudio-probe`（response_anomaly extra） | 参考[安装指南](../get_started/install.md)安装可选依赖后重跑 |
| `failed` | 调用或输入转换发生异常 | 查看该 Case 结果中的 `reason` 字段（保存错误类型与摘要）及检测日志 |

检测专属日志位于 `<work_dir>/logs/response_anomaly/<模型>/<数据集>.out`，检测进度与类型统计也可在 `<work_dir>/status_tmp/tmp_ResponseAnomaly.json` 状态文件中查看。

---

## 断点续跑

使用 `--reuse` 时，已有检测结果按 Case 的 `id` + `uuid` 双键匹配继承（`uuid` 变化说明该 Case 已重新推理，不会错挂旧结果）：

- `completed` 状态的 Case 不会重复检测；
- `skipped` / `failed` / `unavailable` 状态的 Case 会在续跑中重新检测。

`--reuse` 续跑必须沿用原工作目录的 payload 保留策略。
