# Session 感知并发与 Agent Hint 测评指南

## 特性简介

在 agentic（智能体）场景下，LLM 推理不再是孤立的单轮请求，而是在一个长会话（session）中持续多轮交互：每轮由"已有上下文 + 少量新增 token"构成，上下文逐轮累积，可达数十甚至上百轮。这与传统的"用户问一句、模型答一句"截然不同：

- **请求属于同一个 session**：同一会话的多轮请求必须串行（前一轮生成完才能追加下一轮），不同 session 之间才能并发
- **需要传递 agent hint**：每条样本携带 `agent_hint` 嵌套字段，描述 `session_id`、`stage`、`parent_session_id`、`context_management` 等控制信息，供推理服务侧做 KV cache 复用、上下文管理

AISBench 的 **session-aware 并发**特性针对这种工作负载设计：把"按请求维度并发"改为"按 session 维度并发"，同一 session 内请求严格串行，跨 session 并行；同时支持把 `agent_hint` 透传到 vLLM 请求体，让推理服务感知会话边界。

## 测评能力介绍

服务化支持列表：
+ ✅ vLLM（`v1/chat/completions` 或 `v1/completions`）

数据集支持列表：
+ ✅ 自定义 JSONL 数据集（`CustomDataset`，需含 `agent_hint` 字段）

配套工具：
+ `ais_bench/datasets/agent/build_agent_dataset.py` — 从原始 agent 轨迹构造符合 session 模式的 JSONL 数据集
+ `ais_bench/datasets/agent/build_agent_dataset_from_swe.py` — 针对 SWE-bench 类数据的构造脚本

## 数据格式要求

每行一条 JSON 记录，采用 SWE-bench 风格的多字段结构。顶层字段包括 `question`、`answer`、`tool_choice`、`messages`、`tools`、`agent_hint`、`max_tokens`：

```json
{
  "question": "",
  "answer": "",
  "tool_choice": "auto",
  "messages": [
    {"role": "system", "content": "You are a coding agent..."},
    {"role": "user", "content": "You receive a new message: ..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "...", "tool_call_id": "..."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "read_pdf", "description": "..."}}
  ],
  "max_tokens": 706,
  "agent_hint": {
    "session_id": "bench_20260903T061701Z_0001_astropy__astropy-14995_1a065eb97fd_3779ef995722",
    "parent_session_id": "bench_20260903T061701Z_0001_astropy__astropy-14995_1a065eb97fd_3779ef995722",
    "stage": "stage-3",
    "cache_control": {},
    "context_management": {
      "manage_request": false,
      "edits": []
    }
  }
}
```

> 💡 **SWE-bench 风格数据**：`question`/`answer` 通常留空，实际 prompt 内容放在 `messages` 里，由 `build_agent_dataset_from_swe.py` 构造。`tools` 列出可用工具函数，`tool_choice="auto"` 让模型自主决定是否调用。

### `agent_hint` 字段说明

| 字段 | 作用 |
|---|---|
| `session_id` | 会话标识，相同 session_id 的请求串行执行 |
| `parent_session_id` | 父会话标识，用于 offload 分裂时把新子任务关联到原会话（通常与 `session_id` 相同）|
| `stage` | 轮次标记，取值为 `start` / `stage-1` / `stage-2` / ... / `stage-N` / `end` / `""`；`stage="end"` 表示该 session 的最后一条请求，触发子任务切分 |
| `cache_control` | 缓存控制占位字段（当前为空字典，预留扩展）|
| `context_management.manage_request` | 为 `true` 时该请求被排除出性能统计（控制面请求，不产生 token 输出）|
| `context_management.edits[].type` | 上下文编辑类型，如 `evict`（驱逐）、`offload`（卸载）；dispatcher 会根据 type 切分新子任务 |
| `context_management.edits[].target` | 编辑目标，如 `session`（整个会话）|

### manage_request 与 edits 的真实样例

manage_request 请求（控制面，不产生 token 输出）：

```json
{
  "agent_hint": {
    "session_id": "bench_..._astropy-14995_...",
    "parent_session_id": "bench_..._astropy-14995_...",
    "cache_control": {},
    "context_management": {
      "manage_request": true,
      "edits": [{"type": "evict", "target": "session"}]
    },
    "stage": ""
  }
}
```

普通轮次请求（数据面，产生 token 输出）：

```json
{
  "agent_hint": {
    "session_id": "bench_..._django-12209_...",
    "parent_session_id": "bench_..._django-12209_...",
    "cache_control": {},
    "context_management": {
      "manage_request": false,
      "edits": []
    },
    "stage": "end"
  }
}
```

> 💡 **session_id 不需要写在 `input_columns` 里**：`GenInferencer.get_data_list()` 会自动从 `agent_hint` 中提取。`agent_hint` 本身作为额外字段透传到 vLLM 请求体（当 `pass_agent_hint=True` 时）。

## 快速入门

### 命令含义

以 vLLM `v1/chat/completions` 流式接口的 session 模式性能测评为例：

```
ais_bench --models vllm_api_general_stream_session --datasets custom_agent_session_gen --debug -m perf
```

其中：
- `--models` 指定模型任务 `vllm_api_general_stream_session`（在 `vllm_api_general_stream_session.py` 基础上加了 session 控制参数）
- `--datasets` 指定数据集任务 `custom_agent_session_gen`（自定义 agent session 数据集）

### 运行命令前置准备

- **推理服务**：启动支持 `v1/chat/completions` 的 vLLM 服务，可参考 [vLLM OpenAI 兼容服务器](https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html#openai-compatible-server)
- **数据集**：准备一份符合上述格式的 JSONL 数据集，放置在 `custom_agent_session_gen.py` 中 `path` 字段指向的路径

### 任务对应配置文件修改

加上 `--search` 查询配置文件路径：

```shell
ais_bench --models vllm_api_general_stream_session --datasets custom_agent_session_gen --mode perf --search
```

需要修改的关键配置文件有两个：

#### 1. 模型配置文件 `vllm_api_general_stream_session.py`

```python
from ais_bench.benchmark.models import VLLMCustomAPI

models = [
    dict(
        attr="service",
        type=VLLMCustomAPI,
        abbr="vllm-api-general-stream-session",
        path="/mnt/weight/Qwen3-8B",       # 模型权重路径
        model="qwen",                       # 服务端已加载模型名称
        stream=True,
        use_chat_api=True,                  # 发送到 v1/chat/completions
        session_aware=True,                 # 启用 session 模式
        session_concurrency=4,              # 同时活跃的 session 数上限
        pass_agent_hint=True,               # 把 agent_hint 合并进 vLLM 请求 body
        batch_size=4,                       # session 模式下退化为 janus 队列容量
        request_rate=0,
        host_ip="localhost",
        host_port=31036,                    # vLLM 服务端口
        max_out_len=256,
        generation_kwargs=dict(temperature=0, ignore_eos=False),
    )
]
```

**Session 模式专属参数**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `session_aware` | `False` | 是否启用 session 感知并发；`True` 后按 session 维度调度 |
| `session_concurrency` | `1` | 同时活跃的 session 数上限（跨 session 在途总并发）|
| `pass_agent_hint` | `False` | `False` 时 agent_hint 仅用于内部（提取 session_id/stage），不透传到 API；`True` 时把 agent_hint 作为额外字段合并进请求体 |
| `batch_size` | `1` | session 模式下退化为 janus 队列容量（producer → dispatcher 的缓冲），推荐设为 `session_concurrency` |

> ⚠️ **注意**：session 模式下 `_get_workers_num()` 强制返回 1（单进程），因为跨进程无法共享 session 队列状态。`runner.max_num_workers` 也应置 1。

#### 2. 数据集配置文件 `custom_agent_session_gen.py`

```python
agent_reader_cfg = dict(
    input_columns=['question'],           # prompt 模板用到的字段；max_out_len 走单独机制
    output_column=None,
)

agent_demo_datasets = [
    dict(
        abbr='session-agent',
        type=CustomDataset,
        path='ais_bench/datasets/agent/example.jsonl',  # 改成你的 JSONL 数据路径
        reader_cfg=agent_reader_cfg,
        infer_cfg=agent_infer_cfg,
        eval_cfg=agent_eval_cfg,
    )
]
```

### 最大输出长度（max_out_len / max_tokens）的优先级

AISBench 内部用 `max_out_len` 这个名字控制单次生成的最大 token 数。它有两条生效路径：

**路径 A：AISBench 自己的 `max_out_len` 机制**
- 默认值来自模型配置 `vllm_api_general_stream_session.py` 中的 `max_out_len`（如 256）
- 若数据集 JSONL 中有 `max_out_len` 列、且 `input_columns` 包含 `max_out_len`，则数据集值**优先级最高**，按样本覆盖模型配置
- 若某条样本的 `max_out_len` 为 0/None/空，则 fallback 回模型配置的 `max_out_len`
- 该值最终被写入 vLLM 请求体的 `max_tokens` 字段（vLLM API 标准字段名）

**路径 B：直接透传到 vLLM API 的 `max_tokens` 字段**
- SWE-bench 风格数据中的 `max_tokens` 字段**不被** AISBench 的 `get_max_out_len()` 识别（该方法硬编码检查 `max_out_len`）
- 该字段会作为额外字段透传到 vLLM 请求体，直接由 vLLM 端按 `max_tokens` 生效
- 此时模型配置中的 `max_out_len` 不会进入请求体（被 `max_tokens` 覆盖）

| 数据集情况 | 生效的 max tokens |
|---|---|
| 有 `max_out_len` 列且在 `input_columns` | 数据集的 `max_out_len`（AISBench 机制）|
| 有 `max_tokens` 列（SWE-bench 风格）| 数据集的 `max_tokens`（vLLM API 直接生效）|
| 两者都没有 | 模型配置的 `max_out_len`（默认 256）|

> 💡 **推荐**：如果数据集是 SWE-bench 风格（字段名为 `max_tokens`），无需在 `input_columns` 里显式声明，它会自动作为额外字段透传。模型配置里的 `max_out_len` 此时被 `max_tokens` 覆盖，无需额外处理。

## 调度行为详解

### Session 维度并发控制

启用 `session_aware=True` 后，调度器从"请求维度"切换到"session 维度"：

1. **Dispatcher 实时路由**：读取数据集 → 按 `parent_session_id`（fallback `session_id`）分桶到 per-session 队列 → 创建 `session_worker` 任务
2. **同 session 串行**：同一 session 内的请求按顺序发送，前一轮返回后才发下一轮
3. **跨 session 并行**：最多同时运行 `session_concurrency` 个 session_worker
4. **stage=end 切分子任务**：遇到 `agent_hint.stage="end"` 时，当前 session_worker 结束，dispatcher 准备接收该 session 的下一批请求（如果有）

### Offload / Evict 分裂

当 `agent_hint.context_management.edits` 非空（如 `edits[].type == "offload"` 或 `"evict"`）时，表示该 session 触发了上下文编辑（KV cache 部分移出 HBM 或驱逐整个会话）。调度器会：

1. 当前 `session_worker` 处理完 edit 标记前的请求后退出
2. 创建新的 `session_worker` 处理剩余请求，通过 `parent_session_id` 关联到原会话
3. `wait_all_tasks()` 手动循环等待所有 tasks，因为分裂会动态追加 task

### 管理请求剔除

`agent_hint.context_management.manage_request == true` 的请求是控制面请求（如 KV cache 预热、上下文压缩触发），不产生用户可见 token。这些请求：

- 在 `GenInferencer.do_request()` 中被跳过（当 `pass_agent_hint=False` 时），不发送到推理服务
- 当 `pass_agent_hint=True` 时发送到服务，但 `output.extra_perf_data["manage_request"]` 被标记为 `True`
- `default_perf.py` summarizer 把 manage_request 请求标记为 `excluded=True`
- `DefaultPerfMetricCalculator` 在计算 `data_count` 时扣除 excluded 请求数，`Failed = Total - Success` 不含 manage_request

## 性能指标说明

session 模式下指标含义与常规模式一致，但需注意：

| 指标 | session 模式语义 |
|---|---|
| Total Requests | 仅计入非 manage_request 请求（excluded 已扣除）|
| Failed Requests | `Total - Success`，不含 manage_request |
| TTFT / TPOT | 按非流式场景会被置 0（`default_perf.py` 中 `if not model_cfg.get("stream", True)` 分支）|
| Concurrency | 实际并发度受 `session_concurrency` 限制，不会超过该值 |
| Stable Stage | 稳定阶段识别算法同常规 perf，但并发爬坡阶段可能更长（要等 session 数爬到上限）|

## 数据集构造工具

### `build_agent_dataset.py`

从原始 agent 轨迹（每条记录含完整上下文 + 新增 token）构造符合 session 模式格式的 JSONL：

```shell
python ais_bench/datasets/agent/build_agent_dataset.py \
    --input trajectory.jsonl \
    --output dataset.jsonl \
    --tokenizer /path/to/tokenizer.json \
    --max-token-len 32768
```

主要参数：
- `--input` / `--output`：输入轨迹 / 输出数据集路径
- `--tokenizer`：用于 token 长度过滤的 tokenizer 文件
- `--max-token-len`：单条样本最大 token 长度
- `--copy n`：对大组样本复制 n 次（扩大样本量）

### `build_agent_dataset_from_swe.py`

针对 SWE-bench 类数据（每条记录用 `messages` + `tools` 字段描述任务，`question` 留空）：

```shell
python ais_bench/datasets/agent/build_agent_dataset_from_swe.py \
    --input swe_trajectories.jsonl \
    --output dataset.jsonl
```

> ⚠️ **注意**：`from_swe` 变体中 `question` 始终为空字符串，`--max-token-len` 参数对该变体无效（过滤逻辑检查的是 `question` 字段长度）。

## 常见问题

### Q: 同一 session 的请求会乱序吗？
不会。dispatcher 按 `parent_session_id` 分桶，同一 session 队列严格 FIFO，`session_worker` 顺序发送。

### Q: 不同 session 之间会争抢 GPU 吗？
会。最多 `session_concurrency` 个 session 同时活跃，它们共享 GPU 资源。`session_concurrency` 应根据 GPU 显存和 KV cache 容量调整，避免 OOM。

### Q: `pass_agent_hint=True` 会影响推理服务性能吗？
会略微增加请求体大小（多了 `agent_hint` 字段），但 vLLM 端会忽略未识别字段，不影响推理本身。如果推理服务侧实现了对 `agent_hint` 的处理（如 prefix cache 命中、上下文管理），则能显著提升性能。

### Q: manage_request 请求会影响 TTFT/TPOT 统计吗？
不会。manage_request 请求被标记为 `excluded=True`，从 `data_count` 中扣除，不参与任何 token/延迟指标计算。

### Q: 如何调试 session 调度行为？
加 `--debug` 参数，dispatcher 会打印每个 session 的请求分配、stage=end 切分、offload 分裂等日志。
