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
+ `ais_bench/datasets/agent/build_agent_dataset.py` — 从原始 agent 轨迹构造符合 session 模式的 JSONL 数据集（同一脚本通过 `--swe-format` 开关切换普通格式与 SWE-bench 风格输出）

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

> 💡 **SWE-bench 风格数据**：`question`/`answer` 通常留空，实际 prompt 内容放在 `messages` 里，由 `build_agent_dataset.py --swe-format` 构造。`tools` 列出可用工具函数，`tool_choice="auto"` 让模型自主决定是否调用。

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

`ais_bench/datasets/agent/build_agent_dataset.py` 把原始数据转成符合 session 模式格式的 JSONL。同一个脚本通过 `-t/--src-dataset-type` 切换数据源类型，目前支持：

- `default`（默认）：从一个 JSONL 文件读 `question` 字段，按 tokenizer 分词后用 trie 做前缀分组，命中率高（共享前缀长）的记录归到同一个 session。适合「问句有大量共享前缀」的 prompt 工作负载（如同一模板的多次微调、不同变体的同类问题）。
- `Jiuwen`：从目录里读 Jiuwen agent 轨迹 JSON 文件，按 `parent_session_id` 聚合大组、按 `session_id` 切分小组。适合真实 agent 多轮轨迹回放。

### `default` 模式：从普通 JSONL 构造

输入是一个 JSONL 文件，每行至少包含 `question` 字段。脚本会按 token 前缀把记录分桶，同一桶内的记录顺序追加为同一 session 的多轮，并在每个 session 末尾追加一条 `evict` 控制面记录。

```shell
python ais_bench/datasets/agent/build_agent_dataset.py \
    -t default \
    -s input_questions.jsonl \
    -o dataset.jsonl \
    --tokenizer /path/to/tokenizer.json \
    -r 0.5
```

主要参数（`default` 相关）：
- `-s/--src`：输入 JSONL 文件路径（每行一条记录，至少含 `question` 字段）
- `-o/--output`：输出 JSONL 路径
- `--tokenizer`：**必填**，HuggingFace `tokenizer.json` 文件，用于分词和前缀分组
- `-r/--hit-rate`：新 session 触发阈值 ∈ [0.0, 1.0]，记录对当前 session trie 的命中率 ≤ 该值就开新 session。默认 `0.0`（只有 0% 命中才切分）

输入示例（`input_questions.jsonl`）：

```jsonl
{"question": "Please summarize the following article: <article A...>"}
{"question": "Please summarize the following article: <article B...>"}
{"question": "Please summarize the following article: <article C...>"}
{"question": "Translate the following English text to French: <text...>"}
```

输出示例（`dataset.jsonl`，前 3 条同 session，第 4 条新 session，末尾各跟 evict）：

```jsonl
{"question": "Please summarize the following article: <article A...>", "agent_hint": {"session_id": "sid_00001", "parent_session_id": "", "stage": "start", "cache_control": {}, "context_management": {"manage_request": false, "edits": []}}}
{"question": "Please summarize the following article: <article B...>", "agent_hint": {"session_id": "sid_00001", "parent_session_id": "", "stage": "stage-1", "cache_control": {}, "context_management": {"manage_request": false, "edits": []}}}
{"question": "Please summarize the following article: <article C...>", "agent_hint": {"session_id": "sid_00001", "parent_session_id": "", "stage": "stage-2", "cache_control": {}, "context_management": {"manage_request": false, "edits": []}}}
{"question": "", "agent_hint": {"session_id": "sid_00001", "parent_session_id": "", "stage": "stage-3", "cache_control": {}, "context_management": {"manage_request": true, "edits": [{"type": "evict", "target": "session"}]}}}
{"question": "Translate the following English text to French: <text...>", "agent_hint": {"session_id": "sid_00002", "parent_session_id": "", "stage": "start", "cache_control": {}, "context_management": {"manage_request": false, "edits": []}}}
{"question": "", "agent_hint": {"session_id": "sid_00002", "parent_session_id": "", "stage": "stage-1", "cache_control": {}, "context_management": {"manage_request": true, "edits": [{"type": "evict", "target": "session"}]}}}
```

### `Jiuwen` 模式：从真实 agent 轨迹构造

```shell
python ais_bench/datasets/agent/build_agent_dataset.py \
    -t Jiuwen --src swe_trajectories_dir \
    --output dataset.jsonl \
    --cut-count 4 --session-count 6 --evict
```

主要参数（`Jiuwen` 相关）：
- `-s/--src`：输入目录路径，目录里是 Jiuwen agent 轨迹 JSON 文件
- `-c/--cut-count`：每个 session 只保留最后 N 条记录
- `-n/--session-count`：只保留前 N 个 session（按首次出现顺序）
- `-p/--add-pause <fraction>`：在指定比例的 session 中随机插入一对 offload + prefetch 暂停点（小写 p）
- `-P/--preview`：预览模式，把 question 文本替换成 `"preview text"` 并打印 ParentGroup/SessionGroup 结构（大写 P）
- `--show-agent-hint`：配合 `--preview` 使用，额外打印每条记录的完整 `agent_hint` JSON
- `--evict`：用 `group_by_parent_sessions_evict` 替代默认逻辑——中断小组末尾追加 offload，未中断小组末尾追加 evict + end 两条记录，大组最后小组仅追加 evict
- `--max-token-len`：丢弃 `question` token 数超过该值的记录（需配合 `--tokenizer`）
- `--copy n`：每个大组（按 `parent_session_id`）复制 n 份，复本的 `session_id`/`parent_session_id` 加 8 字符随机后缀以区分

### `--swe-format` 输出格式

`default` 和 `Jiuwen` 都支持 `--swe-format` 开关，把数据面记录的 `question` 平铺字段改成 SWE-bench 风格的 `messages` + `tools` + `tool_choice`，`question` 留空。控制面记录（evict/offload/end）也会带上占位 `messages`/`tools` 以保持形状一致。用于对接 `v1/chat/completions` 端点：

```shell
python ais_bench/datasets/agent/build_agent_dataset.py \
    -t Jiuwen --src swe_trajectories_dir \
    --output dataset.jsonl --swe-format
```

> ⚠️ **注意**：开启 `--swe-format` 时数据记录的 `question` 字段为空字符串，`--max-token-len` 在该模式下无效（过滤逻辑检查的是 `question` 字段长度）。

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
