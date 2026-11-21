# Session-Aware Concurrency and Agent Hint Benchmarking Guide

## Feature Overview

In agentic scenarios, LLM inference is no longer isolated single-turn requests. Instead, it involves long-running sessions with many turns: each turn consists of "existing context + a small number of newly appended tokens", and the context accumulates turn by turn, potentially reaching dozens or even hundreds of turns. This is fundamentally different from traditional "user asks one question, model answers one question":

- **Requests belong to the same session**: multiple turns within the same session must be serialized (the next turn can only be appended after the previous turn completes), while concurrency is only possible across sessions
- **Agent hint passthrough**: each sample carries an `agent_hint` nested field describing `session_id`, `stage`, `parent_session_id`, `context_management`, and other control information. The inference server uses this for KV cache reuse and context management

AISBench's **session-aware concurrency** feature is designed for this workload: it switches from "request-level concurrency" to "session-level concurrency" — requests within the same session are strictly serialized, while different sessions run in parallel. It also supports passing `agent_hint` through to the vLLM request body so the inference server can perceive session boundaries.

## Benchmarking Capabilities

Supported serving backends:
+ ✅ vLLM (`v1/chat/completions` or `v1/completions`)

Supported datasets:
+ ✅ Custom JSONL datasets (`CustomDataset`, must contain `agent_hint` field)

Companion tools:
+ `ais_bench/datasets/agent/build_agent_dataset.py` — builds session-format JSONL datasets from raw agent trajectories (the same script switches between plain and SWE-bench-style output via the `--swe-format` flag)

## Data Format Requirements

Each line is a JSON record using a SWE-bench-style multi-field structure. Top-level fields include `question`, `answer`, `tool_choice`, `messages`, `tools`, `agent_hint`, `max_tokens`:

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

> 💡 **SWE-bench-style data**: `question`/`answer` are typically left empty; the actual prompt content is placed in `messages`, constructed by `build_agent_dataset.py --swe-format`. `tools` lists available tool functions, and `tool_choice="auto"` lets the model decide whether to call them.

### `agent_hint` Field Descriptions

| Field | Purpose |
|---|---|
| `session_id` | Session identifier; requests with the same session_id are serialized |
| `parent_session_id` | Parent session identifier; used to associate new sub-tasks with the original session during offload/evict splits (typically identical to `session_id`) |
| `stage` | Turn marker; values are `start` / `stage-1` / `stage-2` / ... / `stage-N` / `end` / `""`; `stage="end"` indicates the last request of this session, triggering sub-task splitting |
| `cache_control` | Cache-control placeholder field (currently an empty object, reserved for future extension) |
| `context_management.manage_request` | When `true`, the request is excluded from performance statistics (control-plane request, produces no token output) |
| `context_management.edits[].type` | Context-edit type, e.g. `evict` (eviction) or `offload` (offloading); the dispatcher splits a new sub-task based on the type |
| `context_management.edits[].target` | Edit target, e.g. `session` (entire session) |

### Real Examples of manage_request and edits

manage_request (control-plane, produces no token output):

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

Normal turn (data-plane, produces token output):

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

> 💡 **Note**: `session_id` does not need to be listed in `input_columns`. `GenInferencer.get_data_list()` automatically extracts it from `agent_hint`. The `agent_hint` itself is passed through to the vLLM request body as an extra field (when `pass_agent_hint=True`).

## Quick Start

### Command Example

Running a session-mode performance benchmark against a vLLM `v1/chat/completions` streaming endpoint:

```
ais_bench --models vllm_api_general_stream_session --datasets custom_agent_session_gen --debug -m perf
```

Where:
- `--models` specifies the model task `vllm_api_general_stream_session` (adds session control parameters on top of `vllm_api_general_stream_session.py`)
- `--datasets` specifies the dataset task `custom_agent_session_gen` (custom agent session dataset)

### Prerequisites

- **Inference service**: start a vLLM service supporting `v1/chat/completions`; see [vLLM OpenAI-Compatible Server](https://docs.vllm.com/en/latest/getting_started/quickstart.html#openai-compatible-server)
- **Dataset**: prepare a JSONL dataset conforming to the format above, placed at the path indicated by the `path` field in `custom_agent_session_gen.py`

### Configuration File Modifications

Use `--search` to locate config file paths:

```shell
ais_bench --models vllm_api_general_stream_session --datasets custom_agent_session_gen --mode perf --search
```

Two key config files need modification:

#### 1. Model config `vllm_api_general_stream_session.py`

```python
from ais_bench.benchmark.models import VLLMCustomAPI

models = [
    dict(
        attr="service",
        type=VLLMCustomAPI,
        abbr="vllm-api-general-stream-session",
        path="/mnt/weight/Qwen3-8B",       # model weights path
        model="qwen",                       # model name loaded on the server
        stream=True,
        use_chat_api=True,                  # send to v1/chat/completions
        session_aware=True,                 # enable session mode
        session_concurrency=4,              # max concurrently active sessions
        pass_agent_hint=True,               # merge agent_hint into vLLM request body
        batch_size=4,                       # degenerates to janus queue capacity in session mode
        request_rate=0,
        host_ip="localhost",
        host_port=31036,                    # vLLM service port
        max_out_len=256,
        generation_kwargs=dict(temperature=0, ignore_eos=False),
    )
]
```

**Session-mode-specific parameters**:

| Parameter | Default | Description |
|---|---|---|
| `session_aware` | `False` | Enable session-aware concurrency; when `True`, scheduling is session-level |
| `session_concurrency` | `1` | Max number of concurrently active sessions (cross-session in-flight total) |
| `pass_agent_hint` | `False` | `False`: agent_hint is only used internally (extracting session_id/stage), not passed to the API; `True`: agent_hint is merged into the request body as an extra field |
| `batch_size` | `1` | Degenerates to janus queue capacity (producer → dispatcher buffer) in session mode; recommended to set equal to `session_concurrency` |

> ⚠️ **Note**: in session mode, `_get_workers_num()` is forced to return 1 (single process) because session queue state cannot be shared across processes. `runner.max_num_workers` should also be set to 1.

#### 2. Dataset config `custom_agent_session_gen.py`

```python
agent_reader_cfg = dict(
    input_columns=['question'],           # fields used in the prompt template; max_out_len has a separate mechanism
    output_column=None,
)

agent_demo_datasets = [
    dict(
        abbr='session-agent',
        type=CustomDataset,
        path='ais_bench/datasets/agent/example.jsonl',  # change to your JSONL data path
        reader_cfg=agent_reader_cfg,
        infer_cfg=agent_infer_cfg,
        eval_cfg=agent_eval_cfg,
    )
]
```

### Maximum Output Length (max_out_len / max_tokens) Priority

AISBench internally uses the name `max_out_len` to control the maximum number of tokens per generation. It has two enforcement paths:

**Path A: AISBench's own `max_out_len` mechanism**
- Default value comes from the model config `vllm_api_general_stream_session.py` (e.g. 256)
- If the JSONL dataset has a `max_out_len` column AND `input_columns` includes `max_out_len`, the dataset value takes **highest priority**, overriding the model config per-sample
- If a sample's `max_out_len` is 0/None/empty, it falls back to the model config's `max_out_len`
- This value is ultimately written into the vLLM request body as `max_tokens` (the standard vLLM API field name)

**Path B: direct passthrough to the vLLM API's `max_tokens` field**
- The `max_tokens` field in SWE-bench-style data is **not** recognized by AISBench's `get_max_out_len()` (which hard-codes checking for `max_out_len`)
- This field is passed through as an extra field to the vLLM request body, taking effect directly as `max_tokens` on the vLLM side
- In this case the model config's `max_out_len` does not enter the request body (overridden by `max_tokens`)

| Dataset situation | Effective max tokens |
|---|---|
| Has `max_out_len` column AND in `input_columns` | Dataset's `max_out_len` (AISBench mechanism) |
| Has `max_tokens` column (SWE-bench style) | Dataset's `max_tokens` (vLLM API direct) |
| Neither | Model config's `max_out_len` (default 256) |

> 💡 **Recommendation**: if your dataset is SWE-bench-style (field name `max_tokens`), there is no need to declare it in `input_columns` — it is automatically passed through as an extra field. The model config's `max_out_len` is then overridden by `max_tokens` with no extra work.

## Scheduling Behavior Details

### Session-Level Concurrency Control

When `session_aware=True` is enabled, the scheduler switches from "request-level" to "session-level":

1. **Dispatcher real-time routing**: reads the dataset → buckets by `parent_session_id` (fallback `session_id`) into per-session queues → creates `session_worker` tasks
2. **Intra-session serialization**: requests within the same session are sent in order; the next turn is sent only after the previous turn returns
3. **Cross-session parallelism**: at most `session_concurrency` session_workers run simultaneously
4. **stage=end sub-task splitting**: when `agent_hint.stage="end"` is encountered, the current session_worker ends, and the dispatcher prepares to receive the next batch of requests for that session (if any)

### Offload / Evict Splitting

When `agent_hint.context_management.edits` is non-empty (e.g. `edits[].type == "offload"` or `"evict"`), the session has triggered a context edit (partial KV cache eviction from HBM or full-session eviction). The scheduler:

1. The current `session_worker` processes requests up to the edit marker, then exits
2. A new `session_worker` is created to handle remaining requests, associated with the original session via `parent_session_id`
3. `wait_all_tasks()` manually loops to wait for all tasks, because splitting dynamically appends tasks

### Manage Request Exclusion

Requests with `agent_hint.context_management.manage_request == true` are control-plane requests (e.g., KV cache warmup, context compression triggers) that produce no user-visible tokens. These requests:

- Are skipped in `GenInferencer.do_request()` (when `pass_agent_hint=False`), not sent to the inference service
- When `pass_agent_hint=True`, are sent to the service but `output.extra_perf_data["manage_request"]` is marked `True`
- `default_perf.py` summarizer marks manage_request requests as `excluded=True`
- `DefaultPerfMetricCalculator` deducts the excluded count from `data_count`; `Failed = Total - Success` excludes manage_request

## Performance Metrics

Metric semantics in session mode are consistent with regular mode, with these notes:

| Metric | Session-mode semantics |
|---|---|
| Total Requests | Counts only non-manage_request requests (excluded are deducted) |
| Failed Requests | `Total - Success`, excluding manage_request |
| TTFT / TPOT | Set to 0 in non-streaming scenarios (the `if not model_cfg.get("stream", True)` branch in `default_perf.py`) |
| Concurrency | Actual concurrency is bounded by `session_concurrency` |
| Stable Stage | Stable-stage identification algorithm is the same as regular perf, but the ramp-up phase may be longer (sessions need to ramp up to the concurrency limit) |

## Dataset Construction Tools

`ais_bench/datasets/agent/build_agent_dataset.py` converts raw data into session-format JSONL. A single script switches between source types via `-t/--src-dataset-type`; currently supported:

- `default` (default): reads a JSONL file's `question` field, tokenizes it, and groups records by token-prefix trie. Records sharing long prefixes land in the same session. Suitable for prompt workloads where questions share large common prefixes (e.g., variations of the same template, multi-turn refinements of one question).
- `Jiuwen`: reads Jiuwen agent-trajectory JSON files from a directory, groups by `parent_session_id` into big groups and by `session_id` into sub-sessions. Suitable for replaying real multi-turn agent trajectories.

### `default` mode: build from a plain JSONL file

Input is a JSONL file where each line has at least a `question` field. The script buckets records by token prefix, appends records within the same bucket as sequential turns of one session, and appends an `evict` control-plane record at the end of each session.

```shell
python ais_bench/datasets/agent/build_agent_dataset.py \
    -t default \
    -s input_questions.jsonl \
    -o dataset.jsonl \
    --tokenizer /path/to/tokenizer.json \
    -r 0.5
```

Main parameters (`default`-specific):
- `-s/--src`: input JSONL file path (one record per line, must contain a `question` field)
- `-o/--output`: output JSONL path
- `--tokenizer`: **required**, HuggingFace `tokenizer.json` file used for tokenization and prefix grouping
- `-r/--hit-rate`: threshold ∈ [0.0, 1.0] for starting a new session. A record whose hit rate against the current session trie is ≤ this value starts a new session. Default `0.0` (only 0% hit breaks)

Input example (`input_questions.jsonl`):

```jsonl
{"question": "Please summarize the following article: <article A...>"}
{"question": "Please summarize the following article: <article B...>"}
{"question": "Please summarize the following article: <article C...>"}
{"question": "Translate the following English text to French: <text...>"}
```

Output example (`dataset.jsonl`; first 3 records share a session, the 4th opens a new session, each session ends with an evict marker):

```jsonl
{"question": "Please summarize the following article: <article A...>", "agent_hint": {"session_id": "sid_00001", "parent_session_id": "", "stage": "start", "cache_control": {}, "context_management": {"manage_request": false, "edits": []}}}
{"question": "Please summarize the following article: <article B...>", "agent_hint": {"session_id": "sid_00001", "parent_session_id": "", "stage": "stage-1", "cache_control": {}, "context_management": {"manage_request": false, "edits": []}}}
{"question": "Please summarize the following article: <article C...>", "agent_hint": {"session_id": "sid_00001", "parent_session_id": "", "stage": "stage-2", "cache_control": {}, "context_management": {"manage_request": false, "edits": []}}}
{"question": "", "agent_hint": {"session_id": "sid_00001", "parent_session_id": "", "stage": "stage-3", "cache_control": {}, "context_management": {"manage_request": true, "edits": [{"type": "evict", "target": "session"}]}}}
{"question": "Translate the following English text to French: <text...>", "agent_hint": {"session_id": "sid_00002", "parent_session_id": "", "stage": "start", "cache_control": {}, "context_management": {"manage_request": false, "edits": []}}}
{"question": "", "agent_hint": {"session_id": "sid_00002", "parent_session_id": "", "stage": "stage-1", "cache_control": {}, "context_management": {"manage_request": true, "edits": [{"type": "evict", "target": "session"}]}}}
```

### `Jiuwen` mode: build from real agent trajectories

```shell
python ais_bench/datasets/agent/build_agent_dataset.py \
    -t Jiuwen --src swe_trajectories_dir \
    --output dataset.jsonl \
    --cut-count 4 --session-count 6 --evict
```

Main parameters (`Jiuwen`-specific):
- `-s/--src`: input directory containing Jiuwen agent-trajectory JSON files
- `-c/--cut-count`: keep only the last N records of each session
- `-n/--session-count`: keep only the first N sessions (in first-seen order)
- `-p/--add-pause <fraction>`: insert an offload + prefetch pause pair at a random position inside the given fraction of sessions (lowercase p)
- `-P/--preview`: preview mode — replace question text with `"preview text"` and print the ParentGroup/SessionGroup structure (uppercase P)
- `--show-agent-hint`: when used with `--preview`, also print the full `agent_hint` JSON for each record
- `--evict`: use `group_by_parent_sessions_evict` instead of the default logic — interrupted sub-sessions get an offload marker at the end; non-interrupted sub-sessions get a paired evict + end record; the last sub-session of a big group gets only an evict
- `--max-token-len`: drop records whose `question` token count exceeds this value (requires `--tokenizer`)
- `--copy n`: replicate each big group (by `parent_session_id`) n times; copies get an 8-char random suffix on `session_id`/`parent_session_id` to distinguish them

### `--swe-format` output format

Both `default` and `Jiuwen` support the `--swe-format` flag, which replaces the flat `question` field of data-plane records with SWE-bench-style `messages` + `tools` + `tool_choice` (and leaves `question` empty). Control-plane records (evict/offload/end) also carry placeholder `messages`/`tools` to keep the shape uniform. Use this when targeting a `v1/chat/completions` endpoint:

```shell
python ais_bench/datasets/agent/build_agent_dataset.py \
    -t Jiuwen --src swe_trajectories_dir \
    --output dataset.jsonl --swe-format
```

> ⚠️ **Note**: when `--swe-format` is enabled, the `question` field is always an empty string, so `--max-token-len` is ineffective in this mode (the filter logic checks the `question` field length).

## FAQ

### Q: Can requests within the same session arrive out of order?
No. The dispatcher buckets by `parent_session_id` with strict FIFO per-session queue, and the `session_worker` sends in order.

### Q: Do different sessions compete for GPU resources?
Yes. Up to `session_concurrency` sessions are active simultaneously and share GPU resources. Tune `session_concurrency` based on GPU memory and KV cache capacity to avoid OOM.

### Q: Does `pass_agent_hint=True` affect inference service performance?
It slightly increases request body size (the `agent_hint` field is added), but vLLM ignores unrecognized fields, so inference itself is unaffected. If the inference service implements `agent_hint` handling (e.g., prefix cache hits, context management), performance can improve significantly.

### Q: Do manage_request requests affect TTFT/TPOT statistics?
No. Manage_request requests are marked `excluded=True` and deducted from `data_count`; they do not participate in any token/latency metric calculations.

### Q: How do I debug session scheduling behavior?
Add the `--debug` flag. The dispatcher will print logs for each session's request allocation, stage=end splitting, offload splits, etc.
