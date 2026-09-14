# AISBench core llm_io replay task

`LLMIOReplayTask` is a custom `BaseTask` following the same execution pattern
as `OneIGEvalTask`: `LocalRunner` starts the task, `run()` owns log loading and
HTTP execution, and the task writes its result under the AISBench work dir.

Each source log line is sent as one complete chat-completions request. Saved
message histories are never split into separate turns.

## Run

From the AISBench repository:

```bash
export AISBENCH_REPLAY_LOG_FILE=/path/to/s9fkf_input_14-25round_150users_2500_redacted.txt
export AISBENCH_REPLAY_URL=http://172.27.13.87:8900/v1/chat/completions
export AISBENCH_REPLAY_MODEL=glm51
export AISBENCH_REPLAY_CONCURRENCY=100
export AISBENCH_REPLAY_REQUESTS=2500
export AISBENCH_REPLAY_TIMEOUT=900

ais_bench ais_bench/configs/performance_benchmark/llm_io_replay_glm51.py \
  --mode infer
```

The intended runtime is Linux. Paths in `AISBENCH_REPLAY_LOG_FILE` may be
absolute Linux paths or paths relative to the AISBench repository.

Set `AISBENCH_REPLAY_REQUESTS=2300` for the 79dw4 file. A value of `0` sends
every loaded record once. If the requested count exceeds the number of loaded
records, records are cycled in source order, matching `glm51_replay_v3.py`.

The task writes JSON and Markdown reports under:

```text
outputs/llm_io_replay/predictions/<model-abbr>/
```

## Compatibility details

- Preserves `messages`, `tools`, `tool_choice`, and `max_tokens`.
- Preserves message `tool_calls`, `tool_call_id`, and `name`.
- Drops `reasoning_content`, `tool_stream`, and `reasoning_effort` from the
  outgoing request, matching the source script.
- Adds the script-start timestamp prefix to message text by default.
- Uses robust SSE event framing rather than TCP-chunk framing.
- Repairs the supplied redacted JSON by default. Placeholders inside strings
  remain unchanged; damaged numeric values outside strings become zero.
- The source script computes an HMAC but discards it. The task therefore sends
  `Authorization: <x_app_id>` and accepts `x_app_key` only for compatibility.

Use `--mode infer`, not `--mode perf`: this custom task generates its own
performance summary and does not use the tokenizer-dependent OpenICL
performance summarizer.
