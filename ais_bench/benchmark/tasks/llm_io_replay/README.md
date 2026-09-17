# AISBench core llm_io replay task

`LLMIOReplayTask` is a custom `BaseTask` following the same execution pattern
as `OneIGEvalTask`: `LocalRunner` starts the task, `run()` owns log loading and
HTTP execution, and the task writes its result under the AISBench work dir.

Each source log line is sent as one complete chat-completions request. Saved
message histories are never split into separate turns.

## Run

All values can be set directly in
`ais_bench/configs/performance_benchmark/llm_io_replay_glm51.py`. For example,
the supplied config already contains `input_log_file`, `url`, `model`,
`x_app_id`, `x_app_key`, `concurrent`, `requests`, and `timeout`. After editing
that file, run from the AISBench repository without any exports:

```bash
ais_bench ais_bench/configs/performance_benchmark/llm_io_replay_glm51.py \
  --mode infer
```

Alternatively, leave the config reusable and override one or more values on
the CLI:

```bash
ais_bench ais_bench/configs/performance_benchmark/llm_io_replay_glm51.py \
  --mode infer \
  --replay-log-file /data/s9fkf_input_14-25round_150users_2500_redacted_fix.txt \
  --replay-url http://172.27.13.87:8900/v1/chat/completions \
  --replay-model glm51 \
  --replay-x-app-id 1111 \
  --replay-x-app-key 22222 \
  --replay-concurrency 100 \
  --replay-requests 2500 \
  --replay-timeout 900
```

CLI values have precedence over the config file. The old
`AISBENCH_REPLAY_*` environment variables remain as a compatibility fallback
only when the matching config field is absent or `None`. Because command-line
arguments may be visible in process listings, keep real credentials in a
protected config file when that matters.

The intended runtime is Linux. Replay log paths may be absolute Linux paths or
paths relative to the AISBench repository.

Set `requests=2300` in the config or pass `--replay-requests 2300` for the
79dw4 file. A value of `0` sends every loaded record once. If the requested
count exceeds the number of loaded records, records are cycled in source order,
matching `glm51_replay_v3.py`.

The task writes JSON and Markdown reports under:

```text
outputs/llm_io_replay/predictions/<model-abbr>/
```

At the end of inference, the main CLI process also prints AISBench-style
`fancy_grid` tables for request latency (E2EL), TTFT, derived TPOT, per-request
TPS, typing speed, prefill throughput, token distributions, request-body size,
common throughput/concurrency metrics, and error groups. The Markdown report
adds configuration, token-source, cache-hit, success-detail, and failure-detail
sections. The JSON report retains every summary value and per-request field.

ITL is intentionally omitted because the replay endpoint does not return a
timestamp for every generated token. TPOT is derived from generation time and
the output-token count.

## Compatibility details

- Preserves `messages`, `tools`, `tool_choice`, and `max_tokens`.
- Preserves message `tool_calls`, `tool_call_id`, and `name`.
- Drops `reasoning_content`, `tool_stream`, and `reasoning_effort` from the
  outgoing request, matching the source script.
- Adds the script-start timestamp prefix to message text by default.
- Uses robust SSE event framing rather than TCP-chunk framing.
- Reads the validated `_fix.txt` files directly by default. Runtime repair can
  still be enabled explicitly for legacy damaged inputs.
- The source script computes an HMAC but discards it. The task therefore sends
  `Authorization: <x_app_id>` and accepts `x_app_key` only for compatibility.

Use `--mode infer`, not `--mode perf`: this custom task generates its own
performance summary and does not use the tokenizer-dependent OpenICL
performance summarizer.
