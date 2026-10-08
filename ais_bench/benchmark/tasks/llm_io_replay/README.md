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
`x_app_id`, `x_app_key`, `concurrent`, `requests`, `timeout`, `max_tokens`, and
`ignore_eos`. After editing that file, run from the AISBench repository without
any exports:

```bash
ais_bench ais_bench/configs/performance_benchmark/llm_io_replay_glm51.py \
  --mode perf
```

Do not append `--datasets` to this command. AISBench treats a complete custom
configuration file and the `--models`/`--datasets` selectors as two alternative
entry points; selectors are ignored when a custom configuration file is
provided. The replay dataset is already declared by the config's `datasets`
list. The old `AISBENCH_REPLAY_*` environment variables remain only as a
compatibility fallback when the matching config field is absent or `None`.

`max_tokens` and `ignore_eos` can also be set under the replay model in the
Python config. An explicit config value overrides every source log payload;
leaving it as `None` preserves the payload value (or omits the field when the
payload does not contain it). Set `ignore_eos=False` in the config to
explicitly send `false` to vLLM.

The intended runtime is Linux. Replay log paths may be absolute Linux paths or
paths relative to the AISBench repository.

Set `requests=2300` in the config for the 79dw4 file. A value of `0` sends every
loaded record once. If the requested count exceeds the number of loaded
records, records are cycled in source order, matching `glm51_replay_v3.py`.

The task writes raw JSON and Markdown reports under:

```text
outputs/llm_io_replay/predictions/<model-abbr>/
```

In `--mode perf`, the replay summarizer also writes the complete report,
request details, and Markdown summary under:

```text
outputs/llm_io_replay/performances/<model-abbr>/
```

During the `PerfViz` stage, the main CLI process prints AISBench-style
`fancy_grid` tables for request latency (E2EL), TTFT, derived TPOT, per-request
TPS, typing speed, prefill throughput, token distributions, request-body size,
common throughput/concurrency metrics, and error groups. The Markdown report
adds configuration, token-source, cache-hit, success-detail, and failure-detail
sections. The JSON report retains every summary value and per-request field.

ITL is intentionally omitted because the replay endpoint does not return a
timestamp for every generated token. TPOT is derived from generation time and
the output-token count.

## Compatibility details

- Preserves `messages`, `tools`, `tool_choice`, `max_tokens`, and `ignore_eos`
  unless an explicit replay override is configured.
- Preserves message `tool_calls`, `tool_call_id`, and `name`.
- Drops `reasoning_content`, `tool_stream`, and `reasoning_effort` from the
  outgoing request, matching the source script.
- Adds the script-start timestamp prefix to message text by default.
- Uses robust SSE event framing rather than TCP-chunk framing.
- Reads the validated `_fix.txt` files directly by default. Runtime repair can
  still be enabled explicitly for legacy damaged inputs.
- The source script computes an HMAC but discards it. The task therefore sends
  `Authorization: <x_app_id>` and accepts `x_app_key` only for compatibility.

The config selects `LLMIOReplayPerfSummarizer`, so `--mode perf` uses the
standard AISBench `Infer -> PerfViz` workflow without invoking the
tokenizer-dependent OpenICL performance summarizer.
