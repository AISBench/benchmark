# Result identity, integrity and performance evidence

Evaluation results pass result.schema.json before publication. The coordinator
binds task/operation/service/case identity, rejecting conflicting adapter identities.
Execution, evidence and performance enums are independent; VERIFIED performance
requires COMPLETE evidence. Native request success alone cannot assert this.

Use the event's result_sha256 when reading a stage result:

    deploy-eval-v2 --config control.json read-result --task-ref TASK --artifact-ref results/OP/stage-result.json --expected-sha256 DIGEST

Verification is optional for ordinary logs and explicit for result artifacts up to
4 MiB. It checks the entire file before returning a byte slice; continuation keeps
the same digest. Corruption returns RECOVERY_REQUIRED. verified_sha256 counts toward
the same final response byte budget. Large logs retain the bounded locator workflow.

Native details are streamed, hashed, and validated against request groups/counts.
Input/output token counts are recorded when present; unknown counts remain absent,
and differences from requested synthetic lengths are not automatic failures.
Native summaries are bounded to 1 MiB and checked for stage, units, finite values,
request consistency, planned maximum concurrency and throughput arithmetic.

When raw start_time/end_time seconds are available, timing analysis independently
checks summary duration. Steady analysis uses the second start reaching configured
concurrency through the final event at that concurrency, and checks selected count
and window duration. It also requires maximum full-run request latency below one
third of the stable window. Timing analysis is capped at 100,000 requests; exceeding
that bound does not reject the load or truncate native evidence, but leaves steady
performance UNVERIFIED. Missing timestamps (including versions storing them only
in another format), unsupported summary formats, inconsistent values and short
windows carry explicit reasons. No fabricated fallback metrics are emitted.

The observed format/semantics reference is AISBench/benchmark commit
aff1554354fbf19dccbdf2199cc3c22119f815a2:

- https://github.com/AISBench/benchmark/blob/aff1554354fbf19dccbdf2199cc3c22119f815a2/ais_bench/benchmark/calculators/base_perf_metric_calculator.py
- https://github.com/AISBench/benchmark/blob/aff1554354fbf19dccbdf2199cc3c22119f815a2/ais_bench/benchmark/calculators/stable_perf_metric_calculator.py
- https://github.com/AISBench/benchmark/blob/aff1554354fbf19dccbdf2199cc3c22119f815a2/docs/source_en/advanced_tutorials/stable_stage.md

The implementation is independent; no upstream source is imported at execution.
Summary evidence checks can verify ordinary or steady native request performance.
Configured server prefix metrics must also be verified. A requested shared-prefix
ratio still requires warmup/routing coverage that before/after aggregate counters
cannot prove; such a result remains UNVERIFIED with that scope explicit. This is an
evidence conclusion, not an assertion of zero cache hits or a reason to change load.

Synthetic postflight runs under the bound tester Python with local-only tokenizer
loading. It streams original native text, hashes details and records count,
min/max/mean token length and common prefix without regenerating data. Deviation
is diagnostic only; missing text/tokenizer is UNVERIFIED. The compatibility
reader delegates to this bounded code. External dataset rows are limited to
1 MiB, generation manifests to 16 MiB, within the preflight stage deadline.
