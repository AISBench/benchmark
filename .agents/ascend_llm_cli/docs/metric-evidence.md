# Independent prefix metric evidence

The v2 metric adapter preserves baseline counter semantics from commit
74a7feffb76c32b23658d37ea681784b3a9b2474, scripts/metric_evidence.py, without importing
the stable module. Unlike that baseline reader, HTTP response bodies are bounded
to 64 KiB and use the remaining case deadline. Failed/oversized captures are
explicitly UNVERIFIED and cannot become zero counters.

External plans may supply service.metrics with reviewed id/role/url and optional
node/dp. Owned service handles carry the compiled per-instance metric bindings.
No metric endpoint is inferred for external services. Capture runs immediately
before and after the native CLI, so the window includes native warmups.

Raw Prometheus files are fsynced and hashed before their capture report is written.
Summarization verifies hashes, identical instance/series coverage, nonnegative
counter deltas and hits not exceeding queries. It aggregates prefill/mixed
instances only. Changed, missing, duplicate, NaN or reset series invalidate the
prefix report. The returned case result locates the report with host plus path.

Verified prefix deltas alone do not establish steady-state throughput. Routing
coverage is verified only for explicit single-DP instances with positive query
deltas on every required prefill/mixed target. Multiple DP without identifiable
series remains UNVERIFIED. The native CLI window combines warmup and formal load:
nonzero warmups therefore have UNVERIFIED warmup-route coverage; zero warmups are
NOT_REQUESTED. These are evidence limits, never a reason to alter the frozen load.
Overall validity also requires native performance and audited prefix dataset
evidence. Local tests inject metric responses; no real cache measurements exist.
