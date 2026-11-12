# Explicit cache reset

Semantics adapted independently from stable cache_reset.py at baseline
74a7feffb76c32b23658d37ea681784b3a9b2474; no stable module is imported or changed.

External plans must provide service.permissions.reset_cache=true and a nonempty
service.reset_targets array of {id,url,metrics_url}. The two URLs must address
the same instance origin. These reviewed targets are not inferred from a proxy.
Owned evaluation derives per-instance targets from frozen topology metrics when
the reviewed case explicitly requests reset_cache=true. Endpoint permission alone
does not cause a reset when the selected case has reset_cache=false.

After health and benchmark preflight, before native warmups, every target must
report finite zero num_requests_running and num_requests_waiting metrics.
Only after all idle checks succeed are reset requests sent, in order. Each must
return HTTP 200 with JSON success=true. A lost response is UNKNOWN_OUTCOME;
it is never automatically retried, later targets and the benchmark do not run.
Cancellation and the shared stage deadline apply to all requests.

cache-reset.json records intent, per-target states and SHA-256 references to raw
idle/response evidence. HTTP success does not assert distributed/external cache
invalidation; the scope is explicitly local_prefix_cache. Live endpoint behavior
and concurrency with workloads outside this task must be validated on the target.
