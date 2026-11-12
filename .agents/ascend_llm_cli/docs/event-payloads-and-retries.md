# Known event payloads and deployment attempts

event.schema.json now specifies required payloads and status values for all known
boundary types, command acceptance, phase changes and task initialization. Journal
append/replay and response validation enforce these contracts. Command acceptance
also binds operation_id and after_seq; evidence references reject traversal, and
task versions cannot regress. Unknown extension events remain readable and do not
apply lifecycle transitions. Large-event projections retain required decision data.

Failure/cancellation boundaries without an executor-supplied evidence artifact get
a persisted terminal record before the event is written. This records actual
cleanup uncertainty rather than fabricating successful cleanup. Fake stage events
remain explicitly simulated but conform to the same payload shapes.

The reviewed max_retries (0–3) is now supported. Deployment attempts share one
startup deadline and use different durable effect command IDs. Every attempt has
its own report under results/OP/attempts/N; the aggregate stage result references
all attempts. Only confirmed INVALID_INPUT/TimeoutError failures in dependency,
health or smoke phases may retry, and only after VERIFIED cleanup and while time
remains. UNKNOWN_OUTCOME, transport/recovery errors, uncertain resource absence,
and cancellation never cause automatic replay. Retry count is a maximum, not a
requirement to repeat an unsafe or deterministic resource operation.

Tests cover missing evidence, invalid states, command identity/cursor mismatch,
confirmed failed smoke followed by cleanup/new attempt, and unknown response with
known-resource cleanup that still does not retry.
