# Event delivery and budgets

Journal events remain the source for recovery (16 KiB per internal event). Agent
responses project large details into results/events/SEQ.json with an artifact_ref
and details_externalized flag; identity/type/time/status remain on the event.
Decision fields including service/case identities, execution/performance/cleanup
status, error codes and evidence references remain visible. Oversized essential
fields fail with a protocol error rather than moving the cursor past the event.

Final event pages include metadata before checking the 8 KiB serialized UTF-8
budget. At most ten boundaries and 1 KiB total summary/error prose are delivered
per page. Unknown event types remain boundaries; the Agent must not infer actions
from them. command.accepted and phase.changed do not end watches and can advance
a safe cursor only until the next undelivered boundary. After loss of a response,
the original cursor can replay the same identities.

Ordinary WAIT_TIMEOUT carries no history/configuration. Submit uses the same watch
mechanism and converts timeout to RUNNING with operation identity. Tool output
does not stream progress into model context. Internal producer events exceeding
16 KiB must already reference a separate artifact rather than embedding raw logs.
