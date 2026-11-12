# Event and response envelope validation

event.schema.json defines required identity, sequence/version, UTC timestamp,
operation, type, status and object data. Journal append validates before any write;
replay maps invalid records to RECOVERY_REQUIRED. Unknown event names remain legal
for forward compatibility, while runtime policy does not transition on them.

response.schema.json includes the event schema and validates final stage/query
envelopes. The disposable client also checks per-page task identity, strictly
increasing delivered sequences, sequences not beyond next_cursor, at most ten
events, error shape and empty WAIT_TIMEOUT history. Serialized UTF-8 response and
aggregate summary limits are enforced. This is distinct from the journal's larger
internal event size budget.

Known-event payload semantics and full service/result schemas remain additional
work; structural envelope validation alone is not evidence of deployment health,
successful cleanup or valid benchmark performance. Publication/startup bootstrap
responses have their own bounded contracts and do not masquerade as stage events.
