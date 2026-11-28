# Cancellation and uncertain outcomes

Stage result publication and command admission share the runtime lock. If cancel
is accepted before deployment.ready/evaluation.completed is published, the late
success takes the owned executor cleanup path before publishing cancellation.
Cleanup runs outside the runtime lock while the original operation remains active;
status/watch clients remain usable and new stage admission remains blocked.
If the success boundary wins the lock first, cancellation sees the idle service
and uses the existing cancel_idle path. External services are never released.

An unexpected adapter exception preserves its full traceback under logs and emits
an evidence reference. Failed/cancelled stages with UNVERIFIED cleanup enter
RECOVERY_REQUIRED; they cannot silently resume work or reuse unknown resources.

After an exclusive runtime restart, a journal acceptance without terminal evidence
produces one persisted runtime.recovery_required boundary. Original request identity
is retained and never re-executed. Repeated restarts do not append repeated recovery
boundaries. Clients reconnect using their prior cursor to observe the uncertainty.
This is the first-version conservative recovery contract, not automatic remediation
of containers or process groups whose state has not been attested.

After owned containers are confirmed absent, release checks each frozen NPU
with bounded fuser and each TCP port by exclusive bind. Errors, occupancy or
timeout retain host claims. Per-resource evidence is persisted before unlocking.
These observations do not reserve against non-v2 users; new v2 deployments
still claim and independently check occupancy before creating containers.
