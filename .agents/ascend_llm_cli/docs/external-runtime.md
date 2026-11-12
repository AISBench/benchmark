# External service runtime integration

`ExternalExecutor(root, frozen, bundle_digest, runtime_digest, verified_host)`
loads and verifies the input bundle. The launcher must supply a verified host
identity; a string argument alone is not attestation. Production server wiring
remains disabled until that launcher/preflight exists.

Runtime writes `task.initialized` with bundle digest and external service identity
before accepting commands. This internal event is an implementation addition to
design 0.3's non-exhaustive event list; `EXTERNAL_READY` means registered for
evaluation, never health-checked. It increments the decision version, so clients
must read status before their first eval. No health/deployment evidence is invented.

The executor rechecks frozen inputs before admission and before execution.
Deploy/release are forbidden for this service. Unsupported cache reset and unknown
cases are rejected before command.accepted. Successful result evidence is fsynced
and renamed before its digest/reference is published. Failures preserve traceback
in a file and report external_service_action=NONE. Cleanup currently remains
UNVERIFIED when full client-group evidence is unavailable.

Tests inject a runner to exercise API → runtime → executor → result evidence and
recovery without a server. This is not CLI/SSH/socket integration or AISBench
acceptance. The production CLI server remains explicitly gated off.
