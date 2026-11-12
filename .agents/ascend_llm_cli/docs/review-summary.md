# Review summary binding

`draft-review --request draft.json` accepts plan, bindings, runtime_sha256 and
destination paths. It creates a new directory containing review-summary.json and
review.json with status=DRAFT, evidence_refs=[] and approved=false in the CLI
response. It does not authorize execution or contact hosts.

Summary material includes task/intent, full input hashes, ownership and cleanup
policy, execution settings, deployment parameters or external service/cases,
site bindings and unresolved items. Deployment material explicitly shows effective
container defaults and which fields were defaulted. Large details stay in the
JSON file; the tool response contains a reference and digest, not the full summary.

review.inputs now requires summary_sha256 alongside plan/bindings/runtime hashes.
After resolving/reviewing inputs, the approval record must contain current hashes,
APPROVED status, no unresolved items and evidence references. Freeze regenerates
the deterministic summary into the bundle. Read/installation verification repeats
that comparison; rehashing an altered summary in manifest.json cannot approve it.

This tightens the unfinished v2 alpha contract. Earlier alpha review records and
bundles lacking summary_sha256/review-summary.json must be regenerated and reviewed;
there is no silent compatibility bypass. Stable-version inputs are unchanged.

Runtime also rebuilds operations/OPERATION_ID.json from journal acceptance and
known terminal events. Missing index files are not evidence that an operation did
not run. Unknown event types remain observable but cannot clear an active operation
or replace its service/state.
