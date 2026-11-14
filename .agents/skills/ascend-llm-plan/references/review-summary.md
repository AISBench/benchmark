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

## 本地请求精确字段
freeze-inputs的request为plan、bindings、review（三者为JSON文件路径）、destination（新的冻结目录）、runtime_sha256（64位摘要），不得额外增加字段。verify-inputs为directory、sha256、runtime_sha256。draft-review字段见上文。请求字段正确仍可能因计划、绑定或审阅内容不满足Schema或语义约束而失败，须依据error.message定位，不把所有INVALID_INPUT都解释为请求字段错误。
模型实例的checks必须有health、prefix_cache_reset、metrics三个HTTP路径；proxy只需health。这些是检查/指标/缓存接口的路径元数据，不授予清缓存权限；本轮reset_cache=false时不得调用缓存清理接口。freeze会验证完整模型launch与这些检查元数据。实际路径应从所选框架公开接口确认；vLLM通常为/health、/reset_prefix_cache、/metrics。
