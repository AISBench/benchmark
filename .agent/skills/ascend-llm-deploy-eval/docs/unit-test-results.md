# 当前单元测试验证记录

由 tests/render_unit_test_report.py 根据原始 JSON 报告生成；不要手改计数。

测试内容 SHA256：`f11b2470ea59d23995765325d89b9c054ff464857c5e04c40b9b11515e98b15a`。文本换行统一为 LF 后计算，避免 Git 的 CRLF/LF 转换产生伪差异。摘要覆盖 scripts、tests、schemas、examples 的 Python/JSON 文件及 requirements.txt；不含本文和其他说明文档。

| UTC 时间 | 系统 | Python | 总数 | 通过 | 失败 | 错误 | 跳过 | 结果 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-11-10T01:53:26.753485+00:00 | Windows | 3.11.4 | 240 | 228 | 0 | 0 | 12 | PASS |

## 平台及跳过原因

### Windows

`Windows-10-10.0.26200-SP0`

- `test_compact_entry.CompactTests.test_distribution_shared_path_reuses_and_changed_bundle_rejected`：Receiver uses Linux flock
- `test_container_dependencies.DependencyTests.test_timeout_terminates_only_worker_process_group`：Linux process groups
- `test_group_worker_process.WorkerProcessTests.test_retired_worker_rejects_execution_and_terminal_evidence_remains_strict`：Detached host worker requires Linux
- `test_lifecycle.LockTests.test_container_lock_excludes_other_process_and_releases`：Linux flock integration
- `test_mount_profiles.MountProfileTests.test_missing_profile_source_never_creates_container`：Requires Linux bash
- `test_overall_review.OverallReviewTests.test_distribution_failures_and_competing_publishers`：Linux receiver publication
- `test_port_preflight.PortCheckTests.test_listener_remains_blocked_with_and_without_reuseaddr`：Linux TCP reuse semantics
- `test_port_preflight.PortCheckTests.test_nonreusable_bound_socket_is_blocked`：Linux TCP reuse semantics
- `test_port_preflight.PortCheckTests.test_time_wait_allows_restart_but_bare_bind_fails`：Linux TCP reuse semantics
- `test_port_recovery.PortRecovery.test_time_wait_does_not_block_redeployment`：Linux server socket semantics
- `test_proxy_gate.GateTests.test_generated_gate_waits_for_both_models_and_times_out`：Bash/curl behavior
- `test_runtime_publication.RuntimePublicationTests.test_receiver_idempotence_conflict_and_symlink_escape`：Linux receiver uses flock

本报告仅记录离线回归，不启动真实容器或模型，也不证明真实硬件性能；是否需要实机复验须按具体修改影响另行评估。
