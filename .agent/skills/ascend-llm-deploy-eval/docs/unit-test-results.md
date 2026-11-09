# 当前单元测试验证记录

由 tests/render_unit_test_report.py 根据原始 JSON 报告生成；不要手改计数。

测试内容 SHA256：`3a2a8778588292c6f2c038c892467181e428ab1ec7df0ecc4b85e4a37cd57d0a`。文本换行统一为 LF 后计算，避免 Git 的 CRLF/LF 转换产生伪差异。摘要覆盖 scripts、tests、schemas、examples 的 Python/JSON 文件及 requirements.txt；不含本文和其他说明文档。

| UTC 时间 | 系统 | Python | 总数 | 通过 | 失败 | 错误 | 跳过 | 结果 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-11-09T11:46:33.756708+00:00 | Windows | 3.11.4 | 217 | 206 | 0 | 0 | 11 | PASS |
| 2026-11-09T11:45:19.443553+00:00 | Linux | 3.11.6 | 217 | 217 | 0 | 0 | 0 | PASS |

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

### Linux

`Linux-6.6.0-159.4.16.170.oe2403sp4.aarch64-aarch64-with-glibc2.38`

无跳过。


离线测试不启动真实容器或模型，不证明真实硬件性能；本轮为测试基础设施和文档修改，无需重新执行实机部署。
