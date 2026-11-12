---
name: ascend-llm-eval
description: Deploy and Eval v2 的服务测试规划、结果证据分析与阶段交接；支持已有产物离线分析及本地/远端测试端，实机验证待完成。
---

# Eval

依赖 deploy-eval-v2 CLI 0.1 与协议 1.0，先执行 `deploy-eval-v2 --version` 核对。共享 CLI 契约随 ascend_llm_cli 包提供；缺失依赖时明确报告，不切换到旧版。自有服务支持绑定远端测试机，需发布冻结包并核对 machine_id、独立 Python 进程能力 和主节点到测试机的 SSH；外部服务的 runtime 在测试机运行。远端停止未核验时保留服务和资源声明，进入 RECOVERY_REQUIRED。显式缓存重置须有冻结权限及逐实例目标；全部实例核验空闲后执行，结果不明不重试、不启动负载。原生汇总及稳态窗口会按证据判定；缺失时间戳、未知汇总格式或前缀路由覆盖不完整时保持 UNVERIFIED，不将请求成功写成性能有效。

输入为 service handle 和一个冻结 case；不依赖 deploy 私有 dispatch.json。外部端点通过 external_eval 计划登记，不要求先 deploy，也不自动获得清缓存、重启、停止或释放权限。不用 fake-service 冒充外部服务。

目标执行顺序：重新检查服务健康 → AISBench/数据核验 → 显式获授权的缓存操作 → 预热 → 正式测试 → 自有客户端清理 → 原生结果及指标证据核验。每次使用阶段剩余时间，不能把多次调用各自的完整超时累加突破 deadline。

保留用户显式负载；默认普通请求数为并发×4、稳态为×10。不得为演示减少请求数、改变端口或丢弃未知参数。记录合成长度及前缀偏差，不自动判失败，不把目标前缀比例写成缓存命中率。

分析时分别给出 execution_status、evidence_status、performance_validity。检查 AISBench 退出码、预期模型×数据集结果组、请求数量/成功状态、worker 日志和稳态窗口。请求完成但没有有效稳态窗口可为 SUCCEEDED + UNVERIFIED；未知指标不填零。

evaluation.completed 仅表示阶段结束。按结果决定下一 case 或收尾，不在子技能内递归调用 deploy。测试失败先核验自有客户端清理，服务处理遵循冻结策略；外部服务始终不随测试失败释放。交付结果引用、实际负载、证据状态和限制，不粘贴完整日志。

## 本阶段参考与交付

依赖 CLI >=0.1.0a6、协议 1.0。[测试环境预检](references/benchmark-preflight.md)用于解释环境门控；远端测试器见 [远端测试](references/remote-tester.md)，结果核验见 [结果契约](references/result-validation.md)和[原生证据](references/native-result-evidence.md)，仅清缓存任务读取 [缓存重置](references/cache-reset.md)。共享请求字段见已安装 ascend-llm-router 的 references/cli-contract.md。

从 get-status 获取当前版本，使用冻结 service_id/case_id 提交 eval；RUNNING 时按 next_cursor 等待。保持 tester.environment 一致，进程使用直接 Python 启动。交付结果引用、实际负载、三项状态与清理证据；外部服务不交给 release 停止。缺少性能证据时保留 UNVERIFIED。
