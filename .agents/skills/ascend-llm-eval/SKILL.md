---
name: ascend-llm-eval
description: Deploy and Eval v2 的服务测试规划、结果证据分析与阶段交接；支持已有产物离线分析、本地/远端测试端及既有容器内 AISBench。
---

# Eval

依赖 deploy-eval-v2 CLI 0.1 与协议 1.0，先执行 `deploy-eval-v2 --version` 核对。共享 CLI 契约随 ascend_llm_cli 包提供；缺失依赖时明确报告，不切换到旧版。自有服务支持绑定远端测试机，需发布冻结包并核对 machine_id、独立 Python 进程能力 和主节点到测试机的 SSH；外部服务的 runtime 在测试机运行。远端停止未核验时保留服务和资源声明，进入 RECOVERY_REQUIRED。显式缓存重置须有冻结权限及逐实例目标；全部实例核验空闲后执行，结果不明不重试、不启动负载。原生汇总及稳态窗口会按证据判定；缺失时间戳、未知汇总格式或前缀路由覆盖不完整时保持 UNVERIFIED，不将请求成功写成性能有效。

独立入口接受 endpoint+model 或部署产出的 service.json，加 cases 与 tester JSON；通过 plan 的 prepare-inputs 准备新评测任务，并重新健康检查，不继承部署任务释放权限。执行输入为 service handle 和一个冻结 case；不依赖 deploy 私有 dispatch.json。外部端点通过 external_eval 计划登记，不要求先 deploy，也不自动获得清缓存、重启、停止或释放权限。不用 fake-service 冒充外部服务。

目标执行顺序：重新检查服务健康 → AISBench/数据核验 → 显式获授权的缓存操作 → 预热 → 正式测试 → 自有客户端清理 → 原生结果及指标证据核验。每次使用阶段剩余时间，不能把多次调用各自的完整超时累加突破 deadline。

保留用户显式负载；默认普通请求数为并发×4、稳态为×10。不得为演示减少请求数、改变端口或丢弃未知参数。记录合成长度及前缀偏差，不自动判失败，不把目标前缀比例写成缓存命中率。

分析时分别给出 execution_status、evidence_status、performance_validity。检查 AISBench 退出码、预期模型×数据集结果组、请求数量/成功状态、worker 日志和稳态窗口。请求完成但没有有效稳态窗口可为 SUCCEEDED + UNVERIFIED；未知指标不填零。

evaluation.completed 仅表示阶段结束。按结果决定下一 case 或收尾，不在子技能内递归调用 deploy。测试失败先核验自有客户端清理，服务处理遵循冻结策略；外部服务始终不随测试失败释放。交付结果引用、实际负载、证据状态和限制，不粘贴完整日志。

## 本阶段参考与交付

依赖 CLI >=0.1.0a7、协议 1.0。[测试环境预检](references/benchmark-preflight.md)用于解释环境门控；远端测试器见 [远端测试](references/remote-tester.md)，结果核验见 [结果契约](references/result-validation.md)和[原生证据](references/native-result-evidence.md)，仅清缓存任务读取 [缓存重置](references/cache-reset.md)。共享请求字段见已安装 ascend-llm-router 的 references/cli-contract.md。

从 get-status 获取当前版本，使用冻结 service_id/case_id 提交 eval；RUNNING 时按 next_cursor 等待。保持 tester.environment 一致，进程使用直接 Python 启动。交付结果引用、实际负载、三项状态与清理证据；外部服务不交给 release 停止。缺少性能证据时保留 UNVERIFIED。

独立 external_eval 的框架收尾：先核验 evaluation.completed、原生结果和被测服务仍健康，再确认 get-status 无活动 operation。当前CLI不允许直接停止EVALUATED runtime；如无需继续case，使用原评测task_ref和当前expected_version提交 `cancel`（scope=task，新的request_id），只终结本评测会话。external_eval契约不具备部署或服务释放权，此cancel不会停止被测服务。等 operation.cancelled 且 cleanup_status=VERIFIED 后，用原启动nonce调用stop-runtime；再核验外部服务仍健康，补收最终task.jsonl。此流程不是release服务，不需要也不得取得服务部署归属身份。若清理未知或仍有活动操作，报告并按原身份查询，不以手工kill或换任务掩盖未知状态。保留evaluation.completed成功结果；随后task会话终结的CANCELLED不等于性能请求被取消。

## 容器内测试工具
AISBench 可位于宿主或已有运行容器。宿主未找到工具时，继续只读检查已有容器的工具、解释器、依赖、tokenizer和网络；不要要求用户重复提供可自行发现的绑定。
容器 tester 在原有 host/work_root/machine_id 字段之外设置 container={"id":"完整64位容器ID","image":"sha256:完整镜像摘要"}；python、aisbench、model_path、environment、datasets 路径均属于容器内环境。host_python（默认 /usr/bin/python3）和 control.json 的 python 属于宿主，用于框架，不要求容器安装框架依赖。work_root 是已授权宿主工作根，执行器在容器同路径下建立独立操作目录，通过公开CLI发布工具代码与收回原生产物，不要求已有容器挂载该目录。
冻结前 docker inspect 核对完整ID、镜像、运行状态；只读检查容器内Python、AISBench版本及依赖、权重可见性、被测endpoint从容器可达。工具存在不代表依赖完整。使用公开CLI执行，不手写docker exec负载封装。容器仅提供测试环境，不授予停止、重启、删除容器或被测服务的权限。执行器只停止本任务工具进程组；证据回收失败或清理未知必须报告，不能标记完整。

## 服务健康路径
模型和代理健康路由必须分别从本轮参考或实际接口确认，不统一套用/health。deploy实例checks.health绑定各自路径；service.json携带入口health_path；部署后的eval沿用该路径。独立prepare-inputs(stage=eval)允许health_path字段，与endpoint同源；默认/health只适用于已确认提供该路由的服务。用户提供service对象时沿用其health_path。仅设置健康路径不授予服务管理权限。AISBench预检也从容器实际测试环境访问同一路由。

## 有效性与日志归集
普通原生性能case没有shared_prefix_ratio时，性能有效性由原生请求记录、负载和性能汇总证据决定；附加prefix指标保留各自验证状态，不作为普通case的必需门槛。有shared_prefix_ratio时仍须核验数据、前缀指标、路由和预热覆盖，不因普通case规则放宽。原生合成input_tokens是工具配置目标；实际重新分词长度另列synthetic-evidence，不把目标长度冒充实测长度，不自行改造合成数据。

本地tester（含同宿主容器tester）的case目录为results/<eval operation_id>/benchmark，原生AISBench控制台日志是该目录/aisbench.log；原生summary/details/worker日志由该目录/validated.json和performance-evidence.json中的引用定位，逐一归集。不要误读logs/aisbench.log。preflight.log、postflight.log、container.log是框架诊断来源标签，内容已流入唯一task.jsonl，并非必须存在的独立文件；JSON证据preflight.json、synthetic-evidence.json等仍在case目录。服务原生日志为logs/<instance>.service.log。远端tester使用响应的remote_evidence定位，不能把主机路径猜为本地路径。

所有read-result分页读到total_bytes，保存原始JSON/退出码并核对连续偏移。停止runtime后可按当前CLI的只读产物接口补收最终task.jsonl。缺少实际要求的AISBench控制台或原生日志时标PARTIAL；不把按设计合并的框架日志标签误当缺失，也不把尚未回收的远端文件标为已归集。
