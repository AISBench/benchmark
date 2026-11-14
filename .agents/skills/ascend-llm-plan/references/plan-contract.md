# Plan and bindings contract

The accompanying schemas define deploy-plan, deploy-bindings, external-eval-plan and eval-bindings. Review and freezing perform schema and semantic validation, compile topology and scripts, and bind all input and runtime digests. These local checks do not prove host readiness.

Owned deploy plans may contain evaluation {cases, test_binding}; deploy bindings then include testers. Each tester binds host, machine_id, work_root, python, aisbench, model_path and datasets. Native synthetic datasets use type=aisbench_synthetic. Explicit load values are preserved; defaults must be resolved before review.

Optional tester.environment is a map of explicit environment variable strings applied to preflight, AISBench and postflight subprocesses. No shell expansion occurs. Freeze actual absolute paths, not $VARIABLE expressions. Discover required CANN environment from the selected installed toolkit using read-only checks; validate imports under those values before deployment. Do not install or upgrade a user's environment to resolve missing libraries. Record chosen toolkit, environment values and import evidence. Never infer that an executable's presence means its imports work. Environment values must not contain secrets, since frozen inputs and evidence retain them.

Production preflight checks interpreter and console entrypoint ownership, tokenizer assets, config imports and dataset capability. Explicit authorized cache reset requires frozen per-instance targets; never infer these from the service URL. Execution and evidence/performance validity remain separate outcomes.

## 容器内测试工具
AISBench 可位于宿主或已有运行容器。宿主未找到工具时，继续只读检查已有容器的工具、解释器、依赖、tokenizer和网络；不要要求用户重复提供可自行发现的绑定。
容器 tester 在原有 host/work_root/machine_id 字段之外设置 container={"id":"完整64位容器ID","image":"sha256:完整镜像摘要"}；python、aisbench、model_path、environment、datasets 路径均属于容器内环境。host_python（默认 /usr/bin/python3）和 control.json 的 python 属于宿主，用于框架，不要求容器安装框架依赖。work_root 是已授权宿主工作根，执行器在容器同路径下建立独立操作目录，通过公开CLI发布工具代码与收回原生产物，不要求已有容器挂载该目录。
冻结前 docker inspect 核对完整ID、镜像、运行状态；只读检查容器内Python、AISBench版本及依赖、权重可见性、被测endpoint从容器可达。工具存在不代表依赖完整。使用公开CLI执行，不手写docker exec负载封装。容器仅提供测试环境，不授予停止、重启、删除容器或被测服务的权限。执行器只停止本任务工具进程组；证据回收失败或清理未知必须报告，不能标记完整。

## 服务健康路径
模型和代理健康路由必须分别从本轮参考或实际接口确认，不统一套用/health。deploy实例checks.health绑定各自路径；service.json携带入口health_path；部署后的eval沿用该路径。独立prepare-inputs(stage=eval)允许health_path字段，与endpoint同源；默认/health只适用于已确认提供该路由的服务。用户提供service对象时沿用其health_path。仅设置健康路径不授予服务管理权限。AISBench预检也从容器实际测试环境访问同一路由。
