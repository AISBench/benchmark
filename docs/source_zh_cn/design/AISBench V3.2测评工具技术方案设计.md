**状态 (Status):** Draft / Reviewing / Approved / Rejected / Superseded
**作者 (Authors):** @Your_Community
**创建日期 (Created):** 2026-05-09
**更新日期 (Updated):** 2026-05-09
**相关 Issue/PR:** #285,#286,#279

---

# 1. 概述

## 1.1 简介

在 AISBench 中统一适配接入 SWE-Bench、BFCLV4、Tau2-Bench、Terminal-Bench 2，建设覆盖“数据读取 → 推理执行 → 结果判定 → 指标汇总 → 样本审计”的标准化精度测评链路，形成可配置、可复现、可追溯的评测体系。

## 1.2 动机

SWE-Bench 聚焦真实软件工程问题修复，BFCLV4 聚焦函数/工具调用准确性，Tau2-Bench 聚焦对话式 Agent 的工具使用与状态交互，Terminal-Bench 2（terminal-bench-2）在容器化终端环境中评测 Agent 完成真实工作流（安全修复、数据处理、系统调试、模型训练等）。以上均为 Agent 能力评估的主流公开基准。AISBench 当前缺少对上述基准的统一支持，导致评测口径分散、交付成本高、横向对比困难。

## 1.3 目标

通过支持SWE-Bench-Verified、SWE-Bench-Pro、Terminal-Bench、Tau2-Bench、BFCL-V4等数据集，覆盖Agent的coding、function call、tool call等场景。

# 2. 用例分析

通过支持SWE-Bench-Verified、SWE-Bench-Pro、Terminal-Bench、Tau2-Bench、BFCL-V4等数据集，覆盖Agent的coding、function call、tool call等场景。

# 3. 方案设计

## 3.1 总体方案

三套基准差异大（代码修复 vs 函数调用 vs 对话 Agent），在 AISBench 内采用 统一外壳（注册、调度、落盘、报表）+ 数据集专用适配（读取器、执行器、评测器） 的分层设计；三个子模块可独立交付，通过各自验收后整体目标达成。

## 3.2 技术选型
基于开源数据集的实现，适配工具功能。

## 3.3 功能与性能设计

### 3.3.1 SWE-Bench 数据集接入与测评能力建设
背景：以真实 GitHub issue 为样本，依赖代码仓上下文、补丁产出与测试通过率判定。

#### 方案要点：

- 数据：issue 描述、仓库元数据、任务标识等映射到 AISBench 统一任务输入；读取器与字段标准化。
- 执行：infer/eval 配置模板；对接补丁产物与 harness（如 mini-swe-agent）；测试型判定结果采集。
- 评测：聚合 pass 类核心指标；样本级判定与失败原因审计。
##### 验收口径：

- 验收模型：Qwen3-Coder-32B；平台：NPU/GPU。
- 数据集：swebench-lite、swebench-verify、swebench-multi。
- 指标：使用 mini-swe-agent，后处理 temperature=0，每个子数据集 随机固定 10% 子集，计算 pass@5，与 SWE-Bench 官方 测试方法对比，精度差异小于 1%。
sequenceDiagram
  participant Bench as AISBench调度
  participant Data as SWE-Bench数据
  participant Agent as 推理与补丁生成
  participant Harness as 测试Harness
  participant Eval as 评测与审计
  Bench->>Data: 加载issue样本
  Bench->>Agent: infer生成patch
  Agent->>Harness: 提交patch执行测试
  Harness->>Eval: 通过/失败与日志
  Eval->>Bench: 指标与样本审计落盘

#### 使用说明
1. SWE-Bench：配置数据集子集（lite/verified/multi）、mini-swe-agent 参数、temperature=0、10% 固定子集抽样策略（需在配置或文档中固化 seed 与样本列表路径）
2. 通用约束：NPU/GPU 平台；产物目录符合 AISBench 规范；版本信息（数据集 commit、工具链版本）写入运行元数据。

#### 测试设计
##### 单元测试
- 各数据集读取器：关键字段完整率、异常样本报告。
- SWE-Bench：补丁路径与样本 ID 映射；BFCLV4：函数调用解析；Tau2-Bench：多轮消息序列完整性。

##### 集成测试
- 各子模块 infer + eval 一键跑通；日志与结果落盘完整；失败场景错误信息明确。

##### 端到端 / 官方对齐
- SWE-Bench：pass@5 与官方方法差异 < 1%（给定验收模型与子集）。

##### 需求拆分验收点（通过测试体现）
- 数据标准化、执行链路接入、评测器与审计输出三类能力在各子模块内分别覆盖（详见内部需求拆分）。

### 3.3.2：SWE-Bench-Pro 数据集设计
Pro 与常规 SWE-Bench 的差异在 eval：必须拉取镜像、挂载 workspace、执行每样本脚本并由 parser 解析测试结果。AISBench 侧封装 数据标准化 + patch 协议 + 执行后端适配 + 结果 schema。


#### 方案范围：SWE-bench Pro 数据集接入与测评能力建设
Pro 与常规 SWE-Bench 的差异在 eval：必须拉取镜像、挂载 workspace、执行每样本脚本并由 parser 解析测试结果。AISBench 侧封装 数据标准化 + patch 协议 + 执行后端适配 + 结果 schema。

sequenceDiagram
  participant AIS as AISBench
  participant DS as 数据集CSV或JSONL
  participant PJ as patch JSON
  participant Dock as Docker或Modal
  participant TS as run_script与parser
  AIS->>DS: 加载样本元数据
  AIS->>PJ: 读取预测patch
  AIS->>Dock: 启动容器dockerhub_tag
  Dock->>TS: 执行run_script
  TS->>AIS: 测试结果与日志
  AIS->>AIS: FAIL或PASS覆盖判定
  AIS->>AIS: eval_results与审计落盘

##### 数据读取与字段标准化

- 支持 HF ScaleAI/SWE-bench_Pro 或本地导出 CSV/JSONL。
- 至少包含：instance_id；dockerhub_tag（构造 dockerhub_username/sweap-images:{dockerhub_tag}）；before_repo_set_cmd、selected_test_files_to_run；fail_to_pass、pass_to_pass（或 FAIL_TO_PASS/PASS_TO_PASS）；可选 base_commit、repo、base_dockerfile、instance_dockerfile。
字段显式标准化，避免评测脚本隐式列名猜测。

##### Patch 产物协议

与 swe_bench_pro_eval.py 互通：列表 [{instance_id, patch, prefix}]，兼容 model_patch 字段名。
明确空 patch 策略（是否计入 submitted、是否直接失败、审计方式）；prefix 语义（多运行/多模型比较、落盘路径索引）。
对齐 gather_patches.py 等汇聚工具约定。
##### 评测执行链路

后端至少一种：local Docker 或 Modal（推荐）；参数对齐参考脚本：dockerhub_username、scripts_dir（默认 SWE-bench_Pro-os/run_scripts 或可配置）、num_workers、block_network、timeout、失败重试/跳过。
Linux + Docker 环境下固定子集稳定跑通；失败保留 stdout/stderr、patch 快照、entryscript 快照。

##### 指标与审计

汇总：resolved accuracy（可选 submitted accuracy）。
样本级：instance_id → resolved(bool) + 审计路径（stdout/stderr、output.json、patch.diff、entryscript.sh 等）。
可选失败分类：拉镜像失败、脚本缺失、patch apply 失败、测试超时、parser 失败。
##### 最小验收与复现

固定子集策略（排序取前 N 或固定 seed 抽样）；Docker/Modal/数据来源版本；输入输出目录结构；gold patch 验收集。

#### 使用说明
- 前置条件：Linux、Docker（及可选 Modal 账号与配置）；可访问 DockerHub 镜像；磁盘与网络满足镜像拉取。
- 配置项：数据路径、patch JSON 路径、执行后端、dockerhub_username、scripts_dir、并发、超时、网络隔离。
- 输入：符合协议的 patch JSON；样本表与 HF snapshot 版本记录在元数据中。
- 约束：资源消耗高；首次冷启动耗时大；权限不足时需清晰报错与文档指引。

#### 测试设计
##### 单元测试
- CSV/JSONL/HF 字段解析与缺失字段检测。
- Patch 列表 schema 校验与空 patch 分支。

##### 集成测试
- 最小子集 + gold patches：eval 全链路跑通，eval_results.json 结构符合约定。
- 模拟失败：镜像不存在、脚本缺失，验证失败分类与审计字段。

##### 端到端 / 对齐验收
- 同一 patch 输入下，resolved 判定与参考 swe_bench_pro_eval.py 一致。
- 重复运行（同输入同配置）判定一致。


### 3.3.3：BFCLV4 数据集接入与测评能力建设
背景：Berkeley Function Calling Leaderboard V4，评估函数选择、参数构造、格式遵循与多轮/Agent 任务中的工具调用。

#### 方案要点：

- 数据：自然语言请求、函数 schema、调用约束与多类子集字段的标准化映射。
- 执行：函数调用推理入口；输出解析与多轮轨迹采集。
- 评测：BFCLV4 专用评测器与子任务维度指标聚合。
#### 验收口径：

验收模型：Qwen3-235B-A22B-Instruct-2507；平台：NPU/GPU。
temperature=0；单轮对话与 官方 Gorilla README 测试方式精度差异 < 1%；多轮对话与官方 leaderboard 精度差异 < 1%。

#### 使用说明
1. BFCLV4：区分单轮与多轮评测配置；指定模型与温度；对接官方评分脚本或等价实现。
2. 通用约束：NPU/GPU 平台；产物目录符合 AISBench 规范；版本信息（数据集 commit、工具链版本）写入运行元数据。

#### 测试设计
##### 单元测试
- 各数据集读取器：关键字段完整率、异常样本报告。

##### 集成测试
- 各子模块 infer + eval 一键跑通；日志与结果落盘完整；失败场景错误信息明确。

##### 端到端 / 官方对齐
- BFCLV4：单轮、多轮分别 < 1%。

##### 需求拆分验收点（通过测试体现）
- 数据标准化、执行链路接入、评测器与审计输出三类能力在各子模块内分别覆盖（详见内部需求拆分）。


### 3.3.4：Tau2-Bench 数据集接入与测评能力建设
背景：对话式 Agent，强调工具可调用环境中的多轮决策与执行可靠性。

#### 方案要点：

- 数据：多轮上下文、工具定义、环境状态相关字段解析与注册。
- 执行：多轮 Agent 执行对接；交互轨迹与调用结果结构化记录。
- 评测：任务完成类指标；样本级执行审计。

#### 验收口径：

验收模型：Qwen3-Coder-32B；平台：NPU/GPU。
数据集：airline、retail、telecom。
temperature=0；每个子类别选取 前 20% case（固定），与官方测试方式 pass@5 精度差异 < 1%（需求文档中官方链接与 BFCL 同源引用，实施时以 Tau2-Bench 官方流程为准并对齐 pass@5 定义）。
影响范围
新增三套数据集任务包、配置模板与评测器；可能扩展通用“函数调用/Agent”执行抽象。
与现有精度评测报表字段对齐，避免破坏历史纯问答类任务。

#### 使用说明
1. Tau2-Bench：指定 domain（airline/retail/telecom）、20% 子集规则、pass@5 计算方式与引用版本。
2. 通用约束：NPU/GPU 平台；产物目录符合 AISBench 规范；版本信息（数据集 commit、工具链版本）写入运行元数据。

#### 测试设计
##### 单元测试
- 各数据集读取器：关键字段完整率、异常样本报告。

##### 集成测试
- 各子模块 infer + eval 一键跑通；日志与结果落盘完整；失败场景错误信息明确。

##### 端到端 / 官方对齐
- Tau2-Bench：pass@5 < 1%（给定验收模型与子集）。

##### 需求拆分验收点（通过测试体现）
- 数据标准化、执行链路接入、评测器与审计输出三类能力在各子模块内分别覆盖（详见内部需求拆分）。


### 3.3.5：Terminal-Bench 数据集接入与测评能力建设

#### 任务发现与字段标准化（需求拆分项汇总：任务加载）

- task_id / task_path；instruction（instruction.md）。
- 环境：docker_image 或 environment/Dockerfile；cpus、memory、storage、build_timeout_sec。
- 超时：agent_timeout_sec、verifier_timeout_sec。
- 验证：verifier_script（默认 tests/test.sh）；产物 /logs/verifier/reward.txt、/logs/verifier/ctrf.json。
- 元数据：difficulty、category、tags、作者等（task.toml 的 [metadata]、[agent]、[verifier]、[environment]）。

#### 容器执行与编排（需求拆分项汇总：容器编排）

- 启动方式：拉取 docker_image 或从 Dockerfile 构建。
- 挂载：任务工作目录、结果路径、/logs 可收集；保证 verifier 可写。
- Agent 输出协议：任务要求的关键产物路径（如 /app/results.txt）在文档与校验中明确。
- 资源与隔离：task.toml 限制与并发隔离策略（避免任务间污染、日志覆盖）。

#### 评测判定（需求拆分项汇总：reward/CTRF 判定）

- 最终以 /logs/verifier/reward.txt 数值（0/1）为准。
- 保留 ctrf.json 作为测试级审计（失败用例、栈、耗时）。
- 超时/环境失败分类：镜像拉取失败、容器启动失败、依赖安装失败等，错误信息可观测。

#### 指标与审计（需求拆分项汇总：汇总与审计落盘）

- 汇总：pass rate（可选加权/分层）。
- 分组：按 category、difficulty、tags。
- 样本级：task_id、判定、reward/CTRF 路径、关键日志路径、失败类型（timeout / env_error / verifier_error 等）。
#### 最小验收集与复现（需求拆分项汇总：验收集与复现约束）

- 固定子集策略（排序取前 N 或固定 seed）；固定镜像 tag 与仓库 commit；固定资源与超时；并发与隔离策略。
- 建议验收集覆盖多个 category（如 security、model-training、data-processing）。
- 外部参考：TB2 官方可用 Harbor（harbor run --dataset terminal-bench@2.0 ...）；AISBench 侧须保证行为、判定口径、审计产物清晰一致。

#### 使用说明
1. 前置：Docker 可用；磁盘与内存满足最差任务配置；Agent 执行器需支持在容器内非交互或伪终端运行（按所选 Agent 集成方式）。
2. 配置：任务子集列表或筛选规则、镜像拉取策略、资源上限、超时、并发数、日志宿主机落盘根目录。
4. 判定：禁止绕过 test.sh 自行判定 pass/fail；除非文档明确“调试模式”且不计入正式评测。
5. 复现：每次运行记录任务仓库 commit、镜像 digest、AISBench 版本。


#### 测试设计
##### 单元测试
- task.toml 解析与各字段默认值；缺失必填字段报错。
- reward 解析：mock reward.txt 为 0/1 与非数字异常。

##### 集成测试
- 固定最小子集：全链路 infer（或 mock Agent）+ verifier；收集 reward 与 ctrf。
- 失败注入：镜像不存在、超时，验证失败类型与审计字段。

##### 端到端 / 官方对齐
- 同输入同配置多次运行判定一致；差异可通过日志解释。
- 依赖缺失时用户文档中的修复指引可验证（检查清单）。

##### 需求拆分验收点（通过测试体现）
- 任务加载、编排约定、reward/CTRF 判定、汇总审计、最小验收集（见内部需求拆分）。


## 3.4 安全隐私与DFX设计

本版本特性主要涉及以下安全方面：

1. **API Key安全管理**：API Key通过环境变量或配置文件传递，不在日志中打印
2. **网络传输安全**：支持HTTPS协议，保护数据传输安全
3. **文件访问安全**：验证文件路径，防止路径遍历攻击
4. **服务端安全**：验证服务端URL，防止SSRF攻击

### 3.4.1 API Key安全管理

**安全措施**：

- API Key不在日志中打印，避免泄露
- API Key通过环境变量传递，不在命令行参数中暴露
- 支持HTTPS协议，保护API Key传输安全
- API Key错误时，提供明确的错误提示，但不泄露API Key信息

**风险分析**：

- 风险级别：中
- 影响：API Key泄露可能导致未授权访问
- 消减措施：使用环境变量、HTTPS协议、日志脱敏

### 3.4.2 网络传输安全

**安全措施**：

- 支持HTTPS协议，加密数据传输
- 支持SSL证书验证，防止中间人攻击
- 支持超时控制，防止长时间连接

**风险分析**：

- 风险级别：中
- 影响：网络传输被窃听可能导致数据泄露
- 消减措施：使用HTTPS协议、SSL证书验证

### 3.4.3 文件访问安全

**安全措施**：

- 验证文件路径，防止路径遍历攻击
- 限制文件访问范围，仅访问用户指定的文件
- 验证文件格式，防止恶意文件

**风险分析**：

- 风险级别：低
- 影响：路径遍历攻击可能导致未授权文件访问
- 消减措施：路径验证、访问范围限制

### 3.4.4 隐私保护

**隐私数据**：

- 本工具不收集个人数据
- 评估结果仅包含评估指标，不包含原始数据
- 日志中不包含敏感信息

**隐私保护措施**：

- 不收集、不存储个人数据
- 评估结果文件仅包含评估指标
- 日志脱敏，不打印敏感信息

### 3.4.5 韧性设计

**容错机制**：

- 网络请求失败自动重试
- 单条数据推理失败不影响其他数据
- 部分功能失败不影响核心功能

**恢复机制**：

- 支持从检查点恢复
- 支持结果文件备份和恢复
- 支持配置文件的备份和恢复

### 3.4.5可靠性&amp;可用性设计

#### 3.4.5.1冗余设计

**配置参数备份**：

- 关键配置参数（模型配置、数据集配置等）支持配置文件备份
- 支持配置文件的版本管理，便于回滚

**数据备份**：

- 评估结果自动保存，支持结果文件的备份和恢复
- 支持评估结果的版本管理，便于对比分析

**恢复策略**：

- 配置文件损坏时，使用默认配置或提示用户修复
- 评估结果丢失时，支持重新执行测评任务
- 支持从检查点恢复，避免重复计算

#### 3.4.5.2故障管理

**故障检测**：

- 网络连接故障：自动检测vLLM服务连接状态，连接失败时自动重试
- 数据集加载故障：检测数据集文件是否存在和格式是否正确
- 评估计算故障：检测评估指标计算过程中的异常，记录错误信息

**故障隔离**：

- 单条数据推理失败不影响其他数据的处理
- 单个数据集加载失败不影响其他数据集的处理
- 单个评估器计算失败不影响其他评估器的执行

**故障定位**：

- 详细的错误日志记录，包括错误类型、错误位置、错误原因
- 支持debug模式，输出更详细的调试信息
- 支持错误堆栈跟踪，便于定位问题

**故障恢复**：

- 网络请求失败：自动重试（默认2次），重试失败后记录错误
- 数据集加载失败：提示用户检查数据集文件，不继续执行
- 评估计算失败：记录失败的数据，继续处理其他数据

**告警设计**：

- 关键错误（如服务连接失败）记录ERROR级别日志
- 一般错误（如单条数据推理失败）记录WARNING级别日志
- 支持日志级别配置，控制日志输出

#### 3.4.5.3过载控制设计

**流量检测**：

- 监控并发请求数，避免超过系统承载能力
- 监控内存使用，避免内存溢出
- 监控CPU使用，避免CPU过载

**限速机制**：

- 支持通过`--max-num-workers`参数限制并行任务数
- 支持通过`request_rate`参数控制请求发送速率
- 支持通过`batch_size`参数控制批量大小

**过载处理**：

- 当并发数过高时，自动排队等待
- 当内存不足时，提示用户减少并发数或数据集大小
- 当CPU过载时，自动降低并发数

**优雅降级**：

- 非关键功能（如详细日志）可以降级
- 核心功能（如推理和评估）优先保障
- 支持部分结果输出，即使部分数据失败也能输出已处理的结果

#### 3.4.5.4升级不中断业务

**版本兼容**：

- 新版本支持老版本的配置文件和命令行参数
- 新版本支持老版本的数据集格式
- 新版本支持老版本的评估结果格式

**升级策略**：

- 支持平滑升级，不需要停止现有任务
- 支持配置迁移，自动将老版本配置转换为新版本格式
- 支持回滚，升级失败时可以回滚到老版本

**数据兼容**：

- 评估结果格式向前兼容，新版本可以读取老版本结果
- 配置文件格式向前兼容，新版本可以读取老版本配置
- 数据集格式向前兼容，新版本可以读取老版本数据集

#### 3.4.5.5人因差错设计

**配置错误防护**：

- 参数验证：所有命令行参数和配置参数都进行验证
- 默认值设置：提供合理的默认值，减少配置错误
- 错误提示：配置错误时提供明确的错误提示和处理建议

**操作错误防护**：

- 高危操作提示：对于可能影响数据的操作，提供确认提示
- 操作日志：所有操作都记录日志，便于追溯
- 快速回退：支持操作回退，配置错误时可以快速恢复

**数据保护**：

- 评估结果自动保存，避免数据丢失
- 支持结果文件备份，防止意外删除
- 支持结果文件版本管理，便于对比分析

#### 3.4.5.6故障预测预防设计

**资源监控**：

- 监控内存使用，提前预警内存不足
- 监控磁盘空间，提前预警磁盘空间不足
- 监控网络连接，提前预警网络问题

**健康检查**：

- 定期检查vLLM服务健康状态
- 定期检查数据集文件完整性
- 定期检查配置文件有效性

**预防措施**：

- 支持资源使用限制，防止资源耗尽
- 支持自动清理临时文件，防止磁盘空间不足
- 支持连接池管理，防止连接泄漏


## 3.5 编程与调用设计
### 3.5.1 接口定义与设计

本次方案里，主要的变动点在Agent测评时，需要适配的外部接口和内部输入和输出文件的定义。

1. 外部接口：
- HuggingFace datasets：load_dataset(hf_id, split=...)，在线拉取数据集。
- mini-swe-agent：process_instance(...)，生成实例补丁。
- SWE-bench harness：run_instance(...)，执行实例评测。
- Docker：image inspect/pull与容器清理接口。

2. 内部接口：
- Infer输出：predictions/<model>/<dataset>.json（instance_id到预测对象映射）。
- Eval输入：预测文件；Eval输出：results/<model>/<dataset>.json聚合报告。
- Summarizer输入：Eval聚合结果；输出：统一accuracy视图与统计字段。


---

附录

* **参考资料链接。**
1. 【SWE-Bench: Can Language Models Resolve Real-World GitHub Issues?】 https://arxiv.org/abs/2310.06770
2. 【Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces】：https://arxiv.org/abs/2601.11868
3. 【Humanity's Last Exam】：https://arxiv.org/abs/2501.14249
4. mini-swe-agent: https://github.com/SWE-agent/mini-swe-agent
5. terminal-bench:https://github.com/harbor-framework/terminal-bench

* **术语表。**
1. HLE: Humanity's Last Exam

* **文档更新计划**
1. 06.30：更新多模态生成数据集接入的详细设计；当前SWE-Bench测评效率提升的解决方案


