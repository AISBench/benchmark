# SKILL 测评报告：ascend-llm-deploy-eval

| 项 | 值 |
|---|---|
| 被测 SKILL | `AISBench/benchmark::.agents/skills/ascend-llm-deploy-eval` |
| 版本锚点 | ref `master` / commit `1aead1f12bd35746672bda5a8a2330f58e7ba651` |
| 测评集版本 | `1.0.0` |
| 执行时间 | 2026-11-14 22:27:57 中国标准时间 |
| 被测解释器 | `python.exe 3.11.0`（执行器 Python 3.14.6 / Windows） |
| 执行方式 | 离线黑盒：只调用 SKILL 自带入口，不需要 SSH、Docker、NPU |
| 依赖路径 | `.deps` |

## 总评

**综合得分 100.0 / 100**（27/27 用例通过，权重 120/120）

| 维度 | 用例 | 通过 | 权重 | 得分 |
|---|---:|---:|---:|---:|
| D1 契约与状态机 | 5 | 5 | 19 | 100.0 |
| D2 规范化与派生默认值 | 3 | 3 | 17 | 100.0 |
| D3 绑定保真与阻断 | 6 | 6 | 26 | 100.0 |
| D4 生成物完整性与可审计性 | 3 | 3 | 20 | 100.0 |
| D5 AISBench 测试语义 | 3 | 3 | 13 | 100.0 |
| D6 并行与阶段 | 4 | 4 | 15 | 100.0 |
| D7 内容与格式规范 | 3 | 3 | 10 | 100.0 |

## 用例明细

| 用例 | 名称 | 维度 | 权重 | 结果 | 失败首因 |
|---|---|---|---:|---|---|
| C01 | draft 任务拒绝生成可执行脚本，且不留半成品 | D1 契约与状态机 | 5 | PASS |  |
| C02 | ready 但 pending 未闭环（unresolved 非空）拒绝生成 | D1 契约与状态机 | 4 | PASS |  |
| C03 | accuracy 模式明确拒绝且不回退成性能测试 | D1 契约与状态机 | 4 | PASS |  |
| C04 | task_end=keep_service 明确拒绝（当前只支持 stop_container） | D1 契约与状态机 | 3 | PASS |  |
| C05 | after_case=stop_service 明确拒绝 | D1 契约与状态机 | 3 | PASS |  |
| C06 | 稳态用例派生 formal_requests=并发×10 并选用 stable_stage 统计 | D2 规范化与派生默认值 | 6 | PASS |  |
| C07 | 非稳态用例派生 4× 并发，零散默认值（预热/超时/收尾/执行/容器）全部落盘 | D2 规范化与派生默认值 | 6 | PASS |  |
| C08 | enable_cache_reset_api=false：不注入 DEV_MODE、也不删除用户显式值 | D2 规范化与派生默认值 | 5 | PASS |  |
| C09 | 未解析绑定 {{paths.missing}} 阻断生成 | D3 绑定保真与阻断 | 4 | PASS |  |
| C10 | 参考脚本里写死的 HCCL_IF_IP 与现场绑定不一致时阻断 | D3 绑定保真与阻断 | 5 | PASS |  |
| C11 | 参考脚本里写死的 data-parallel-address 与绑定不一致时阻断 | D3 绑定保真与阻断 | 4 | PASS |  |
| C12 | 把任务目录显式挂成只读时阻断 | D3 绑定保真与阻断 | 4 | PASS |  |
| C13 | 绑定路径含 .. 时阻断（拒绝可能逃逸工作根的路径） | D3 绑定保真与阻断 | 4 | PASS |  |
| C14 | 生成物清单完整、status 为 SCRIPTS_READY、manifest 摘要自洽 | D4 生成物完整性与可审计性 | 8 | PASS |  |
| C15 | 容器脚本：特权基线、任务/模型挂载、全量绑定端口、无旧基线主机目录 | D4 生成物完整性与可审计性 | 6 | PASS |  |
| C16 | 实例脚本：绑定值展开、显存设备选择、host/port 与 KV 参数由绑派生，代理按路由生成 | D4 生成物完整性与可审计性 | 6 | PASS |  |
| C17 | 外部数据集生成 CustomDataset 配置，端点/负载/重试语义与 job_spec 对齐 | D5 AISBench 测试语义 | 5 | PASS |  |
| C18 | 原生合成数据用 SyntheticDataset，按输入长度与前缀比例生成 RequestSize/PrefixLen | D5 AISBench 测试语义 | 5 | PASS |  |
| C19 | 外部数据集声明前缀比例但缺 generation_manifest 时阻断 | D5 AISBench 测试语义 | 3 | PASS |  |
| C20 | 双机 PD：两个节点各自生成容器/实例脚本，跨节点 IP 与端口不串台 | D6 并行与阶段 | 5 | PASS |  |
| C21 | 显式 stages：两个独立分支各自生成，dispatch 覆盖全部子任务 | D6 并行与阶段 | 4 | PASS |  |
| C22 | 并行分支共用同一物理节点时阻断 | D6 并行与阶段 | 3 | PASS |  |
| C23 | 两个节点槽位绑定同一台主机（别名）时阻断 | D6 并行与阶段 | 3 | PASS |  |
| C24 | 现场证据与绑定不一致时评审判定 BLOCKED；一致时只停在待 Agent 审阅 | D3 绑定保真与阻断 | 5 | PASS |  |
| S01 | SKILL.md 前置元数据完整，name 与目录名一致 | D7 内容与格式规范 | 3 | PASS |  |
| S02 | SKILL.md 引用的相对文件全部存在（references/docs/schemas/scripts） | D7 内容与格式规范 | 4 | PASS |  |
| S03 | 文档承诺的 CLI 形态与 run.py 实际参数一致（唯一入口、两种模式） | D7 内容与格式规范 | 3 | PASS |  |

## 失败详情

无失败用例。

## 覆盖边界（未纳入自动判定的部分）

- FULL 模式实机链路（SSH 免密、Docker 建容器、昇腾 NPU、AISBench 实装、真实权重）离线不可执行；本测评只判定「同一份 job_spec/bindings 生成的脚本是否正确、可审计」，不判定实机是否跑通吞吐/时延。
- review_job.py 的现场证据采集需要真实主机；本测评用 review_job.assess() 的判定契约（C24）替代，不覆盖 PROBE 采集实现。
- 代理脚本、镜像内 vLLM/AISBench 版本行为、指标采集正确性依赖现场版本，不作为通过条件。
- 被测 SKILL 自带 tests/ 单元测试不计入本测评分数（避免用自己的测试给自己的行为打分），仅作参考。

## 复现

```sh
python -m skill_eval --suite suites/ascend-llm-deploy-eval --skill-dir ../skills/ascend-llm-deploy-eval --python python.exe --deps .deps --work .eval-work --md reports/report.md --json reports/result.json
```

用例输入（job_spec / bindings）与断言全部在测评集内，执行不修改被测 SKILL 目录。
