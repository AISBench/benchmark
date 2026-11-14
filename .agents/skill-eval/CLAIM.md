# 揭榜回复（可直接贴到评论区）

> 以下内容只引用公开仓库与自造占位数据（`10.0.0.x`、`registry.example`、`/mnt/models/demo-model`），
> 不含项目代号、客户信息或未公开产品信息。

```text
📌 揭榜
- 认领内容：在 AISBench/benchmark 仓为 .agents/skills/ascend-llm-deploy-eval 搭建 SKILL 测评
  （测评集 27 例 + 执行脚本 + 报告 + 可迁移模板），评测维度选「功能效果」
- 计划时间：已跑通，可直接评审
- 风险/依赖：需要 Linux + Python 3.10+ 评审环境复跑（纯本地、约 30 秒）；
  实机 FULL 链路（SSH/Docker/昇腾 NPU）不在本轮范围，如需扩展需借用一台昇腾机器
```

## 一、做了什么

选了一个**真实存在、且天然适合被测评**的 SKILL：`ascend-llm-deploy-eval`（昇腾模型部署与性能测试）。
它自己把生命周期切成「离线生成脚本（`--mode scripts`）」和「实机执行（`--mode FULL`）」两半，
前半段是确定性契约、可离线判定，正好是代码演进中最容易被改坏的部分。

于是针对前半段做了一套**黑盒回归测评**：喂 `job_spec` + `bindings`，看退出码、看生成的脚本、看产物清单。
不需要 SSH、不需要 Docker、不需要 NPU，30 秒出分。

结果：**27/27 通过（100/100）**；并且用 8 条注入退化验证了测评集本身有效（**8/8 检出**）。

## 二、对照「有效参与判据」

| 判据 | 对应证据 |
|---|---|
| ① 测评流程在**某仓实际跑通** | 27 个用例在 `AISBench/benchmark` 的 SKILL 上实际执行，产物在 `.eval-work/runs/*/run`；报告 `reports/report.md` |
| ② 产出**可复用的**测评集 / 脚本 / 报告模板 | `suites/ascend-llm-deploy-eval/`（测评集）、`skill_eval/` + `tools/`（脚本，零第三方依赖）、`reports/report.md`（报告模板）、`template/`（换一个 SKILL 的最小样例，2/2 跑通） |
| ③ 经评审确认有效 | 待评审。可先看 `reports/mutation.md`：8 条退化全部被对应用例抓到，说明断言不是空转 |

## 三、这套测评到底在测什么

7 个维度、27 条用例，每条用例都写了「为什么这么期望」并指向 `SKILL.md` / `docs/` 原文：

1. **契约与状态机**：draft 不生成、`ready` 但 `unresolved` 非空不生成、`accuracy` 明确拒绝而不是偷偷跑成 perf。
2. **规范化与派生默认值**：稳态 `10×并发`、普通 `4×并发`、预热/超时/收尾/容器默认值是否真的落盘。
3. **绑定保真与阻断**：未解析占位符、写死 IP/RPC 地址与现场绑定冲突、任务目录只读挂载、路径逃逸。
4. **生成物完整性与可审计性**：27 个产物一个不多一个不少、`manifest` 摘要自洽、容器/实例脚本逐项内容。
5. **AISBench 测试语义**：`CustomDataset` / 原生 `SyntheticDataset`（`RequestSize`/`PrefixLen`）、`retry=1`、参数对齐。
6. **并行与阶段**：双机 PD 跨节点不串台、`stages` 全覆盖、并行分支共用物理节点/同 host 别名必须拦。
7. **内容与格式规范**：frontmatter 合法、SKILL.md 引用文件全部存在、文档承诺的 CLI 与实现一致。

设计上坚持三条：**黑盒**（只调 CLI，不 import 被测代码）、**正负成对**（能生成与必须拒绝都要测）、
**不越权**（不判定实测吞吐、权重正确性、镜像内版本行为——那些属于现场，不属于 SKILL 的契约）。

## 四、边界（不吹）

- `FULL` 实机链路未覆盖：SSH、建容器、NPU 占用、AISBench 实跑、释放核验。要测这一段，把 `adapter.command`
  换成 `--mode FULL` 并接一台昇腾机即可，框架不用改。
- `review_job.py` 的现场证据**采集**（需真机）未覆盖，只验证了它的**判定契约**（证据与绑定不一致即 `BLOCKED`）。
- SKILL 自带的 `tests/` 单元测试不计分，避免「用自己的测试给自己打分」。

顺带记录两条改进建议（非功能缺陷，详见 `README.md` 第 8 节）：生成期对临时目录权限有隐性要求；
`manifest.json` 摘要不覆盖 `status.json` 而文档表述容易被误读。

## 五、评委怎么复跑

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/AISBench/benchmark.git upstream
git -C upstream sparse-checkout set .agents/skills/ascend-llm-deploy-eval
uv pip install --target .deps --python /usr/bin/python3.11 "jsonschema>=4.18,<5" "packaging>=23"
python -m skill_eval --suite suites/ascend-llm-deploy-eval \
    --python /usr/bin/python3.11 --deps .deps --md reports/report.md
python tools/mutation_check.py --python /usr/bin/python3.11 --deps .deps --md reports/mutation.md
```

## 六、如果想把这个能力铺到全部门

`template/` 里已经把「换一个 SKILL 要改什么」压到三件事：拷框架、写 `suite.json`、
写用例（`fixtures` 放基线输入 + `cases` 写 patch 与断言）。断言类型有 20 种，覆盖进程/文件/JSON/文本/静态检查。
对没有 CLI、只输出自然语言的 SKILL，把 `adapter` 换成「拼 prompt 调模型」、`expect` 换成
`probe_json` 读结构化输出即可，装载、打分、报告、自检都不需要改。
