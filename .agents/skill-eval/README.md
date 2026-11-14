# SKILL 测评：ascend-llm-deploy-eval

针对本仓 [`.agents/skills/ascend-llm-deploy-eval`](../skills/ascend-llm-deploy-eval/SKILL.md)
（上游 `AISBench/benchmark` 同名 skill）做的一套**可执行、可复现、可迁移**的 SKILL 测评。

选的是「⚡ 功能效果」这一档：不看 SKILL 写得漂不漂亮，看它承诺的行为在真实输入下**还成立不成立**。

结论先行：**27 个用例全通过（100/100），且测评集自身经 8 条注入退化的自检全部检出**。
换句话说，这个 SKILL 当前的确定性契约是完整的；后续任何人改 `scripts/`、`schemas/` 或 `references/`
把契约改坏，这套测评会在几十秒内给出具体是哪一条坏了。

---

## 1. 为什么这个 SKILL 能离线评

`ascend-llm-deploy-eval` 的职责是把自然语言需求变成 `job_spec` + `bindings`，再据此生成
可人工复核、可复现的部署与测试脚本。它自己就把生命周期切成了两半：

| 模式 | 是否需要服务器 | 本测评 |
|---|---|---|
| `--mode scripts` | **不需要**。纯本地校验 + 生成脚本，状态 `SCRIPTS_READY` | 全部覆盖（24 个行为用例 + 3 个静态用例） |
| `--mode FULL` | 需要 SSH / Docker / 昇腾 NPU / AISBench 实装 | 不覆盖，见「覆盖边界」 |

而真正容易在代码演进中被改坏的，恰好是前半段那些**确定性契约**：默认值怎么派生、绑定冲突要不要拦、
生成的脚本里 IP/端口/挂载对不对、`accuracy` 会不会被静默降级成 `perf`……
这些用 `job_spec`+`bindings` 喂进去、读退出码和产物就能判定，不需要一张昇腾卡。

所以本测评的定位是：**「这个 SKILL 还算不算它自己」的回归网**，而不是「这次部署性能好不好」。

---

## 2. 交付物

```text
.agents/skill-eval/                 # ← 本目录（仓库内位置）
├── README.md                       # 本文：方案 + 结论 + 迁移方法
├── CLAIM.md                        # 揭榜回复草稿与判据对照
├── skill_eval/                     # 测评框架（与具体 SKILL 解耦，可直接搬到别的仓）
│   ├── cli.py                      #   入口：python -m skill_eval --suite ...
│   ├── harness.py                  #   用例装载 / 调用被测 SKILL / 断言 / 打分
│   └── render.py                   #   Markdown 报告渲染
├── suites/ascend-llm-deploy-eval/  # ① 测评集（本 SKILL 专用）
│   ├── suite.json                  #   被测对象锚点、调用方式、覆盖边界
│   ├── fixtures/{base,two-node,multi}/  # 3 套基线 job_spec + bindings
│   ├── cases/                      # 27 个用例（输入补丁 + 期望输出）
│   └── probes/review_gate.py       #   替代 SSH 探测的判定契约探针
├── tools/
│   ├── fetch_skill.py              # 按 commit 拉取被测 SKILL（可复现）
│   └── mutation_check.py           # ② 测评集自检：注入退化，验证用例会红
├── template/                       # ④ 复用模板：换一个 SKILL 的最小样例
│   ├── demo-skill/                 #   一个形状完全不同的 SKILL
│   └── demo-suite/                 #   它的测评集（2 用例，已跑通）
└── reports/
    ├── report.md                   # ③ 测评报告（人读）
    ├── result.json                 #   机器可读结果（CI 用）
    └── mutation.md                 #   测评集自检报告
```

对应活动要求的「测评集 / 脚本 / 报告 / 复用模板」四件套，见第 6 节。

---

## 3. 快速开始

### 在 AISBench 仓内（`.agents/skill-eval/`，推荐）

```sh
cd .agents/skill-eval

# 被测 SKILL 的依赖（本仓 `.agents/skills/ascend-llm-deploy-eval` 要求 Python >= 3.10 + jsonschema）
uv pip install --target .deps --python /usr/bin/python3.11 "jsonschema>=4.18,<5" "packaging>=23"

# 跑测评（约 30 秒，纯本地）
PYTHONIOENCODING=utf-8 python -m skill_eval \
    --suite suites/ascend-llm-deploy-eval \
    --python /usr/bin/python3.11 --deps .deps \
    --work .eval-work --md reports/report.md --json reports/result.json

# 测评集自检（改动用例后建议跑）
python tools/mutation_check.py --python /usr/bin/python3.11 --deps .deps --md reports/mutation.md
```

### 作为独立目录使用

把 `skill_eval/`、`suites/`、`tools/`、`template/` 拷到任意位置，再用
`tools/fetch_skill.py` 按 commit 拉取被测 SKILL：

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/AISBench/benchmark.git upstream
git -C upstream sparse-checkout set .agents/skills/ascend-llm-deploy-eval
```

把 `suites/<suite>/suite.json` 里的 `skill.local_dir` 指到拉下来的目录即可。

跑完看 `reports/report.md`：维度得分、每条用例结果、失败首因、复现命令。

> 平台提示：在 Windows + Python 3.13/3.14 上，`tempfile` 会用 0o700 建临时目录，若工作区被 ACL 限制
> 会报 `Permission denied`（与被测 SKILL 无关）。本仓库实测用 Python 3.11 执行被测 SKILL；
> Linux 上直接用系统 Python 3.10+ 即可。

---

## 4. 测评集设计

27 个用例、7 个维度，每个用例都写明「为什么这么期望」（`why` 字段直接指向 SKILL.md / docs 原文），
避免把测评写成「照着实现抄期望」。

| 维度 | 用例 | 权重 | 评什么 |
|---|---|---:|---|
| D1 契约与状态机 | C01–C05 | 19 | draft 不生成、`ready` 但有 `unresolved` 不生成、`accuracy` 明确拒绝不回退、未接入的收尾策略明确拒绝 |
| D2 规范化与派生默认值 | C06–C08 | 17 | 稳态 `10×并发` + `stable_stage`、普通 `4×并发`、预热/超时/收尾/容器默认值全部落盘、`enable_cache_reset_api` 关掉后不注入也不删用户值 |
| D3 绑定保真与阻断 | C09–C13、C24 | 26 | 未解析占位符、写死 IP 与绑定冲突、写死 RPC 地址冲突、任务目录只读挂载、路径逃逸、现场证据与绑定不一致时 `BLOCKED` |
| D4 生成物完整性与可审计性 | C14–C16 | 20 | 产物清单一个不多一个不少、`manifest` 摘要自洽、容器脚本（特权/挂载/全量端口/无旧基线目录）、实例脚本（绑定展开/设备选择/KV 参数/代理路由） |
| D5 AISBench 测试语义 | C17–C19 | 13 | 外部数据 `CustomDataset`、原生合成 `SyntheticDataset`（`RequestSize/PrefixLen`）、`retry=1`、`--num-prompts/--num-warmups`、缺 `generation_manifest` 时拒绝前缀声明 |
| D6 并行与阶段 | C20–C23 | 15 | 双机 PD 跨节点不串台、显式 `stages` 全覆盖、并行分支共用物理节点/同一 host 别名必须拦 |
| D7 内容与格式规范 | S01–S03 | 10 | frontmatter 合法且 name 与目录一致、SKILL.md 引用文件全部存在、文档承诺的 CLI 与 `run.py` 参数一致 |

用例组织方式（每个 `cases/*.json`）：

```json
{
  "id": "C10",
  "title": "参考脚本里写死的 HCCL_IF_IP 与现场绑定不一致时阻断",
  "dimension": "D3 绑定保真与阻断",
  "weight": 5,
  "fixture": "base",
  "why": "SKILL.md「现场绑定差异必须展示原值、解析值及依据」…",
  "patch": [{"file": "job_spec.json", "pointer": "/deployments/0/instances/0/launch/env/HCCL_IF_IP", "value": "10.0.0.99"}],
  "expect": [
    {"type": "exit_code", "value": 1},
    {"type": "stderr_contains", "value": "Confirmed environment conflicts with binding: prefill.HCCL_IF_IP"},
    {"type": "dir_absent", "value": "{run_dir}"}
  ]
}
```

三个设计原则：

1. **黑盒**：只调 SKILL 自带 CLI，不 import、不 mock 内部函数。内部重构只要行为不变，测评就仍然通过。
2. **正向 + 负向成对**：能生成（C14）与必须拒绝（C09–C13、C22–C23）都要有。只测 happy path 的测评抓不到「该拦没拦」。
3. **越权的期望不写**：不判定实测吞吐、不判定权重正确性、不判定镜像里 vLLM/AISBench 的版本行为——那些属于现场，不属于 SKILL 的契约。

---

## 5. 评分与自检

得分 = `Σ(权重 × 通过) / Σ权重 × 100`，维度分同理。任一断言失败该用例即 FAIL，报告给出首个失败原因与现场目录。

「全绿」本身不是证据：断言写松了也会全绿。所以配了 `tools/mutation_check.py`——把 SKILL 复制一份，
注入 8 条**真实可能发生的退化**，要求对应用例必须变红：

| 编号 | 注入的退化 | 期望失败 | 结果 |
|---|---|---|---|
| M1 | 稳态正式请求数从 10× 退回 4× | C06 | 检出 |
| M2 | 去掉 `ready` 任务的 `unresolved` 拦截 | C02 | 检出 |
| M3 | 关闭开关时仍然注入 `VLLM_SERVER_DEV_MODE` | C08 | 检出 |
| M4 | 删除「参考脚本环境 vs 绑定」一致性校验（两处） | C10 | 检出 |
| M5 | 代理入口标记 `# @proxy-command` 消失 | C16 | 检出 |
| M6 | 模型目录挂载丢掉 `:ro` | C15 | 检出 |
| M7 | AISBench `retry` 从 1 改成 3 | C17 | 检出 |
| M8 | 删除 `references/mount-profiles.md`（SKILL.md 断链） | S02 | 检出 |

**8/8 检出**，报告见 `reports/mutation.md`。这一步同时证明测评集对「代码演进把 SKILL 改坏」这个真实场景是有效的。

---

## 6. 盖住和没盖住

**盖住**（离线可判定，100% 通过）：

- 状态机与拒绝语义；默认值派生；绑定冲突与未解析占位符；
- 生成脚本的逐项内容（特权、挂载、端口、设备、KV、代理路由、AISBench 配置）；
- 产物清单与 `manifest` 摘要；跨节点不串台；并行分支隔离；
- 文档与实现的 CLI 一致性、SKILL.md 链接完整性。

**没盖住**（诚实声明，已在 `suite.json.limits` 与报告里同步）：

- `FULL` 实机链路：SSH、Docker 建容器、NPU 占用检查、AISBench 实跑、真实权重、释放核验；
- `review_job.py` 的现场证据**采集**（需要真机）：本测评只验证它的**判定契约**（C24，证据与绑定不符即 `BLOCKED`）；
- 代理脚本、镜像内版本行为、指标采集正确性——依赖现场版本，不作为通过条件；
- SKILL 自带 `tests/` 单元测试不计分（避免「用自己的测试给自己打分」）。

要补上实机那一段，只需再加一个 suite，把 `adapter.command` 换成 `--mode FULL` 并给一台可用的昇腾机；
用例的 `expect` 换成 `logs/`、`results/`、`status.json` 的产物断言即可，框架不用改。

---

## 7. 换成别的仓怎么用（3 步）

`template/` 里有一个形状完全不同的 demo SKILL 和它的测评集（2 用例，已跑通），可以直接照抄。

1. **拷框架**：`skill_eval/` + `tools/` 整个目录拷走（无第三方依赖，标准库即可）。
2. **写 `suite.json`**：填被测对象锚点（repo/path/ref/commit）和调用方式：

   ```json
   "adapter": {
     "type": "command",
     "workdir": "{skill_dir}",
     "command": ["{python}", "scripts/run.py", "--job-spec", "{job_spec}", "--bindings", "{bindings}", "--run-dir", "{run_dir}", "--mode", "{mode}"]
   }
   ```
   可用占位符：`{python}` `{skill_dir}` `{suite_dir}` `{job_spec}` `{bindings}` `{run_dir}` `{case_dir}` `{mode}`。
3. **写用例**：`fixtures/<name>/{job_spec,bindings}.json` 放基线输入，`cases/*.json` 用 `patch`（JSON Pointer 的 set/remove/append）造场景，`expect` 写断言。

断言类型（当前实现）：

| 类别 | 类型 |
|---|---|
| 进程 | `exit_code`、`stdout_contains`、`stderr_contains`、`stderr_not_contains`、`stderr_regex`、`stderr_any_of` |
| 文件 | `file_exists`、`file_absent`、`file_set_equals`、`dir_absent`、`no_temp_leftover`、`manifest_integrity` |
| 内容 | `json_equals`、`json_present`、`text_contains`、`text_not_contains`（`path` 支持 `skill:` 前缀读被测目录） |
| 探针 | `probe_json`（跑一个脚本读 stdout JSON，按指针断言，用于替代需要 SSH 的入口） |
| 静态 | `frontmatter`、`relative_links_exist`、`cli_flags`（不执行 SKILL，只读文件） |

对「LLM 类 SKILL」（没有 CLI、只有自然语言输出）也一样能评：把 `adapter` 换成「拼 prompt 调模型」，
把 `expect` 换成 `probe_json` 读模型输出的结构化字段 + 人工仲裁的关键点即可；框架的装载、打分、报告、自检都不用动。

---

## 8. 顺带发现的两点（非功能缺陷）

跑到 27/27 全绿的过程中，记了两条值得改的地方，都不是行为错误：

1. **生成期对临时目录权限有隐性要求**。`run.py` 用 `TemporaryDirectory` + 目录 rename 保证「不留半成品」，
   在 Linux 上没问题；但在受限 ACL 环境（如 Windows + Python 3.13+，`tempfile` 会建 0o700 目录）会
   以 `Permission denied` 失败，且异常被清理逻辑盖住，报错指向清理而不是根因。
   建议 `README` 的「快速开始」里补一句平台前提，或给临时目录显式传 `dir=` 并降到 0o755。
2. **`manifest.json` 的摘要不覆盖 `status.json`**。`status.json` 是在 manifest 生成之后写入的，属设计选择
   （status 由协调器运行时重写），但文档里「脚本、输入和生成器摘要」容易被读成「全部产物」。
   建议在 `docs/compact-cli.md` 一句话点明。

两条都写在报告里，供维护者取舍。
