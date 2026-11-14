# aisbench-devflow 快速开始

中文 | [English](README_en.md)

## 用途与边界

把 AISBench 的 Issue → 实现 → PR → CI → 检视闭环做成可暂停、可恢复、带证据的流程，并把**能本地判定的合入门禁前移**，缩短一次 PR 往返 20~40 分钟的反馈回路。

本技能**不做**：自动合入、自动 approve、代签 DCO/CLA、自动建 Issue、修改 PR 模板或标签工作流、NPU 真机评测复现、全量本地测试复跑。

## 它到底帮你做什么

| 场景 | 没有技能 | 有技能 |
|---|---|---|
| 提交前判门禁 | push 后等自托管 runner 20~40 分钟 | 3 秒本地得到 `PASS/FAIL/UNKNOWN` |
| 覆盖率 | CI 挂才发现 | 得到该补的测试目录与建议文件路径 |
| PR 标签 | 勾了但没打上，不知为何 | 直接指出模板文案与 workflow 映射不一致 |
| 双语文档 | 漏了英文，合入前被退回 | 本地即报缺对侧文件 |
| DCO | 忘了 `-s` | 本地指出哪些提交缺签署行 |

## 依赖

- Git，Python 3.10+（运行期仅标准库）
- 远端读写：已登录的 GitHub CLI（`gh`），或宿主提供的 GitHub 工具

## 快速开始

```bash
# 1. 读取仓库画像（阈值/模板/标签映射/文档树，全部现读，不硬编码）
python .agents/skills/aisbench-devflow/scripts/gate_check.py --profile

# 2. 提交前跑本地门禁
python .agents/skills/aisbench-devflow/scripts/gate_check.py \
  --base origin/master --pr-type feature --issue 651 \
  --title "[feat][datasets]支持 X" --pr-body-file /tmp/pr-body.md --out /tmp/gates.json

# 3. 需要补测试时，问该写到哪里
python .agents/skills/aisbench-devflow/scripts/ut_map.py suggest \
  --changed ais_bench/benchmark/datasets/hle.py

# 4. 按模板渲染 PR 正文（勾选精确、附审计标记）
python .agents/skills/aisbench-devflow/scripts/pr_body.py render \
  --template .github/pull_request_template.md \
  --workflow .github/workflows/auto_label_pr.yml \
  --pr-type feature --issue 651 --sections sections.json --out /tmp/pr-body.md

# 5. 自检正文
python .agents/skills/aisbench-devflow/scripts/pr_body.py check \
  --body-file /tmp/pr-body.md --workflow .github/workflows/auto_label_pr.yml --issue 651

# 6. 记录/恢复进度（只写 Git 元数据目录，不进出 diff）
python .agents/skills/aisbench-devflow/scripts/ledger.py show
python .agents/skills/aisbench-devflow/scripts/ledger.py set --repo AISBench/benchmark \
  --field run_id=pr-651-20261114-a1b2 --field last_completed_state=pr-open --head
python .agents/skills/aisbench-devflow/scripts/ledger.py gate --name size --status PASS --evidence "412 lines"
```

`--sections` 是 JSON：`{"Motivation": "变更动机正文", "Modification": "修改内容"}`，键按标题模糊匹配。

## 判定语义（重要）

| 值 | 含义 |
|---|---|
| `PASS` | 有证据表明门禁成立 |
| `FAIL` | 有证据表明门禁不成立，附 `remedy` |
| `UNKNOWN` | 本地无法判定（远端 review、thread 状态、全量覆盖率、外部构建、自托管冒烟） |
| `NOT_APPLICABLE` | 不适用，附适用性理由 |

- **退出码 0 只表示“没有已知 FAIL”**，不代表可以推送或推进。
- 推进前要求 `promotion_blockers` 为空；`UNKNOWN` 必须披露。
- 禁止用 `PASS` 表示“没检查”。

## 已知限制

1. **覆盖率只跑子集**：默认不安装依赖、不跑全量；子集结果标 `scope=subset`，全量由 CI 的 `pr_run_test` 给出。
2. **文档内容同步不自动判定**：脚本只做路径配对，内容是否一致标 `UNKNOWN`，需人工/Agent 复核。
3. **合入永不执行**：`merge` 只输出 `ready-to-merge` 清单。
4. **标签文案不一致**：仓库现有模板与 workflow 有 CI/CD、Dependency 两处不一致，技能只告警、保留模板原文，不代为修复。
5. **环境差异**：自托管 runner 与 NPU 冒烟无法本地等价复现，一律 `UNKNOWN`。

## 目录结构

```
aisbench-devflow/
├── SKILL.md              # 规则入口：action 路由、契约、状态机
├── README.md / README_en.md
├── references/           # 按 action 渐进加载的细则（单文件双语）
├── scripts/              # 只判定、不修改（账本仅写 Git 元数据目录）
│   ├── gate_check.py     # 本地门禁预检
│   ├── pr_body.py        # 模板渲染与自检
│   ├── ut_map.py         # 源码↔UT 映射、子集覆盖率
│   └── ledger.py         # 恢复账本：原子写、加锁、worktree 兼容、身份隔离
└── tests/                # unittest（含真实 git 仓夹具）
```

## 账本

- 位置：`<git rev-parse --absolute-git-dir>/aisbench-devflow/ledger.json`，兼容 worktree/submodule（`.git` 为文件）。
- 只存标识与结论：`head_sha`、门禁状态、`unverified`；**不存 Token、绝对路径、日志**。
- 原子写 + `ledger.lock`，过期锁需显式 `--break-lock`；身份不匹配时拒绝覆盖（`--reinit` 才重建）。
- `show` 会与当前 `HEAD` 比对并输出 `stale`：账本陈旧时，其中门禁结论一律作废。

## 与只读技能的分工

- `pr-checklist`：规范判定（标题/模板/DCO/分类）
- `ci-failure-triage`：CI 失败只读诊断
- `aisbench-devflow`：诊断之后的**修复**、以及上述之外的**写路径**

## 开发与测试

```bash
python tests/run_unit_tests.py --output /tmp/devflow-tests.json
python -m ruff check --config ruff.toml scripts tests   # 需要 ruff==0.17.4
```

测试会在临时目录里构造真实 git 仓库（大 PR、缺 DCO、单侧双语、删除无配对等），不访问网络、不安装依赖。
