# din_integration — SWE-bench DinD Evaluation Pipeline

> **状态**: 本目录是对 SWE-bench DinD (Docker-in-Docker) 多 agent × 多 case 评测流水线的**整包搬运**,
> 作为 AISBench 仓库根目录下的独立子目录提交。
>
> **历史**: 代码原位于 `/home/zengziyu/mini_matrix/cli/` + `mini_matrix/scripts/` + `mini_matrix/config/`,
> 经评审后整合到本目录,便于在 AISBench 仓内统一治理。

---

## ⚠ 路径假设(本包当前已知局限)

本包直接拷贝自 `mini_matrix/`,**未做 env-var 化路径重构**。这意味着:

| 路径硬编码 | 假设值 |
|---|---|
| `swebench_dind/launcher.py:58` api_key.env | `/home/zengziyu/mini_matrix/scripts/api_key.env` |
| `swebench_dind/config.py:14-17` ROOT | 由 `Path(__file__).parent.parent` 自动推导 → 当前为 `din_integration/` |
| `swebench_dind/config.py:27-35` jobs/tasks/logs | 位于 ROOT 同级 → 当前为 `din_integration/{jobs,tasks,logs}/` |
| `swebench_dind/config.py:31` api_key.env | `ROOT/scripts/api_key.env` → 当前为 `din_integration/scripts/api_key.env` |

**实测工作方式**(以最小集 PR 为目标):

```bash
# 1. 把 secrets 放在 launcher.py 期望的位置(原 mini_matrix/scripts/api_key.env)
ln -sf /path/to/your/api_key.env /home/zengziyu/mini_matrix/scripts/api_key.env

# 2. 在 din_integration/ 下装包
cd din_integration
pip install -e .

# 3. 准备运行时数据目录(可指向任意位置)
mkdir -p ~/swebench_dind_{jobs,tasks,logs}

# 4. 跑 CLI(注意 launcher.py 仍会从 mini_matrix/scripts/ 读 key)
swebench-dind --version
```

**完整 env-var 重构方案**见设计文档 [方案 A §5](file:///home/zengziyu/aisbench/docs/research/24-swebench-dind-integration-plan-A-slim-2026-08.md#5-代码修改清单共-9-处),
本次未执行(以最小集 PR 为目标)。

---

## 📂 本目录结构

```
din_integration/
├── README.md                                       ← 本文件
├── pyproject.toml                                  ← Python 包元数据 (name=swebench-dind)
├── bin/swebench-dind                               ← shell wrapper
├── docs/
│   ├── CLI-USAGE.md                                ← CLI 完整命令参考
│   └── MIGRATION.md                                ← 老脚本 → CLI 对照表
├── swebench_dind/                                  ← 核心 Python 包
│   ├── __init__.py                                 ← __version__ = "0.1.0"
│   ├── __main__.py
│   ├── cli.py                                      ← Typer 7 子命令入口
│   ├── config.py                                   ← 单一真相源(常量 + tag 推导)
│   ├── container.py                                ← DinD 容器生命周期
│   ├── builder.py                                  ← L3/L4 镜像烤制(Jinja2)
│   ├── launcher.py                                 ← harbor jobs start 拼装 + Rich 进度
│   ├── patcher.py                                  ← idempotent install probe 注入
│   ├── summarizer.py                               ← result.json → md/csv/json
│   ├── dockerfiles/                                ← L1/L2 Dockerfile 模板 (5 个 .j2)
│   │   ├── Dockerfile.l1-base.j2
│   │   ├── Dockerfile.l2-agent-aider.j2
│   │   ├── Dockerfile.l2-agent-msa.j2
│   │   ├── Dockerfile.l2-agent-oh.j2
│   │   └── Dockerfile.l2-agent-qwen.j2
│   └── aisbench_adapter/                           ← AISBench BaseTask 适配
│       ├── __init__.py
│       ├── task.py                                 ← SwebenchDindTask (BaseTask 子类)
│       ├── result_writer.py                        ← harbor → AISBench schema
│       └── runner.py                               ← subprocess 入口
├── configs/
│   ├── matrix.yaml                                 ← Harbor JobConfig (15 task × 4 agent = 60 trial)
│   └── swebench_dind_3x3.py                        ← AISBench config 示例 (3 cases × 3 agents)
└── scripts/
    ├── start_orchestrator.sh                       ← 启动 DinD 容器 + bind mount
    ├── summarize.py                                ← 汇总 jobs/*/result.json
    └── filter_matrix.py                            ← 子集过滤
```

---

## 🚀 快速使用(在 din_integration/ 内)

```bash
cd din_integration
pip install -e .

# 启 DinD (需要 host 已安装 docker + qemu binfmt)
bash scripts/start_orchestrator.sh

# 跑单个 trial
swebench-dind launch trial --case 11099 --agent aider --wait

# 汇总结果
swebench-dind summarize
```

---

## 📦 跟 AISBench 的集成方式

`aisbench_adapter/task.py` 实现 `SwebenchDindTask`,继承自 `ais_bench.benchmark.tasks.base.BaseTask`,
通过 `ais_bench.benchmark.registry.TASKS.register_module()` 注册。

**AISBench config 示例**见 [configs/swebench_dind_3x3.py](configs/swebench_dind_3x3.py),
3 cases × 3 agents = 9 trial 的最小矩阵。

⚠ **本包不通过 `setup.py` entry_point 注册到 `ais_bench.benchmark_plugins`** —— 因为:
1. swebench-dind 是**重量级 CLI + DinD 镜像**集成,不是传统意义上的 plugin (单文件 import)
2. AISBench plugin 接口需要 `pip install` 后才能 import,本包需要 host 上 docker + QEMU 准备
3. 用户显式选择 "最小集 PR" 路径(参见 doc 23/24 讨论)

如需 AISBench 标准 plugin 形式接入,后续可加 `setup.py` + `entry_points`.

---

## 📚 关联文档

- [mini_matrix/docs/research/18-swebench-dind-complete-project-doc-2026-08.md](../../../mini_matrix/docs/research/18-swebench-dind-complete-project-doc-2026-08.md) — 工程实现细节
- [aisbench/docs/research/24-swebench-dind-integration-plan-A-slim-2026-08.md](../../../aisbench/docs/research/24-swebench-dind-integration-plan-A-slim-2026-08.md) — 方案 A 设计文档(精简版)
- [aisbench/docs/research/23-swebench-dind-integration-into-aisbench-2026-08.md](../../../aisbench/docs/research/23-swebench-dind-integration-into-aisbench-2026-08.md) — 方案 B 设计文档(完整版)

---

## 📊 已验证

- 12 trial 历史 (9 PASS / 75% pass@1) + 1 次 e2e-cli-11099-aider PASS (4min 49s)
- 详见 [mini_matrix/docs/research/18 §11](../mini_matrix/docs/research/18-swebench-dind-complete-project-doc-2026-08.md)

---

## 📝 License

MIT (沿用 swebench-dind 包原始 license)