# AISBench Agent Runtime — 5 层 DinD 统一测评环境

为 AISBench Agent 测评（Harbor Terminal-Bench、SWE-bench、SWE-bench Pro）提供
**镜像构建 + 容器启动 + 命令执行** 的完整运行时方案。

> 本 PR 是 AISBench/benchmark 的社区运行时补充，不改动核心评测逻辑。

## 架构总览

```
宿主机用户
  │
  ├─ ais_bench_agent.sh build     ← 一键构建 (v3 H 批)
  │   └─ build_image_agent_runtime.sh [→ build_l2_baked_image.sh]
  │
  └─ ais_bench_agent.sh run       ← 拉起 + 自检 + 跑测试 (v3 H 批)
      │
      └─ bootstrap.sh → docker run 启动 runtime 容器
            │
            ├─ 模式 A (DinD): --privileged --net=host
            │     └─ 容器内 dockerd → harbor → trial 容器
            └─ 模式 B (Socket): -v /var/run/docker.sock
                  └─ 共享宿主 dockerd → harbor → trial 容器
            │
            ├─ datasets bind mount (宿主路径 = 容器内路径)
            ├─ case 镜像 tar 自动 docker load
            └─ 3 个隔离 venv: /opt/venvs/{harbor, swebench, swebench_pro}

容器内
  ├─ ais_bench_agent_doctor.sh <pack>    ← L1 自检
  ├─ agent_env <pack>                    ← 激活 venv
  ├─ ais_bench_agent_run.sh              ← harbor jobs start 包装
  ├─ ais_bench_agent_watch.sh            ← 阻塞等 results.json
  ├─ ais_bench_agent_summarize.sh        ← 聚合 → md/csv/json
  └─ ais_bench_agent_orchestrator_status.sh ← 5 段状态查询
```

## 分层职责

| 层 | 脚本/文件 | 职责 |
|---|---|---|
| **入口** | `ais_bench_agent.sh` | 统一 facade：build / run / status / watch / summarize / doctor |
| **L4 镜像** | `Dockerfile.agent-runtime` + `build_image_agent_runtime.sh` | 在 aisbench_benchmark 基镜上追加 3 个隔离 venv |
| **L5 启动** | `ais_bench_agent_bootstrap.sh` + `safe_start_...` | 一键起 runtime 容器 (DinD/Socket + 挂数据集 + 加载 case tar) |
| **L3 调度** | `ais_bench_agent_{run,watch,summarize,orchestrator_status}.sh` | 在容器内调 harbor jobs / 等结果 / 汇总 |
| **L2 加速** | `build_l2_baked_image.sh` + 5 个 Jinja2 模板 | 预烤 agent 到 case 镜像，跳过 trial 内装包 |
| **L1 自检** | `doctor.sh` | 静态校验 docker / venv / pack / 资源（秒级） |

## 快速拉起流程

> 📘 端到端复现手册（含 harbor offline + mini-swe-agent tarball + 真实 trial 验证 + 失败诚实声明）见 [REPRODUCE.md](REPRODUCE.md)。

### 1. 构建 runtime 镜像（宿主机）

```bash
# 标准构建（用默认 harbor==0.20.0）
bash docker/agent_runtime/ais_bench_agent.sh build \
    --base-tag v3.1-20260522-master --push

# 用自定义 harbor wheel（如 harbor-offline，替换默认 harbor）
bash docker/agent_runtime/ais_bench_agent.sh build \
    --base-tag v3.1-20260522-master \
    --harbor-wheel /path/to/harbor-offline.whl --push

# 同时构建 L2 baked images
bash docker/agent_runtime/ais_bench_agent.sh build \
    --base-tag v3.1-20260522-master --l2
```

### 2. 跑测评（宿主机）

```bash
# Harbor Terminal-Bench（自动启容器 + doctor 自检 + 执行测试）
bash docker/agent_runtime/ais_bench_agent.sh run \
    --pack harbor \
    --datasets /data/harbor/mini-0.10/terminal-bench-2-offline-selected_0.10 \
    --matrix-yaml /opt/config/matrix.yaml \
    --api-key-file /opt/config/api_key.env

# SWE-bench verified mini（--split 自动派生 config 文件）
bash docker/agent_runtime/ais_bench_agent.sh run \
    --pack swebench --split verified_mini \
    --datasets /data/swebench/verified \
    --matrix-yaml /opt/config/matrix.yaml

# 自定义命令（透传到容器内，不自动推导）
bash docker/agent_runtime/ais_bench_agent.sh run \
    --pack harbor \
    --datasets /data/harbor/mini \
    --command "harbor jobs start -c /opt/swebench/config/my_matrix.yaml -n 2"

# 离线模式（内网 / 无 ghcr.io 访问）
bash docker/agent_runtime/ais_bench_agent.sh run \
    --pack harbor \
    --datasets /data/harbor \
    --runtime-tar /opt/aisbench/agent-runtime.tar.gz \
    --case-tar /opt/aisbench/case-images.tar.gz
```

### 3. 查询 / 等待 / 汇总（宿主机）

```bash
bash docker/agent_runtime/ais_bench_agent.sh status
bash docker/agent_runtime/ais_bench_agent.sh watch <job-name>
bash docker/agent_runtime/ais_bench_agent.sh summarize <job-name>
```

## Pack 智能化：`--pack` + `--split` 自动派生 config

用户只需传 `--pack` 和 `--split`，wrapper 自动选择正确的 ais_bench config 文件：

| `--pack` | `--split` | 自动推导 config 路径 |
|---|---|---|
| `harbor` | (不需要) | `ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py` |
| `swebench` | (默认) / `lite` | `.../swe_bench_examples/mini_swe_agent_swe_bench_lite.py` |
| `swebench` | `verified` | `.../swe_bench_examples/mini_swe_agent_swe_bench_verified.py` |
| `swebench` | `verified_mini` | `.../swe_bench_examples/mini_swe_agent_swe_bench_verified_mini.py` |
| `swebench` | `full` | `.../swe_bench_examples/mini_swe_agent_swe_bench_full.py` |
| `swebench` | `multilingual` | `.../swe_bench_examples/mini_swe_agent_swe_bench_multilingual.py` |
| `swebench_pro` | (默认) / `mini` | `.../swe_bench_pro_examples/mini_swe_agent_swe_bench_pro_mini.py` |
| `swebench_pro` | `full` | `.../swe_bench_pro_examples/mini_swe_agent_swe_bench_pro_full.py` |

非法 `--split` 会明确报错并提示合法值列表。

## 命令透传设计

`ais_bench_agent.sh run --command "..."` 将用户在宿主机交付的任意命令原样透传到容器内执行：

```
用户 CLI (宿主机)
  │  ais_bench_agent.sh run --command "harbor jobs start ..."
  ▼
wrapper 自动完成: 派生 config → 检查/启容器 → bootstrap → doctor → docker exec
  │
  ▼
容器内: bash -c "harbor jobs start ..."
  │
  ▼
harbor → DinD → trial 容器 → /opt/swebench/jobs/<job>/result.json
```

## 目录结构（核心文件）

```
docker/agent_runtime/
├── ais_bench_agent.sh                  ← 统一入口 (NEW)
├── Dockerfile.agent-runtime             ← L4 runtime 镜像
├── build_image_agent_runtime.sh         ← 镜像构建脚本
├── build_l2_baked_image.sh              ← L2 baked image 构建
├── ais_bench_agent_bootstrap.sh         ← L5 一键起容器
├── ais_bench_agent_entrypoint.sh        ← 容器 ENTRYPOINT
├── ais_bench_agent_run.sh               ← L3 harbor jobs 包装
├── ais_bench_agent_watch.sh             ← L3 等结果
├── ais_bench_agent_summarize.sh         ← L3 汇总
├── ais_bench_agent_orchestrator_status.sh ← L3 状态查询
├── safe_start_ais_bench_agent_bootstrap.sh ← watchdog 包装
├── doctor.sh                            ← L1 自检
├── scripts/
│   ├── filter_matrix.py                 ← matrix 过滤
│   └── summarize.py                     ← 汇总逻辑
├── packs/
│   ├── harbor.yaml
│   ├── swebench.yaml
│   └── swebench_pro.yaml
├── patches/
│   └── harbor_compose_patch.py
└── dockerfiles/
    ├── Dockerfile.l1-base.j2
    ├── Dockerfile.l2-agent-aider.j2
    ├── Dockerfile.l2-agent-msa.j2
    ├── Dockerfile.l2-agent-oh.j2
    ├── Dockerfile.l2-agent-qwen.j2
    └── README.md
```

## 所有提交概览

### PR410 baseline (12 commits)
`b92b18c` → `25af9e0`: 基镜像 docker 安装、ais_bench configs 更新、Dockerfile + build 脚本 + bootstrap.sh + doctor.sh + packs + patches + 文档

### v3 A 批 — L4 镜像改造 (5 commits)
`7875023` A1: ARM64 host 适配 · `abb9024` A2: harbor 0.6.1 → 0.20.0 · `c2663f9` A3: DinD registry-mirrors 参数化 · `848ec98` A4: ENTRYPOINT + binfmt + daemon.json · `ce8a1f9` A5: /opt/swebench 预创建

### v3 B 批 — L5 launcher 改造 (6 commits)
`21200a4` B1: bootstrap.sh 新增 8 个参数 · `7f36db1` B2: --data-image · `3f89ec6` B3: bind mount + env 接入 · `ead9f51` B4: --production --restart unless-stopped · `c6fe5ad` B5: DOCKER_DEFAULT_PLATFORM · `3a6ded5` B6: 推荐 harbor jobs start

### v3 C 批 — L3 调度接入 (4 commits)
`05e6221` C1: run.sh + filter_matrix.py · `6b17bcd` C2: watch.sh · `08e3f0f` C3: summarize.sh + summarize.py · `638d17b` C4: orchestrator_status.sh

### v3 D 批 — L2 baked image (3 commits)
`d65f44d` D1: build_l2_baked_image.sh · `f0d0871` D2: 5 个 Jinja2 模板 · `c59c4f8` D3: bootstrap.sh --l2-image

### v3 E/F/G/H 批 (4 commits)
`6cf5edb` E: bootstrap.sh --qemu · `43ebb27` F: 移除 --cgroupns=host · `085a826` G: safe_start watchdog · `1e523aa` **H: ais_bench_agent.sh 统一入口 + harbor wheel 可替换**

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)