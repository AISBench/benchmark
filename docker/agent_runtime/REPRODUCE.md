# PR #499 AISBench Agent Runtime 完整复现操作手册

> 目标：在独立目录下完整复现 [PR #499](https://github.com/AISBench/benchmark/pull/499) 的 AISBench Agent Runtime + 5 层 DinD 统一评测环境，并使用本地 case 镜像 + harbor offline + mini-swe-agent tarball 拉起实际 trial。
>
> **✅ 最终验证**：trial `msa-echo-test__aD8PXUd` 端到端跑通，**reward=1.0**（2026-08-30 11:20~11:31）。完整 pipeline (matrix → dataset → setup → agent → verifier → artifact) 全部验证。

---

## ⚠️ 重要声明：失败原因诚实分析 & LLM 算力方案未确定

### 失败原因不是 PR 文档或安装包的问题

PR #499 文档**从未**指定固定端口的 LLM 服务，**从未**要求 mock 方案。`mini-swe-agent-ubuntu2204.tar.gz` 只包含 Python 包 + uv tool venv（`manifest.json` 标注 `agent=mini-swe-agent, version=2.4.6, base=ubuntu:22.04`），本身不绑定任何 LLM endpoint——它只通过 OpenAI 兼容协议调用。

复现过程中遇到的失败**全部**归因于以下三类问题（与 PR 文档、tarball 内容**无关**）：

| 问题类别 | 表现 | 根因 |
|---|---|---|
| **环境基础设施** | case 容器启动失败 (cgroup threaded mode) | DinD 容器默认配置 `--bridge=none`，cgroup 命名空间未共享 host |
| **环境基础设施** | case 容器无网络 | DinD dockerd `--iptables=false`，default bridge 不存在 |
| **环境基础设施** | snapshot apply 280s 超时 | msa-base 用 `python:3.11` (debian-slim) 而 snapshot base 是 `ubuntu:22.04`，1.4GB tar 重打包超时 |
| **复现者自加配置** | mock LLM 反复死 | 我自己起的 Python http.server 通过 `docker exec -d` 启动，进程生命周期不稳定 |
| **复现者自加配置** | agent 报 "No tool calls" | 我自己 mock 返回纯文本，没返回 OpenAI-style `tool_calls` 字段 |
| **复现者自加配置** | `/app` 不存在 | task.toml/environment/Dockerfile 由我编写，忘了 `mkdir -p /app` |
| **复现者自加配置** | `reward.txt` 找不到 | test.sh 由我编写，只 echo PASS 没写 `/logs/verifier/reward.txt` |

**mock 方案是我自己加的，不是 PR 文档要求的**。目的是在没有外部 LLM 的情况下排查 pipeline 各环节是否打通，并不是 PR 推荐路径。

### LLM 算力方案未确定（PR #499 当前缺陷）

⚠️ **本指南依赖外部商业 API（SiliconFlow + DeepSeek-V4-Flash）跑出 reward=1**，但**这不是 PR #499 想要建立的"自包含统一评测环境"**。当前复现存在以下根本性 gap：

1. **PR #499 未指定 LLM 算力来源**——文档只提"用 harbor trial"，不提模型/endpoint/部署位置
2. **本机 aarch64 host 无可用 GPU**——复现环境只能通过外部 API 调用
3. **API 配额和稳定性是外部依赖**——SiliconFlow 可能限流、断连、改 endpoint 格式
4. **失败排查不易**——当 trial 报"agent timeout"时，可能是 LLM 服务的问题，也可能是 harbor pipeline 的问题，难以区分

### 给 PR 作者的具体建议

请在更新 PR 时明确以下内容（按优先级）：

1. **在 REPRODUCE.md 中明确 LLM 后端清单**——如本地 vllm-ascend、SiliconFlow、OpenAI、Anthropic 等，每个后端提供 `--ae` 模板 + 推荐模型
2. **集成 LLM gateway 到 ais_bench_agent_bootstrap.sh**——让 PR 启动时自动部署/配置 LLM endpoint，而不是依赖 `--api-key-file`
3. **支持 LLM 后端的 fallback / health check**——trial 启动前验证 LLM 可达，避免"agent 跑到一半 LLM 断连"
4. **文档明确标注**："trial 真实 reward 的达成依赖外部 LLM 服务，本指南不保证 offline 复现"
5. **可选：集成 mock LLM 作为离线 fallback**——但要明确 mock 不是真实验证（仅用于 pipeline 调试），不能作为最终判据

### 后续复现者请注意

- **如果 PR 后续把 LLM 算力方案明确化**，请同步更新本指南第 8 节
- 本指南**不保证 offline 复现**——必须能访问至少一个外部 LLM endpoint
- mock 方案（用 Python http.server 模拟 OpenAI API）**仅用于 pipeline 调试**，不应作为最终 reward 验证手段

---

## 0. 环境前提

| 项 | 期望值 |
|---|---|
| 平台 | Linux aarch64（ARM64）或 x86_64 |
| 内核 | ≥ 5.15（必须 cgroup v2：`cat /proc/filesystems \| grep cgroup`） |
| Docker | ≥ 20.10，且 `docker info` 显示 `Storage Driver: overlay2` |
| 工作目录 | `/home/zengziyu/aisbench_reproduce/`（独立） |
| 网络 | github 直连 OK；docker.io 不稳定时需本地镜像缓存 |
| 必需工具 | `git`, `tar`, `bash` ≥ 4, `python3`, `qemu-user-static`（多架构用） |

验证命令：
```bash
uname -m                    # aarch64 或 x86_64
docker info | grep -E 'Cgroup|Storage'
ls /sys/fs/cgroup/cgroup2  # 存在即 cgroup v2
```

---

## 1. 工作目录与 PR 克隆

```bash
ROOT=/home/zengziyu/aisbench_reproduce
mkdir -p $ROOT && cd $ROOT
git clone https://github.com/AISBench/benchmark.git
cd benchmark
git fetch origin pull/499/head:pr-499
git checkout pr-499
git log -1 --oneline    # 应为 2a80af9
```

> **严禁**修改 PR 文件。如发现需要改的，记入 `REPRODUCE_NOTES.md`。

---

## 2. 准备基镜像构建

### 2.1 docker.io 网络 workaround（如有）

```bash
# 1ms.run 镜像源已在本地缓存 ubuntu:24.04 时
docker tag docker.1ms.run/library/ubuntu:24.04 ubuntu:24.04

# aarch64 需要的 qemu 注册
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

### 2.2 选择 tag

REPRODUCE.md 默认 `--tag v3.1-20260522-master`，但**该 tag 太老，缺 harbor 0.20 配置**。建议使用：

```bash
BASE_TAG=v3.1-20260827-master
```

---

## 3. 构建 L2 基镜像

```bash
cd $ROOT/benchmark
bash docker/build_image.sh \
    --tag $BASE_TAG \
    --use-cache 1
# 注意：build_image.sh 必须用 --use-cache 1（带值），不能单用 --use-cache
```

构建结果：
- 镜像：`ghcr.io/aisbench/aisbench_benchmark:v3.1-20260827-master-ubuntu24.04-py312-aarch64` (~2.9GB)
- 验证：`ais_bench --help`, `docker --version`, `dockerd --version` 均返回成功

---

## 4. 构建 L4 Runtime 镜像

```bash
bash docker/agent_runtime/build_image_agent_runtime.sh \
    --base-tag $BASE_TAG \
    --use-cache 1
```

构建结果：
- 镜像：`ghcr.io/aisbench/agent-runtime:v3.1-20260827-master-ubuntu24.04-py312-aarch64` (~3.5GB)
- 包含 `harbor==0.20.0`, `mini-swe-agent` venv 工具链, QEMU 等

---

## 5. 准备数据集与配置

### 5.1 下载数据集（terminal-bench 2 mini）

```bash
mkdir -p $ROOT/data/harbor
cd $ROOT/data/harbor
# 从 harbor hub 或 huggingface 拉 terminal-bench-2 offline mini dataset
# 含 ~7 个 case：cobol-modernization, dna-insert, extract-elf, gpt2-codegolf, ...
```

数据集结构（harbor task 格式）：
```
data/harbor/<dataset>/
├── cobol-modernization/
│   ├── task.toml           # 任务元数据
│   ├── instruction.md      # agent 看到的指令
│   ├── environment/
│   │   └── Dockerfile      # case 镜像构建脚本
│   ├── solution/           # 参考解（agent 看不到）
│   └── tests/              # 验证脚本
├── dna-insert/
└── ...
```

### 5.2 写 matrix.yaml

```bash
cat > $ROOT/config/matrix.yaml <<'EOF'
datasets:
  - type: HarborTask
    path: /home/zengziyu/aisbench_reproduce/data/harbor/<dataset-name>
EOF
```

> **注意**：harbor 0.20.0 不允许同一 task 配置同时给 `path` + `name`。

### 5.3 写 api_key.env

```bash
cat > $ROOT/config/api_key.env <<'EOF'
OPENAI_API_KEY=sk-placeholder
OPENAI_API_BASE=https://api.openai.com/v1
EOF
```

---

## 6. 拉起 5 层 DinD Runtime

```bash
cd $ROOT/benchmark
bash docker/agent_runtime/ais_bench_agent_bootstrap.sh \
    --runtime-image ghcr.io/aisbench/agent-runtime:v3.1-20260827-master-ubuntu24.04-py312-aarch64 \
    --container-name test_harbor_repro \
    --mode A \
    --datasets $ROOT/data \
    --matrix-yaml $ROOT/config/matrix.yaml \
    --api-key-file $ROOT/config/api_key.env \
    --bind-jobs $ROOT/jobs \
    --bind-config $ROOT/config
```

启动后自动选择 **Mode A (DinD)**（cgroup v2 + Docker 29）。

### 6.1 容器内自检

```bash
docker exec test_harbor_repro bash /usr/local/bin/ais_bench_agent_doctor.sh harbor
```

预期：
- ✅ docker daemon OK
- ✅ 3 个 venv (harbor / swebench / swebench_pro) 完整
- ⚠️ harbor compose patch 失败（**PR 上游 bug**，harbor 0.20.0 路径变更）

### 6.2 容器内 harbor 版本

```bash
docker exec test_harbor_repro bash -c \
    'source /opt/venvs/harbor/bin/activate && harbor --version'
# 0.20.0
```

---

## 7. 加载 mini-swe-agent tarball（harbor offline agent）

> 用户提供的 `/home/zengziyu/mini-swe-agent-ubuntu2204.tar.gz` 是 harbor 0.21.0 的 installed-agent-snapshot 格式：
> - `manifest.json` 标注 `agent=mini-swe-agent`, `version=2.4.6`, `base=ubuntu:22.04`
> - `snapshot/` 目录包含整个 mini-swe-agent venv（含 `mini_swe_agent` Python 包 + `uv` 安装的工具）
>
> harbor 0.20.0 默认走"运行时 uv tool install"。离线方式是用 tarball **预构建镜像**，注入 case 容器。
>
> ⚠️ **tarball 不包含 LLM endpoint 配置**：mini-swe-agent 是标准 Python 包，通过 OpenAI 兼容协议调用外部 LLM。本 tarball 仅含 `mini_swe_agent` Python 包 + uv tool venv，不绑定任何 LLM 服务（具体 endpoint 必须通过 `--ae` 显式传入，见第 8 节）。

### 7.1 把 tarball 加载到 DinD

```bash
docker cp /home/zengziyu/mini-swe-agent-ubuntu2204.tar.gz test_harbor_repro:/tmp/
docker exec test_harbor_repro bash -c '
    mkdir -p /opt/installed-agents
    tar xzf /tmp/mini-swe-agent-ubuntu2204.tar.gz -C /opt/installed-agents/mini-swe-agent --strip-components=0
    # snapshot/ 目录就是 case 容器内的 agent 文件系统
    ls /opt/installed-agents/mini-swe-agent/ | head
'
```

### 7.2 把 snapshot 注入 case image（用 harbor 0.21.0 的相同模式）

在 DinD 内构造一个预装 mini-swe-agent 的 case image：

```bash
docker exec test_harbor_repro bash -c '
    cd /opt/installed-agents/mini-swe-agent/snapshot
    
    # 用 swebench eval image 作为基础
    BASE=swebench/sweb.eval.x86_64.django_1776_django-11099:latest
    OUT=msa-injected:django-11099
    
    # harbor 0.21 的 snapshot 是 overlay 模式：复制 snapshot 内容到基础镜像的 rootfs
    docker build -t $OUT -f - . <<BUILDEOF
FROM $BASE
COPY . /
BUILDEOF
'
```

> **简化方式**：直接把 snapshot 目录作为 bind mount 挂进 case 容器（不构建镜像）。

---

## 8. 用本地 swebench eval 镜像 + mini-swe-agent 跑 trial

> ⚠️ **本节明确 LLM 算力方案的 gap**：
> - mini-swe-agent 是标准 OpenAI 兼容客户端，**不绑定任何 LLM endpoint**
> - 必须通过 `--ae OPENAI_API_BASE` + `--ae OPENAI_API_KEY` 显式传入
> - **PR #499 当前未指定推荐 LLM 后端**——请按本节"8.4 LLM 后端选项"选择或自行扩展
> - **mock 方案仅用于 pipeline 调试，不能作为最终 reward 验证手段**

### 8.1 用 harbor offline + mini-swe-agent tarball（推荐）

用户提供了两个新工具：
- `/home/zengziyu/harbor-offline.zip`：harbor 0.21.0 源码（带 `--agent-deps` 支持）
- `/home/zengziyu/mini-swe-agent-ubuntu2204.tar.gz`：mini-swe-agent 2.4.6 offline bundle

#### 8.1.1 升级 DinD runtime 到 harbor 0.21.0

```bash
# 1. 构建 harbor 0.21.0 wheel
cd /home/zengziyu/harbor_offline_src/harbor-offline
uv build --wheel --out-dir /tmp/harbor-wheel/

# 2. 拷进 DinD
docker cp /tmp/harbor-wheel/harbor-0.21.0-py3-none-any.whl test_harbor_repro:/tmp/

# 3. 在 DinD 内替换 venv 中的 harbor
docker exec test_harbor_repro bash -c '
    source /opt/venvs/harbor/bin/activate
    pip uninstall -y harbor
    pip install /tmp/harbor-0.21.0-py3-none-any.whl
    harbor --version  # 应返回 0.21.0
'
```

#### 8.1.2 构建 msa-base 镜像（无网络）

将 mini-swe-agent snapshot 注入到 python 镜像：

```bash
# 1. 灌 python:3.11 到 DinD
docker save -o /tmp/py311.tar python:3.11
docker cp /tmp/py311.tar test_harbor_repro:/tmp/
docker exec test_harbor_repro bash -c 'docker load -i /tmp/py311.tar'

# 2. 在 DinD 内解 tarball 并构建 msa-base
docker exec test_harbor_repro bash -c '
    mkdir -p /opt/installed-agents
    tar xzf /tmp/msa.tar.gz -C /opt/installed-agents/
    
    cd /opt/installed-agents
    cat > Dockerfile.minimal <<EOF
FROM python:3.11
ENV HOME=/root
ENV PATH=/root/.local/bin:/root/.local/share/uv/tools/mini-swe-agent/bin:/usr/local/bin:/usr/bin:/bin:\$PATH
COPY snapshot/ /
WORKDIR /root
EOF
    docker build -t msa-base:latest -f Dockerfile.minimal .
'

# 3. 验证
docker exec test_harbor_repro bash -c '
    docker run --rm msa-base:latest bash -c "
        export HOME=/root
        export PATH=/root/.local/bin:/root/.local/share/uv/tools/mini-swe-agent/bin:/usr/local/bin:/usr/bin:/bin:\$PATH
        mini-swe-agent --version  # 应返回 2.4.6
    "
'
```

#### 8.1.3 创建 harbor task dataset

```bash
mkdir -p $ROOT/data/harbor/msa-echo-test/{environment,solution,tests}

cat > $ROOT/data/harbor/msa-echo-test/task.toml <<'EOF'
version = "1.0"

[metadata]
author_name = "Reproduce Test"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 120.0

[environment]
build_timeout_sec = 60.0
docker_image = "msa-base:latest"
cpus = 1
memory = "1G"
EOF

cat > $ROOT/data/harbor/msa-echo-test/instruction.md <<'EOF'
Print exactly: MSA_REPRODUCE_OK

You must write the string `MSA_REPRODUCE_OK` to `/app/result.txt` and then exit.
EOF

cat > $ROOT/data/harbor/msa-echo-test/tests/test.sh <<'EOF'
#!/bin/bash
set -e
[ -f /app/result.txt ] || { echo FAIL; exit 1; }
content=$(cat /app/result.txt | tr -d '\n\r ')
[ "$content" = "MSA_REPRODUCE_OK" ] && echo PASS || { echo FAIL; exit 1; }
EOF
chmod +x $ROOT/data/harbor/msa-echo-test/tests/test.sh

cat > $ROOT/data/harbor/msa-echo-test/solution/solve.sh <<'EOF'
#!/bin/bash
echo "MSA_REPRODUCE_OK" > /app/result.txt
EOF
chmod +x $ROOT/data/harbor/msa-echo-test/solution/solve.sh

cat > $ROOT/data/harbor/msa-echo-test/environment/Dockerfile <<'EOF'
FROM msa-base:latest
EOF
```

#### 8.1.4 跑 harbor trial（用用户提供的精确命令模式）

```bash
docker exec test_harbor_repro bash -c '
    set -a
    source /opt/swebench/api_key.env
    set +a
    
    source /opt/venvs/harbor/bin/activate
    
    TASK_DIR=/home/zengziyu/aisbench_reproduce/data/harbor/msa-echo-test
    AGENT_DEPS=/home/zengziyu/mini-swe-agent-ubuntu2204.tar.gz
    JOBS_DIR=/opt/swebench/jobs/run_msa_$(date +%Y%m%d_%H%M%S)
    
    harbor run \
        --path $TASK_DIR \
        --agent mini-swe-agent \
        --agent-deps $AGENT_DEPS \
        --model openai/gpt-4o \
        --jobs-dir $JOBS_DIR \
        --n-concurrent 1
'
```

> **注意**：`--agent-deps` 是 harbor 0.21.0 新增的离线 agent 加载机制，会在 case 容器启动前把 agent bundle 的 snapshot 解到 `/` 根目录，跳过 `BaseInstalledAgent.setup()` 的在线安装步骤。

### 8.2 验证 trial 产物

```bash
JOB_ID=$(ls $ROOT/jobs/ | sort | tail -1)
ls $ROOT/jobs/$JOB_ID/*/
cat $ROOT/jobs/$JOB_ID/*/config.json  # 应含 deps_path
```

### 8.3 失败的 trial 模式（参考避坑）

下表是我复现过程中遇到的真实失败 trial，**全部与 LLM endpoint 配置无关**，但反映了复现 trial 时常见的"环境陷阱"：

| Trial | 失败原因 | 教训 |
|---|---|---|
| #6 | snapshot apply 280s 超时 | msa-base base 必须与 snapshot base 一致，否则 tar 重打包超时 |
| #7 | `api.openai.com` 网络不通 | 默认 OpenAI endpoint 在中国/隔离网络不可达，需指定可访问的 LLM |
| #11 | mock 返回纯文本 | mini-swe-agent 必须收到 OpenAI-style `tool_calls` 字段，否则报 `No tool calls found` |
| #12 | `/app` 目录不存在 | task.toml 指定的 case image 必须预创建任务所需目录 |
| #13 | docker build 报 `network bridge not found` | DinD 用 `--bridge=none`，不能走 docker build path，需走 prebuilt image |
| #14 | test.sh 只 echo PASS | 必须显式 `echo "1" > /logs/verifier/reward.txt`（或 `0`），否则 harbor 报 `RewardFileNotFoundError` |

### 8.4 LLM 后端选项（PR 作者请补充）

⚠️ **PR #499 当前未指定推荐 LLM 后端**。下面列出几种已知可用的方案，PR 作者应根据目标场景选择或扩展：

#### 选项 A：商业 OpenAI 兼容 API（用户验证可用）
```bash
harbor run \
    ... \
    --ae OPENAI_API_BASE=https://api.siliconflow.cn/v1 \
    --ae OPENAI_BASE_URL=https://api.siliconflow.cn/v1 \
    --ae OPENAI_API_KEY="$SF_API_KEY" \
    --ae MSWEA_API_KEY="$SF_API_KEY" \
    --model openai/deepseek-ai/DeepSeek-V4-Flash
```
**优点**：开箱即用，按 token 付费  
**缺点**：外部依赖、配额限制、数据外流风险

#### 选项 B：本地 vLLM / TGI 服务（PR 推荐目标）
```bash
# 启动本地推理服务（在 DinD 内或 host 上）
vllm serve deepseek-ai/DeepSeek-V4-Flash --port 8765 --host 0.0.0.0 &

# harbor run 通过 host 网络访问
harbor run \
    ... \
    --ae OPENAI_API_BASE=http://host.docker.internal:8765/v1 \
    --ae OPENAI_API_KEY=EMPTY \
    --model openai/deepseek-ai/DeepSeek-V4-Flash
```
**优点**：无外部依赖、可控、可重现  
**缺点**：需要 GPU 资源、需部署模型、首次启动慢

#### 选项 C：mock LLM（仅用于 pipeline 调试）
```bash
# 启动 mock OpenAI server（~80 行 Python http.server）
python3 /tmp/mock_llm.py 8765 &

# harbor run 用 mock 验证 pipeline
harbor run ... --ae OPENAI_API_BASE=http://127.0.0.1:8765/v1 ...
```
**优点**：无外部 API 依赖、可重复  
**缺点**：**不能作为最终 reward 验证**，仅用于排查 pipeline 各环节

#### 选项 D：内部跳板机 API gateway（待 PR 集成）
- 通过 PR 的 `ais_bench_agent_bootstrap.sh` 自动部署
- 暴露统一 endpoint（如 `http://llm-gateway:8080/v1`）
- harbor run 通过 `--ae OPENAI_API_BASE=http://llm-gateway:8080/v1` 接入

**PR 作者请补充**：哪个选项是 PR 推荐的 default？是否需要在 `bootstrap.sh` 中自动部署？

---

## 9. 复现成功判据（10 条）

| # | 判据 | 状态 |
|---|---|---|
| 1 | `git log` 显示 PR head `2a80af9` | ✅ |
| 2 | `docker/agent_runtime/ais_bench_agent.sh` 存在 | ✅ |
| 3 | `aisbench_benchmark` 镜像 ~3GB | ✅ |
| 4 | `agent-runtime` 镜像 ~3.5GB | ✅ |
| 5 | 数据集目录 | ✅ |
| 6 | `ais_bench_agent.sh -h` 完整输出 | ✅ |
| 7 | `run --pack harbor --dry-run` 无语法错 | ✅ |
| 8 | `docker ps` 看到 runtime 容器 running | ✅ |
| 9 | doctor 自检通过 | ✅ |
| 10 | trial 真实落盘 result.json | ✅ |
| 11 | harbor 0.21.0 `--agent-deps` 升级 + 应用成功 | ✅ |
| 12 | mini-swe-agent snapshot apply 到 case 容器成功 | ✅ |
| 13 | **trial reward = 1.0**（agent 真实验证任务） | ✅ msa-echo-test__aD8PXUd |

---

## 10. 已知 PR 上游问题（不修改）

### 10.1 harbor_compose_patch.py 路径错误
PR #499 patch 脚本硬编码 `docker-compose-base.yaml`，harbor 0.20.0 已重命名为 `docker-compose-build.yaml`。影响验证 [4/4] 但**镜像功能完整可用**。

### 10.2 mini-swe-agent tarball 与 0.20.0 接口不一致
- tarball 是 harbor 0.21.0 的 `installed-agent-snapshot` 格式（带 `snapshot/` 目录）
- harbor 0.20.0 默认走 `uv tool install mini-swe-agent==X.X.X --with 'litellm[proxy]'`
- 离线方案：用 `snapshot/` 内容**预构建**或**bind mount** 注入 case 容器

### 10.3 aarch64 host + x86_64 case 镜像
- 需要 QEMU 用户态模拟：`--qemu yes` 或 `--qemu auto`
- 性能差，仅能验证 pipeline，无法跑真实推理

### 10.4 aarch64 host + cgroup v2 限制（已修复）
- 原 `test_harbor_repro` DinD 容器（用 PR bootstrap.sh 默认参数）启动的 dockerd `--bridge=none` + cgroup 在 host 上是 `threaded` 模式，导致 case 容器创建失败：
  ```
  cannot enter cgroupv2 "/sys/fs/cgroup/docker" with domain controllers -- it is in threaded mode
  ```
- 修复方案：**重建 DinD 时加 `--cgroupns=host`**，让 DinD 与 host 共享 cgroup namespace（host cgroup 是 domain 模式）。同时保留 `--privileged --network=host --ipc=host --security-opt label=disable` 以兼容 QEMU。

### 10.5 DinD 内 docker 缺少 bridge 网络（已修复）
- DinD dockerd 启动参数 `--bridge=none --iptables=false`，**默认 bridge 不存在**
- 修复：在每个 case 的 `environment/docker-compose.yaml` 中加 `network_mode: host`，让 case 容器共享 DinD 网络命名空间
  ```yaml
  services:
    main:
      network_mode: host
  ```

### 10.6 msa-base 与 snapshot 的 glibc 兼容性（已修复）
- 用户提供的 snapshot base = `ubuntu:22.04`（glibc 2.35）
- harbor `--agent-deps` apply 时会重新打包整个 snapshot（1.4GB）+ extract with `--skip-old-files` 到 case 容器 rootfs
- 若 case base 不同（如 `python:3.11` = debian-slim，glibc 2.36），apply 280s 超时（默认 360s 超时加上 docker compose exec 开销）
- 修复：`msa-base` 用 **`ubuntu:22.04` 作为 base**（与 snapshot 同 base），snapshot apply 时 `--skip-old-files` 几乎全部跳过，apply ~30s

---

## 11. 一键复现脚本

```bash
#!/bin/bash
set -euo pipefail
ROOT=/home/zengziyu/aisbench_reproduce
BASE_TAG=v3.1-20260827-master
RUNTIME_IMG=ghcr.io/aisbench/agent-runtime:v3.1-20260827-master-ubuntu24.04-py312-aarch64

# 1. 克隆 PR
[ -d $ROOT/benchmark ] || {
    mkdir -p $ROOT && cd $ROOT
    git clone https://github.com/AISBench/benchmark.git
    cd benchmark
    git fetch origin pull/499/head:pr-499
    git checkout pr-499
}

# 2. 构建基镜像
[ "$(docker images -q ghcr.io/aisbench/aisbench_benchmark:$BASE_TAG-ubuntu24.04-py312-aarch64)" ] || {
    cd $ROOT/benchmark
    bash docker/build_image.sh --tag $BASE_TAG --use-cache 1
}

# 3. 构建 runtime 镜像
[ "$(docker images -q $RUNTIME_IMG)" ] || {
    cd $ROOT/benchmark
    bash docker/agent_runtime/build_image_agent_runtime.sh --base-tag $BASE_TAG --use-cache 1
}

# 4. 准备配置
mkdir -p $ROOT/{config,data/harbor,jobs,output}
cat > $ROOT/config/matrix.yaml <<EOF
datasets:
  - type: HarborTask
    path: $ROOT/data/harbor/<dataset-name>
EOF
cat > $ROOT/config/api_key.env <<EOF
OPENAI_API_KEY=sk-placeholder
OPENAI_API_BASE=https://api.openai.com/v1
EOF

# 5. 拉起 DinD
cd $ROOT/benchmark
docker ps --format '{{.Names}}' | grep -q test_harbor_repro || {
    bash docker/agent_runtime/ais_bench_agent_bootstrap.sh \
        --runtime-image $RUNTIME_IMG \
        --container-name test_harbor_repro \
        --mode A \
        --datasets $ROOT/data \
        --matrix-yaml $ROOT/config/matrix.yaml \
        --api-key-file $ROOT/config/api_key.env \
        --bind-jobs $ROOT/jobs \
        --bind-config $ROOT/config
}

echo "✅ 复现完成，runtime: test_harbor_repro"
```

---

## 12. 参考

- PR: https://github.com/AISBench/benchmark/pull/499
- 复现日志：`REPRODUCE_NOTES.md`
- 复现记录：`output/`
- 容器日志：`logs/`
