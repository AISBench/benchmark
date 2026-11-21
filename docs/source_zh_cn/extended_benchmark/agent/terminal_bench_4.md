# Terminal-Bench 4.0 操作适配

Terminal-Bench 4.0（官方 release `v4.0.0`）共 **66 个任务（63 CPU + 3 GPU）**，配套 **157 个任务镜像 + 2 个外部镜像共 159 个**。本文档给出在 AISBench 中基于 Harbor 评测 Terminal-Bench 4.0 的操作步骤，覆盖 **socket（DooD）** 与 **dind（DinD）** 两种运行模式。

> - 通用 Agent 测评的参数说明、实时监控、结果落盘结构见 📚 [基于Harbor框架的Agent 测评](../../base_tutorials/scenes_intro/agent_benchmark.md)，本文只写 TB4.0 特有的适配步骤。
> - 步骤整理自 TB4.0 全流程适配记录 `TB4_RUNTIME_SETUP_FULL_FLOW.md`，章节对应关系见文末附录。

## 1. 资源清单

| 资源 | 获取链接 | 说明 |
| --- | --- | --- |
| 数据集源码 | https://github.com/harbor-framework/terminal-bench/archive/refs/tags/v4.0.0.zip | v4.0.0 源码（tasks/tools），用于核对任务结构，不参与运行期 |
| 预构建任务包 | https://github.com/harbor-framework/terminal-bench/releases/download/v4.0.0/terminal-bench-prebuilt-v4.0.0.tar.gz | 66 个任务的 `task.toml`/`instruction.md`/`tests/`，**解压后即为 Harbor 格式数据集** |
| 官方镜像清单 | https://github.com/harbor-framework/terminal-bench/releases/download/v4.0.0/prebuilt-images.json | 157 个镜像的 digest 清单，用于物料校验 |
| 数据集对应镜像 | https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-4-images/terminal-bench-4.0-images-x86_64.tar | 159 个镜像打包（约 177G，x86_64） |
| 依赖文档 | https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-4-images/dep.zip | 全流程脚本与配置（release 校验、agent-deps 修补、payments verifier、运行脚本等） |

Agent 离线依赖包按 [Agent 依赖准备](../../base_tutorials/scenes_intro/agent_benchmark.md) 中对应 OS 的链接下载，统一放入 `agent-deps/` 目录；TB4.0 任务镜像基础 OS 覆盖 `ubuntu:20.04/22.04/24.04`、`debian:12/13`、`fedora:42`。已验证 agent：`claude-code`（socket 模式）、`mini-swe-agent`（dind 模式）。

**环境要求**：x86_64（镜像包不支持 aarch64）；磁盘 ≥ 400G；docker compose > 2.0.0、buildx > 0.18.0（dind 模式另要求 docker > 20.0.0）；离线环境需自备 PyPI/apt 镜像源（dep.zip 内含相关 patch 脚本）。

**本文实际用到的 dep.zip 内容**（`unzip dep.zip -d dep` 后按下表的相对路径引用）：

| dep 内路径 | 用途 | 使用位置 |
| --- | --- | --- |
| `dep/01-release-validation/validate_release.py` | 校验 release tag、任务数、镜像数与 digest pin | §3 步骤 1 |
| `dep/01-release-validation/check_package_consistency.py` | 交叉核对任务包 / manifest / compose 三者一致 | §3 步骤 1 |
| `dep/03-runtime-container/start_tb4_agent_runtime.sh` | TB4 dep 版 runtime 启动脚本（旧名 `start_agent_runtime.sh` 保留为兼容 wrapper）：默认容器名 `tb4-agent-runtime`，dind 模式会把 dataset/agent-deps/trials 同时同路径挂载（步骤 7 的内层构建依赖此挂载） | §4 步骤 5a、§5 步骤 4b |
| `dep/05-agent-deps/post.sh` | 修补 claude-code 离线 bundle 的 `/root/.local` 路径问题 | §3 步骤 3 |
| `dep/07-payments-verifier/install_payments_verifier.sh` | 幂等安装 payments verifier 的 pip 镜像源补丁；脚本从自身目录取 `payments_pip.py`/`__init__.py`，不再依赖 `/root/code/...` | §6 步骤 6 |
| `dep/runtime-patches/install_runtime_patches.sh` | 安装官方 runtime 镜像缺失的两个 Python 补丁（mini-swe-agent 的 ConfigDict 归一化、logger.exception）；测模型前必做 | §6 步骤 7 |
| `dep/runtime-patches/task-fixes/apply_task_fixes.sh` | 去 `task.toml` digest、修 3 个任务配置、重建 `local/tb4-fixes:*` 验证器镜像 | §6 步骤 7 |
| `dep/run/run_tb4_full_eval.sh` | **推荐的一键启动器**：默认 `N=6`，跑前逐项校验容器、66 个任务、去 digest、任务补丁、runtime 补丁、payments verifier、159 个镜像与磁盘；再做余额/参数探针并启动全量，日志打码、产物脱敏；`CHECK_ONLY=1` 只预检 | §6 步骤 11 |
| `dep/run/run_mini_swe_full_n8.sh`（及 `run/` 下 helper） | 63 任务全量运行封装：镜像校验、余额/参数探针、日志与产物打码；`BASE` 从脚本位置推导 | §6 步骤 11 |
| `dep/build-patch/tb4-manifest.json` | 159 个镜像 RepoTag 清单，供 `run/verify_dind_images.sh` 校验内层 daemon | §6 步骤 11 |
| `dep/run/probe_deepseek_balance.py` | 查询 DeepSeek 账户可用状态和余额 | §6 步骤 8 |
| `dep/README.md` | 换机三个适配点（路径 / 代理地址 / 镜像名） | §3 步骤 1 注 |

## 2. 两种运行模式

| 维度 | socket 模式（默认，推荐） | dind 模式（强隔离） |
| --- | --- | --- |
| 原理 | 挂载宿主机 `/var/run/docker.sock`（DooD），容器内 docker 命令直接操作宿主机 daemon | `--privileged` 在容器内拉起独立 dockerd（DinD），trial 容器只出现在内层 daemon |
| 题目镜像 | 提前在**宿主机** `docker load`，容器内直接可见 | 通过 `--case-images` 传入镜像包（必选），启动脚本自动 load 进**内层** daemon |
| 数据挂载 | 数据集 / agent-deps / 产物目录按宿主机原路径**同路径透传**（保证 trial 容器的 bind mount 在宿主 daemon 上可解析） | 数据集、agent-deps 和产物目录均按宿主机原路径挂载，同时可通过 `/opt/data/terminal-bench-2`、`/opt/agent-resources/deps` 和 `/benchmark/outputs` 访问。数据集和 agent-deps 只读 |
| docker 版本 | 无特殊要求 | > 20.0.0（启动脚本使用 `--cgroupns=host`；宿主 Docker 过老如 18.09 时不可直接使用） |
| 适用场景 | 宿主 daemon 可直接使用，镜像与排障最省事 | 需要强隔离，或宿主 daemon 不宜被评测直接操作 |
| 安全 | 容器内持有 docker.sock 等价于宿主 root 级 daemon 控制权，勿在容器内执行不受信任指令 | trial 容器不影响宿主 daemon，隔离性好 |

## 3. 公共准备步骤（两种模式都执行）

### 步骤 1：物料校验与任务包解压

在存放下载物料的宿主机目录中执行。后续宿主机命令沿用这里定义的变量。

```bash
TB4_WORKDIR="$(pwd -P)"
export TB4_WORKDIR
export TB4_DEP_DIR="$TB4_WORKDIR/dep"
export TB4_TASKS="$TB4_WORKDIR/prebuilt/tasks"
export TB4_AGENT_DEPS="$TB4_WORKDIR/agent-deps"
export TB4_TRIALS="$TB4_WORKDIR/trials"

# 1) 下载第 1 节 5 份物料后，先比对 SHA256（与 GitHub 发布页一致）
sha256sum terminal-bench-prebuilt-v4.0.0.tar.gz prebuilt-images.json

# 2) 解压依赖文档（以下校验/修补/运行脚本均由 dep.zip 提供）
unzip dep.zip -d "$TB4_DEP_DIR"

# 3) 校验 release：release_tag=v4.0.0、66 任务、157 镜像、digest 全 pin
#    脚本来自 dep.zip 的 01-release-validation/ 目录
python3 "$TB4_DEP_DIR/01-release-validation/validate_release.py"

# 4) 解压任务包（prebuilt/tasks/ 即为数据集，含 66 个任务目录）
mkdir -p prebuilt
tar -xzf terminal-bench-prebuilt-v4.0.0.tar.gz -C prebuilt/

# 5) 交叉核对任务包 / manifest / compose 三者一致
#    同样来自 dep.zip 的 01-release-validation/ 目录
python3 "$TB4_DEP_DIR/01-release-validation/check_package_consistency.py"
```

> 步骤 1 用到的 `validate_release.py`、`check_package_consistency.py`，以及下面引用的 `dep/README.md`，都是 `dep.zip` 解压出来的文件；校验脚本默认从当前目录读取 `prebuilt-images.json` 和 `prebuilt/tasks`。一致性检查脚本需要 PyYAML。其他环境适配见 `dep/README.md`。

### 步骤 2：task.toml 去 digest（使用镜像包必做）

镜像包是 `docker save` 产物，`docker load` 后**不保留 RepoDigests**；而 `task.toml` 中 `docker_image = "repo:tag@sha256:..."` 带 digest 引用，加载后 Compose 解析不到会尝试 pull 并失败。Harbor 实际读取的是每个任务的 `task.toml`（只改 `prebuilt-images.json` 无效）。

```bash
# 处理前：应为 132 个带 @sha256 的引用
grep -R 'docker_image.*@sha256' prebuilt/tasks/*/task.toml | wc -l

# 批量去掉 digest（保留 repo:tag）
sed -i -E 's/(docker_image[[:space:]]*=[[:space:]]*"[^"]+)@sha256:[0-9a-f]+"/\1"/' prebuilt/tasks/*/task.toml

# 处理后：应为 0
grep -R 'docker_image.*@sha256' prebuilt/tasks/*/task.toml | wc -l
```

> 如果使用步骤 7 的 `apply_task_fixes.sh all`，脚本会再次执行同样的去 digest；本步骤可以跳过。

### 步骤 3：准备 agent 离线依赖

1. 从 [Agent 依赖准备](../../base_tutorials/scenes_intro/agent_benchmark.md) 的表格下载所选 agent 对应 OS 的离线包，统一放入 `agent-deps/` 目录。
2. 若 bundle 把 agent 装在 `/root/.local`，任务容器以非 root（`nobody`）运行时找不到可执行文件：用 `dep.zip` 内的 `dep/05-agent-deps/post.sh` 方案修补 bundle（迁到 `/opt` 并在 `/usr/local/bin` 生成 wrapper）。

## 4. 模式一：socket（DooD）操作步骤

### 步骤 4a：宿主机导入题目镜像

```bash
docker load -i terminal-bench-4.0-images-x86_64.tar   # 159 个镜像，约 177G，耗时较长

# 镜像包中 2 个外部镜像的 repo 前缀有误，重打 tag：
docker tag harborframework/terminal-bench:4.3.1 apache/kafka-native:4.3.1
docker tag harborframework/terminal-bench:7-alpine redis:7-alpine
```

### 步骤 5a：启动 runtime 容器

```bash
# 脚本来源二选一：
#   - OBS 官方版（socket/DooD 足够）：下面的 wget，文件名 start_agent_runtime.sh
#   - dep 副本（推荐，含 dind 同路径挂载）：dep/03-runtime-container/start_tb4_agent_runtime.sh
wget -O start_agent_runtime.sh https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/scripts/start_agent_runtime.sh

bash ./start_agent_runtime.sh \
    --image ghcr.io/aisbench/agent-runtime:v3.1-20261013-master-ubuntu24.04-py312-x86_64 \
    --dataset "$TB4_TASKS" \
    --agent-deps "$TB4_AGENT_DEPS" \
    --agent mini-swe-agent \
    --name tb4-agent-runtime \
    --trials-dir "$TB4_TRIALS"
```

- socket 模式题目镜像已在宿主机，无需 `--case-images`。
- runtime 镜像 tag 以 [Agent runtime 镜像列表](https://github.com/AISBench/benchmark/wiki/Agent-runtime-images) 最新为准。
- 若直接用 dep 副本，把 `bash ./start_agent_runtime.sh` 换成 `bash "$TB4_DEP_DIR/03-runtime-container/start_tb4_agent_runtime.sh"`，参数相同；包内旧名 `start_agent_runtime.sh` 是兼容 wrapper。

## 5. 模式二：dind（DinD）操作步骤

### 步骤 4b：启动 runtime 容器（自动导入镜像）

```bash
# 必须用 dep.zip 内的 TB4 版（含 dataset/agent-deps/trials 同路径挂载）；
# OBS 上的官方版只挂 /opt/data 等固定路径，步骤 7 的内层构建会找不到上下文
bash "$TB4_DEP_DIR/03-runtime-container/start_tb4_agent_runtime.sh" --mode dind \
    --name tb4-agent-runtime \
    --image ghcr.io/aisbench/agent-runtime:v3.1-20261013-master-ubuntu24.04-py312-x86_64 \
    --dataset "$TB4_TASKS" \
    --case-images "$TB4_WORKDIR/terminal-bench-4.0-images-x86_64.tar" \
    --agent-deps "$TB4_AGENT_DEPS" \
    --agent mini-swe-agent \
    --trials-dir "$TB4_TRIALS"
```

脚本会在容器内拉起独立 dockerd，并自动把镜像包 `docker load` 进内层 daemon（约 177G，耗时数分钟）。dep 副本还会把 dataset / agent-deps / trials **各以同路径再挂载一次**；步骤 7 的 `apply_task_fixes.sh` 通过内层 daemon 构建验证器镜像时，构建上下文必须能在容器内解析到同一绝对路径，因此不要删这三组同路径 `-v`。

### 步骤 5b：校验内层 daemon 与镜像

```bash
docker exec tb4-agent-runtime docker info \
  --format 'version={{.ServerVersion}} driver={{.Driver}}'

# 重打 2 个外部镜像 tag（内层 daemon）
docker exec tb4-agent-runtime docker tag harborframework/terminal-bench:4.3.1 apache/kafka-native:4.3.1
docker exec tb4-agent-runtime docker tag harborframework/terminal-bench:7-alpine redis:7-alpine

# 校验：157 个任务镜像 + 2 个外部镜像
docker exec tb4-agent-runtime docker images --format '{{.Repository}}:{{.Tag}}' \
  | grep -c '^harborframework/terminal-bench:'               # 应为 157
docker exec tb4-agent-runtime docker images --format '{{.Repository}}:{{.Tag}}' \
  | grep -cE '^(apache/kafka-native:4.3.1|redis:7-alpine)$'   # 应为 2
```

## 6. 评测执行

### 步骤 6：安装 payments verifier（出分关键）

`payments-pipeline-fix` 的 verifier 以 `env -i` 执行 pip，会清空 pip 镜像源环境变量导致必挂；`PipConfVerifier` 在 verify 前把华为云镜像源写进 `/etc/pip.conf`（文件可扛过 `env -i`）。

```bash
# 宿主机执行（幂等；runtime 容器每次重建后重跑；脚本已从自身目录取 verifier 文件）
# 脚本来自 dep.zip 的 07-payments-verifier/ 目录
bash "$TB4_DEP_DIR/07-payments-verifier/install_payments_verifier.sh"
```

### 步骤 7：运行前补丁（runtime Python 补丁 + 任务配置，Harbor 运行前必做）

> 本节命令在**宿主机**执行，不要先 `docker exec` 进入 runtime 容器：`runtime-patches/` 没有挂载进容器，容器内看不到这些脚本。脚本本身在宿主机运行；`--mode dind` 时只有 `docker build` / `docker image inspect` 通过 `docker exec tb4-agent-runtime docker ...` 转发到内层 daemon。

先安装 runtime Python 补丁（**测模型前必做**，oracle 不需要）：

```bash
bash "$TB4_DEP_DIR/runtime-patches/install_runtime_patches.sh"
```

该脚本把官方 runtime 镜像缺失的两个文件补进容器：`mini_swe_agent.py`（把 `mmengine.ConfigDict` 经 JSON 归一化后再 `yaml.safe_dump`，否则步骤 11 的嵌套 `llm_kwargs.config` 会报 "cannot represent an object"）和 `logger.py`（补 `Logger.exception()`）。脚本按 sha256 幂等，原文件备份在 `runtime-patches/backups/<时间戳>/`；**容器每次重建后要重跑**。

然后应用任务配置补丁。脚本会：

1. 去掉所有 `task.toml` 中 `docker_image` 的 `@sha256`（镜像包 `docker load` 后不保留 RepoDigests，带 digest 会触发 pull 失败）；
2. 修改 3 个任务：
   - `sglang-qwen-burst`：`tests/test.sh` 的 gitconfig 写法改为 `GIT_CONFIG_*` 环境变量；
   - `photonic-waveguide-routing`：`[verifier].timeout_sec` 300 → 600；
   - `retro-console-soc`：`tests/test_state.py` 的 nextpnr 超时 300 → 600；
3. 在目标 daemon（默认 DinD 内层）重建 `sglang-qwen-burst`、`retro-console-soc` 两个本地验证器镜像，并把对应 `task.toml` 指向 `local/tb4-fixes:*`。

```bash
# MODE=dind or MODE=dood
MODE=dind
BASE="$TB4_WORKDIR" bash "$TB4_DEP_DIR/runtime-patches/task-fixes/apply_task_fixes.sh" \
    all --mode "$MODE" --tasks "$TB4_TASKS"
BASE="$TB4_WORKDIR" bash "$TB4_DEP_DIR/runtime-patches/task-fixes/apply_task_fixes.sh" \
    check --mode "$MODE" --tasks "$TB4_TASKS"
```

DooD 模式将 `MODE` 改为 `dood`，脚本会在宿主 daemon 构建镜像。DinD 模式在内层 daemon 构建。首次应用后再执行 `check`。可用 `--force-build` 重建已有镜像，或用 `all --dry-run` 预览。修改前的备份保存在 `dep/runtime-patches/task-fixes/backups/<时间戳>/`。

### 步骤 8：余额与模型参数探针

手动启动或续测 DeepSeek 评测前，在宿主机检查余额和推理参数。步骤 11 的 DinD 启动脚本会自动做这两项检查。

```bash
read -rsp "DeepSeek API key: " AIS_AGENT_API_KEY; echo
export AIS_AGENT_API_KEY
export API_BASE=https://api.deepseek.com/v1
export MODEL=openai/deepseek-chat
python3 -I "$TB4_DEP_DIR/run/probe_deepseek_balance.py" &&
    python3 -I "$TB4_DEP_DIR/run/probe_model_params.py" "$API_BASE" "$MODEL"
```

余额探针调用 [DeepSeek 余额接口](https://api-docs.deepseek.com/zh-cn/api/get-user-balance/)。确认 `is_available=true`，并核对各币种的 `total_balance` 是否覆盖计划用量。`GET /models` 可用于鉴权、连通性及查看模型列表。参数探针发送一次实际推理请求，使用示例的 mini-swe-agent 参数。换模型服务时，按该服务的方式查询余额，并用实际配置的参数验证推理请求。

### 步骤 9：进入 runtime 容器

```bash
docker exec -it -e TB4_TASKS -e TB4_AGENT_DEPS -e TB4_TRIALS tb4-agent-runtime bash
agent_env harbor
cd "$TB4_TRIALS"
```

两种模式均可在容器内使用上面传入的同路径变量。DinD 的固定挂载路径见第 2 节。oracle 冒烟、手动全量和手动续测在此 shell 执行。使用宿主机启动脚本时，先退出此 shell。

### 步骤 10：oracle canary 冒烟（不消耗模型费用）

```bash
# 容器内执行；oracle 不调用模型，key 仅需非空占位
ais_bench /benchmark/ais_bench/configs/agent_example/harbor_agent_task.py --mode agent \
    -a oracle --model openai/any --api-base http://127.0.0.1:1/v1 --agent-api-key unused \
    -p "$TB4_TASKS" --agent-deps "$TB4_AGENT_DEPS" \
    --include-task-name batched-eval-parity -n 1 -k 1 -y
# 期望 reward 1.0（覆盖 compose/sidecar 链路，verifier 约 10 分钟）；
# 再跑一次 --include-task-name payments-pipeline-fix 验证步骤 6 生效，期望 reward 1.0
# 注：ctr-optimization 的 oracle 在当前环境会挂 2 个断言（任务自身问题），不要用它做 canary
```

### 步骤 11：全量运行（63 个 CPU 任务）

#### 测模型时：修改 `harbor_agent_task.py` 的上下文与推理强度

`ais_bench` 命令里的 `--model`、`--api-base` 只决定模型名和服务地址；**上下文上限、推理强度和单次输出上限写在 `harbor_agent_task.py` 的 `models` 列表里**。测真实模型（不是 oracle）时，先修改容器内：

```text
/benchmark/ais_bench/configs/agent_example/harbor_agent_task.py
```

参考写法：

```python
models = [
    dict(
        abbr="terminus-2",
        agent_name="terminus-2",  # -a/--agent: harbor AgentName 或 module.path:ClassName
        # agent_import_path=None,  # 自定义 agent（module.path:ClassName）
        model_names=["hosted_vllm/qwen3"],  # --model: 模型名称
        api_base="http://0.0.0.0:8080/v1",  # --api-base: 模型服务 base url（统一语义）
        # api_key=None,            # --api-key: 模型服务 API key
        llm_kwargs={  # LLM 调用参数，合并进 agent kwargs
            "config": {  # mini-swe-agent YAML 配置覆盖（写入容器后以 -c 传入）
                "model": {
                    "model_class": "litellm",  # 走标准 chat/completions，不使用 litellm_response
                    "model_kwargs": {
                        "temperature": 1.0,  # 放这里才会真正传给 API
                        "top_p": 1.0,
                        "reasoning_effort": "max",
                        # litellm 1.96.x 的 OpenAI provider 默认不把
                        # reasoning_effort 放进请求；显式放行后原样发送 "max"。
                        "allowed_openai_params": ["reasoning_effort"],
                        "max_tokens": 256000,
                    },
                }
            },
        },
        model_info={  # 模型 token 限制与成本信息
            "max_input_tokens": 128000,
            "max_output_tokens": 256000,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        # 原始 agent 私有参数/环境变量（优先级高于统一参数转换结果）
        # agent_kwargs={},
        # agent_env=None,          # --ae/--agent-env: 传递给 agent 的环境变量
        # ---- 以下为可选的高级 agent 参数（均有对应 CLI 或配置项）----
        # deps_path=None,          # --agent-deps: 离线 agent 依赖包路径
        # n_concurrent=None,       # --n-concurrent-agents: 每个 agent 并发上限
        # skills=None,             # skill 目录或 git 源，可多值
        # mcp_servers=None,        # MCP 服务器配置列表
        # resume_trajectory=False, # 多步任务跨步恢复 agent 会话
        # load_trajectory=None,    # 预加载轨迹文件路径
        # extra_allowed_hosts=None,# 额外允许的 host/IP
        # include_logs=None,       # 需要保留的 agent 日志 glob
        # exclude_logs=None,       # 需要排除的 agent 日志 glob
        # override_timeout_sec=None,
        # override_setup_timeout_sec=None,
        # max_timeout_sec=None,
    )
]
```

关键点：

- `reasoning_effort` 必须放在 `config.model.model_kwargs` 里，并加上 `allowed_openai_params=["reasoning_effort"]`，否则 litellm 的 OpenAI provider 不会把该字段发给服务端。
- `max_tokens` 是单次输出上限；`model_info.max_output_tokens` 要与之一致。
- `model_info.max_input_tokens` 要按服务端实际上下文长度填写，不要照抄示例里的 `128000`；以 `/models` 返回或报错信息为准。
- `temperature`、`top_p` 也必须放在 `model_kwargs` 内才会真正传给 API。
- `agent_name`、`model_names`、`api_base` 按实际使用的 agent 和服务填写，并与命令行 `-a`、`--model`、`--api-base` 保持一致；oracle 冒烟不需要改这些。

#### 方式 A：在容器内手动启动（DooD 或 DinD）

```bash
read -rsp "Model API key: " AIS_AGENT_API_KEY; echo
export API_BASE=https://api.deepseek.com/v1
export MODEL=openai/deepseek-chat
ais_bench /benchmark/ais_bench/configs/agent_example/harbor_agent_task.py --mode agent \
    -a mini-swe-agent \
    --model "$MODEL" \
    --api-base "$API_BASE" \
    --agent-api-key "$AIS_AGENT_API_KEY" \
    -p "$TB4_TASKS" \
    --agent-deps "$TB4_AGENT_DEPS" \
    --exclude-task-name fp8-rmsnorm-gemm jax-speedrun-gpu math-eval-grader \
    -n 6 -k 1 \
    --no-force-build --delete -y
```

#### 方式 B：在宿主机用脚本启动（DinD）

配置好容器内的模型参数后，回到宿主机执行以下命令。脚本会通过 `docker exec` 启动评测。全量运行选 A 或 B 即可。

```bash
BASE="$TB4_WORKDIR" bash "$TB4_DEP_DIR/run/run_tb4_full_eval.sh"
```

默认 `N=6`。脚本会检查数据、补丁、镜像和磁盘，再做余额及参数探针并启动评测。设 `CHECK_ONLY=1` 可只检查环境。也可以用 `BASE="$TB4_WORKDIR" N=6 bash "$TB4_DEP_DIR/run/run_mini_swe_full_n8.sh"` 启动旧版封装。

- 66 任务中 3 个 GPU 任务（`fp8-rmsnorm-gemm`、`jax-speedrun-gpu`、`math-eval-grader`）默认排除；**Pass@1 分数 = 通过数 / 63**。
- 结果目录里的 `harbor_terminal-bench-2` 是 AISBench/Harbor 的 `sub_tasks` 键，**不是 TB 2.x 数据集**；本次实际任务是 v4.0.0 的 66 个 task。不要与 `terminal-bench-2.1-images-x86_64.tar` 混用。
- 换用 `claude-code` 时已验证参数：`--ak reasoning_effort=max --ak max_turns=500`（模型服务走 anthropic 格式接口）。
- 首次运行建议先 `--include-task-name <单个任务>` 或 `--n-tasks 1` 跑通流程。

### 步骤 12：中断续测与补跑

```bash
# 处理异常原因。若为 HTTP 402，充值后重新确认 is_available=true 和余额，再续测。
ais_bench /benchmark/ais_bench/configs/agent_example/harbor_agent_task.py --mode agent \
    ...（同步骤 11 的手动运行参数）... --reuse <时间戳> --purge-exception-cases
```

- verifier 超时类失败用 `harbor trial regrade` 只重跑 verifier（可加 `--verifier-timeout-multiplier`），不要重跑 agent。

### 步骤 13：读取结果

- 汇总：`outputs/default/<时间戳>/summary/summary_<时间戳>.csv`（`correct` / `wrong` / `exception` / `avg_score`）。
- 明细：`results/<模型>/<数据集>/details/result.json` 的 `stats.evals`；逐 case 的 `trial_*/verifier/reward.json`、`ctrf.json`。
- 落盘结构详见 📚 [执行结果与落盘文件](../../base_tutorials/scenes_intro/agent_benchmark.md#执行结果与落盘文件)。

## 附录：与全流程记录的章节对应

| 本文步骤 | `TB4_RUNTIME_SETUP_FULL_FLOW.md` |
| --- | --- |
| 物料校验与任务解压 | §1、§3.1 |
| 镜像导入与去 digest | §3.2、§10.2 |
| socket 模式 | §3.4、§6 |
| dind 模式 | §0.1、§10 |
| payments verifier | §9.1 |
| canary / 余额探针 / 全量 / resume | §9.2~9.5、§9.7 |
