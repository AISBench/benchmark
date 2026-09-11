# AISBench harbor agent-runtime 操作文档（harbor-only / 双模式）

本文档面向在**目标机**上从零跑通 harbor（terminal-bench-2）agent 测评的操作者。

`start_agent_runtime.sh` 支持两种模式（`--mode` 参数）：

| 模式 | 默认 | 原理 | 隔离性 | case 镜像准备 |
|---|---|---|---|---|
| `socket`（DooD） | **是** | 挂载宿主机 `/var/run/docker.sock`，trial 容器由**宿主机 daemon** 拉起 | 弱（容器内等价于宿主机 daemon 控制权） | **零准备**，宿主机已有镜像直接可见 |
| `dind`（DinD） | 否 | `--privileged` 容器内拉起独立 dockerd，trial 容器只存在于里层 | 强（与宿主机 Docker 环境完全隔离） | 需要 case 镜像包 |

无强隔离需求直接用默认 socket 模式（最简路径）；安全要求高的环境用 `--mode dind`。

---

## 0. 前提与资产

### 0.1 目标机环境要求

| 项 | 要求 |
|---|---|
| OS | Linux（宿主 cgroup v2；openEuler/RHEL 系已验证 seccomp 兼容） |
| Docker | 已安装并可运行（`docker info` 正常） |
| 架构 | aarch64（镜像为 arm64 版） |
| 网络 | 目标机需能访问模型服务（`--api-base`）；**评测过程本身不需要外网** |
| 权限 | socket 模式：普通 docker 权限即可；dind 模式：需能执行 `docker run --privileged` |

自检命令：

```bash
docker info | grep -i "cgroup version"    # 预期：Cgroup Version: 2
uname -m                                   # 预期：aarch64
```

### 0.2 资产清单（仓库 `docker/agent_runtime/` 目录 + 内部网盘镜像包）

| 文件 | 用途 | 是否必需 |
|---|---|---|
| `aisbench-agent-runtime-harbor-only.tar.gz` | 打包好的 agent-runtime 镜像（860MB，内部网盘下载） | 路径 A（推荐）必需 |
| `start_agent_runtime.sh` | 一键启动脚本（默认 socket 模式，`--mode dind` 切强隔离） | 必需 |
| `sync_images.sh` | dind 模式增量补充：宿主机 → 容器内 daemon 镜像直传 | 可选（仅 dind 用） |
| `Dockerfile.agent-runtime` | 镜像构建文件 | 仅路径 B（重建）需要 |
| `patches/harbor_compose_patch.py` | harbor compose 模板 seccomp patch | 仅路径 B 需要（已烧进镜像） |
| 基镜像 `ghcr.io/aisbench/aisbench_benchmark:v3.1-20260903-master-ubuntu24.04-py312-aarch64` | 构建 agent-runtime 镜像的底座 | 仅路径 B 需要 |

### 0.3 需要自备的素材

| 素材 | 说明 | 传入脚本的参数 | 适用模式 |
|---|---|---|---|
| tb2 数据集目录 | terminal-bench-2 任务集目录 | `--dataset` | 两模式 |
| case 镜像包 | `docker save` 打出的 tar / tar.gz | `--case-images` | **仅 dind**（socket 直接用宿主机镜像） |
| agent deps bundle 目录 | agent 离线依赖包（可现做，见第 5 步） | `--agent-deps` | 两模式 |

> **数据集目录结构**：每个任务一个子目录，内含 `task.toml`、`instruction.md`、
> `environment/`（docker-compose 或 Dockerfile）、`tests/`。
>
> **镜像包内容**（仅 dind）：数据集中所有 case 用到的运行镜像（task.toml
> `[environment]` `docker_image` 字段对应的镜像），供内层 daemon 离线使用。

---

## 1. 导入 agent-runtime 镜像（路径 A，推荐）

```bash
gunzip -c aisbench-agent-runtime-harbor-only.tar.gz | docker load
```

预期输出：

```
Loaded image: aisbench/agent-runtime:harbor-only
```

确认：

```bash
docker images aisbench/agent-runtime
# REPOSITORY                  TAG           IMAGE ID       CREATED         SIZE
# aisbench/agent-runtime      harbor-only   ...            ...             约 2.5GB
```

### 1b. （可选）路径 B：从基镜像重建镜像

仅当需要改动 Dockerfile 或无法使用离线包时执行。**构建过程需访问华为 PyPI 镜像源与 gh-proxy.com（GitHub 镜像加速）**：

```bash
cd docker/agent_runtime          # 仓库 checkout 目录内
docker build --network host \
    -f Dockerfile.agent-runtime \
    -t aisbench/agent-runtime:harbor-only \
    --build-arg BASE_IMAGE=ghcr.io/aisbench/aisbench_benchmark:v3.1-20260903-master-ubuntu24.04-py312-aarch64 \
    .
```

构建完成后跳过第 1 步，直接进入第 2 步。

---

## 2. 准备素材

### 2.1 tb2 数据集目录

放到任意宿主路径，例如 `/data/terminal-bench-2`。要求目录下直接是各任务子目录。

> 挂载方式按模式不同：
> - **socket 模式**：按「宿主机原路径透传」挂载（容器内外同路径，如 `/data/terminal-bench-2`），
>   因此 `--dataset` 必须传**真实绝对路径**（不能是含符号链接的缩写路径，脚本会校验）。
> - **dind 模式**：挂到容器内固定路径 `/opt/data/terminal-bench-2`（只读）。

### 2.2 case 镜像包（仅 `--mode dind` 需要）

> **socket 模式完全跳过本节**：trial 容器由宿主机 daemon 拉起，宿主机上已有的
> case 镜像直接可见，零导入。宿主机缺镜像时 `docker pull` 或 `docker load` 离线包即可。

在有这些镜像的机器上打包（任意一台能 `docker pull`/已有镜像的机器）：

```bash
# 单镜像
docker save alexgshaw/headless-terminal:20260430 | gzip > case-images.tar.gz

# 多镜像合并一个包
docker save -o case-images.tar \
    alexgshaw/headless-terminal:20260430 \
    <其他 case 镜像>...
gzip case-images.tar
```

> gzip 压缩与否均可，脚本按 `/opt/agent-resources/images/case-images.tar` 挂载后
> 直接 `docker load -i`（load 原生支持读 gzip 流）。

**增量补充（仅 dind）**：不想重打全量包时，宿主机执行 `sync_images.sh`，通过
`docker save | docker exec -i docker load` 管道把缺失镜像直传进容器内 daemon
（不在宿主机落中间文件）：

```bash
./sync_images.sh --list              # 只看差异清单
./sync_images.sh --all               # 宿主机全量差集直传
./sync_images.sh fix-git regex-log   # 按任务名同步
```

### 2.3 agent deps bundle 目录

先建一个空目录即可（推荐在容器内现做，见第 5 步）：

```bash
mkdir -p /data/agent-deps
```

若已有离线 bundle（如 `terminus-2-debian-12-aarch64.tar.gz`），直接放进该目录。

**bundle 命名规则（重要）**：`<agent>-<os-id>-<version-id>-<arch>.tar.gz`，
运行时 harbor 按 trial 容器内实测的 `/etc/os-release` + `uname -m` 精确匹配文件名，
例如 case 镜像是 Debian 12 底 → 需要 `terminus-2-debian-12-aarch64.tar.gz`。
bundle 构建时的 `--base-image` 必须与 case 运行镜像**同族**（glibc/系统库一致）。

---

## 3. 一键启动 agent-runtime 容器

> **注意**：下文命令中的 `/data/terminal-bench-2`、`/data/agent-deps` 均为**示例路径**，
> 请替换成你机器上真实存在的绝对路径（脚本会预检，目录不存在会直接报错退出）。

```bash
cd docker/agent_runtime          # 仓库 checkout 目录内
chmod +x start_agent_runtime.sh   # 首次执行前

# ---- 默认 socket 模式（推荐，无需镜像包）----
./start_agent_runtime.sh \
    --image aisbench/agent-runtime:harbor-only \
    --dataset /data/terminal-bench-2 \
    --agent-deps /data/agent-deps \
    --agent terminus-2

# ---- dind 模式（强隔离，需要镜像包）----
./start_agent_runtime.sh --mode dind \
    --image aisbench/agent-runtime:harbor-only \
    --dataset /data/terminal-bench-2 \
    --case-images /data/case-images.tar.gz \
    --agent-deps /data/agent-deps \
    --agent terminus-2
```

可选参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mode` | `socket` | `socket`（DooD，复用宿主机 daemon）/ `dind`（强隔离，独立 daemon） |
| `--name` | `agent-runtime` | 容器名 |
| `--trials-dir` | `~/harbor-trials` | 评测产物持久化到宿主机的目录（dind 模式对应容器内 `/benchmark/outputs`；socket 模式容器内外同路径） |
| `--docker-sock` | `/var/run/docker.sock` | 仅 socket 模式：宿主机 docker socket 路径 |

预期输出（关键行）——socket 模式：

```
==> 启动容器 agent-runtime（socket 模式，复用宿主机 daemon）
<容器ID>
==> 校验容器内 docker 通道（应直连宿主机 daemon）
==> 宿主机 daemon 就绪（容器内可见 alexgshaw/*:20260430 镜像 89 个）

======================================================================
 模式: socket ｜ 容器已就绪，进入容器：
     docker exec -it agent-runtime bash
======================================================================
```

dind 模式额外输出：

```
==> 启动容器 agent-runtime（dind 模式，privileged / 独立 daemon）
<容器ID>
==> 容器内启动 dockerd
==> 内层 dockerd 就绪
==> 导入 case 镜像包（可能需要数分钟）
Loaded image: alexgshaw/headless-terminal:20260430
==> 内层 daemon 镜像列表：
REPOSITORY                    TAG        SIZE
alexgshaw/headless-terminal   20260430   174MB
```

脚本自动完成的事：

- 公共：写 `/root/RUN_HINT.txt`（登录容器自动打印，内含按模式生成的命令模板）。
- socket：挂载 docker.sock → 校验容器内 `docker info` → 统计可见 case 镜像数
  （为 0 时 WARN 提醒宿主机先备镜像）。
- dind：`--privileged + --cgroupns=host` 启动 → 容器内拉起 dockerd（180s 就绪等待）
  → `docker load` case 镜像包。

> **socket 模式路径约束**：数据集 / agent deps / 产物目录按「宿主机原路径透传」挂载
> （容器内外同路径），保证 harbor 为 trial 容器生成的 bind mount 在宿主机 daemon 上
> 能正确解析。因此 `--dataset`、`--agent-deps` 必须传真实绝对路径（脚本会校验）。

---

## 4. 进入容器并自检

```bash
docker exec -it agent-runtime bash
```

登录后会自动打印 RUN_HINT（含你的 agent 名与命令模板）。依次自检：

```bash
# 1) docker 通道正常
docker info | head -5        # socket 模式：宿主机 daemon 信息；dind 模式：内层独立 daemon
docker images                # case 镜像在列表里（socket 模式应看到宿主机全部镜像）

# 2) harbor venv 可用
agent_env harbor
harbor --version                                    # 0.21.0（AISBench fork）
python -c "from harbor.models.trial.config import AgentConfig; AgentConfig(deps_path='/tmp'); print('deps_path OK')"

# 3) 挂载素材在位（路径按模式不同，以 RUN_HINT 打印为准）
# socket 模式：宿主机原路径（示例值 = 启动参数）
ls /data/terminal-bench-2            # --dataset 的值
ls /data/agent-deps                  # --agent-deps 的值
# dind 模式：固定挂载点
ls /opt/data/terminal-bench-2        # 数据集任务目录
ls /opt/agent-resources/deps         # deps bundle（可为空，下一步构建）
ls /opt/agent-resources/images/      # case-images.tar
```

---

## 5. （bundle 目录为空时）容器内构建 agent deps bundle

在容器内（`agent_env harbor` 已激活）执行（`--out` 按模式取对应 deps 目录）：

```bash
# dind 模式（固定挂载点）
harbor agent-deps build \
    -a terminus-2 \
    --base-image alexgshaw/headless-terminal:20260430 \
    --out /opt/agent-resources/deps \
    -m mock-model \
    --host-network

# socket 模式（--out 填启动时 --agent-deps 的宿主机原路径）
harbor agent-deps build \
    -a terminus-2 \
    --base-image alexgshaw/headless-terminal:20260430 \
    --out /data/agent-deps \
    -m mock-model \
    --host-network
```

说明：

| 参数 | 说明 |
|---|---|
| `-a` | agent 名，与后续 ais_bench `-a` 一致 |
| `--base-image` | **与 tb2 case 运行镜像同族**（最稳妥：直接用 case 镜像本身。dind 模式下已 load 进内层 daemon 不会重新拉取；socket 模式下直接用宿主机镜像） |
| `--out` | 输出目录（按模式取上面对应路径）；产物自动命名为 `terminus-2-<os>-<ver>-<arch>.tar.gz` |
| `-m` | terminus-2 必填（构造要求），构建过程**不调用 LLM**，任意值即可 |
| `--host-network` | 构建容器共享网络，安装依赖可出网 |

预期输出末尾：

```
Built agent deps bundle: /data/agent-deps/terminus-2-debian-12-aarch64.tar.gz
```

构建产物落在宿主机 `--agent-deps` 挂载目录（该挂载为读写），下次可直接复用。

> 若多种 case 底镜像族不同（如 debian-12 和 ubuntu-24.04 混杂），按各底镜像分别构建，
> bundle 全放同一目录，运行时按 os-release 自动匹配。

---

## 6. 运行评测

仍在容器内、`agent_env harbor` 已激活。

**socket 模式**：先 `cd` 进产物目录（保证 ais_bench work_dir 落在同路径透传挂载上，
trial 容器才能按宿主机路径正确 bind mount），配置路径用容器内绝对路径，
`--agent-deps` / `-p` 用启动时的宿主机原路径：

```bash
cd <启动时 --trials-dir 的宿主机路径>   # 如 ~/harbor-trials
ais_bench /benchmark/ais_bench/configs/agent_example/harbor_agent_task.py \
    --mode agent \
    -a terminus-2 \
    --agent-deps /data/agent-deps \
    -p /data/terminal-bench-2 \
    --n-tasks 1 \
    --api-base http://<模型服务IP>:<端口>/v1 \
    --model openai/<模型名>
```

**dind 模式**（容器内固定路径）：

```bash
ais_bench ais_bench/configs/agent_example/harbor_agent_task.py \
    --mode agent \
    -a terminus-2 \
    --agent-deps /opt/agent-resources/deps \
    -p /opt/data/terminal-bench-2 \
    --n-tasks 1 \
    --api-base http://<模型服务IP>:<端口>/v1 \
    --model openai/<模型名>
```

**只跑指定的 case**（任选其一，可与上面参数自由组合）：

```bash
# 方式 1：--include-task-name 按 task 名过滤（推荐）
#   匹配对象 = task.toml 的 [task] name 字段（如 sys-report、terminal-bench/fix-git）
#   支持 glob 通配；可空格分隔多个，也可重复传参
--include-task-name sys-report
--include-task-name "terminal-bench/fix-git" "*hello*"

# 方式 2：--exclude-task-name 反向排除（匹配规则同上），可与 include 组合

# 方式 3：-p 直接指向单个 case 目录（等效于只跑该 case）
-p /data/terminal-bench-2/sys-report
```

> `--n-tasks N` 表示"最多取 N 个任务"，与过滤参数叠加生效：
> 先按 include/exclude 筛选，再截取前 N 个。要精确跑某一个 case，
> 用 `--include-task-name` 而不要只靠 `--n-tasks 1`（后者取的是数据集里第一个任务）。

参数说明：

| 参数 | 说明 |
|---|---|
| `-a` | agent 名（须与 deps bundle 前缀、第 5 步构建的 agent 一致） |
| `--agent-deps` | deps bundle 目录（dind 固定 `/opt/agent-resources/deps`；socket 为启动时的宿主原路径，文件夹模式自动匹配 bundle） |
| `-p` | 数据集目录（dind 固定 `/opt/data/terminal-bench-2`；socket 为启动时的宿主原路径）；**也支持直接指向单个 case 子目录** |
| `--include-task-name <名>` | **只跑指定 case**：匹配 `task.toml` 的 `[task] name`，支持 glob、可多个（如 `--include-task-name sys-report` 或 `"terminal-bench/*"`） |
| `--exclude-task-name <名>` | 排除匹配的 case（规则同 include），可与 include 组合 |
| `--n-tasks` | 最多抽取的任务数（与过滤参数叠加）；**首次冒烟务必 `1`** |
| `--api-base` | OpenAI 兼容服务地址（以 `/v1` 结尾） |
| `--model` | **必须带 litellm provider 前缀**；OpenAI 兼容接口一律 `openai/<模型名>`，如 `openai/Qwen2.5-7B-Instruct` |
| `--agent-api-key <key>` | 模型服务需要鉴权时传（可选） |
| `--n-concurrent N` | 并发 trial 数（默认配置 5，可按资源调整） |
| `-k N` | 每个 task 的尝试次数 |
| `--ae KEY=VALUE` | 注入 trial 容器/tmux 会话的环境变量，可重复（如 `--ae OPENAI_API_KEY=xxx`） |
| `--ak KEY=VALUE` | 传给 agent 的额外 kwarg，可重复（进阶用法） |
| `--agent-import-path <m:C>` | 自定义 agent 导入路径（`module.path:ClassName`），配合 `-a` 使用 |
| `-d <name@version>` | 远端数据集（registry 或包），与 `-p` 二选一 |
| `-e <环境>` | harbor 环境类型（docker/daytona/e2b/modal…，默认 docker） |
| `--timeout-multiplier X` | case 超时时间倍率（慢模型/长任务可调大） |
| `--max-retries N` | trial 失败自动重试次数 |
| `--host-network` | trial 容器共享宿主网络栈（镜像内 compose 模板已默认 host，一般无需再传） |
| `--force-build/--no-force-build` | 是否强制重建 case 环境（prebuilt 镜像场景无需关心） |
| `--delete/--no-delete` | 结束后是否删除 trial 容器（排查时 `--no-delete` 保留现场） |
| `--purge-exception-cases` | 配合 `--reuse`：先清掉上次异常退出的 case 再自动重跑 |
| `-q` / `-y` | 抑制单 trial 进度显示 / 自动确认环境变量提示 |
| `--env-file <path>` | 从 .env 文件读取环境变量 |
| `--monitor-port <port>` | harbor monitor HTTP 服务端口（默认 0 = 关闭） |
| `--disable-verification` | 跳过 verifier 打分（链路调试用；正常评测勿加） |

预期日志关键行（以 dind 模式路径为例，socket 模式对应路径为宿主原路径）：

```
Running Harbor Job:   0%|  | 0/1
Installing offline agent deps from /opt/agent-resources/deps
Selected offline deps bundle /opt/agent-resources/deps/terminus-2-debian-12-aarch64.tar.gz
Installed offline agent deps for terminus-2 ...
<agent 与 LLM 交互日志>
Evaluation results saved to outputs/default/<时间戳>/results/...
+------------+-----------+-------------------------+-----------+ ...
| agent      | model_name| dataset                 | avg_score | ...
+------------+-----------+-------------------------+-----------+ ...
write summary csv to /benchmark/outputs/default/<时间戳>/summary/...
```

> 依赖服务无鉴权时可不传 key（ais_bench 有默认占位 key）。
> 若服务仅限特定 Authorization，务必传 `--agent-api-key` 或 `--ae OPENAI_API_KEY=<值>`。

---

## 7. 查看结果

**宿主机**（`--trials-dir` 对应目录，默认 `~/harbor-trials`）：

```
~/harbor-trials/default/<时间戳>/
├── configs/            # 本次运行的配置快照
├── results/
│   └── terminus-2/
│       ├── harbor_terminal-bench-2.json      # 汇总结果
│       └── harbor_terminal-bench-2/
│           └── details/                       # harbor 原生产物（trajectory、verifier 日志等）
└── summary/summary_<时间戳>.csv              # 汇总表
```

**容器内**：dind 模式对应 `/benchmark/outputs/`（同一份数据）；
socket 模式下容器内路径与宿主机路径相同（同路径透传挂载）。

单条 trial 详细产物（agent 执行轨迹、verifier 输出）在
`results/terminus-2/harbor_terminal-bench-2/details/<task>__<随机串>/` 下。

> socket 模式下产物由 trial 容器内 root 写入，宿主机普通用户删除可能报
> Permission denied，清理方法见第 9 节。

---

## 8. 停止与重跑

```bash
# 宿主机执行
docker rm -f agent-runtime
```

- **dind 模式**：内层 dockerd 与 trial 容器随外层容器一起销毁，无需单独清理；
  镜像缓存随之清空（重建容器后重新 load）。
- **socket 模式**：trial 容器由宿主机 daemon 拉起、独立于外层容器。harbor 正常结束
  会自动回收 trial 容器；异常残留时 `docker ps -a | grep <task名>` 查到后
  `docker rm -f` 清理。宿主机镜像不受任何影响。
- 重跑评测：直接重新执行第 3 步一键脚本（产物目录 `--trials-dir` 内容保留，多次运行按时间戳隔离）。
- 换数据集/镜像包/agent：改对应参数重跑第 3 步即可。
- 镜像无需重复导入（socket 模式永远不需要；dind 模式仅在外层容器被删后需要）。

---

## 9. 故障排查

| 症状 | 原因与处理 |
|---|---|
| （socket）启动报「容器内 docker 不可用（socket 挂载失败？）」 | docker.sock 挂载失败：确认宿主机 `/var/run/docker.sock` 存在且可读；非默认路径用 `--docker-sock` 指定 |
| （socket）启动 WARN「宿主机上未发现 tb2 case 镜像」 | 宿主机没有 case 镜像：`docker pull` 或 `docker load` 离线包（tag 形如 `alexgshaw/<case>:20260430`） |
| （socket）宿主机删除产物报 Permission denied | trial 容器内 root 写入的产物文件：`docker run --rm -v <产物目录>:/x alpine rm -rf /x/<子目录>` 清理 |
| （socket）trial 容器内数据集目录为空 | `--dataset`/`--agent-deps` 传了符号链接缩写路径，宿主机 daemon 解析不到：改用真实绝对路径（脚本会校验，一般不会发生） |
| `cannot enter cgroupv2 "/sys/fs/cgroup/docker" ... threaded mode`（dind） | 启动脚本版本过旧。确认用带 `--cgroupns=host -v /sys/fs/cgroup:/sys/fs/cgroup:rw` 的 `start_agent_runtime.sh` |
| 卡在 `==> 容器内启动 dockerd` 180s 超时（dind） | 进容器看 `cat /tmp/dockerd.log`；确认外层容器是 `--privileged` 启动 |
| `No offline deps bundle for agent 'terminus-2' on <os> <ver> (<arch>)` | bundle 缺失或命名不匹配。报错会给出期望文件名，按第 5 步用同族 base-image 补建 |
| `litellm.BadRequestError: LLM Provider NOT provided` | `--model` 未带 provider 前缀，改为 `--model openai/<模型名>` |
| `litellm.InternalServerError: ... Connection error` | 模型服务地址不通：检查 `--api-base` 是否以 `/v1` 结尾、IP/端口可达（容器内 `curl http://<ip>:<port>/v1/models` 验证） |
| 内层拉镜像超时/失败（dind） | case 镜像不在 case-images 包里：把缺失镜像 `docker save` 合入镜像包重建容器，或用 `sync_images.sh` 增量直传 |
| `pthread_create failed: Operation not permitted`（openEuler/RHEL 宿主） | compose 模板 seccomp patch 未生效（不应发生，已烧进镜像）；进容器检查 `grep -r seccomp /opt/venvs/harbor/lib/python3.12/site-packages/harbor/environments/docker/docker-compose-prebuilt.yaml` |
| 容器已存在报错 | `docker rm -f agent-runtime` 或换 `--name` |
| 产物没出现在宿主机 trials 目录 | 确认启动脚本用了 `--trials-dir` 挂载（dind 模式为 `-v <trials-dir>:/benchmark/outputs`）；socket 模式下确认 ais_bench 是在 `cd <trials-dir>` 之后运行的 |
