# AISBench Agent Runtime

为 AISBench Agent 测评（Harbor Terminal-Bench、SWE-bench、SWE-bench Pro 等）提供运行环境容器的镜像与脚本。

> 本仓库目录是 AISBench/benchmark 的社区运行时补充，不属于 benchmark 核心评测逻辑。用户使用本目录的脚本和镜像准备好 runtime 容器后，仍用原生 `ais_bench` 命令执行测评，原理与各 benchmark 文档完全一致。

> 📘 想从零跑通本 PR（含 harbor offline + mini-swe-agent tarball + 真实 trial 验证）？看 [REPRODUCE.md](REPRODUCE.md) — 618 行端到端复现手册，含失败原因诚实声明与 LLM 算力方案 gap 说明。

## 解决什么问题

Agent 测评的环境准备存在三大痛点：

1. **依赖冲突**：harbor 强制升级 datasets 到 4.0+；SWE-bench 与 SWE-bench Pro 各需要一个不同 fork 的 `mini-swe-agent`，同包名互相覆盖。
2. **容器配置易错**：DinD 模式 A/B、`--cgroupns=host`、`daemon.json`、seccomp，任一步漏配都会在跑测评时才报错。
3. **数据集 / case 镜像版本变化频繁**：数据集和 case 镜像都有大量版本，烤入 runtime 镜像会很快过期。

本包通过分层解决这些问题：

| 层 | 内容 | 解决 |
|---|---|---|
| `Dockerfile.agent-runtime` | 在 `aisbench_benchmark` 基镜之上追加 3 个 venv 隔离层（harbor / swebench / swebench_pro） | 依赖冲突 |
| `bootstrap.sh` | 一键起 runtime 容器，自动选 DinD/Socket 模式 + 挂载数据集 + 加载 case tar | 容器配置易错 + 数据集接入 |
| `doctor.sh` | 静态自检（L1，秒级）— 校验 docker / venv / pack / 资源 | 跑前验证 runtime 配置 |
| `packs/<name>.yaml` | 各 benchmark 的元数据（venv 名 / 原生配置 / 文档） | 工具链与 benchmark 解耦 |

## 数据集 / case 镜像由谁负责

**刻意不做的事**：本方案**不**预置 agent benchmark 的数据集和 case 沙箱镜像到 runtime 镜像中。原因是这两者版本变化频繁，烤入镜像后：

- 数据集每次更新都要重新构建 runtime 镜像，对维护者负担大、对用户下载体积大
- case 沙箱镜像一个 full 集 ~71GB，不能烤进基础镜像

**谁负责什么**：

| 项 | 谁准备 | 怎么接入 runtime 容器 |
|---|---|---|
| runtime 镜像 | AISBench 维护者 | `docker pull ghcr.io/aisbench/agent-runtime:latest-...`（或 `--runtime-tar` 离线） |
| 数据集（task.toml 等） | 用户在物理机上准备好 | `bootstrap.sh --datasets <完整数据集路径>`（挂载到容器内同路径 + 注入 env var） |
| case 沙箱镜像 | 用户在物理机上准备好 tar | `bootstrap.sh --case-tar <tar>`（自动 `docker cp` 进容器 + `docker load`） |
| 模型调用参数 (api_base / model_names) | 用户改原生配置 | 容器内 `vim ais_bench/configs/agent_example/...` |

## 快速入门(以 Harbor Terminal-Bench 为例 aarch64)
快速入门针对物理机20.0.0以下版本docker环境，其他环境请参考对应agent测评文档。

```bash
# 1. 物理机上准备数据集与镜像 tar（已有可跳过）

git clone https://modelers.cn/AISBench/terminal-bench-2-offline-mini.git # 数据集准备
wget https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/agent_runtime_image_v3.1-20260701-master-ubuntu24.04-py312-aarch64.tar.gz # 测评镜像准备（可选，不准备则自动获取最新）
wget https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-offline-prepared-images-selected-0.10_aarch64.tar # case 镜像准备，按需从对应agent测评文档获取链接
mkdir /path/to/test_wkp/ # 物理机创建一个空的工作目录

# 2. 物理机上一键起 runtime 容器（自动选 DinD/Socket 模式，自动挂载数据集，如果执行环境不通外网，可以先从其他环境获取ais_bench_agent_bootstrap.sh再bash执行
#    自动把 case 镜像 tar 拷进容器内部 docker load 完）
curl -fsSL https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/ais_bench_agent_bootstrap.sh \
    | bash -s -- \
        --datasets /path/to/terminal-bench-2-offline-mini/terminal-bench-2-offline-selected_0.10/ \
        --runtime-tar /path/to/agent_runtime_image_v3.1-20260701-master-ubuntu24.04-py312-aarch64.tar.gz \
        --case-tar /path/to/terminal-bench-2-offline-prepared-images-selected-0.10.tar \
        --host-path /path/to/test_wkp/ \
        --container-name test_agent_run
# --datasets 指向的目录结构需与 terminal-bench-2-offline-mini 仓库的 terminal-bench-2-offline-selected_0.10/ 子目录结构一致
# --runtime-tar （可选）提前准备的测评镜像
# --case-tar 指向的 tar 结构需与对应 agent 测评文档的 case 镜像 tar 结构一致
# --host-path 指向的目录需为空目录，容器内会自动创建同名目录挂载数据集和 case 镜像
# --container-name 指向的容器名需唯一，否则会覆盖旧容器

# 3. 进入容器（case 镜像已在内部，直接可用）
docker exec -it test_agent_run bash

# 4. （无需 vim）原生配置 path 自动从 AISBENCH_AGENT_DATASET_PATH 读
#    仅需 vim 改 model_names / api_base
vim ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py

# 5. 验证 runtime 就绪
ais_bench_agent_doctor.sh harbor

# 6. 跑测评
agent_env harbor
ais_bench ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py --debug
```

切换数据集：销毁旧容器 + 重启 bootstrap（数据集路径 / case tar 一起更新）：

```bash
docker rm -f test_agent_run
bash ais_bench_agent_bootstrap.sh \
    --datasets /data/datasets/harbor/full/terminal-bench-2 \
    --case-tar /data/cases/terminal-bench-2-prepared-images_x86_64.tar.gz
```

## 目录结构

```
agent_runtime/
├── README.md                       # 本文件
├── Dockerfile.agent-runtime        # runtime 镜像构建文件（BASE_IMAGE 通过 --build-arg 传入，不写死）
├── build_image_agent_runtime.sh    # 构建脚本（支持 --base-tag/--push/--upload/--multi-arch）
├── ais_bench_agent_bootstrap.sh                    # 一键起 runtime 容器（用户侧入口，需上传到 OBS）
├── doctor.sh                       # runtime 就绪验证（容器内，仅校验 docker/venv/config 不校验数据集/cases）
├── packs/                          # 各 benchmark 的清单（name/runtime_venv/native_config/native_doc）
│   ├── harbor.yaml                 # Harbor Terminal-Bench
│   ├── swebench.yaml               # SWE-bench（mini_swe_agent + SWE-bench harness）
│   └── swebench_pro.yaml           # SWE-bench Pro（scaleapi 适配版）
└── patches/                        # 构建期 / 启动期用的补丁脚本
    └── harbor_compose_patch.py     # 给 harbor 的 docker-compose-base.yaml 加 seccomp=unconfined + network_mode=host
```

## ais_bench_agent_bootstrap.sh 用法

```bash
# 最简调用：挂载一个数据集目录
bash ais_bench_agent_bootstrap.sh --datasets /data/datasets

# 挂载多个目录
bash ais_bench_agent_bootstrap.sh --datasets /data/datasets --datasets /data/extra

# 强制模式 A/B
bash ais_bench_agent_bootstrap.sh --mode A --datasets /data/datasets

# 自定义容器名（一台机器同时跑多个 runtime 时区分）
bash ais_bench_agent_bootstrap.sh --container-name my_eval_1 --datasets /data/datasets

# 自定义 runtime 镜像（推荐显式传 tag，保证可复现性）
bash ais_bench_agent_bootstrap.sh \
    --runtime-image ghcr.io/aisbench/agent-runtime:v3.1-20260522-master-ubuntu24.04-py312-x86_64 \
    --datasets /data/datasets

# 模式 B + 自定义 /benchmark 提取目标（仅 /opt 不可写时用）
bash ais_bench_agent_bootstrap.sh --mode B --host-path /data/ais_bench_host --datasets /data/datasets

# 离线模式（内网/隔离环境）：用宿主上已下载好的 tar 包加载 runtime 镜像
#   完全跳过 docker pull / OBS 下载
#   适用：内网部署机无法访问 ghcr.io，也无法访问外网 OBS
bash ais_bench_agent_bootstrap.sh --datasets /data/datasets --runtime-tar /opt/aisbench/agent-runtime-ubuntu24-py312-x86_64.tar.gz

# 完全离线：runtime tar + case 镜像 tar 一起传入
#   容器启动后会自动把 case tar 拷进容器并 docker load
bash ais_bench_agent_bootstrap.sh \
    --datasets /data/datasets \
    --runtime-tar /opt/aisbench/agent-runtime.tar.gz \
    --case-tar /opt/aisbench/case-tb2-mini-0.10.tar.gz

# 一次加载多个 case 镜像（可多次 --case-tar，也可传一个目录）
bash ais_bench_agent_bootstrap.sh \
    --datasets /data/datasets \
    --runtime-tar /opt/aisbench/agent-runtime.tar.gz \
    --case-tar /opt/aisbench/case-tb2-mini-0.10.tar.gz \
    --case-tar /opt/aisbench/case-tb2-mini-0.14.tar.gz \
    --case-tar /opt/aisbench/case-tars/         # 目录下所有 .tar/.tar.gz/.tgz 都会被加载
```

`--datasets` / `--host-path` / `--runtime-tar` / `--case-tar` 必须是**绝对路径**，且在**物理机上必须存在**（脚本会校验）。容器内路径与宿主路径相同。

`--datasets` 传入的完整路径会原封不动注入为容器内环境变量 `AISBENCH_AGENT_DATASET_PATH`，原生 ais_bench 配置（如 `harbor_terminal_bench_2_task.py`）直接把这个 env var 作为数据集 `path` 字段使用——**不拼接、不转换、完全一致**。所以：

- **推荐**：把 `--datasets` 传成你准备好的 harbor benchmark 数据集完整路径（如 `/data/datasets/harbor/mini-0.10/terminal-bench-2-offline-selected_0.10`），这样配置无需 vim 改 path
- **多目录**：可多次传 `--datasets`，但 env var 只用首个；用户可手动 `export AISBENCH_AGENT_DATASET_PATH=...` 覆盖

### 命令行参数一览

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--datasets <HOST_PATH>` | 无（不挂载） | 数据集目录，可多次 |
| `--runtime-tar <HOST_PATH>` | 无（走 pull） | runtime 镜像 tar，离线场景用 |
| `--case-tar <HOST_PATH>` | 无（容器内手动准备） | case 镜像 tar，文件或目录，可多次 |
| `--mode A\|B` | 自动判断 | 强制 DinD (A) 或 Socket 代理 (B) |
| `--container-name <NAME>` | `ais_bench_agent` | runtime 容器名 |
| `--runtime-image <TAG>` | `ghcr.io/aisbench/agent-runtime:latest-ubuntu24.04-py312-${ARCH}` | runtime 镜像 tag |
| `--host-path <ABS_PATH>` | `/opt/ais_bench_agent` | 模式 B 时 `/benchmark` 提取目标 |

### 环境变量

仅保留一个 env 变量（其它配置请用 CLI 参数）：

| env | 默认值 | 说明 |
|---|---|---|
| `OBS_RUNTIME_TAR_BASE` | `https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/runtime` | OBS runtime tar 下载基址（一般无需改） |

### 离线场景

内网/隔离环境的部署机无法访问 `ghcr.io/aisbench/agent-runtime` 或 OBS，但已通过 U 盘、内网代理等方式拿到了 runtime tar 包和 case 镜像 tar：

1. **获取 tar 包**（任选其一）：
   - 让维护者跑 `build_image_agent_runtime.sh --upload 1` 上传到 OBS，内网用户从 OBS 下载
   - 在能访问外网的机器上 `docker save ghcr.io/aisbench/agent-runtime:<tag> -o agent-runtime.tar.gz` 后拷贝进来
2. **部署机执行**：
   ```bash
   # 最小：只传 runtime tar（case 镜像仍需在容器内手动 pull / load）
   bash ais_bench_agent_bootstrap.sh --datasets /data/datasets --runtime-tar /path/to/agent-runtime.tar.gz

   # 完全离线：runtime + case tar 都传
   bash ais_bench_agent_bootstrap.sh \
       --datasets /data/datasets \
       --runtime-tar /path/to/agent-runtime.tar.gz \
       --case-tar /path/to/case-tb2-mini-0.10.tar.gz
   ```
3. **行为**：
   - `--runtime-tar`：完全跳过 `docker pull` 与 OBS 的 `curl` 下载；`docker load -i <tar>` 后自动检测 tag（grep `agent-runtime`），优先匹配 `RUNTIME_IMAGE`；检测失败给精确错误
   - `--case-tar <PATH>`：支持传单个 tar 或一个目录（目录会递归加载所有 `.tar` / `.tar.gz` / `.tgz`）。脚本会 `docker cp` 进容器，再用 `docker load -i` 加载。**支持 A/B 两种模式**（A 模式加载到容器内 DinD；B 模式加载到容器内，但因 socket 与宿主共享实际也加载到了宿主），模式A|B的具体介绍参考[OVERVIEW.zh.md](../OVERVIEW.zh.md#运行-agent--沙箱类测评在容器内使用-docker)
   - 可多次 `--case-tar`

模式 B（Socket 代理）默认会把 `/benchmark` 提取到宿主 `/opt/ais_bench_agent`。若你的环境 `/opt` 不可写（如某些只读根容器/沙箱），用 `--host-path` 改写到可写路径：

```bash
bash ais_bench_agent_bootstrap.sh --datasets /data/datasets --host-path /data/ais_bench_host
```

正常物理机无需设置该变量。其余可配置环境变量见 `bootstrap.sh` 头部注释。

## 镜像构建

runtime 镜像基于 `aisbench_benchmark` 基镜构建，基镜像 tag 通过参数传入，不写死：

```bash
# 基础构建（本地，当前架构）
bash docker/agent_runtime/build_image_agent_runtime.sh \
    --base-tag v3.1-20260522-master

# 指定 OS/Python（默认 ubuntu24.04 + py312）
bash docker/agent_runtime/build_image_agent_runtime.sh \
    --base-tag v3.1-20260522-master --os ubuntu24.04 --py-version py312

# 构建并推送到远程仓库
bash docker/agent_runtime/build_image_agent_runtime.sh \
    --base-tag v3.1-20260522-master --push 1

# 多架构构建并推送
bash docker/agent_runtime/build_image_agent_runtime.sh \
    --base-tag v3.1-20260522-master --multi-arch 1 --push 1

# 构建、推送、并上传离线包到 OBS（供 ais_bench_agent_bootstrap.sh 回退下载）
bash docker/agent_runtime/build_image_agent_runtime.sh \
    --base-tag v3.1-20260522-master --push 1 --upload 1
```

构建脚本会自动校验（4 项）：
1. ais_bench 可用
2. 3 个 venv（harbor / swebench / swebench_pro）都完整 + 3 个 venv 内都有 ais_bench wrapper + 两个 swebench venv 能 import minisweagent
3. doctor.sh / packs 就位
4. harbor compose 模板已 patch `seccomp=unconfined`

## 已支持的 pack

| pack 名 | runtime_venv | 文档 | 说明 |
|---|---|---|---|
| `harbor` | harbor | [harbor_bench.md](../../docs/source_zh_cn/extended_benchmark/agent/harbor_bench.md) | Harbor Terminal-Bench 2.0 |
| `swebench` | swebench | [swe_bench.md](../../docs/source_zh_cn/extended_benchmark/agent/swe_bench.md) | SWE-bench（lite/verified/full/multilingual 等） |
| `swebench_pro` | swebench_pro | [swe_bench_pro.md](../../docs/source_zh_cn/extended_benchmark/agent/swe_bench_pro.md) | SWE-bench Pro（仅 x86） |

pack.yaml 不声明数据集路径、不声明 case 镜像获取方式——这些完全交给用户掌控：
- 数据集路径：用户 `bootstrap.sh --datasets <完整数据集路径>` 时显式指定（要哪个就跑哪个）
- case 镜像：用户按 pack.yaml 的 `native_doc` 指向的文档自行 `docker pull` 或 `docker load`

如果以后接更多 benchmark，每加一个 `packs/<name>.yaml` 即可。

> 常用的 harbor 数据集目录名（仅供参考，与工具无关）：
> - `/data/datasets/harbor/full/terminal-bench-2`（89 case，含少量外网任务）
> - `/data/datasets/harbor/mini-0.10/terminal-bench-2-offline-selected_0.10`（7 case）
> - `/data/datasets/harbor/mini-0.14/terminal-bench-2-offline-selected_0.14`（10 case）
> - `/data/datasets/harbor/mini-0.20/terminal-bench-2-offline-selected_0.20`（14 case）
>
> 用户在 `bootstrap.sh --datasets` 传入哪个路径，就由 env var `AISBENCH_AGENT_DATASET_PATH` 把哪个路径注入容器。
> mini-* 系列基于 `terminal-bench-2-offline`（剔除外网任务后的 70 个 case）做 K-means 采样，完全离线可跑。
>
> SWE-bench 数据集说明见 [swe_bench.md](../../docs/source_zh_cn/extended_benchmark/agent/swe_bench.md)（HF 下载）；SWE-bench Pro 见 [swe_bench_pro.md](../../docs/source_zh_cn/extended_benchmark/agent/swe_bench_pro.md)。

## 详细方案

完整设计与各脚本参数说明见各脚本头部注释。