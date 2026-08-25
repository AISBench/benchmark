#!/bin/bash
# ais_bench_agent_bootstrap.sh
#
# 一键准备 AISBench Agent 测评运行环境（runtime 容器）
#
# 本脚本在物理机上执行，完成：
#   1. 探测宿主 docker / cgroup / arch
#   2. 自动选择 DinD（模式 A）或 Socket 代理（模式 B）
#   3. 拉取 agent-runtime 镜像（dockerhub 优先，OBS tar 回退）
#   4. 启动 runtime 容器（自动带 --privileged / --cgroupns=host / daemon.json）
#   5. 容器内启动 dockerd（模式 A）或重链 ais_bench（模式 B）
#   6. （可选）把宿主的数据集目录挂载进容器（--datasets <PATH>）
#   7. 自检并打印下一步（doctor → 原生 ais_bench）
#
# 本脚本不接管测评执行，也不预置数据集 / case 镜像：
#   - 数据集由用户在物理机上准备好，通过 --datasets <HOST_PATH> 挂载到容器内同路径
#   - case 沙箱镜像由用户自行 docker pull / docker load 后再跑测评
#   环境就绪后，用户用原生 ais_bench 命令跑测评，原理与各 benchmark 文档完全一致。
#
# 用法:
#   curl -fsSL <OBS_URL>/ais_bench_agent_bootstrap.sh | bash -s -- --datasets /data/datasets
#   bash ais_bench_agent_bootstrap.sh --datasets /data/datasets
#
# 环境变量（仅 OBS_RUNTIME_TAR_BASE 保留为 env，其他配置请用 CLI 参数）:
#   OBS_RUNTIME_TAR_BASE      OBS runtime tar 下载基址（一般无需改，正常物理机无需设置）
#
# 命令行参数:
#   --datasets <HOST_PATH>    把宿主 <HOST_PATH> 挂载到容器内的相同路径 <HOST_PATH>
#                             同时把首个 --datasets 路径作为环境变量
#                             AISBENCH_AGENT_DATASET_PATH 注入容器（这就是
#                             原生配置 path 的值，不在容器内做任何拼接）。
#
#                             直接传 harbor benchmark 数据集完整目录即可：
#                               --datasets /data/datasets/harbor/mini-0.10/terminal-bench-2-offline-selected_0.10
#                             或 full：
#                               --datasets /data/datasets/harbor/full/terminal-bench-2
#                             原生配置（如 harbor_terminal_bench_2_task.py）从
#                             env var 直接读 path，无需 vim 修改。
#
#                             可多次指定挂载多个目录，但只有首个路径会注入 env var。
#                             多目录场景下用户可手动 export AISBENCH_AGENT_DATASET_PATH
#                             覆盖。
#   --runtime-tar <HOST_PATH> 离线模式：用宿主上已下载好的 tar 包加载 runtime 镜像，
#                             完全跳过 docker pull / OBS 下载。
#                             适用场景：内网/隔离环境部署机无法访问 registry。
#                             tar 可通过以下方式获取：
#                               - 维护者用 build_image_agent_runtime.sh --upload 1 上传 OBS 后下载
#                               - 拷贝到 U 盘/内网代理服务器后再 wget
#                               - 镜像构建产物（docker save）直接拷贝
#                             tar 内镜像 tag 默认从 tar 内检测（grep agent-runtime），
#                             若 RUNTIME_IMAGE 已在宿主机存在则优先匹配。
#                             例: --runtime-tar /opt/aisbench/agent-runtime-ubuntu24-py312-x86_64.tar.gz
#   --case-tar <HOST_PATH>    离线场景：把宿主上已下载好的 case 沙箱镜像 tar 加载进 runtime 容器内
#                             的 docker daemon。路径可以是单个 tar 文件，也可以是一个文件夹
#                             （脚本会递归加载其中所有 .tar / .tar.gz / .tgz 文件）。
#                             适用于 case 镜像提前下载到本地、内网无法 pull 的场景。
#                             加载完成后镜像就在 runtime 容器内可见，可直接跑测评。
#                             （可多次指定）
#                             例: --case-tar /opt/aisbench/case-tb2-mini-0.10.tar.gz
#                                 --case-tar /opt/aisbench/case-tars/
#   --mode A|B                强制指定 DinD (A) 或 Socket 代理 (B)。不传则按
#                             docker 版本 + cgroup 类型自动判断。
#   --container-name <NAME>   runtime 容器名（默认 ais_bench_agent）。用于一台机器上
#                             同时跑多个独立 runtime 容器时区分。
#   --runtime-image <TAG>     指定 runtime 镜像 tag（默认 latest-ubuntu24.04-py312-${ARCH}）。
#                             推荐显式传具体 commit tag 以保证可复现性：
#                               --runtime-image ghcr.io/aisbench/agent-runtime:v3.1-20260522-master-ubuntu24.04-py312-x86_64
#   --host-path <ABS_PATH>    仅模式 B 生效。/benchmark 的提取目标目录，
#                             默认 /opt/ais_bench_agent。仅 /opt 不可写的受限环境需要改。
#
# v3 B 批新增参数（5 层 DinD 接通）:
#   --matrix-yaml <HOST_PATH>  harbor 矩阵文件 host 路径（绝对路径），
#                             bind mount 到容器内 /opt/swebench/config/matrix.yaml
#                             （J2 决策：宿主 mount，非 image 内置）
#                             例: --matrix-yaml /opt/swebench/config/matrix.yaml
#   --bind-jobs <HOST_DIR>     5 层 DinD 中 trial 产物目录 host 路径，
#                             bind mount 到 /opt/swebench/jobs（harbor jobs 写入此目录）
#                             例: --bind-jobs /opt/swebench/jobs
#   --bind-tasks <HOST_DIR>    SWE-bench / terminal-bench task.toml 目录 host 路径，
#                             bind mount 到 /opt/swebench/tasks
#                             例: --bind-tasks /opt/swebench/tasks
#   --bind-config <HOST_DIR>   harbor 配置目录 host 路径，
#                             bind mount 到 /opt/swebench/config
#                             例: --bind-config /opt/swebench/config
#   --api-key-file <HOST_PATH> api_key.env host 路径（含 OPENAI_API_KEY / OPENAI_API_BASE），
#                             bind mount 到 /opt/swebench/api_key.env，
#                             同时转成 -e OPENAI_API_KEY / -e OPENAI_API_BASE 注入容器
#                             （替代 inline env var，符合 12-factor）
#                             例: --api-key-file /opt/swebench/config/api_key.env
#   --registry-mirror <URL>    DinD inner dockerd 用的 registry mirror，
#                             注入 -e AIS_BENCH_AGENT_REGISTRY_MIRROR=<URL> 到容器
#                             （A3 镜像层会用此 env 写 daemon.json）
#                             多 mirror 用英文逗号分隔
#                             例: --registry-mirror https://docker.1ms.run
#   --data-image <IMAGE[:TAG]> 创建只读 data 容器（`docker create`），
#                             容器启动时 `--volumes-from <data>:ro`
#                             （B2 batch：替代 image 内置数据集场景）
#                             例: --data-image swebench/swebench-data:v0.1
#   --production               启用生产模式：容器加 --restart unless-stopped
#                             （J4 决策：默认不加，避免 dev 时无法 stop）

set -e

# ============ 默认配置 ============
# runtime 镜像 tag 格式: <target_hub_repo>:<base-tag>-<os>-<py_version>-<arch>
#   例如 ghcr.io/aisbench/agent-runtime:v3.1-20260522-master-ubuntu24.04-py312-x86_64
#
# 默认用 "latest" tag（CI 构建时打 latest，指向最新版本），用户推荐显式传 --runtime-image
# 指定具体版本，以保证可复现性:
#   bash bootstrap.sh --runtime-image ghcr.io/aisbench/agent-runtime:v3.1-20260522-master-ubuntu24.04-py312-x86_64 --datasets /data/datasets
#
# 构建脚本（build_image_agent_runtime.sh）产出的 tag 形如
#   ghcr.io/aisbench/agent-runtime:<base-tag>-<os>-<py_version>-<arch>
# 推送时若同时打了 latest-<os>-<py_version>-<arch>，本脚本默认拉 latest。
ARCH=$(uname -m)

# 仅保留的 env 变量（CLI 参数已替代其他所有 env 配置）
OBS_RUNTIME_TAR_BASE="${OBS_RUNTIME_TAR_BASE:-https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/runtime}"

# CLI 参数暂存变量（由 --xxx 填充；后续 [应用默认值] 段统一给最终值）
MODE_FORCE=""
CONTAINER_NAME_CLI=""
RUNTIME_IMAGE_CLI=""
HOST_PATH_CLI=""

# 用户要挂载进容器的宿主数据集路径（可多个）
DATASET_MOUNTS=()
# 离线模式：用户提供的 runtime tar 包路径（绝对路径）
RUNTIME_TAR=""
# 离线模式：用户提供的 case 沙箱镜像 tar（文件或目录，可多个）
CASE_TAR_PATHS=()

# ============================================================
# v3 B 批新增参数（B1: bind mount + matrix + api_key + registry mirror）
#
# 全部可选，无默认值（J2 决策：用户主动指定 host bind mount 路径）。
# 用户不传时，对应的 bind mount 跳过，运行时用容器内默认路径或环境变量。
# ============================================================
# Harbor 矩阵文件（host bind mount → /opt/swebench/config/matrix.yaml）
MATRIX_YAML=""
# 5 层 DinD bind mount 路径（host → 容器内 /opt/swebench/{jobs,tasks,config}）
BIND_JOBS=""
BIND_TASKS=""
BIND_CONFIG=""
# api_key.env（host bind mount → /opt/swebench/api_key.env，自动 source OPENAI_API_KEY/BASE）
API_KEY_FILE=""
# registry-mirror（AIS_BENCH_AGENT_REGISTRY_MIRROR env 注入 A3 image 的 daemon.json）
REGISTRY_MIRROR=""

# v3 B 批其他新增参数（B2/B4 在后续 commit 启用）
# --data-image：可选 data container image（创建 volumes-from 只读容器）
DATA_IMAGE=""
# --production：仅加 --restart unless-stopped（J4 决策，默认不加）
PRODUCTION=0

# ============ 参数解析 ============
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)
            [ -z "${2:-}" ] && { echo "[错误] --datasets 需要一个绝对路径" >&2; exit 1; }
            [[ "$2" != /* ]] && { echo "[错误] --datasets 必须是绝对路径: $2" >&2; exit 1; }
            [ ! -d "$2" ] && { echo "[错误] --datasets 路径在宿主上不存在: $2" >&2; exit 1; }
            DATASET_MOUNTS+=("$2")
            shift 2
            ;;
        --runtime-tar)
            [ -z "${2:-}" ] && { echo "[错误] --runtime-tar 需要一个绝对路径" >&2; exit 1; }
            [[ "$2" != /* ]] && { echo "[错误] --runtime-tar 必须是绝对路径: $2" >&2; exit 1; }
            [ ! -f "$2" ] && { echo "[错误] --runtime-tar 文件在宿主上不存在: $2" >&2; exit 1; }
            RUNTIME_TAR="$2"
            shift 2
            ;;
        --case-tar)
            [ -z "${2:-}" ] && { echo "[错误] --case-tar 需要一个绝对路径" >&2; exit 1; }
            [[ "$2" != /* ]] && { echo "[错误] --case-tar 必须是绝对路径: $2" >&2; exit 1; }
            [ ! -e "$2" ] && { echo "[错误] --case-tar 路径在宿主上不存在: $2" >&2; exit 1; }
            CASE_TAR_PATHS+=("$2")
            shift 2
            ;;
        --mode)
            [ -z "${2:-}" ] && { echo "[错误] --mode 需要 A 或 B" >&2; exit 1; }
            case "$2" in
                A|B) MODE_FORCE="$2" ;;
                *) echo "[错误] --mode 必须是 A 或 B，收到: $2" >&2; exit 1 ;;
            esac
            shift 2
            ;;
        --container-name)
            [ -z "${2:-}" ] && { echo "[错误] --container-name 不能为空" >&2; exit 1; }
            CONTAINER_NAME_CLI="$2"
            shift 2
            ;;
        --runtime-image)
            [ -z "${2:-}" ] && { echo "[错误] --runtime-image 不能为空" >&2; exit 1; }
            RUNTIME_IMAGE_CLI="$2"
            shift 2
            ;;
        --host-path)
            [ -z "${2:-}" ] && { echo "[错误] --host-path 需要一个绝对路径" >&2; exit 1; }
            [[ "$2" != /* ]] && { echo "[错误] --host-path 必须是绝对路径: $2" >&2; exit 1; }
            HOST_PATH_CLI="$2"
            shift 2
            ;;
        # ============================================================
        # v3 B 批新增参数 (B1): bind mount + matrix + api_key + registry mirror
        # ============================================================
        --matrix-yaml)
            [ -z "${2:-}" ] && { echo "[错误] --matrix-yaml 需要一个绝对路径" >&2; exit 1; }
            [[ "$2" != /* ]] && { echo "[错误] --matrix-yaml 必须是绝对路径: $2" >&2; exit 1; }
            [ ! -f "$2" ] && { echo "[错误] --matrix-yaml 文件在宿主上不存在: $2" >&2; exit 1; }
            MATRIX_YAML="$2"
            shift 2
            ;;
        --bind-jobs)
            [ -z "${2:-}" ] && { echo "[错误] --bind-jobs 需要一个绝对路径" >&2; exit 1; }
            [[ "$2" != /* ]] && { echo "[错误] --bind-jobs 必须是绝对路径: $2" >&2; exit 1; }
            [ ! -d "$2" ] && { echo "[错误] --bind-jobs 路径在宿主上不是目录: $2" >&2; exit 1; }
            BIND_JOBS="$2"
            shift 2
            ;;
        --bind-tasks)
            [ -z "${2:-}" ] && { echo "[错误] --bind-tasks 需要一个绝对路径" >&2; exit 1; }
            [[ "$2" != /* ]] && { echo "[错误] --bind-tasks 必须是绝对路径: $2" >&2; exit 1; }
            [ ! -d "$2" ] && { echo "[错误] --bind-tasks 路径在宿主上不是目录: $2" >&2; exit 1; }
            BIND_TASKS="$2"
            shift 2
            ;;
        --bind-config)
            [ -z "${2:-}" ] && { echo "[错误] --bind-config 需要一个绝对路径" >&2; exit 1; }
            [[ "$2" != /* ]] && { echo "[错误] --bind-config 必须是绝对路径: $2" >&2; exit 1; }
            [ ! -d "$2" ] && { echo "[错误] --bind-config 路径在宿主上不是目录: $2" >&2; exit 1; }
            BIND_CONFIG="$2"
            shift 2
            ;;
        --api-key-file)
            [ -z "${2:-}" ] && { echo "[错误] --api-key-file 需要一个绝对路径" >&2; exit 1; }
            [[ "$2" != /* ]] && { echo "[错误] --api-key-file 必须是绝对路径: $2" >&2; exit 1; }
            [ ! -f "$2" ] && { echo "[错误] --api-key-file 文件在宿主上不存在: $2" >&2; exit 1; }
            API_KEY_FILE="$2"
            shift 2
            ;;
        --registry-mirror)
            [ -z "${2:-}" ] && { echo "[错误] --registry-mirror 不能为空" >&2; exit 1; }
            REGISTRY_MIRROR="$2"
            shift 2
            ;;
        # ============================================================
        # v3 B 批新增参数 (B2/B4): data-image + production
        # ============================================================
        --data-image)
            [ -z "${2:-}" ] && { echo "[错误] --data-image 不能为空" >&2; exit 1; }
            DATA_IMAGE="$2"
            shift 2
            ;;
        --production)
            PRODUCTION=1
            shift
            ;;
        -h|--help)
            sed -n '2,80p' "$0" | sed 's/^# *//'
            exit 0
            ;;
        *) echo "[错误] 未知参数: $1" >&2; exit 1 ;;
    esac
done

# ============ 应用默认值 ============
# 仅 CLI 参数与 hardcoded 默认，不读 env

# --mode：默认空（自动判断）
MODE="${MODE_FORCE:-}"

# --container-name（默认 ais_bench_agent）
CONTAINER_NAME="${CONTAINER_NAME_CLI:-ais_bench_agent}"

# --runtime-image（默认 latest-${OS}-${PY}-${ARCH}）
RUNTIME_IMAGE="${RUNTIME_IMAGE_CLI:-ghcr.io/aisbench/agent-runtime:latest-ubuntu24.04-py312-${ARCH}}"

# --host-path（仅模式 B，默认 /opt/ais_bench_agent）
HOST_PATH="${HOST_PATH_CLI:-/opt/ais_bench_agent}"

# ============ 工具函数 ============
log()   { echo "[$(date +%H:%M:%S)] $*"; }
fail()  { echo "[错误] $*" >&2; exit 1; }

# ============ [1/6] 探测宿主环境 ============
log "=== [1/6] 探测宿主环境 ==="

command -v docker >/dev/null 2>&1 || fail "宿主未安装 docker，请先安装 docker（>= 20.10）"

DOCKER_VER=$(docker version --format '{{.Server.Version}}' 2>/dev/null | head -1)
[ -z "$DOCKER_VER" ] && fail "无法获取 docker 版本，请确认 docker daemon 已启动"
DOCKER_MAJOR=$(echo "$DOCKER_VER" | cut -d. -f1)

CGROUP_TYPE=$(stat -fc %T /sys/fs/cgroup 2>/dev/null || echo "unknown")

log "  Docker:  ${DOCKER_VER} (major=${DOCKER_MAJOR})"
log "  cgroup:  ${CGROUP_TYPE}"
log "  arch:    ${ARCH}"
if [ "${#DATASET_MOUNTS[@]}" -gt 0 ]; then
    log "  datasets挂载（${#DATASET_MOUNTS[@]} 个）:"
    for p in "${DATASET_MOUNTS[@]}"; do
        log "    - 宿主 ${p}  →  容器内 ${p}"
    done
    log "  环境变量 AISBENCH_AGENT_DATASET_PATH=${DATASET_MOUNTS[0]}（容器内可读）"
else
    log "  datasets挂载: 无（用户未传 --datasets；配置中 path 字段需自行保证可用）"
fi
if [ -n "${RUNTIME_TAR}" ]; then
    SIZE=$(du -h "${RUNTIME_TAR}" 2>/dev/null | cut -f1)
    log "  runtime来源: 离线 tar  ${RUNTIME_TAR}  (${SIZE})"
else
    log "  runtime来源: registry pull / OBS 回退（默认行为）"
fi
if [ "${#CASE_TAR_PATHS[@]}" -gt 0 ]; then
    log "  case镜像来源（${#CASE_TAR_PATHS[@]} 个，运行时加载进容器 docker daemon）:"
    for p in "${CASE_TAR_PATHS[@]}"; do
        log "    - ${p}  ($(if [ -d "$p" ]; then echo dir; else echo "file $(du -h "$p" 2>/dev/null | cut -f1)"; fi))"
    done
else
    log "  case镜像来源: 容器内手动 docker pull / load（默认行为）"
fi

# v3 B 批新参数 logging
if [ -n "${MATRIX_YAML}" ]; then
    log "  matrix-yaml: ${MATRIX_YAML}  →  /opt/swebench/config/matrix.yaml"
fi
if [ -n "${BIND_JOBS}${BIND_TASKS}${BIND_CONFIG}" ]; then
    log "  Bind mount (5 层 DinD L5 bind):"
    [ -n "${BIND_JOBS}" ]   && log "    jobs:    ${BIND_JOBS}  →  /opt/swebench/jobs"
    [ -n "${BIND_TASKS}" ]  && log "    tasks:   ${BIND_TASKS}  →  /opt/swebench/tasks"
    [ -n "${BIND_CONFIG}" ] && log "    config:  ${BIND_CONFIG}  →  /opt/swebench/config"
fi
if [ -n "${API_KEY_FILE}" ]; then
    log "  api-key-file: ${API_KEY_FILE}  →  /opt/swebench/api_key.env"
fi
if [ -n "${REGISTRY_MIRROR}" ]; then
    log "  registry-mirror: ${REGISTRY_MIRROR}"
fi
if [ -n "${DATA_IMAGE}" ]; then
    log "  data-image: ${DATA_IMAGE}（创建只读 data 容器 + --volumes-from :ro）"
fi
[ "${PRODUCTION}" = "1" ] && log "  production: enabled（容器加 --restart unless-stopped）"

# ============ [2/6] 选择 DinD/Socket 模式 ============
log "=== [2/6] 选择 DinD/Socket 模式 ==="

if [ -n "${MODE:-}" ]; then
    log "  → 强制模式 ${MODE}（--mode）"
else
    # 模式 A 推荐：docker >= 20.10 + cgroup v2
    # 模式 B 兼容：任意 docker 版本
    # 详见 docker/OVERVIEW.zh.md "运行 Agent / 沙箱类测评"
    if [ "${DOCKER_MAJOR}" -ge 20 ] && [ "${CGROUP_TYPE}" = "cgroup2fs" ]; then
        MODE="A"
        log "  → 模式 A (Docker-in-Docker，推荐，子容器隔离)"
    else
        MODE="B"
        log "  → 模式 B (Socket 代理，兼容任意 docker 版本)"
        log "    原因: docker_major=${DOCKER_MAJOR} (<20) 或 cgroup=${CGROUP_TYPE} (!=cgroup2fs)"
    fi
fi

# ============ [3/6] 拉取 agent-runtime 镜像 ============
log "=== [3/6] 拉取 agent-runtime 镜像 ==="

if [ -n "${RUNTIME_TAR}" ]; then
    # ---------- 离线模式：用户提供的 tar ----------
    log "  离线模式：跳过 docker pull / OBS 下载"
    log "  docker load -i ${RUNTIME_TAR}"
    docker load -i "${RUNTIME_TAR}" || fail "docker load 失败: ${RUNTIME_TAR}"
    # 检测 tar 加载后的镜像 tag
    if docker image inspect "${RUNTIME_IMAGE}" >/dev/null 2>&1; then
        log "  ✓ 使用 RUNTIME_IMAGE 指定的 tag: ${RUNTIME_IMAGE}"
    else
        DETECTED=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -i 'agent-runtime' | head -1 || true)
        if [ -z "${DETECTED}" ]; then
            fail "tar 加载完成但未发现 agent-runtime 镜像；tar 内容或 RUNTIME_IMAGE 可能不匹配"
        fi
        log "  ✓ tar 中检测到镜像: ${DETECTED}"
        RUNTIME_IMAGE="${DETECTED}"
    fi
elif docker image inspect "${RUNTIME_IMAGE}" >/dev/null 2>&1; then
    log "  ✓ runtime 镜像已存在本地: ${RUNTIME_IMAGE}"
else
    log "  尝试 docker pull ${RUNTIME_IMAGE} ..."
    if docker pull "${RUNTIME_IMAGE}" 2>/dev/null; then
        log "  ✓ docker pull 成功"
    else
        log "  docker pull 失败，回退 OBS tar"
        TAR_URL="${OBS_RUNTIME_TAR_BASE}/agent-runtime-ubuntu24-py312-${ARCH}.tar.gz"
        log "  下载: ${TAR_URL}"
        curl -fsSL "${TAR_URL}" -o /tmp/agent-runtime.tar.gz || fail "OBS 下载失败: ${TAR_URL}"
        log "  docker load ..."
        docker load -i /tmp/agent-runtime.tar.gz || fail "docker load 失败"
        rm -f /tmp/agent-runtime.tar.gz
        log "  ✓ OBS tar 加载成功"
    fi
fi

# ============ [4/6] 启动容器 ============
log "=== [4/6] 启动容器（模式 ${MODE}）==="

# 清理同名旧容器
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "  清理同名旧容器: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

# v3 B 批 B2: 处理 data container（仅 --data-image 传入时执行）
# data container 是只读空容器，仅用于携带 image 内置的数据卷
# runtime 容器 --volumes-from <data>:ro 继承这些数据卷，但运行时不能改
DATA_VOLUMES_ARG=""
if [ -n "${DATA_IMAGE}" ]; then
    DATA_CONTAINER_NAME="${CONTAINER_NAME}-data"
    if docker ps -a --format '{{.Names}}' | grep -q "^${DATA_CONTAINER_NAME}$"; then
        log "  ✓ data container 已存在: ${DATA_CONTAINER_NAME}（复用）"
    else
        log "  创建 data container: ${DATA_CONTAINER_NAME} ← ${DATA_IMAGE}"
        if ! docker create --name "${DATA_CONTAINER_NAME}" "${DATA_IMAGE}" >/dev/null 2>&1; then
            fail "data container 创建失败（镜像不存在或已损坏）: ${DATA_IMAGE}"
        fi
        log "    ✓ ${DATA_CONTAINER_NAME} 已创建"
    fi
    DATA_VOLUMES_ARG="--volumes-from ${DATA_CONTAINER_NAME}:ro"
fi

# 拼装数据集挂载参数
MOUNT_ARGS=""
for p in "${DATASET_MOUNTS[@]+"${DATASET_MOUNTS[@]}"}"; do
    MOUNT_ARGS="$MOUNT_ARGS -v ${p}:${p}"
done

# 把首个 --datasets 路径持久化为容器内环境变量
# 原生配置（如 harbor_terminal_bench_2_task.py）通过读取此变量得到数据集路径，
# 无需用户在容器内额外配置，符合"一键准备"的设计目标。
# 多 --datasets 时仅首个生效，其余需用户自行改 DEFAULT_DATASET_PATH。
DATASET_ENV=""
if [ "${#DATASET_MOUNTS[@]}" -gt 0 ]; then
    DATASET_ENV="-e AISBENCH_AGENT_DATASET_PATH=${DATASET_MOUNTS[0]}"
fi

# v3 B 批 B3: 拼装 5 层 DinD L5 接入所需的 bind mount + -e 注入
# 配合 B1 新增参数：--matrix-yaml / --bind-jobs / --bind-tasks / --bind-config
#                  / --api-key-file / --registry-mirror
# 不传任意参数 → SWEBENCH_BINDS / SWEBENCH_ENVS 均为空 → 行为退化到 PR #410 原版
SWEBENCH_BINDS=""
SWEBENCH_ENVS=""
if [ -n "${MATRIX_YAML:-}" ]; then
    SWEBENCH_BINDS="${SWEBENCH_BINDS} -v ${MATRIX_YAML}:/opt/swebench/config/matrix.yaml:ro"
fi
if [ -n "${BIND_JOBS:-}" ]; then
    SWEBENCH_BINDS="${SWEBENCH_BINDS} -v ${BIND_JOBS}:/opt/swebench/jobs"
fi
if [ -n "${BIND_TASKS:-}" ]; then
    SWEBENCH_BINDS="${SWEBENCH_BINDS} -v ${BIND_TASKS}:/opt/swebench/tasks"
fi
if [ -n "${BIND_CONFIG:-}" ]; then
    SWEBENCH_BINDS="${SWEBENCH_BINDS} -v ${BIND_CONFIG}:/opt/swebench/config"
fi
if [ -n "${API_KEY_FILE:-}" ]; then
    SWEBENCH_BINDS="${SWEBENCH_BINDS} -v ${API_KEY_FILE}:/opt/swebench/api_key.env:ro"
    # source api_key.env 后把 OPENAI_API_KEY / OPENAI_API_BASE 注入 -e
    # 用 set -a 让 source 进来的变量自动 export，避免 shell 子进程丢失
    set -a
    # shellcheck disable=SC1090
    . "${API_KEY_FILE}" 2>/dev/null || log "  ⚠ source ${API_KEY_FILE} 失败（-e 注入可能不全）"
    set +a
    if [ -n "${OPENAI_API_KEY:-}" ]; then
        SWEBENCH_ENVS="${SWEBENCH_ENVS} -e OPENAI_API_KEY=${OPENAI_API_KEY}"
    fi
    if [ -n "${OPENAI_API_BASE:-}" ]; then
        SWEBENCH_ENVS="${SWEBENCH_ENVS} -e OPENAI_API_BASE=${OPENAI_API_BASE}"
    fi
fi
if [ -n "${REGISTRY_MIRROR:-}" ]; then
    SWEBENCH_ENVS="${SWEBENCH_ENVS} -e AIS_BENCH_AGENT_REGISTRY_MIRROR=${REGISTRY_MIRROR}"
fi
if [ -n "${SWEBENCH_BINDS}" ] || [ -n "${SWEBENCH_ENVS}" ]; then
    log "  5 层 DinD L5 接入:"
    [ -n "${SWEBENCH_BINDS}" ] && log "    bind mounts: 容器内 /opt/swebench/{jobs,tasks,config,api_key.env} ← host bind"
    [ -n "${SWEBENCH_ENVS}" ] && log "    env 注入:   OPENAI_API_KEY/BASE + AIS_BENCH_AGENT_REGISTRY_MIRROR"
fi

if [ "${MODE}" = "A" ]; then
    # 模式 A：Docker-in-Docker
    # cgroup v2 宿主机 --privileged + --cgroupns=host 必须同时使用
    # 缺少 --cgroupns=host 会报 "cannot enter cgroupv2 ... invalid state"
    # 详见 docker/OVERVIEW.zh.md 模式 A
    docker run --name "${CONTAINER_NAME}" -it -d \
        --net=host --ipc=host \
        --privileged --cgroupns=host \
        -w /benchmark \
        ${MOUNT_ARGS} \
        ${DATA_VOLUMES_ARG} \
        ${SWEBENCH_BINDS} \
        ${DATASET_ENV} \
        ${SWEBENCH_ENVS} \
        "${RUNTIME_IMAGE}" bash
else
    # 模式 B：Socket 代理
    # 挂载宿主 docker socket，子容器由宿主 daemon 创建
    # 需要把 /benchmark 拷贝到宿主路径再挂回去（避免 docker cp 语义）
    # 详见 docker/OVERVIEW.zh.md 模式 B
    # HOST_PATH 已在前面 [应用默认值] 段统一计算
    log "  模式 B 准备 HOST_PATH: ${HOST_PATH}"
    mkdir -p "${HOST_PATH}"

    # 从 runtime 镜像拷贝 /benchmark 内容到 HOST_PATH
    log "  提取 /benchmark 到 ${HOST_PATH} ..."
    docker run -d --name tmp_extract "${RUNTIME_IMAGE}" bash >/dev/null
    docker cp tmp_extract:/benchmark/. "${HOST_PATH}/" || { docker rm -f tmp_extract; fail "docker cp 失败"; }
    docker rm -f tmp_extract >/dev/null

    docker run --name "${CONTAINER_NAME}" -it -d \
        --net=host --privileged \
        -w "${HOST_PATH}" \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v "${HOST_PATH}":"${HOST_PATH}" \
        ${MOUNT_ARGS} \
        ${DATA_VOLUMES_ARG} \
        ${SWEBENCH_BINDS} \
        ${DATASET_ENV} \
        ${SWEBENCH_ENVS} \
        "${RUNTIME_IMAGE}" bash
fi

log "  ✓ 容器已启动: ${CONTAINER_NAME}"

# ============ [5/6] 容器内配置 docker ============
log "=== [5/6] 容器内配置 docker ==="

if [ "${MODE}" = "A" ]; then
    # 模式 A：容器内启动 dockerd
    # 必须写 daemon.json：
    #   - cgroupfs driver：DinD 在 cgroup v2 宿主上的必需配置（docker 27.x 默认 systemd，容器内无 systemd）
    #   - vfs 存储驱动：DinD 通用性最高；若宿主内核与容器根fs支持，overlay2 性能更好
    # 详见 docker/OVERVIEW.zh.md 模式 A 步骤二
    docker exec "${CONTAINER_NAME}" bash -c '
        set -e
        mkdir -p /etc/docker
        cat > /etc/docker/daemon.json <<EOF
{
  "exec-opts": ["native.cgroupdriver=cgroupfs"],
  "storage-driver": "vfs"
}
EOF
        nohup dockerd > /tmp/dockerd.log 2>&1 &
        for i in $(seq 1 30); do
            [ -S /var/run/docker.sock ] && break
            sleep 1
        done
        if ! docker info >/dev/null 2>&1; then
            echo "[错误] 容器内 dockerd 启动失败，日志："
            tail -30 /tmp/dockerd.log
            exit 1
        fi
        echo "  ✓ 容器内 dockerd ready"
    ' || fail "模式 A dockerd 启动失败"
else
    # 模式 B：socket 已挂载，重链 ais_bench
    # 因为 WORKDIR 改到了 HOST_PATH，需要在该路径重新以 editable 模式安装 ais_bench
    # 只链接不改依赖（--no-deps）
    # 详见 docker/OVERVIEW.zh.md 模式 B 步骤二
    docker exec "${CONTAINER_NAME}" bash -c '
        pip3 install -e ./ --use-pep517 --no-deps --no-build-isolation --break-system-packages >/dev/null 2>&1 \
            && echo "  ✓ ais_bench relinked" \
            || echo "  ⚠ ais_bench relink 失败（可忽略，若 ais_bench 命令可用即可）"
    '
fi

# ============ [6/7] 加载 case 镜像（可选） ============
log "=== [6/7] 加载 case 镜像（可选）==="

if [ "${#CASE_TAR_PATHS[@]}" -gt 0 ]; then
    # 把 --case-tar 展开成具体的 tar 文件列表
    log "  收集 case tar 文件..."
    CASE_TAR_FILES=()
    for p in "${CASE_TAR_PATHS[@]}"; do
        if [ -f "$p" ]; then
            # 单个文件
            case "$p" in
                *.tar|*.tar.gz|*.tgz) CASE_TAR_FILES+=("$p") ;;
                *) log "    [跳过] 不是 docker tar: $p"; ;;
            esac
        elif [ -d "$p" ]; then
            # 目录：递归收集所有 .tar/.tar.gz/.tgz
            while IFS= read -r -d '' f; do
                CASE_TAR_FILES+=("$f")
            done < <(find "$p" -type f \( -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" \) -print0 2>/dev/null)
        fi
    done

    if [ "${#CASE_TAR_FILES[@]}" -eq 0 ]; then
        log "  ⚠ --case-tar 路径下未发现 .tar/.tar.gz/.tgz 文件，跳过加载"
    else
        log "  共 ${#CASE_TAR_FILES[@]} 个 case tar 待加载"

        # 在容器内建暂存目录
        docker exec "${CONTAINER_NAME}" mkdir -p /tmp/case-tars

        # 用 docker cp 把每个 tar 拷进容器
        i=0
        for f in "${CASE_TAR_FILES[@]}"; do
            i=$((i+1))
            BN=$(basename "$f")
            SIZE=$(du -h "$f" 2>/dev/null | cut -f1)
            log "  [${i}/${#CASE_TAR_FILES[@]}] docker cp  ${f}  (${SIZE})"
            docker cp "$f" "${CONTAINER_NAME}:/tmp/case-tars/${BN}" || {
                log "    ✗ docker cp 失败，跳过该文件"
                continue
            }
        done

        # 容器内 docker load 每一个
        log "  容器内 docker load ..."
        docker exec "${CONTAINER_NAME}" bash -c '
            cd /tmp/case-tars
            loaded=0
            failed=0
            for tf in *.tar *.tar.gz *.tgz; do
                [ -f "$tf" ] || continue
                sz=$(du -h "$tf" | cut -f1)
                echo "    loading $tf ($sz) ..."
                if out=$(docker load -i "$tf" 2>&1); then
                    loaded=$((loaded+1))
                    echo "      ✓ $(echo "$out" | tail -1)"
                else
                    failed=$((failed+1))
                    echo "      ✗ docker load 失败: $out"
                fi
            done
            echo "  docker load 完成: ${loaded} 个成功, ${failed} 个失败"
            rm -rf /tmp/case-tars
        ' || log "    ⚠ 容器内 docker load 阶段出错（请人工 docker exec 进容器检查）"
    fi
else
    log "  未传 --case-tar，跳过（容器内用户手动 docker pull / docker load）"
fi

# ============ [7/7] 自检 + 打印下一步 ============
log "=== [7/7] 自检 ==="

docker exec "${CONTAINER_NAME}" bash -c '
    echo "  docker:    $(docker --version 2>/dev/null || echo 不可用)"
    echo "  compose:   $(docker compose version 2>/dev/null | head -1 || echo 不可用)"
    echo "  ais_bench: $(ais_bench --version 2>/dev/null || echo unknown)"
    echo "  venvs:"
    for v in harbor swebench swebench_pro; do
        [ -d /opt/venvs/$v ] && echo "    ✓ $v" || echo "    ✗ $v 缺失"
    done
    echo "  packs:"
    ls /opt/agent-resources/packs/*.yaml 2>/dev/null | while read f; do
        echo "    ✓ $(basename $f .yaml)"
    done
    echo "  用户挂载目录（容器内可见性自检）:"
'

# 自检：用户传入的挂载路径在容器内是否可见
if [ "${#DATASET_MOUNTS[@]}" -gt 0 ]; then
    for p in "${DATASET_MOUNTS[@]}"; do
        if docker exec "${CONTAINER_NAME}" bash -c "[ -d '$p' ]" >/dev/null 2>&1; then
            SIZE=$(docker exec "${CONTAINER_NAME}" bash -c "du -sh '$p' 2>/dev/null | cut -f1")
            log "    ✓ ${p}  (${SIZE})"
        else
            log "    ✗ ${p}  在容器内不可见（挂载失败）"
        fi
    done
fi

echo ""
echo "============================================================"
echo "✓ Agent 测评运行环境容器已就绪（模式 ${MODE}）"
echo "============================================================"
cat <<EOF

下一步：

1. 进入容器
   docker exec -it ${CONTAINER_NAME} bash

2. case 沙箱镜像状态：
EOF
if [ "${#CASE_TAR_PATHS[@]}" -gt 0 ]; then
    N_CASES=${#CASE_TAR_FILES[@]}
    cat <<EOF
   - bootstrap 已加载 ${N_CASES} 个 tar（见上文 [6/7] 日志）
   - 可用 'docker images' 验证是否都加载成功
   - 若有失败的，按需手动补 load
EOF
else
cat <<EOF
   - 未传 --case-tar，需手动准备 case 镜像：
       在线: docker pull <registry>:<tag>     # 注册表与 tag 见各 benchmark 文档
       离线: 从 OBS 下载 tar 后 docker load -i
EOF
fi
cat <<EOF
3. 验证环境就绪（按你要跑的 benchmark 选 pack）
   ais_bench_agent_doctor.sh harbor        # Harbor Terminal-Bench 2.0
   ais_bench_agent_doctor.sh swebench      # SWE-bench（mini_swe_agent + SWE-bench harness）
   ais_bench_agent_doctor.sh swebench_pro  # SWE-bench Pro（scaleapi 适配版）

4. 仅需 vim 改 model_names / api_base（path 已自动从环境变量 AISBENCH_AGENT_DATASET_PATH 读）
   对应原生配置文件见各 pack.yaml 的 native_config 字段，例如：
     vim ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py
     vim ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_lite.py
     vim ais_bench/configs/swe_bench_pro_examples/mini_swe_agent_swe_bench_pro_mini.py

5. 激活对应 venv 跑测评
   agent_env harbor       && ais_bench ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py --debug
   agent_env swebench     && ais_bench ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_lite.py --debug
   agent_env swebench_pro && ais_bench ais_bench/configs/swe_bench_pro_examples/mini_swe_agent_swe_bench_pro_mini.py --debug

详见各 benchmark 文档：
  - harbor:       docs/source_zh_cn/extended_benchmark/agent/harbor_bench.md
  - swebench:     docs/source_zh_cn/extended_benchmark/agent/swe_bench.md
  - swebench_pro: docs/source_zh_cn/extended_benchmark/agent/swe_bench_pro.md

环境准备原理（容器模式 A/B）详见：
  - docker/OVERVIEW.zh.md "运行 Agent / 沙箱类测评" 章节
EOF
