#!/bin/bash
# ============================================================================
# ais_bench_agent.sh — AISBench Agent Runtime 统一入口 (v3 wrapper)
# ============================================================================
# 本脚本在物理机上执行，是 docker/agent_runtime/ 下所有脚本的顶层 facade。
#
# 命令:
#   ais_bench_agent.sh build    构建 L4 runtime 镜像（可选 L2 baked image）
#   ais_bench_agent.sh run      启容器 + doctor 自检 + 透传命令执行
#   ais_bench_agent.sh status   委托 ais_bench_agent_orchestrator_status.sh
#   ais_bench_agent.sh watch    委托 ais_bench_agent_watch.sh
#   ais_bench_agent.sh summarize 委托 ais_bench_agent_summarize.sh
#   ais_bench_agent.sh doctor   委托 doctor.sh
#
# 向下兼容: 现有 10 个脚本的独立使用方式完全不受影响。
#
# 架构:
#   宿主机用户 → ais_bench_agent.sh (wrapper)
#                ├─ build → build_image_agent_runtime.sh [→ build_l2_baked_image.sh]
#                └─ run   → bootstrap.sh → 容器 → doctor → docker exec <command>
#                              ↑ DinD (模式A) / Socket (模式B)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- 颜色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[info]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[错误]${NC} $*" >&2; }
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }

# ---- 默认值 ----
CONTAINER_NAME="ais_bench_agent"
DRY_RUN=0

# ============================================================================
# 使用说明
# ============================================================================
usage() {
    cat <<'EOF'
ais_bench_agent.sh — AISBench Agent Runtime 统一入口

用法:
  ais_bench_agent.sh build   [options]   构建 L4 runtime 镜像（可选 L2）
  ais_bench_agent.sh run     [options]   启容器 + doctor 自检 + 执行测试命令
  ais_bench_agent.sh status  [options]   5 段 DinD runtime 容器状态查询
  ais_bench_agent.sh watch   [options]   阻塞等 harbor job 完成
  ais_bench_agent.sh summarize [options] 聚合 jobs → md/csv/json
  ais_bench_agent.sh doctor  [options]   验证指定 pack 的 runtime 是否就绪

build 参数:
  --base-tag <TAG>          基镜像 tag（必填）
  --os <OS>                 操作系统（默认 ubuntu24.04）
  --py-version <VER>        Python 版本（默认 py312）
  --push                    推送镜像到 ghcr.io
  --upload                  上传离线 tar 到 OBS
  --multi-arch              多架构构建
  --use-cache               使用 Docker 缓存
  --harbor-wheel <PATH>     本地 harbor wheel 文件（替换默认 harbor==0.20.0）
  --l2                      同时构建 L2 baked images
  --l2-cases <LIST>         L2 case ID 列表，逗号分隔（默认 11099,12308,13741）
  --l2-agents <LIST>        L2 agent 列表，逗号分隔（默认 aider,msa,oh）

run 参数:
  --pack <NAME>             pack 名: harbor | swebench | swebench_pro（必填）
  --split <SPEC>            dataset/split 标识（自动派生 config 文件）
  --agent <NAME>            agent 名（matrix filter 用）
  --command <CMD>           透传到容器内的完整 shell 命令（自动推导时可选）
  --container-name <NAME>   runtime 容器名（默认 ais_bench_agent）
  --mode A|B                DinD / Socket（默认自动检测）
  --datasets <PATH>         宿主数据集路径（可多次，绝对路径）
  --runtime-image <TAG>     runtime 镜像 tag
  --runtime-tar <PATH>      离线 runtime tar 包
  --case-tar <PATH>         离线 case 镜像 tar（可多次，支持目录）
  --matrix-yaml <PATH>      matrix.yaml 宿主路径
  --bind-jobs <DIR>         jobs 宿主目录
  --bind-tasks <DIR>        tasks 宿主目录
  --bind-config <DIR>       config 宿主目录
  --api-key-file <PATH>     api_key.env 宿主路径
  --registry-mirror <URL>   registry 镜像站点
  --l2-image <TAG>          L2 baked image tag
  --data-image <TAG>        只读 data container 镜像
  --host-path <PATH>        模式 B 时 /benchmark 提取目标
  --production              生产模式（容器 --restart unless-stopped）
  --qemu auto|yes|no        QEMU 用户态模拟（默认 auto）
  --skip-doctor             跳过 doctor 自检
  --dry-run                 只打印不执行

示例:
  # 构建 runtime 镜像（用自定义 harbor wheel）
  ais_bench_agent.sh build --base-tag v3.1-20260522-master \
      --harbor-wheel /path/to/harbor-offline.whl --push

  # 构建 runtime + L2 baked images
  ais_bench_agent.sh build --base-tag v3.1-20260522-master --l2

  # 跑 SWE-bench verified mini（自动启容器 + 自检 + harbor jobs start）
  ais_bench_agent.sh run --pack swebench --split verified_mini \
      --datasets /data/swebench/verified \
      --matrix-yaml /opt/config/matrix.yaml \
      --api-key-file /opt/config/api_key.env

  # 跑 harbor terminal-bench-2（自定义命令）
  ais_bench_agent.sh run --pack harbor \
      --datasets /data/harbor/mini-0.10/terminal-bench-2-offline-selected_0.10 \
      --command "harbor jobs start -c /opt/swebench/config/matrix.yaml -n 2"

  # 查询状态
  ais_bench_agent.sh status --container-name ais_bench_agent
EOF
    exit 0
}

# ============================================================================
# pack → config 自动派生规则（粒度 B）
# ============================================================================
derive_config() {
    local pack="$1"
    local split="${2:-}"

    case "${pack}" in
        harbor)
            # harbor 只有一个 config，忽略 --split
            echo "ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py"
            ;;
        swebench)
            case "${split}" in
                ""|lite)
                    echo "ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_lite.py" ;;
                verified)
                    echo "ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_verified.py" ;;
                verified_mini)
                    echo "ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_verified_mini.py" ;;
                full)
                    echo "ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_full.py" ;;
                multilingual)
                    echo "ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_multilingual.py" ;;
                multilingual_mini)
                    echo "ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_multilingual_mini.py" ;;
                *)
                    err "--split 无效: ${split}"
                    echo "  swebench 支持: lite | verified | verified_mini | full | multilingual | multilingual_mini" >&2
                    exit 1 ;;
            esac
            ;;
        swebench_pro)
            case "${split}" in
                ""|mini)
                    echo "ais_bench/configs/swe_bench_pro_examples/mini_swe_agent_swe_bench_pro_mini.py" ;;
                full)
                    echo "ais_bench/configs/swe_bench_pro_examples/mini_swe_agent_swe_bench_pro_full.py" ;;
                *)
                    err "--split 无效: ${split}"
                    echo "  swebench_pro 支持: mini | full" >&2
                    exit 1 ;;
            esac
            ;;
        *)
            err "--pack 无效: ${pack}"
            echo "  支持: harbor | swebench | swebench_pro" >&2
            exit 1 ;;
    esac
}

# ============================================================================
# 自动推导 run 命令（用户未传 --command 时使用）
# ============================================================================
derive_run_command() {
    local pack="$1"
    local config="$2"
    local agent="${3:-}"

    # 优先 matrix.yaml 调度模式
    local matrix_yaml="${MATRIX_YAML:-/opt/swebench/config/matrix.yaml}"
    local job_name="job-${pack}-$(date +%Y%m%d-%H%M%S)"

    if [ -n "${MATRIX_YAML:-}" ]; then
        # 矩阵调度模式（推荐）：harbor jobs start
        echo "cd /benchmark && source /opt/swebench/api_key.env 2>/dev/null || true && harbor jobs start -c ${matrix_yaml} --job-name ${job_name} --jobs-dir /opt/swebench/jobs -n 2"
    else
        # 回退：原生 ais_bench 单 config 模式
        case "${pack}" in
            harbor)
                echo "agent_env harbor && ais_bench ${config} --debug" ;;
            swebench)
                echo "agent_env swebench && ais_bench ${config} --debug" ;;
            swebench_pro)
                echo "agent_env swebench_pro && ais_bench ${config} --debug" ;;
        esac
    fi
}

# ============================================================================
# build 子命令
# ============================================================================
cmd_build() {
    local BASE_TAG=""
    local OS="ubuntu24.04"
    local py_version="py312"
    local push=0
    local upload=0
    local multi_arch=0
    local use_cache=0
    local harbor_wheel=""
    local build_l2=0
    local l2_cases="11099,12308,13741"
    local l2_agents="aider,msa,oh"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --base-tag)       BASE_TAG="$2"; shift 2 ;;
            --os)             OS="$2"; shift 2 ;;
            --py-version)     py_version="$2"; shift 2 ;;
            --push)           push=1; shift ;;
            --upload)         upload=1; shift ;;
            --multi-arch)     multi_arch=1; shift ;;
            --use-cache)      use_cache=1; shift ;;
            --harbor-wheel)   harbor_wheel="$2"; shift 2 ;;
            --l2)             build_l2=1; shift ;;
            --l2-cases)       l2_cases="$2"; shift 2 ;;
            --l2-agents)      l2_agents="$2"; shift 2 ;;
            --dry-run)        DRY_RUN=1; shift ;;
            -h|--help)        usage ;;
            *) err "未知参数: $1"; usage ;;
        esac
    done

    if [ -z "${BASE_TAG}" ]; then
        err "--base-tag 是必填参数"
        echo "  示例: --base-tag v3.1-20260522-master"
        exit 1
    fi

    log "=== 构建 L4 runtime 镜像 ==="
    log "  base-tag:     ${BASE_TAG}"
    log "  os:           ${OS}"
    log "  py-version:   ${py_version}"
    log "  push:         ${push}"
    log "  upload:       ${upload}"
    log "  multi-arch:   ${multi_arch}"
    log "  use-cache:    ${use_cache}"
    [ -n "${harbor_wheel}" ] && log "  harbor-wheel: ${harbor_wheel}"
    [ "${build_l2}" = "1" ] && log "  L2:           是 (cases=${l2_cases}, agents=${l2_agents})"

    if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] bash ${SCRIPT_DIR}/build_image_agent_runtime.sh ..."
        return
    fi

    # 拼装 build_image_agent_runtime.sh 参数
    local build_args=(
        --base-tag "${BASE_TAG}"
        --os "${OS}"
        --py-version "${py_version}"
    )
    [ -n "${harbor_wheel}" ] && build_args+=(--harbor-wheel "${harbor_wheel}")
    [ "${push}" = "1" ]       && build_args+=(--push 1)
    [ "${upload}" = "1" ]     && build_args+=(--upload 1)
    [ "${multi_arch}" = "1" ] && build_args+=(--multi-arch 1)
    [ "${use_cache}" = "1" ]  && build_args+=(--use-cache 1)

    bash "${SCRIPT_DIR}/build_image_agent_runtime.sh" "${build_args[@]}"

    # 可选：构建 L2 baked images
    if [ "${build_l2}" = "1" ]; then
        log "=== 构建 L2 baked images ==="
        bash "${SCRIPT_DIR}/build_l2_baked_image.sh" \
            --case "${l2_cases}" \
            --agent "${l2_agents}"
    fi

    ok "build 完成"
}

# ============================================================================
# run 子命令
# ============================================================================
cmd_run() {
    # ---- 参数 ----
    local PACK=""
    local SPLIT=""
    local AGENT=""
    local COMMAND=""
    local MODE=""
    local DATASETS=()
    local RUNTIME_IMAGE=""
    local RUNTIME_TAR=""
    local CASE_TARS=()
    local MATRIX_YAML=""
    local BIND_JOBS=""
    local BIND_TASKS=""
    local BIND_CONFIG=""
    local API_KEY_FILE=""
    local REGISTRY_MIRROR=""
    local L2_IMAGE=""
    local DATA_IMAGE=""
    local HOST_PATH=""
    local PRODUCTION=0
    local QEMU=""
    local SKIP_DOCTOR=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --pack)             PACK="$2"; shift 2 ;;
            --split)            SPLIT="$2"; shift 2 ;;
            --agent)            AGENT="$2"; shift 2 ;;
            --command)          COMMAND="$2"; shift 2 ;;
            --container-name)   CONTAINER_NAME="$2"; shift 2 ;;
            --mode)             MODE="$2"; shift 2 ;;
            --datasets)
                [ -z "${2:-}" ] && { err "--datasets 需要一个绝对路径"; exit 1; }
                [[ "$2" != /* ]] && { err "--datasets 必须是绝对路径: $2"; exit 1; }
                DATASETS+=("$2")
                shift 2 ;;
            --runtime-image)    RUNTIME_IMAGE="$2"; shift 2 ;;
            --runtime-tar)      RUNTIME_TAR="$2"; shift 2 ;;
            --case-tar)         CASE_TARS+=("$2"); shift 2 ;;
            --matrix-yaml)      MATRIX_YAML="$2"; shift 2 ;;
            --bind-jobs)        BIND_JOBS="$2"; shift 2 ;;
            --bind-tasks)       BIND_TASKS="$2"; shift 2 ;;
            --bind-config)      BIND_CONFIG="$2"; shift 2 ;;
            --api-key-file)     API_KEY_FILE="$2"; shift 2 ;;
            --registry-mirror)  REGISTRY_MIRROR="$2"; shift 2 ;;
            --l2-image)         L2_IMAGE="$2"; shift 2 ;;
            --data-image)       DATA_IMAGE="$2"; shift 2 ;;
            --host-path)        HOST_PATH="$2"; shift 2 ;;
            --production)       PRODUCTION=1; shift ;;
            --qemu)             QEMU="$2"; shift 2 ;;
            --skip-doctor)      SKIP_DOCTOR=1; shift ;;
            --dry-run)          DRY_RUN=1; shift ;;
            -h|--help)          usage ;;
            *) err "未知参数: $1"; usage ;;
        esac
    done

    # ---- 校验 ----
    if [ -z "${PACK}" ]; then
        err "--pack 是必填参数 (harbor | swebench | swebench_pro)"
        exit 1
    fi

    # ---- Step 1: 派生 config ----
    local DERIVED_CONFIG
    DERIVED_CONFIG=$(derive_config "${PACK}" "${SPLIT}")
    log "pack: ${PACK}, split: ${SPLIT:-默认}"
    log "派生 config: ${DERIVED_CONFIG}"

    # ---- Step 2: 推导命令 ----
    if [ -z "${COMMAND}" ]; then
        COMMAND=$(derive_run_command "${PACK}" "${DERIVED_CONFIG}" "${AGENT}")
        log "自动推导命令: ${COMMAND}"
    else
        log "用户指定命令: ${COMMAND}"
    fi

    # ---- Step 3: 检查容器是否已运行 ----
    local CONTAINER_RUNNING=false
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
        CONTAINER_RUNNING=true
        log "复用已有容器: ${CONTAINER_NAME}"
    fi

    # ---- Step 4: 未运行则 bootstrap ----
    if [ "${CONTAINER_RUNNING}" = "false" ]; then
        log "容器未运行，启动 bootstrap..."

        local BOOTSTRAP_ARGS=()
        [ -n "${MODE}" ]             && BOOTSTRAP_ARGS+=(--mode "${MODE}")
        for d in "${DATASETS[@]}"; do
            BOOTSTRAP_ARGS+=(--datasets "$d")
        done
        [ -n "${RUNTIME_TAR}" ]      && BOOTSTRAP_ARGS+=(--runtime-tar "${RUNTIME_TAR}")
        for c in "${CASE_TARS[@]}"; do
            BOOTSTRAP_ARGS+=(--case-tar "$c")
        done
        BOOTSTRAP_ARGS+=(--container-name "${CONTAINER_NAME}")
        [ -n "${RUNTIME_IMAGE}" ]    && BOOTSTRAP_ARGS+=(--runtime-image "${RUNTIME_IMAGE}")
        [ -n "${MATRIX_YAML}" ]      && BOOTSTRAP_ARGS+=(--matrix-yaml "${MATRIX_YAML}")
        [ -n "${BIND_JOBS}" ]        && BOOTSTRAP_ARGS+=(--bind-jobs "${BIND_JOBS}")
        [ -n "${BIND_TASKS}" ]       && BOOTSTRAP_ARGS+=(--bind-tasks "${BIND_TASKS}")
        [ -n "${BIND_CONFIG}" ]      && BOOTSTRAP_ARGS+=(--bind-config "${BIND_CONFIG}")
        [ -n "${API_KEY_FILE}" ]     && BOOTSTRAP_ARGS+=(--api-key-file "${API_KEY_FILE}")
        [ -n "${REGISTRY_MIRROR}" ]  && BOOTSTRAP_ARGS+=(--registry-mirror "${REGISTRY_MIRROR}")
        [ -n "${L2_IMAGE}" ]         && BOOTSTRAP_ARGS+=(--l2-image "${L2_IMAGE}")
        [ -n "${DATA_IMAGE}" ]       && BOOTSTRAP_ARGS+=(--data-image "${DATA_IMAGE}")
        [ -n "${HOST_PATH}" ]        && BOOTSTRAP_ARGS+=(--host-path "${HOST_PATH}")
        [ "${PRODUCTION}" = "1" ]    && BOOTSTRAP_ARGS+=(--production)
        [ -n "${QEMU}" ]             && BOOTSTRAP_ARGS+=(--qemu "${QEMU}")

        if [ "${DRY_RUN}" = "1" ]; then
            echo "[dry-run] bash ${SCRIPT_DIR}/ais_bench_agent_bootstrap.sh ${BOOTSTRAP_ARGS[*]}"
        else
            bash "${SCRIPT_DIR}/ais_bench_agent_bootstrap.sh" "${BOOTSTRAP_ARGS[@]}"
        fi
    fi

    # ---- Step 5: doctor 自检 ----
    if [ "${SKIP_DOCTOR}" = "0" ] && [ "${DRY_RUN}" = "0" ]; then
        log "执行 doctor 自检 (pack: ${PACK})..."
        docker exec "${CONTAINER_NAME}" \
            bash /usr/local/bin/ais_bench_agent_doctor.sh "${PACK}" || {
            warn "doctor 自检未完全通过，继续执行命令..."
        }
    fi

    # ---- Step 6: 执行测试命令 ----
    log "容器内执行: ${COMMAND}"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] docker exec ${CONTAINER_NAME} bash -c '${COMMAND}'"
    else
        docker exec "${CONTAINER_NAME}" bash -c "${COMMAND}"
        local exit_code=$?
        if [ ${exit_code} -eq 0 ]; then
            ok "命令执行完成 (exit=0)"
        else
            warn "命令执行完成 (exit=${exit_code})"
        fi
    fi
}

# ============================================================================
# status 子命令（委托 orchestrator_status.sh）
# ============================================================================
cmd_status() {
    bash "${SCRIPT_DIR}/ais_bench_agent_orchestrator_status.sh" "$@"
}

# ============================================================================
# watch 子命令（委托 watch.sh）
# ============================================================================
cmd_watch() {
    bash "${SCRIPT_DIR}/ais_bench_agent_watch.sh" "$@"
}

# ============================================================================
# summarize 子命令（委托 summarize.sh）
# ============================================================================
cmd_summarize() {
    bash "${SCRIPT_DIR}/ais_bench_agent_summarize.sh" "$@"
}

# ============================================================================
# doctor 子命令（委托 doctor.sh）
# ============================================================================
cmd_doctor() {
    bash "${SCRIPT_DIR}/doctor.sh" "$@"
}

# ============================================================================
# 主入口
# ============================================================================
main() {
    if [ $# -eq 0 ]; then
        usage
    fi

    local COMMAND="$1"
    shift

    case "${COMMAND}" in
        build)     cmd_build "$@" ;;
        run)       cmd_run "$@" ;;
        status)    cmd_status "$@" ;;
        watch)     cmd_watch "$@" ;;
        summarize) cmd_summarize "$@" ;;
        doctor)    cmd_doctor "$@" ;;
        -h|--help) usage ;;
        *)
            err "未知命令: ${COMMAND}"
            echo "  支持: build | run | status | watch | summarize | doctor"
            usage ;;
    esac
}

main "$@"