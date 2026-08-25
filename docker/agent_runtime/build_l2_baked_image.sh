#!/bin/bash
# ============================================================================
# build_l2_baked_image.sh — 5 层 DinD L1/L2 baked image 一键构建
# ============================================================================
#
# 设计:
#   - 把 mini_matrix 5 层 DinD L1 (case-base) + L2 (agent) baked image 构建
#     搬到 PR #410 runtime 镜像内,容器内用户可一键 build
#   - 模板:/opt/swebench/dockerfiles/Dockerfile.l{1,2}-*.j2
#     - L1:FROM 上游 prebuilt → WORKDIR /testbed + /logs
#     - L2:FROM L1 → 装 agent(aider / msa / oh / qwen)
#   - 与 mini_matrix scripts/build_baked_v2.sh 等价,但:
#     - 默认参数对齐 batch B 的 --bind-tasks 路径(/opt/swebench/tasks)
#     - 输出 image tag 命名沿用 mini_matrix 规范
#     - 默认开 ARM64 emulation (QEMU + binfmt)
#
# 用法:
#   build_l2_baked_image.sh                                    # 9 个 L2 全 build (3 case × 3 agent)
#   build_l2_baked_image.sh --case 11099                       # 单 case 全 agent
#   build_l2_baked_image.sh --agent aider                      # 所有 case 同一 agent
#   build_l2_baked_image.sh --l1-only                          # 只 build L1 (跳过 L2)
#   build_l2_baked_image.sh --l2-only                          # 只 build L2 (假设 L1 已 build)
#   build_l2_baked_image.sh --dataset-prefix django            # 改 dataset 前缀
#   build_l2_baked_image.sh --registry mirror.aliyuncs.com     # 改 prebuilt registry 镜像
#   build_l2_baked_image.sh --skip-arm64                       # 不加 --platform linux/amd64 (host 是 x86 时用)
#
# 适用:
#   - 容器内调用 (L4 镜像内建 docker,直接 docker build)
#   - host 调用:bash docker/agent_runtime/build_l2_baked_image.sh ...
#   - 模板路径自动检测:容器内 /opt/swebench/dockerfiles,host 用 ./docker/agent_runtime/dockerfiles/
#
# ============================================================================
set -euo pipefail

# ============ 默认值 ============
CASES=(11099 12308 13741)
AGENTS=(aider msa oh)
DATASET_PREFIX="django"
TASKS_DIR="/opt/swebench/tasks"        # batch B --bind-tasks 注入
DOCKERFILES_ROOT="/opt/swebench/dockerfiles"  # batch D2 注入
DEFAULT_REGISTRY="docker.1ms.run"
PLATFORM_FLAG="--platform linux/amd64"  # ARM64 host 默认需要
LOGDIR="/tmp/baked-builds-v3"
BUILD_L1=1
BUILD_L2=1

# ============ 参数解析 ============
while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)          shift; CASES=("$@"); break ;;
        --agent)         shift; AGENTS=("$@"); break ;;
        --l1-only)       BUILD_L2=0; shift ;;
        --l2-only)       BUILD_L1=0; shift ;;
        --dataset-prefix) DATASET_PREFIX="$2"; shift 2 ;;
        --tasks-dir)     TASKS_DIR="$2"; shift 2 ;;
        --registry)      DEFAULT_REGISTRY="$2"; shift 2 ;;
        --skip-arm64)    PLATFORM_FLAG=""; shift ;;
        --logdir)        LOGDIR="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "[错误] 未知参数: $1" >&2; exit 1 ;;
    esac
done

# ============ 工具函数 ============
log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "[错误] $*" >&2; exit 1; }

# ============ 自动检测 host vs 容器内 ============
if [ -d "$DOCKERFILES_ROOT" ]; then
    # 容器内视角:模板已由 D2 commit COPY 到 /opt/swebench/dockerfiles
    ACTIVE_DOCKERFILES_ROOT="$DOCKERFILES_ROOT"
    log "[context] 容器内视角,模板路径: $ACTIVE_DOCKERFILES_ROOT"
elif [ -d "./docker/agent_runtime/dockerfiles" ]; then
    # host 视角:从仓库根目录跑
    ACTIVE_DOCKERFILES_ROOT="./docker/agent_runtime/dockerfiles"
    log "[context] host 视角,模板路径: $ACTIVE_DOCKERFILES_ROOT"
else
    fail "找不到 dockerfiles/ 目录(容器内应 /opt/swebench/dockerfiles,host 应 ./docker/agent_runtime/dockerfiles)"
fi

# ============ 检查 docker 可用 ============
command -v docker >/dev/null 2>&1 || fail "docker 命令不可用(需在 L4 runtime 镜像内或 host docker daemon 上)"
docker info >/dev/null 2>&1 || fail "docker daemon 不可达(确认 dockerd 已启动)"

mkdir -p "$LOGDIR"

# ============ 解析 case → 上游 prebuilt image ============
# 命名规范(沿用 mini_matrix):
#   上游:swebench/sweb.eval.<arch>.<dataset>_<dataset_id>_<case_slug>:latest
#   例如:docker.1ms.run/swebench/sweb.eval.x86_64.django_1776_django-11099:latest
#
# 这里采用简单约定:<DATASET_PREFIX>_<case_id>,实际项目里可能需要映射表
prebuilt_for_case() {
    local case_id="$1"
    echo "${DEFAULT_REGISTRY}/swebench/sweb.eval.x86_64.${DATASET_PREFIX}_1776_${DATASET_PREFIX}-${case_id}:latest"
}

# ============ 校验模板存在 ============
[ -f "$ACTIVE_DOCKERFILES_ROOT/Dockerfile.l1-base.j2" ] \
    || fail "缺 L1 模板: $ACTIVE_DOCKERFILES_ROOT/Dockerfile.l1-base.j2"
for agent in "${AGENTS[@]}"; do
    [ -f "$ACTIVE_DOCKERFILES_ROOT/Dockerfile.l2-agent-${agent}.j2" ] \
        || fail "缺 L2 模板 (agent=$agent): $ACTIVE_DOCKERFILES_ROOT/Dockerfile.l2-agent-${agent}.j2"
done

# ============ 检查 TASKS_DIR (L1 需要 /testbed 上下文,但实际 L1 只用 prebuilt image) ============
# L1 j2 模板不需要 host 上下文,只需 docker build -f <template> -t <tag> <empty context>
# 这里我们用空目录 /tmp/l1-empty 兜底
mkdir -p /tmp/l1-empty /tmp/l2-empty

log ""
log "=== build_l2_baked_image.sh ==="
log "  cases:          ${CASES[*]}"
log "  agents:         ${AGENTS[*]}"
log "  tasks-dir:      $TASKS_DIR (not directly used by templates, kept for ref)"
log "  dockerfiles:    $ACTIVE_DOCKERFILES_ROOT"
log "  registry:       $DEFAULT_REGISTRY"
log "  platform:       ${PLATFORM_FLAG:-(host native)}"
log "  build L1:       $BUILD_L1"
log "  build L2:       $BUILD_L2"
log "  logdir:         $LOGDIR"
log ""

# ============ Step 1: build L1 case-base images ============
if [ "$BUILD_L1" = "1" ]; then
    log "=== [L1] Building case-base images ==="
    L1_PIDS=()
    for case in "${CASES[@]}"; do
        base_tag="swebench/${DATASET_PREFIX}-${case}-base:latest"
        base_src=$(prebuilt_for_case "$case")

        if docker images --format "{{.Repository}}:{{.Tag}}" | grep -qx "$base_tag"; then
            log "  [L1 skip] $base_tag already exists"
            continue
        fi

        log "  [L1 build] $base_tag (FROM $base_src)"
        (
            docker build $PLATFORM_FLAG \
                --build-arg "BASE_IMAGE=$base_src" \
                -f "$ACTIVE_DOCKERFILES_ROOT/Dockerfile.l1-base.j2" \
                -t "$base_tag" \
                /tmp/l1-empty \
                > "$LOGDIR/l1-${case}.log" 2>&1
        ) &
        L1_PIDS+=($!)
    done
    if [ ${#L1_PIDS[@]} -gt 0 ]; then
        wait "${L1_PIDS[@]}"
    fi

    log ""
    log "=== [L1] Verify ==="
    for case in "${CASES[@]}"; do
        base_tag="swebench/${DATASET_PREFIX}-${case}-base:latest"
        if docker images --format "{{.Repository}}:{{.Tag}}" | grep -qx "$base_tag"; then
            log "  [L1 ok] $base_tag"
        else
            log "  [L1 FAIL] $base_tag"
            log "    last log:"
            tail -5 "$LOGDIR/l1-${case}.log" 2>/dev/null | sed 's/^/      /'
            exit 1
        fi
    done
fi

# ============ Step 2: build L2 agent images ============
if [ "$BUILD_L2" = "1" ]; then
    log ""
    log "=== [L2] Building agent images ==="
    L2_PIDS=()
    for case in "${CASES[@]}"; do
        base_tag="swebench/${DATASET_PREFIX}-${case}-base:latest"
        for agent in "${AGENTS[@]}"; do
            l2_tag="swebench/${DATASET_PREFIX}-${case}-with-${agent}:latest"

            if docker images --format "{{.Repository}}:{{.Tag}}" | grep -qx "$l2_tag"; then
                log "  [L2 skip] $l2_tag already exists"
                continue
            fi

            log "  [L2 build] $l2_tag (FROM $base_tag, AGENT=$agent)"
            (
                docker build $PLATFORM_FLAG \
                    --build-arg "BASE_IMAGE=$base_tag" \
                    --build-arg "AGENT=$agent" \
                    -f "$ACTIVE_DOCKERFILES_ROOT/Dockerfile.l2-agent-${agent}.j2" \
                    -t "$l2_tag" \
                    /tmp/l2-empty \
                    > "$LOGDIR/l2-${case}-${agent}.log" 2>&1
            ) &
            L2_PIDS+=($!)
        done
    done
    if [ ${#L2_PIDS[@]} -gt 0 ]; then
        wait "${L2_PIDS[@]}"
    fi

    log ""
    log "=== [L2] Final image list ==="
    for case in "${CASES[@]}"; do
        for agent in "${AGENTS[@]}"; do
            l2_tag="swebench/${DATASET_PREFIX}-${case}-with-${agent}:latest"
            if docker images --format "{{.Repository}}:{{.Tag}}" | grep -qx "$l2_tag"; then
                log "  OK: $l2_tag"
            else
                log "  FAIL: $l2_tag"
                log "    last log:"
                tail -5 "$LOGDIR/l2-${case}-${agent}.log" 2>/dev/null | sed 's/^/      /'
            fi
        done
    done
fi

log ""
log "=== build_l2_baked_image.sh done ==="
log ""
log "下一步(接入 D3):"
log "  bootstrap.sh --l2-image 'swebench/${DATASET_PREFIX}-<case>-with-<agent>:latest'"
log "  → harbor trial 容器直接用 L2 baked image 启动,跳过 trial 内 agent install"