#!/bin/bash
# ============================================================================
# ais_bench_agent_orchestrator_status.sh — 5 层 DinD runtime 容器状态查询
# ============================================================================
#
# 设计:
#   - 一键查询 runtime 容器的健康状态（5 层 DinD L5 launcher 拉的容器）
#   - 输出 5 段:容器基础 / 内层 docker / bind mount 可见性 / jobs 数量 / harbor jobs
#   - 既可在容器内跑（用内层 docker CLI），也可在 host 跑（用 docker exec）
#
# 用法:
#   ais_bench_agent_orchestrator_status.sh                       # 自动检测
#   ais_bench_agent_orchestrator_status.sh --container-name X    # 指定 runtime 容器
#   ais_bench_agent_orchestrator_status.sh --jobs-dir /opt/swebench/jobs
#
# 适用:
#   - 5 层 DinD L3/L4/L5 健康检查
#   - bootstrap 完成后第一次验证
#   - 跑 harbor jobs start 前的快速自检
#
# ============================================================================
set -euo pipefail

# ============ 默认值 ============
JOBS_DIR="/opt/swebench/jobs"
API_KEY_FILE="/opt/swebench/api_key.env"
DOCKER_SOCK="/var/run/docker.sock"

# ============ 参数解析 ============
CONTAINER_NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --container-name)  CONTAINER_NAME="$2"; shift 2 ;;
        --jobs-dir)        JOBS_DIR="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "[错误] 未知参数: $1" >&2; exit 1 ;;
    esac
done

# ============ 工具函数 ============
log() { echo "[$(date +%H:%M:%S)] $*"; }
section() {
    echo ""
    echo "─── $* ───"
}

# ============ 自动检测调用上下文 ============
# 检测当前 shell 是否在 runtime 容器内:
#   - /var/run/docker.sock 可访问（内层 daemon 或宿主 socket）
#   - /opt/swebench 路径下有 host bind mount 内容
INSIDE_CONTAINER=0
if [ -S "$DOCKER_SOCK" ] && [ -d /opt/swebench ]; then
    # 检查 /opt/swebench 是否真是 bind mount（有内容或可访问）
    # （PR #410 的 image 默认 /opt/swebench 不存在，由 A5 预创建 + B3 bind mount 内容）
    if mount | grep -q "/opt/swebench" 2>/dev/null || [ -n "$(ls -A /opt/swebench 2>/dev/null)" ]; then
        INSIDE_CONTAINER=1
    fi
fi

if [ "$INSIDE_CONTAINER" = "1" ]; then
    log "=== ais_bench_agent_orchestrator_status.sh（容器内视角）==="
    # 容器内：直接用内层 docker CLI
    DOCKER_CMD=(docker)
    # jobs dir / api key 等用 bind mount 的容器内路径
    EFFECTIVE_JOBS_DIR="$JOBS_DIR"
    EFFECTIVE_API_KEY="$API_KEY_FILE"
else
    log "=== ais_bench_agent_orchestrator_status.sh（host 视角）==="
    # host：自动找 runtime 容器（第一个名字含 ais_bench_agent 的 running 容器）
    if [ -z "$CONTAINER_NAME" ]; then
        CONTAINER_NAME=$(docker ps --format '{{.Names}}' 2>/dev/null \
            | grep -E 'ais_bench_agent|swebench-orchestrator' | head -1 || true)
        if [ -z "$CONTAINER_NAME" ]; then
            log "[错误] 没找到 runtime 容器,显式传 --container-name"
            log "  docker ps -a | grep ais_bench"
            exit 1
        fi
        log "  auto-detected container: $CONTAINER_NAME"
    fi
    # host：docker exec 进容器跑内层检查
    DOCKER_CMD=(docker exec "$CONTAINER_NAME")
    EFFECTIVE_JOBS_DIR="$JOBS_DIR"  # 同路径（jobs bind mount 在两边都可见）
    EFFECTIVE_API_KEY="$API_KEY_FILE"
fi

# ============ 1. 容器基础状态 ============
section "1. 容器基础"
if [ "$INSIDE_CONTAINER" = "1" ]; then
    log "  hostname:    $(hostname)"
    log "  uptime:      $(uptime -p 2>/dev/null || uptime)"
    log "  arch:        $(uname -m)"
    log "  in-container: yes"
else
    log "  container:    $CONTAINER_NAME"
    log "  state:        $(docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo '?')"
    log "  uptime:       $(docker inspect --format '{{.State.StartedAt}}' "$CONTAINER_NAME" 2>/dev/null || echo '?')"
    log "  image:        $(docker inspect --format '{{.Config.Image}}' "$CONTAINER_NAME" 2>/dev/null || echo '?')"
    log "  restart:      $(docker inspect --format '{{.HostConfig.RestartPolicy.Name}}' "$CONTAINER_NAME" 2>/dev/null || echo '?')"
fi

# ============ 2. 内层 docker daemon ============
section "2. 内层 docker daemon"
if [ "$INSIDE_CONTAINER" = "1" ]; then
    if command -v docker >/dev/null 2>&1; then
        log "  docker:       $(docker --version 2>/dev/null || echo 不可用)"
        if docker info >/dev/null 2>&1; then
            log "  server-ver:   $(docker info --format '{{.ServerVersion}}' 2>/dev/null || echo '?')"
            log "  storage:      $(docker info --format '{{.Driver}}' 2>/dev/null || echo '?')"
            log "  cgroup:       $(docker info --format '{{.CgroupDriver}}' 2>/dev/null || echo '?')"
            log "  containers:   $(docker ps -q 2>/dev/null | wc -l) running, $(docker ps -aq 2>/dev/null | wc -l) total"
            log "  images:       $(docker images -q 2>/dev/null | wc -l)"
            log "  default-platform: ${DOCKER_DEFAULT_PLATFORM:-未设置}"
        else
            log "  [错误] docker info 失败,daemon 可能未启动"
        fi
    else
        log "  [错误] docker 命令不可用"
    fi
else
    log "  (host 视角无法直接查内层 daemon,用 docker exec 进入查看)"
    log "  docker exec $CONTAINER_NAME docker info"
fi

# ============ 3. bind mount 可见性 ============
section "3. bind mount 可见性"
for p in /opt/swebench/jobs /opt/swebench/tasks /opt/swebench/config /opt/swebench/logs; do
    if [ -d "$p" ]; then
        SIZE=$("${DOCKER_CMD[@]}" du -sh "$p" 2>/dev/null | cut -f1 || echo "?")
        FILES=$("${DOCKER_CMD[@]}" find "$p" -maxdepth 2 -type f 2>/dev/null | wc -l || echo 0)
        log "  ✓ $p  (size=$SIZE, files=$FILES)"
    else
        log "  ✗ $p  不存在"
    fi
done

# api_key.env 检查
if "${DOCKER_CMD[@]}" test -f "$EFFECTIVE_API_KEY" 2>/dev/null; then
    log "  ✓ api_key.env: $EFFECTIVE_API_KEY"
    KEY_LEN=$("${DOCKER_CMD[@]}" bash -c "grep -c OPENAI_API_KEY= $EFFECTIVE_API_KEY" 2>/dev/null || echo 0)
    log "      OPENAI_API_KEY lines: $KEY_LEN"
else
    log "  ✗ api_key.env: $EFFECTIVE_API_KEY 不存在"
fi

# ============ 4. jobs 数量 ============
section "4. jobs 目录状态"
if "${DOCKER_CMD[@]}" test -d "$EFFECTIVE_JOBS_DIR" 2>/dev/null; then
    N_JOBS=$("${DOCKER_CMD[@]}" find "$EFFECTIVE_JOBS_DIR" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l || echo 0)
    N_DONE=$("${DOCKER_CMD[@]}" find "$EFFECTIVE_JOBS_DIR" -maxdepth 2 -name 'result.json' 2>/dev/null | wc -l || echo 0)
    N_RUN=$("${DOCKER_CMD[@]}" find "$EFFECTIVE_JOBS_DIR" -maxdepth 2 -name 'config.json' ! -exec test -e '{}/result.json' \; 2>/dev/null | wc -l || echo 0)
    log "  total jobs:        $N_JOBS"
    log "  with result.json:  $N_DONE (completed)"
    log "  running (估):      $N_RUN"
    if [ "$N_JOBS" -gt 0 ]; then
        log "  latest jobs:"
        "${DOCKER_CMD[@]}" bash -c "ls -1dt $EFFECTIVE_JOBS_DIR/*/ 2>/dev/null" | head -3 \
            | while read -r jd; do
                log "    - $(basename "$jd")"
            done
    fi
else
    log "  ✗ jobs-dir 不存在: $EFFECTIVE_JOBS_DIR"
fi

# ============ 5. harbor CLI 可用性 ============
section "5. harbor CLI"
if "${DOCKER_CMD[@]}" command -v harbor >/dev/null 2>&1; then
    HARBOR_VER=$("${DOCKER_CMD[@]}" harbor --version 2>&1 | head -1 || echo "?")
    log "  ✓ harbor: $HARBOR_VER"
    N_JOB_NAMES=$("${DOCKER_CMD[@]}" bash -c "harbor jobs list 2>/dev/null | wc -l" 2>/dev/null || echo "?")
    log "  harbor jobs: $N_JOB_NAMES 行(可读=正常)"
else
    log "  ✗ harbor CLI 不可用"
    log "    提示:bootstrap 没装 harbor?(v3 A2 commit 应已装 0.20.x)"
fi

# ============ 收尾 ============
section "✓ status 完成"
log "如需追踪 trial:"
log "  ais_bench_agent_watch.sh <job-name>"
log "  ais_bench_agent_summarize.sh"
log ""
log "如需启动新 trial:"
log "  ais_bench_agent_run.sh [--datasets 11099,12308] [--agents oracle]"