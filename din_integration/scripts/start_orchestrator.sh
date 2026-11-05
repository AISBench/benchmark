#!/bin/bash
# ============================================================================
# start_orchestrator.sh — 启动 DinD orchestrator 容器(bind mount 到 host)
# ============================================================================
#
# 关键设计:
#   - 所有状态(jobs / tasks / scripts / api_key.env)全部 bind mount 到 host
#   - 容器是无状态的:重建容器 = 数据零丢失
#   - harbor 0.20.x 在容器内跑,产物自动落到 /opt/swebench/jobs(已 mount)
#
# 用法:
#   bash scripts/start_orchestrator.sh           # 启动(若已存在则报错)
#   bash scripts/start_orchestrator.sh --recreate # 删除旧容器后重建(数据保留)
#
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ORCHESTRATOR_IMAGE="${ORCHESTRATOR_IMAGE:-swebench/orchestrator:v0.1-2026-08-04-patched-v11}"
CONTAINER_NAME="${CONTAINER_NAME:-swebench-orchestrator}"
DATA_CONTAINER_NAME="${DATA_CONTAINER_NAME:-swebench-data-3demo}"
DATA_IMAGE="${DATA_IMAGE:-swebench/swebench-data:v0.1-2026-07-30-3demo}"

RECREATE=0
for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE=1 ;;
        *) echo "Unknown arg: $arg"; exit 2 ;;
    esac
done

# --- 前置:秘钥文件必须存在 ---
if [[ ! -f "$ROOT/scripts/api_key.env" ]]; then
    echo "[ERROR] $ROOT/scripts/api_key.env not found. Copy from orchestrator or other host first."
    exit 1
fi

# --- 前置:jobs 目录必须存在(mount target) ---
mkdir -p "$ROOT/jobs" "$ROOT/tasks" "$ROOT/logs"

# --- 旧容器处理 ---
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    if [[ $RECREATE -eq 1 ]]; then
        echo "[start_orchestrator] Removing old container $CONTAINER_NAME (data is on host, safe)"
        docker rm -f "$CONTAINER_NAME" >/dev/null
    else
        echo "[start_orchestrator] Container $CONTAINER_NAME already exists. Use --recreate to replace."
        echo "  (Hint: docker exec -it $CONTAINER_NAME bash)"
        exit 0
    fi
fi

# --- 数据卷容器(只读,3 demo task 兜底) ---
if ! docker ps -a --format '{{.Names}}' | grep -qx "$DATA_CONTAINER_NAME"; then
    echo "[start_orchestrator] Creating data container $DATA_CONTAINER_NAME from $DATA_IMAGE"
    docker create --name "$DATA_CONTAINER_NAME" "$DATA_IMAGE"
fi

# --- 加载 API key ---
set -a
# shellcheck disable=SC1091
source "$ROOT/scripts/api_key.env"
set +a

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[ERROR] OPENAI_API_KEY is empty in scripts/api_key.env"
    exit 1
fi

# --- 启动 orchestrator ---
echo "[start_orchestrator] Starting $CONTAINER_NAME from $ORCHESTRATOR_IMAGE"
docker run -d \
    --name "$CONTAINER_NAME" \
    --hostname orchestrator \
    --privileged \
    --cgroupns=host \
    --restart unless-stopped \
    --volumes-from "$DATA_CONTAINER_NAME":ro \
    \
    -v "$ROOT/jobs":/opt/swebench/jobs:rw \
    -v "$ROOT/tasks":/opt/swebench/data/tasks:rw \
    -v "$ROOT/config":/opt/swebench/config:ro \
    -v "$ROOT/scripts/api_key.env":/opt/swebench/api_key.env:ro \
    -v "$ROOT/orchestrator/entrypoint.sh":/opt/swebench/scripts/entrypoint.sh:ro \
    -v "$ROOT/orchestrator/agent-patches":/opt/swebench/agent-patches:ro \
    -v "$ROOT/logs":/opt/swebench/logs:rw \
    \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.siliconflow.cn/v1}" \
    \
    "$ORCHESTRATOR_IMAGE" \
    bash -c 'tail -f /dev/null'

# 等 DinD ready
echo "[start_orchestrator] Waiting for DinD dockerd to be ready..."
for i in {1..60}; do
    if docker exec "$CONTAINER_NAME" docker info >/dev/null 2>&1; then
        echo "[start_orchestrator] dockerd ready"
        break
    fi
    sleep 2
done

echo "[start_orchestrator] Done. Container: $CONTAINER_NAME"
echo "  - jobs:    $ROOT/jobs → /opt/swebench/jobs"
echo "  - tasks:   $ROOT/tasks → /opt/swebench/data/tasks"
echo "  - config:  $ROOT/config → /opt/swebench/config"
echo "  - patches: $ROOT/orchestrator/agent-patches → /opt/swebench/agent-patches"
echo ""
echo "Next: bash scripts/run_matrix.sh"