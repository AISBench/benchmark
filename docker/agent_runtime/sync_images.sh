#!/bin/bash
# sync_images.sh
#
# 将 host dockerd 上的 case 镜像流式导入 runtime 容器的 DinD daemon。
#
# 背景：
#   agent-runtime 容器内置独立 dockerd（DinD），trial 容器由它拉起，
#   因此 case 环境镜像必须存在于容器内 daemon 的本地缓存。
#   宿主机/CI 环境通常无法直连 Docker Hub（registry-1.docker.io 被墙），
#   但 case 镜像可通过离线包（docker load）预先装载到 host dockerd 上
#   （参见 terminal-bench-2-1 仓库的 package_images.sh 产物）。
#   本脚本提供 host -> 容器 的批量搬运通道：docker save | docker exec -i docker load，
#   管道直传，不在 host 落任何中间文件。
#
# 本脚本在 host 上执行（不是容器内）。
#
# 用法:
#   sync_images.sh <case_name>...     # 按目录名同步指定 case，如 sync_images.sh fix-git regex-log
#   sync_images.sh --all              # 同步 host 上全部缺失的 case 镜像
#   sync_images.sh --list             # 只显示差异清单，不导入
#   sync_images.sh -h, --help
#
# 可通过环境变量覆盖默认值:
#   CONTAINER   runtime 容器名（默认 agent-runtime）
#   PREFIX      镜像仓库前缀（默认 alexgshaw/）
#   TAG         镜像 tag（默认 20260430）

set -eo pipefail

CONTAINER="${CONTAINER:-agent-runtime}"
PREFIX="${PREFIX:-alexgshaw/}"
TAG="${TAG:-20260430}"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "[错误] $*" >&2; exit 1; }

usage() { sed -n '2,26p' "$0"; }

# 按前缀 + tag 过滤镜像名（从 stdin 读列表）
filter_images() {
    awk -v p="$PREFIX" -v t=":$TAG" \
        'substr($0,1,length(p))==p && substr($0,length($0)-length(t)+1)==t'
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

# ---- 前置检查：host docker 与 runtime 容器均可用 ----
command -v docker >/dev/null || fail "host 上未找到 docker 命令"
docker info >/dev/null 2>&1 || fail "host dockerd 不可用"
docker exec "$CONTAINER" docker info >/dev/null 2>&1 \
    || fail "runtime 容器 $CONTAINER 不可达或其内部 dockerd 未就绪"

# ---- 计算待同步清单 ----
if [ "${1:-}" = "--all" ] || [ "${1:-}" = "--list" ]; then
    # 差集 = host 全集 - 容器内已有
    mapfile -t pending < <(
        docker images --format '{{.Repository}}:{{.Tag}}' | filter_images | sort -u > /tmp/.sync_host.$$
        docker exec "$CONTAINER" docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | filter_images | sort -u > /tmp/.sync_have.$$
        comm -23 /tmp/.sync_host.$$ /tmp/.sync_have.$$
        rm -f /tmp/.sync_host.$$ /tmp/.sync_have.$$
    )
    mode="${1}"
else
    [ $# -ge 1 ] || { usage; exit 1; }
    pending=()
    for c in "$@"; do
        img="${PREFIX%/}/${c}:${TAG}"
        docker image inspect "$img" >/dev/null 2>&1 || { log "跳过 $c：host 上不存在 $img"; continue; }
        pending+=("$img")
    done
    mode="named"
fi

if [ "${#pending[@]}" -eq 0 ]; then
    log "没有需要同步的镜像（容器内已齐全或 host 上不存在）"
    exit 0
fi

log "待导入 ${#pending[@]} 个镜像："
printf '  - %s\n' "${pending[@]}"

[ "$mode" = "--list" ] && exit 0

# ---- 管道直传：host docker save -> 容器内 docker load ----
log "开始导入（流式传输，不落盘）..."
docker save "${pending[@]}" | docker exec -i "$CONTAINER" docker load
log "完成。容器内当前 ${PREFIX%.}/*:${TAG} 镜像数：$(docker exec "$CONTAINER" docker images --format '{{.Repository}}:{{.Tag}}' | filter_images | wc -l)"
