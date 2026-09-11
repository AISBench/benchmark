#!/usr/bin/env bash
# ============================================================================
# AISBench harbor agent-runtime 一键启动脚本（双模式）
#
# 模式一：socket 模式（默认，DooD——复用宿主机 docker daemon）
#   原理：挂载宿主机 /var/run/docker.sock 进容器，容器内 docker 命令直接操作
#         宿主机 daemon。trial 容器由宿主机 dockerd 拉起，宿主机上已有的
#         case 镜像直接可见，无需任何镜像导入/打包动作。
#   约束：数据集 / agent deps / 产物目录 按「宿主机原路径透传」挂载（容器内
#         外同路径），保证 harbor 为 trial 容器生成的 bind mount 在宿主机
#         daemon 上能正确解析。
#   安全：容器内拿到 docker.sock 等价于宿主机 root 级 daemon 控制权，请勿在
#         该容器内运行不受信任的指令。
#
# 模式二：dind 模式（--mode dind，强隔离——容器内独立 daemon）
#   原理：--privileged + --cgroupns=host 在容器内拉起独立 dockerd，trial 容器
#         只出现在内层 daemon，不触碰宿主机 daemon。
#   代价：需要提前准备 case 镜像包（--case-images），启动时自动
#         docker load 进内层 daemon（全量导入，耗时数分钟；增量补充可用
#         sync_images.sh 从宿主机差集直传）。
#
# 用法：
#   # 默认 socket 模式
#   ./start_agent_runtime.sh \
#       --image         aisbench/agent-runtime:harbor-only \
#       --dataset       /abs/path/terminal-bench-2 \   # tb2 数据集（绝对路径）
#       --agent-deps    /abs/path/agent-deps-dir \     # agent deps bundle
#       --agent         terminus-2 \
#       [--name agent-runtime] [--trials-dir ~/harbor-trials]
#
#   # dind 模式（强隔离）
#   ./start_agent_runtime.sh --mode dind \
#       --image aisbench/agent-runtime:harbor-only \
#       --dataset /abs/path/terminal-bench-2 \
#       --case-images /abs/path/case-images.tar[.gz] \
#       --agent-deps /abs/path/agent-deps-dir \
#       --agent terminus-2 \
#       [--name agent-runtime] [--trials-dir ~/harbor-trials]
# ============================================================================
set -euo pipefail

CONTAINER_NAME="agent-runtime"
MODE="socket"
TRIALS_DIR="${HOME}/harbor-trials"
DOCKER_SOCK="/var/run/docker.sock"

usage() { grep '^#' "$0" | head -46; exit 1; }
die() { echo "[start_agent_runtime] ERROR: $*" >&2; exit 1; }

IMAGE="" DATASET="" CASE_IMAGES="" AGENT_DEPS="" AGENT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --image)      IMAGE="$2"; shift 2 ;;
        --mode)       MODE="$2"; shift 2 ;;
        --dataset)    DATASET="$2"; shift 2 ;;
        --case-images) CASE_IMAGES="$2"; shift 2 ;;
        --agent-deps) AGENT_DEPS="$2"; shift 2 ;;
        --agent)      AGENT="$2"; shift 2 ;;
        --name)       CONTAINER_NAME="$2"; shift 2 ;;
        --trials-dir) TRIALS_DIR="$2"; shift 2 ;;
        --docker-sock) DOCKER_SOCK="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *) die "未知参数: $1（--help 查看用法）" ;;
    esac
done

[ "$MODE" = "socket" ] || [ "$MODE" = "dind" ] \
    || die "--mode 仅支持 socket | dind（默认 socket）"
[ -n "$IMAGE" ]      || die "缺少 --image"
[ -n "$DATASET" ]    || die "缺少 --dataset（tb2 数据集目录，绝对路径）"
[ -n "$AGENT_DEPS" ] || die "缺少 --agent-deps（agent deps bundle 目录，绝对路径）"
[ -n "$AGENT" ]      || die "缺少 --agent（agent 名称，如 terminus-2）"

# 同路径透传挂载要求绝对路径（socket 模式硬约束；dind 模式同样统一）
for p in "$DATASET" "$AGENT_DEPS"; do
    [ -d "$p" ] || die "目录不存在: $p"
    [ "$p" = "$(cd "$p" && pwd)" ] || die "必须使用绝对路径（不含符号链接尾巴）: $p"
done

if [ "$MODE" = "socket" ]; then
    [ -S "$DOCKER_SOCK" ] || die "宿主机 docker socket 不存在: $DOCKER_SOCK"
else
    [ -n "$CASE_IMAGES" ] || die "dind 模式缺少 --case-images（case 镜像包 tar/tar.gz）"
    [ -f "$CASE_IMAGES" ] || die "case 镜像包不存在: $CASE_IMAGES"
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "本机没有镜像 $IMAGE（先 docker load 离线包）"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    die "容器 $CONTAINER_NAME 已存在，请先 docker rm -f $CONTAINER_NAME 或用 --name 换名"
fi

mkdir -p "$TRIALS_DIR"
TRIALS_DIR_ABS="$(cd "$TRIALS_DIR" && pwd)"

# ---------------------------------------------------------------------------
# 1. 启动外层容器
# ---------------------------------------------------------------------------
if [ "$MODE" = "socket" ]; then
    echo "==> 启动容器 $CONTAINER_NAME（socket 模式，复用宿主机 daemon）"
    docker run -d --name "$CONTAINER_NAME" \
        -v "$DOCKER_SOCK":/var/run/docker.sock \
        -v "$DATASET":"$DATASET" \
        -v "$AGENT_DEPS":"$AGENT_DEPS" \
        -v "$TRIALS_DIR_ABS":"$TRIALS_DIR_ABS" \
        "$IMAGE" sleep infinity
else
    echo "==> 启动容器 $CONTAINER_NAME（dind 模式，privileged / 独立 daemon）"
    # --cgroupns=host + rw 挂载宿主 cgroup2：
    #   DinD 下内层 runc 需在外层容器 cgroup 下创建子 cgroup；若外层容器落在
    #   systemd user session（domain threaded）的 cgroup 里，private cgroupns
    #   会让 runc 报 "cannot enter cgroupv2 ... it is in threaded mode"。
    #   共享宿主 cgroupns 并 rw 挂载后，内层 daemon 直接操作宿主 cgroup 树。
    docker run -d --privileged --name "$CONTAINER_NAME" \
        --cgroupns=host \
        -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
        -v "$DATASET":/opt/data/terminal-bench-2:ro \
        -v "$CASE_IMAGES":/opt/agent-resources/images/case-images.tar:ro \
        -v "$AGENT_DEPS":/opt/agent-resources/deps \
        -v "$TRIALS_DIR_ABS":/benchmark/outputs \
        "$IMAGE" sleep infinity
fi

# ---------------------------------------------------------------------------
# 2. 准备容器内 docker daemon
# ---------------------------------------------------------------------------
if [ "$MODE" = "socket" ]; then
    echo "==> 校验容器内 docker 通道（应直连宿主机 daemon）"
    docker exec "$CONTAINER_NAME" docker info >/dev/null 2>&1 \
        || die "容器内 docker 不可用（socket 挂载失败？）"
    N_IMAGES=$(docker exec "$CONTAINER_NAME" docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
        | grep -c '^alexgshaw/.*:20260430$' || true)
    echo "==> 宿主机 daemon 就绪（容器内可见 alexgshaw/*:20260430 镜像 ${N_IMAGES} 个）"
    if [ "${N_IMAGES:-0}" = "0" ]; then
        echo "[start_agent_runtime] WARN: 宿主机上未发现 tb2 case 镜像（alexgshaw/*:20260430），" >&2
        echo "    请先在宿主机准备 case 镜像（docker pull 或 package_images.sh 产物 docker load）。" >&2
    fi
else
    echo "==> 容器内启动 dockerd"
    docker exec -d "$CONTAINER_NAME" bash -c 'nohup dockerd > /tmp/dockerd.log 2>&1 &'

    DEADLINE=$((SECONDS + 180))
    until docker exec "$CONTAINER_NAME" docker info >/dev/null 2>&1; do
        if [ $SECONDS -ge $DEADLINE ]; then
            echo "[start_agent_runtime] WARN: 内层 dockerd 180s 未就绪，请进容器看 /tmp/dockerd.log" >&2
            break
        fi
        sleep 2
    done
    if docker exec "$CONTAINER_NAME" docker info >/dev/null 2>&1; then
        echo "==> 内层 dockerd 就绪"
    fi

    # -----------------------------------------------------------------------
    # 3.（仅 dind）case 镜像包导入内层 daemon
    # -----------------------------------------------------------------------
    if docker exec "$CONTAINER_NAME" docker info >/dev/null 2>&1; then
        echo "==> 导入 case 镜像包（可能需要数分钟）"
        docker exec "$CONTAINER_NAME" docker load -i /opt/agent-resources/images/case-images.tar
        echo "==> 内层 daemon 镜像列表："
        docker exec "$CONTAINER_NAME" docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}'
    else
        echo "[start_agent_runtime] WARN: 跳过 case 镜像导入（dockerd 未就绪），进容器后手动执行:" >&2
        echo "    docker load -i /opt/agent-resources/images/case-images.tar" >&2
    fi
fi

# ---------------------------------------------------------------------------
# 4. 写入运行指引（登录容器时自动打印）
# ---------------------------------------------------------------------------
if [ "$MODE" = "socket" ]; then
    RUN_HINT_BODY=$(cat <<EOF
================ AISBench harbor agent runtime（socket / DooD）================
docker daemon : 复用宿主机 daemon（挂载 $DOCKER_SOCK）
case 镜像     : 直接使用宿主机已有镜像，无需导入
数据集        : $DATASET            （宿主机原路径，容器内同路径可见）
agent deps    : $AGENT_DEPS         （同上，--agent-deps 参数值）
评测产物      : $TRIALS_DIR_ABS     （同上，宿主机直接可见）

注意1：trial 容器由宿主机 dockerd 拉起，宿主机 docker ps 可见
注意2：本容器内 docker 命令等价于宿主机 docker，勿执行不受信任操作
注意3：--model 需带 litellm provider 前缀，OpenAI 兼容接口用 openai/，

运行命令模板（先 cd 进产物目录，保证 work_dir 落在同路径透传挂载上）：
    cd $TRIALS_DIR_ABS
    agent_env harbor
    ais_bench /benchmark/ais_bench/configs/agent_example/harbor_agent_task.py \\
        --mode agent \\
        -a $AGENT \\
        --agent-deps $AGENT_DEPS \\
        -p $DATASET \\
        --n-tasks 1 \\
        --api-base http://xxx.xx.xx.xx:8080/v1 \\
        --model openai/xxx

首次运行建议 --n-tasks 1 只跑一个 case。
================ AISBench harbor agent runtime（socket / DooD）================
EOF
)
else
    RUN_HINT_BODY=$(cat <<EOF
================ AISBench harbor agent runtime（DinD）================
内层 dockerd : 已由启动脚本拉起（日志 /tmp/dockerd.log）
case 镜像     : 已 load 进内层 daemon
agent deps   : /opt/agent-resources/deps          (--agent-deps)
数据集       : /opt/data/terminal-bench-2         (-p)
评测产物     : /benchmark/outputs（宿主机 --trials-dir）

注意：--model 需带 litellm provider 前缀，OpenAI 兼容接口用 openai/，
例如 --model openai/Qwen2.5-7B-Instruct

激活 harbor venv：
    agent_env harbor

运行命令模板（--api-base / --model 请按实际模型服务填写）：
    ais_bench ais_bench/configs/agent_example/harbor_agent_task.py \\
        --mode agent \\
        -a $AGENT \\
        --agent-deps /opt/agent-resources/deps \\
        -p /opt/data/terminal-bench-2 \\
        --n-tasks 1 \\
        --api-base http://xxx.xx.xx.xx:8080/v1 \\
        --model openai/xxx

首次运行建议 --n-tasks 1 只跑一个 case。
================ AISBench harbor agent runtime（DinD）================
EOF
)
fi

docker exec -i "$CONTAINER_NAME" bash -c "cat > /root/RUN_HINT.txt" <<EOF
$RUN_HINT_BODY
EOF
docker exec "$CONTAINER_NAME" bash -c "grep -q RUN_HINT.txt /etc/bash.bashrc || \
    echo '[ -f /root/RUN_HINT.txt ] && cat /root/RUN_HINT.txt' >> /etc/bash.bashrc"

# ---------------------------------------------------------------------------
# 5. 提示用户进入容器
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo " 模式: $MODE ｜ 容器已就绪，进入容器："
echo "     docker exec -it $CONTAINER_NAME bash"
echo " 进入后按 RUN_HINT.txt 中的命令模板执行即可（会自动打印）。"
echo "======================================================================"
