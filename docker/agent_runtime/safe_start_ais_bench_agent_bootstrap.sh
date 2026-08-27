#!/bin/bash
# ============================================================================
# safe_start_ais_bench_agent_bootstrap.sh — 安全的 bootstrap.sh 启动器(带 watchdog)
# ============================================================================
#
# 与 ais_bench_agent_bootstrap.sh 的区别:
#   - 内置 watchdog:即使脚本被杀 / shell 卡死,容器也会被自动 docker rm -f
#   - 启动前预检 host 是否被污染(qemu binfmt 泄漏)
#   - 容器退出后自动清理 host 上的 qemu binfmt(若泄漏)
#   - 默认 3 分钟超时,--keep-alive 可解除
#   - watchdog 顺序关键:先清 binfmt(否则 docker CLI 也会 ELOOP)再 docker rm -f
#
# 用法:
#   bash safe_start_ais_bench_agent_bootstrap.sh --datasets /data/datasets
#   bash safe_start_ais_bench_agent_bootstrap.sh --keep-alive --datasets /data/datasets
#   bash safe_start_ais_bench_agent_bootstrap.sh --watchdog-min 10 --datasets /data/datasets
#   bash safe_start_ais_bench_agent_bootstrap.sh --test
#
# 工作原理:
#   1. 预检 host + docker
#   2. setsid + nohup + disown 起一个 watchdog 子进程
#      (sleep N then 先清 binfmt 再 docker rm -f)
#   3. 调 ais_bench_agent_bootstrap.sh 真启动容器(转发所有参数)
#   4. 启动成功后,用户使用容器;到时间 watchdog 自动 kill
#
# 安全网:
#   - watchdog 独立于父进程,父进程被杀不会影响 watchdog
#   - 即使整个 host bash 都 ELOOP,watchdog 仍能在到时间后
#     先清 host binfmt(写文件,不需要 exec ELF),再 docker rm -f
#   - 若到时间仍 ELOOP,可以另开 ssh session 跑 cleanup 脚本
#
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认值(与 bootstrap.sh 一致)
CONTAINER_NAME="${CONTAINER_NAME:-ais_bench_agent}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-ghcr.io/aisbench/agent-runtime:latest-ubuntu24.04-py312-$(uname -m)}"
WATCHDOG_MINUTES="${WATCHDOG_MINUTES:-3}"
KEEP_ALIVE=0
TEST_ONLY=0
WATCHDOG_LOG="/tmp/safe_start_watchdog_${CONTAINER_NAME}.log"

# 解析 safe_start 自己的参数(--keep-alive / --watchdog-min / --test /
# --container-name / --runtime-image),其它参数原样转发给 bootstrap.sh
SAFESTART_ARGS=()
BOOTSTRAP_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-alive)                 KEEP_ALIVE=1; shift ;;
        --watchdog-min)               WATCHDOG_MINUTES="$2"; shift 2 ;;
        --container-name)             CONTAINER_NAME="$2"; SAFESTART_ARGS+=("$1" "$2"); shift 2 ;;
        --runtime-image)              RUNTIME_IMAGE="$2"; SAFESTART_ARGS+=("$1" "$2"); shift 2 ;;
        --test)                       TEST_ONLY=1; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            echo
            echo "其余参数转发给 ais_bench_agent_bootstrap.sh"
            bash "$SCRIPT_DIR/ais_bench_agent_bootstrap.sh" --help 2>&1 | head -20 || true
            exit 0
            ;;
        *) BOOTSTRAP_ARGS+=("$1"); shift ;;
    esac
done

WATCHDOG_SECS=$((WATCHDOG_MINUTES * 60))

# ===========================================================================
# 0. 预检
# ===========================================================================
echo "============================================================"
echo " safe_start_ais_bench_agent_bootstrap.sh"
echo "============================================================"
echo " container:        $CONTAINER_NAME"
echo " image:            $RUNTIME_IMAGE"
echo " watchdog:         ${WATCHDOG_MINUTES} min"
echo " keep_alive:       $KEEP_ALIVE"
echo " bootstrap_args:   ${BOOTSTRAP_ARGS[*]:-<none>}"
echo "============================================================"
echo

echo "[safe_start] [1/4] pre-check..."

# 0.1 docker daemon
docker info >/dev/null 2>&1 || { echo "[safe_start] [ERROR] docker daemon not working"; exit 1; }
echo "  ✓ docker daemon"

# 0.2 不强求 image loaded(bootstrap.sh 会从 dockerhub pull 或 OBS tar 加载)
# 仅在 --runtime-image 显式传入时检查
if [[ "${SAFESTART_ARGS[*]}" =~ --runtime-image ]]; then
    if docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1; then
        echo "  ✓ image $RUNTIME_IMAGE"
    else
        echo "  [warn] image $RUNTIME_IMAGE not loaded locally"
        echo "         bootstrap.sh 会自动拉取或加载 tar"
    fi
fi

# 0.3 host 是否被前次实验污染
QEMU_POLLUTED=0
for q in qemu-x86_64 qemu-aarch64 qemu-arm qemu-riscv64; do
    if [[ -f "/proc/sys/fs/binfmt_misc/$q" ]]; then
        echo "  [warn] host 已有 $q binfmt(可能是前次实验泄漏)"
        QEMU_POLLUTED=1
    fi
done
if [[ $QEMU_POLLUTED -eq 0 ]]; then
    echo "  ✓ host binfmt 干净(无 qemu 泄漏)"
fi

# 0.4 sleep 能跑(防 ELOOP)
if ! sleep 0.1 2>&1 | head -1 >/dev/null; then
    echo "  [ERROR] host /usr/bin/sleep 出错(可能 ELOOP),不能启动"
    exit 1
fi
echo "  ✓ sleep 正常"

if [[ $TEST_ONLY -eq 1 ]]; then
    echo
    echo "[safe_start] --test 模式:只跑预检,不启动"
    exit 0
fi

# ===========================================================================
# 1. 启动 watchdog(独立进程,setsid + nohup + disown)
# ===========================================================================
echo
echo "[safe_start] [2/4] 启动 watchdog..."

if [[ $KEEP_ALIVE -eq 1 ]]; then
    echo "  [skip] --keep-alive 模式,无 watchdog"
    WATCHDOG_PID=""
else
    # ★ 关键顺序:先清 host binfmt(写文件,不调 ELF),再调 docker
    # 否则 host 已被污染时,docker CLI 自身会 ELOOP
    setsid bash -c "
        sleep $WATCHDOG_SECS

        # 1. 先清 host qemu binfmt(bash 内置 + 重定向,不依赖 ELF exec)
        for q in qemu-x86_64 qemu-aarch64 qemu-arm qemu-riscv64; do
            if [[ -f /proc/sys/fs/binfmt_misc/\$q ]]; then
                echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] watchdog: cleaning host binfmt \$q\" >> $WATCHDOG_LOG
                echo -1 > /proc/sys/fs/binfmt_misc/\$q 2>> $WATCHDOG_LOG || true
            fi
        done

        # 2. 现在 binfmt 干净,docker CLI 可正常 exec
        if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx '$CONTAINER_NAME'; then
            echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] watchdog: auto-killing $CONTAINER_NAME after ${WATCHDOG_MINUTES} min\" >> $WATCHDOG_LOG
            docker rm -f '$CONTAINER_NAME' >> $WATCHDOG_LOG 2>&1
            echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] watchdog: done\" >> $WATCHDOG_LOG
        else
            echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] watchdog: container gone or docker unreachable\" >> $WATCHDOG_LOG
        fi
    " </dev/null >/dev/null 2>&1 &
    WATCHDOG_PID=$!
    disown 2>/dev/null || true
    echo "  ✓ watchdog PID: $WATCHDOG_PID"
    echo "  ✓ watchdog log: $WATCHDOG_LOG"
fi

# ===========================================================================
# 2. cleanup on signal
# ===========================================================================
cleanup() {
    local rc=$?
    if [[ -n "${WATCHDOG_PID:-}" ]]; then
        echo "[safe_start] [cleanup] killing watchdog $WATCHDOG_PID"
        kill "$WATCHDOG_PID" 2>/dev/null || true
    fi
    exit $rc
}
trap cleanup EXIT INT TERM

# ===========================================================================
# 3. 启动容器(转发 BOOTSTRAP_ARGS)
# ===========================================================================
echo
echo "[safe_start] [3/4] 启动 bootstrap 容器..."
echo

export CONTAINER_NAME
export RUNTIME_IMAGE

bash "$SCRIPT_DIR/ais_bench_agent_bootstrap.sh" "${BOOTSTRAP_ARGS[@]}"
RC=$?

if [[ $RC -ne 0 ]]; then
    echo
    echo "[safe_start] [ERROR] bootstrap.sh 失败 rc=$RC"
    exit $RC
fi

# ===========================================================================
# 4. 报告 + 监控
# ===========================================================================
echo
echo "[safe_start] [4/4] 启动成功"
echo "============================================================"
echo " ✓ 容器名:    $CONTAINER_NAME"
echo " ✓ 镜像:      $RUNTIME_IMAGE"
if [[ -n "${WATCHDOG_PID:-}" ]]; then
    echo " ✓ watchdog:  PID $WATCHDOG_PID (${WATCHDOG_MINUTES} min 后自动 docker rm -f)"
fi
echo "============================================================"
echo
echo "进入容器:     docker exec -it $CONTAINER_NAME bash"
echo "查看日志:     docker logs $CONTAINER_NAME"
echo "手动 kill:    docker rm -f $CONTAINER_NAME"
if [[ -n "${WATCHDOG_PID:-}" ]]; then
    echo "取消 watchdog: kill $WATCHDOG_PID"
    echo "看 watchdog:  tail -f $WATCHDOG_LOG"
fi
echo

if [[ $KEEP_ALIVE -eq 0 ]]; then
    echo "[safe_start] 容器将在 ${WATCHDOG_MINUTES} 分钟后自动清理(即使 shell 卡死)"
    echo "[safe_start] 父脚本现在退出,watchdog 独立运行"
    echo "[safe_start] 若想现在退出 shell,直接 Ctrl-C 即可,watchdog 继续"
fi
exit 0