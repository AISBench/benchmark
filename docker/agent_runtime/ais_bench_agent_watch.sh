#!/bin/bash
# ============================================================================
# ais_bench_agent_watch.sh — 5 层 DinD L3 监控:阻塞等 harbor job 完成
# ============================================================================
#
# 设计:
#   - 阻塞等 /opt/swebench/jobs/<job-name>/result.json 出现
#   - 周期调 `harbor jobs view <job-name>` 打印 trial 进度
#   - results.json 出现后:读 stats 输出 pass@1 / completed / errored / finished_at
#   - 完成 / 超时 / 用户 Ctrl-C 都会清退出
#
# 用法:
#   ais_bench_agent_watch.sh <job-name>                              # 阻塞等
#   ais_bench_agent_watch.sh <job-name> --interval 30                # 30s 间隔
#   ais_bench_agent_watch.sh <job-name> --timeout 7200               # 2h 超时
#   ais_bench_agent_watch.sh <job-name> --no-poll                    # 不轮询 harbor jobs view
#   ais_bench_agent_watch.sh <job-name> --jobs-dir /opt/swebench/jobs # 改 jobs 目录
#
# 适用:
#   - 在容器内跑(5 层 DinD L3 入口)
#   - 也可以 host 上跑:`docker exec <container> ais_bench_agent_watch.sh <job-name>`
#
# ============================================================================
set -euo pipefail

# ============ 默认值 ============
JOBS_DIR="/opt/swebench/jobs"
INTERVAL=15          # 秒
TIMEOUT=0            # 0 = 不超时
POLL_HARBOR=1

# ============ 参数解析 ============
JOB_NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)    INTERVAL="$2"; shift 2 ;;
        --timeout)     TIMEOUT="$2"; shift 2 ;;
        --jobs-dir)    JOBS_DIR="$2"; shift 2 ;;
        --no-poll)     POLL_HARBOR=0; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        --*)
            echo "[错误] 未知参数: $1" >&2; exit 1 ;;
        *)
            if [ -z "$JOB_NAME" ]; then
                JOB_NAME="$1"
            else
                echo "[错误] 多余位置参数: $1" >&2; exit 1
            fi
            shift
            ;;
    esac
done

[ -z "$JOB_NAME" ] && { echo "[错误] 必须指定 <job-name>" >&2; exit 1; }

# ============ 工具函数 ============
log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "[错误] $*" >&2; exit 1; }

JOB_DIR="$JOBS_DIR/$JOB_NAME"
RESULT_JSON="$JOB_DIR/result.json"

# ============ 前置检查 ============
log "=== ais_bench_agent_watch.sh ==="
log "  job-name:   $JOB_NAME"
log "  jobs-dir:   $JOBS_DIR"
log "  job-dir:    $JOB_DIR"
log "  interval:   ${INTERVAL}s"
log "  timeout:    $([ "$TIMEOUT" -eq 0 ] && echo none || echo "${TIMEOUT}s")"

if [ ! -d "$JOBS_DIR" ]; then
    fail "jobs-dir 不存在: $JOBS_DIR(提示:bootstrap --bind-jobs)"
fi

START_TIME=$(date +%s)

# ============ Ctrl-C 优雅退出 ============
cleanup() {
    echo ""
    log "  (退出 watch,job 仍在 harbor 后台跑)"
    log "  下次再 watch: ais_bench_agent_watch.sh $JOB_NAME"
}
trap cleanup INT TERM

# ============ 主循环 ============
log ""
log "等待 results.json 出现: $RESULT_JSON"

WAITED=0
while true; do
    # 检查 result.json
    if [ -f "$RESULT_JSON" ]; then
        log "✓ results.json 出现,耗时 $(( $(date +%s) - START_TIME ))s"
        break
    fi

    # 检查 harbor jobs view（仅在 result.json 出现前调）
    if [ "$POLL_HARBOR" = "1" ] && command -v harbor >/dev/null 2>&1; then
        # harbor jobs view <name> 输出有 finished_at 字段时说明完成
        HARBOR_OUT=$(harbor jobs view "$JOB_NAME" 2>&1 || true)
        if echo "$HARBOR_OUT" | grep -q "finished_at:"; then
            log "harbor jobs view 报告 finished_at,但 result.json 尚未出现"
            log "  (可能是 harbor 在写 result.json 之前的瞬态)"
            sleep 2
            if [ -f "$RESULT_JSON" ]; then
                log "✓ results.json 出现,耗时 $(( $(date +%s) - START_TIME ))s"
                break
            fi
        fi
    fi

    # 检查超时
    if [ "$TIMEOUT" -gt 0 ]; then
        ELAPSED=$(( $(date +%s) - START_TIME ))
        if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
            log "⏰ 超时 ${TIMEOUT}s,results.json 仍未出现"
            log "  Harbor 仍可能继续跑,可加大 --timeout 或 --no-poll"
            exit 2
        fi
    fi

    sleep "$INTERVAL"
    WAITED=$((WAITED + 1))
    if [ $((WAITED % 4)) -eq 0 ]; then
        ELAPSED=$(( $(date +%s) - START_TIME ))
        log "  等待中...(${ELAPSED}s,共 $WAITED 轮)"
    fi
done

# ============ 解析 results.json 输出统计 ============
log ""
log "=== job $JOB_NAME 完成统计 ==="

# 用 python 解析 result.json（无需 jq 依赖）
if command -v python3 >/dev/null 2>&1; then
    python3 - "$RESULT_JSON" <<'PYEOF' || log "  ⚠ result.json 解析失败,直接 cat 看原文"
import json
import sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

stats = data.get("stats", {})
evals = stats.get("evals", {})

n_total = stats.get("n_total_trials", 0)
n_completed = stats.get("n_completed_trials", 0)
n_errored = stats.get("n_errored_trials", 0)
n_running = stats.get("n_running_trials", 0)
n_pending = stats.get("n_pending_trials", 0)
finished = data.get("finished_at")
started = data.get("started_at")

rewards = []
for eval_key, eval_data in evals.items():
    buckets = eval_data.get("reward_stats", {}).get("reward", {})
    for r_str, trial_ids in buckets.items():
        try:
            r = float(r_str)
        except (ValueError, TypeError):
            continue
        n = len(trial_ids) if isinstance(trial_ids, list) else 1
        rewards.extend([r] * n)

n_pass = sum(1 for r in rewards if r == 1.0)
pass_at_1 = n_pass / max(len(rewards), 1) if rewards else 0.0

print(f"  started_at:    {started or '?'}")
print(f"  finished_at:   {finished or '?'}")
print(f"  n_total:       {n_total}")
print(f"  n_completed:   {n_completed}")
print(f"  n_errored:     {n_errored}")
print(f"  n_running:     {n_running}")
print(f"  n_pending:     {n_pending}")
print(f"  n_pass(=1.0):  {n_pass} / {len(rewards)} trials")
print(f"  pass@1:        {pass_at_1:.1%}")
PYEOF
else
    log "  ⚠ python3 不可用,直接 cat:"
    cat "$RESULT_JSON" | head -30
fi

log ""
log "下一步:"
echo "  cat $RESULT_JSON        # 完整结果"
echo "  ais_bench_agent_summarize.sh $JOB_NAME   # 聚合到 md/csv/json"
echo "  harbor jobs view $JOB_NAME   # harbor CLI 原生命令"