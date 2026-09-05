#!/bin/bash
# ============================================================================
# ais_bench_agent_summarize.sh — 5 层 DinD L3 聚合:汇总 jobs/*/result.json
# ============================================================================
#
# 设计:
#   - 扫 /opt/swebench/jobs/<job-name>/result.json,聚合 pass@1 by (bench, agent)
#   - 输出 3 份:
#       /opt/swebench/logs/summary-<ts>.md     (人类读)
#       /opt/swebench/logs/summary-<ts>.csv    (Excel/pandas 处理)
#       /opt/swebench/logs/summary-<ts>.json   (raw data)
#   - 三份文件都落在 /opt/swebench/logs/(host bind mount 可读)
#   - 调用 ais_bench_agent_summarize.py 实现聚合(无需 jq)
#
# 用法:
#   ais_bench_agent_summarize.sh                          # 汇总所有 job
#   ais_bench_agent_summarize.sh m3x3-aider-11099 ...     # 汇总指定 job(substring 匹配)
#   ais_bench_agent_summarize.sh --latest                  # 仅最新一个 job
#   ais_bench_agent_summarize.sh --jobs-dir /opt/swebench/jobs
#   ais_bench_agent_summarize.sh --output-dir /opt/swebench/logs
#
# 适用:
#   - 容器内跑,输出日志到 bind-mount 路径(host 可读)
#   - host 调:`docker exec <container> ais_bench_agent_summarize.sh <args>`
#
# ============================================================================
set -euo pipefail

# ============ 默认值 ============
JOBS_DIR="/opt/swebench/jobs"
OUTPUT_DIR="/opt/swebench/logs"
SUMMARY_SCRIPT="/usr/local/bin/ais_bench_agent_summarize.py"
LATEST_ONLY=0
INCLUDE_ARGS=()

# ============ 参数解析 ============
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs-dir)    JOBS_DIR="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --latest)      LATEST_ONLY=1; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        --*)
            echo "[错误] 未知参数: $1" >&2; exit 1 ;;
        *)
            INCLUDE_ARGS+=("$1"); shift
            ;;
    esac
done

# ============ 工具函数 ============
log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "[错误] $*" >&2; exit 1; }

# ============ 前置检查 ============
log "=== ais_bench_agent_summarize.sh ==="
log "  jobs-dir:    $JOBS_DIR"
log "  output-dir:  $OUTPUT_DIR"
log "  include:     ${#INCLUDE_ARGS[@]} 个 substring filter"

if [ ! -d "$JOBS_DIR" ]; then
    fail "jobs-dir 不存在: $JOBS_DIR(提示:bootstrap --bind-jobs)"
fi
if [ ! -f "$SUMMARY_SCRIPT" ]; then
    fail "summarize.py 未找到: $SUMMARY_SCRIPT(image 构建漏装?)"
fi

mkdir -p "$OUTPUT_DIR" || fail "output-dir 创建失败: $OUTPUT_DIR"

# 处理 --latest:只取 jobs-dir 下最新的一个 job 目录
if [ "$LATEST_ONLY" = "1" ]; then
    LATEST_JOB=$(ls -1dt "$JOBS_DIR"/*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_JOB" ]; then
        fail "jobs-dir 下没有任何 job:$JOBS_DIR"
    fi
    LATEST_NAME=$(basename "$LATEST_JOB")
    log "  --latest:仅汇总 $LATEST_NAME"
    INCLUDE_ARGS=("$LATEST_NAME")
fi

# ============ 准备 summarize.py 命令行 ============
PY_ARGS=(--jobs-dir "$JOBS_DIR" --output-dir "$OUTPUT_DIR")

if [ "${#INCLUDE_ARGS[@]}" -gt 0 ]; then
    PY_ARGS+=(--include "$(IFS=, ; echo "${INCLUDE_ARGS[*]}")")
fi

# ============ 触发 ============
log ""
log "调 summarize.py ${PY_ARGS[*]}"
log ""

python3 "$SUMMARY_SCRIPT" "${PY_ARGS[@]}"

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    log "✓ 汇总完成,产物在 $OUTPUT_DIR/"
    ls -la "$OUTPUT_DIR"/summary-*.{md,csv,json} 2>/dev/null | tail -10 || true
else
    fail "summarize.py 退出码 $EXIT_CODE"
fi