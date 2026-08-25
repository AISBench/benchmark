#!/bin/bash
# ============================================================================
# ais_bench_agent_run.sh — 5 层 DinD L3 入口:触发 harbor 矩阵评测
# ============================================================================
#
# 设计:
#   - 简单包装 `harbor jobs start -c matrix.yaml`,让用户避免手写命令行
#   - 支持子集过滤(--datasets / --agents)→ 生成 tmp yaml
#   - 自动 source /opt/swebench/api_key.env 注入 OPENAI_API_KEY/BASE
#     (由 bootstrap.sh --api-key-file bind mount,详见 B1+B3 commit)
#   - job 落 /opt/swebench/jobs/<job-name>/(由 bootstrap.sh --bind-jobs bind mount)
#   - 容器内调用;host 上调用方式:`docker exec <container> ais_bench_agent_run.sh ...`
#
# 适用:
#   - 5 层 DinD L3 调度入口(L4 image + L5 launcher 已由 v3 A+B 批就绪)
#   - 默认参数对应当前 batch A+B 的 bootstrap 默认
#   - 用户在容器内 `ais_bench_agent_run.sh` 即可(已 PATH /usr/local/bin/)
#
# 用法:
#   ais_bench_agent_run.sh                              # 全矩阵,默认 matrix.yaml
#   ais_bench_agent_run.sh --datasets 11099,12308       # 子集
#   ais_bench_agent_run.sh --agents oracle,aider        # 子集
#   ais_bench_agent_run.sh --job-name my-test           # 自定义 job 名
#   ais_bench_agent_run.sh --container-name my-runtime  # 改 runtime 容器
#   ais_bench_agent_run.sh --matrix /path/to/other.yaml # 改 matrix 路径
#   ais_bench_agent_run.sh --dry-run                    # 只打印命令,不执行
#
# 环境变量(可选,优先级低于 CLI):
#   AIS_BENCH_AGENT_CONTAINER   runtime 容器名(默认 ais_bench_agent)
#   AIS_BENCH_AGENT_MATRIX      matrix.yaml 容器内路径(默认 /opt/swebench/config/matrix.yaml)
#
# ============================================================================
set -euo pipefail

# ============ 默认值 ============
CONTAINER_NAME="${AIS_BENCH_AGENT_CONTAINER:-ais_bench_agent}"
MATRIX_YAML="${AIS_BENCH_AGENT_MATRIX:-/opt/swebench/config/matrix.yaml}"
JOBS_DIR="/opt/swebench/jobs"
LOG_DIR="/opt/swebench/logs"
API_KEY_FILE="/opt/swebench/api_key.env"
FILTER_SCRIPT="/usr/local/bin/ais_bench_agent_filter_matrix.py"

# CLI 参数
JOB_NAME=""
DATASETS_FILTER=""
AGENTS_FILTER=""
DRY_RUN=0

# ============ 参数解析 ============
while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-name)        JOB_NAME="$2"; shift 2 ;;
        --datasets)        DATASETS_FILTER="$2"; shift 2 ;;
        --agents)          AGENTS_FILTER="$2"; shift 2 ;;
        --container-name)  CONTAINER_NAME="$2"; shift 2 ;;
        --matrix)          MATRIX_YAML="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "[错误] 未知参数: $1" >&2; exit 1 ;;
    esac
done

JOB_NAME="${JOB_NAME:-matrix-$(date +%Y%m%d-%H%M%S)}"

# ============ 工具函数 ============
log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "[错误] $*" >&2; exit 1; }

# ============ 前置检查 ============
log "=== ais_bench_agent_run.sh ==="
log "  container:    $CONTAINER_NAME"
log "  matrix.yaml:  $MATRIX_YAML"
log "  jobs-dir:      $JOBS_DIR"
log "  job-name:      $JOB_NAME"

# 检查 matrix.yaml 存在
if [ ! -f "$MATRIX_YAML" ]; then
    fail "matrix.yaml 不存在: $MATRIX_YAML
    提示:在 bootstrap 时传 --matrix-yaml <HOST_PATH> 把 matrix bind 进容器"
fi

# 检查 harbor CLI 可用
if ! command -v harbor >/dev/null 2>&1; then
    fail "harbor 命令不可用(请确认 harbor 0.20.x 已装,详见 v3 A2 commit)"
fi

# 检查 jobs-dir / log-dir 可写
for d in "$JOBS_DIR" "$LOG_DIR"; do
    if [ ! -d "$d" ]; then
        # 尝试创建(通常 A5 已预创建)
        mkdir -p "$d" 2>/dev/null || fail "目录不存在且创建失败: $d"
    fi
    [ -w "$d" ] || fail "目录不可写: $d(host bind mount 权限问题)"
done

# api_key.env 可选但推荐
if [ -f "$API_KEY_FILE" ]; then
    log "  api_key.env: 已挂载(将由下方 source 注入 OPENAI_API_KEY/BASE)"
else
    log "  ⚠ api_key.env 未挂载: $API_KEY_FILE"
    log "    harbor CLI 会因 OPENAI_API_KEY 未设置而失败"
    log "    提示:bootstrap 时传 --api-key-file <HOST_PATH>"
fi

# ============ 准备:source api_key.env ============
# set -a 让 source 的变量自动 export 给后续 docker exec / harbor CLI
if [ -f "$API_KEY_FILE" ]; then
    log "  注入 api_key.env → OPENAI_API_KEY / OPENAI_API_BASE"
    set -a
    # shellcheck disable=SC1090
    . "$API_KEY_FILE"
    set +a
fi

# ============ 准备:过滤 tmp yaml ============
CMD_YAML="$MATRIX_YAML"
CLEANUP_TMP=""
if [[ -n "$DATASETS_FILTER" || -n "$AGENTS_FILTER" ]]; then
    # tmp yaml 必须写到 bind-mount 的 log 目录,harbor 才能在容器内看到
    TMP_YAML="$LOG_DIR/_tmp_filtered_$(date +%s).yaml"
    CLEANUP_TMP="$TMP_YAML"
    log "  生成过滤 tmp yaml..."
    log "    datasets: ${DATASETS_FILTER:-(全部)}"
    log "    agents:   ${AGENTS_FILTER:-(全部)}"
    python3 "$FILTER_SCRIPT" \
        --input "$MATRIX_YAML" \
        --output "$TMP_YAML" \
        ${DATASETS_FILTER:+--datasets "$DATASETS_FILTER"} \
        ${AGENTS_FILTER:+--agents "$AGENTS_FILTER"} \
        || fail "filter_matrix.py 失败"
    CMD_YAML="$TMP_YAML"
fi

trap '[[ -n "$CLEANUP_TMP" && -f "$CLEANUP_TMP" ]] && rm -f "$CLEANUP_TMP"' EXIT

# ============ 触发 harbor jobs start ============
HARBOR_CMD=(harbor jobs start
    -c "$CMD_YAML"
    --job-name "$JOB_NAME"
    --jobs-dir "$JOBS_DIR"
    -n 2
)

log ""
log "执行: ${HARBOR_CMD[*]}"
log "  trials 结果将落: $JOBS_DIR/$JOB_NAME/(host bind mount 可见)"
log ""

if [ "$DRY_RUN" = "1" ]; then
    log "[dry-run] 不实际执行 harbor jobs start"
    exit 0
fi

"${HARBOR_CMD[@]}"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    log "✓ 提交成功"
    echo ""
    log "跟踪 job 状态:"
    echo "  harbor jobs view $JOB_NAME"
    echo "  ls -la $JOBS_DIR/$JOB_NAME/"
    echo "  ais_bench_agent_watch.sh $JOB_NAME         # 阻塞等结果"
    echo "  ais_bench_agent_summarize.sh $JOB_NAME     # 聚合 md/csv/json"
else
    fail "harbor jobs start 退出码 $EXIT_CODE"
fi