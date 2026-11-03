#!/usr/bin/env bash

# ============================================================
# AISBench perf benchmark for an already-running vLLM service.
#
# 一次运行覆盖「多组输入/输出长度 × 多档并发」：
#   - 外层 shell 循环遍历 GROUP_SPECS 中的每个输入/输出组合；
#   - 组内并发阶梯由 AISBench 多组合能力一次展开：
#       --batch-size  A B C...
#       --num-prompts A B C...   # 必须与 --batch-size 一一对应（取值个数一致）
#     每个并发档自动生成独立任务（abbr 追加 -bs{N}），档间串行并带 5s 冷却，
#     档内统一汇总为 combos_manifest.json / batch_perf_summary.csv /
#     batch_perf_report.html。
#
# 输入长度与公共前缀通过 CLI 覆盖，不再需要手改数据集配置文件：
#   --tokenid-config-request-size <total input tokens>
#   --tokenid-config-prefix-len   <shared prefix tokens>
#   --request-count               <生成的请求总数，>= 最大并发档>
#
# 使用方式:
#   bash aisbench_vllm_performance_zh_cn.sh 0 # 示例1：关闭前缀缓存，前台交互运行
#   RUN_BG=1 bash aisbench_vllm_performance_zh_cn.sh 80   # 示例2：开启前缀缓存，后台运行，SSH 断开不中断
#
# Mode 0 : 服务端关闭前缀缓存（vllm serve 追加 --no-enable-prefix-caching），
#          公共前缀长度固定为 0。
# Mode 80: 服务端开启前缀缓存（vllm serve 追加 --enable-prefix-caching），
#          每组公共前缀占总输入 80%。
#          不要依赖默认值：vllm-ascend(NPU) 当前默认 enable_prefix_caching=False。
#
# 长度约束：Qwen3.8 服务以 --max-model-len 32768 启动，
# 要求 total_input + output <= 32768（tokenid 模式回解码可能存在少量膨胀，
# 建议总长度留出 ~6% 余量；超过 90% 时脚本给出 WARNING）。
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CACHE_MODE="${1:-}"
if [[ "$CACHE_MODE" != "0" && "$CACHE_MODE" != "80" ]]; then
    echo "用法: $0 {0|80}" >&2
    echo "" >&2
    echo "缓存模式必须与服务端启动参数匹配（两种模式都要求显式设置，不依赖默认值）:" >&2
    echo "  0  = 服务端关闭前缀缓存（vllm serve 追加 --no-enable-prefix-caching）" >&2
    echo "  80 = 服务端开启前缀缓存（vllm serve 追加 --enable-prefix-caching，公共前缀占 80%）" >&2
    exit 2
fi

# ---- 服务默认参数（与 vllm 服务端部署保持一致） ----
MODEL_PATH="${MODEL_PATH:-/mnt/weight/Qwen3.8-27B-w8a8}"
HOST_IP="${HOST_IP:-127.0.0.1}"
HOST_PORT="${HOST_PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-qwen3.8}"

# ---- 服务化后端交互参数（均为 AISBench 现有 CLI） ----
REQUEST_RATE="${REQUEST_RATE:-0}"   # 0 = 不限速；并发由 --batch-size 控制
RETRY="${RETRY:-2}"
API_KEY="${API_KEY:-}"
WARMUP_NUM="${WARMUP_NUM:-16}"      # 每个并发档单独预热，不计入指标
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}" # vllm启动时设置的最大上下文总长度
GEN_KWARGS="${GEN_KWARGS:-}"
if [[ -z "$GEN_KWARGS" ]]; then
    GEN_KWARGS='{"temperature":0.01,"ignore_eos":true}'
fi
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# ---- 模型/数据集配置 ----
# vllm_api_stream_chat = VLLMCustomAPI + stream=True -> /v1/chat/completions
MODEL_CFG="${MODEL_CFG:-vllm_api_stream_chat}"
DATASET_CFG="${DATASET_CFG:-synthetic_gen_tokenid}"

# ---- 任务组合：每行一个「总输入长度 输出长度 并发阶梯」 ----
# 并发阶梯即该组的 --batch-size 列表；--num-prompts 与之一一对应
# NOTE: total_input + output 必须 <= MAX_MODEL_LEN。
GROUP_SPECS=(
"8192 1024 1 8 16 32 40 50 64"
"16384 1024 1 8 16 20 25 32 40"
"30720 1024 1 2 4 8 12 16 20 25"
)

NET_ARGS=(--host-ip "$HOST_IP" --host-port "$HOST_PORT")
MODELS_URL="http://${HOST_IP}:${HOST_PORT}/v1/models"
NET_DESC="host-ip=${HOST_IP}, host-port=${HOST_PORT}"
[[ -n "$API_KEY" ]] && NET_ARGS+=(--api-key "$API_KEY")
NET_ARGS+=(--retry "$RETRY" --request-rate "$REQUEST_RATE")

TRUST_ARGS=()
[[ "$TRUST_REMOTE_CODE" == "1" ]] && TRUST_ARGS+=(--trust-remote-code)

# ---- 前缀缓存模式提醒（模式与服务端启动参数必须一致） ----
if [[ "$CACHE_MODE" == "80" ]]; then
    echo ">>> 提示: 模式 80 要求服务端已开启前缀缓存（vllm serve 追加 --enable-prefix-caching）"
else
    echo ">>> 提示: 模式 0 要求服务端已显式关闭前缀缓存（vllm serve 追加 --no-enable-prefix-caching）"
fi
echo "    验证: docker logs <container> 2>&1 | grep -m1 enable_prefix_caching"

# ---- background mode: survive SSH disconnect ----
if [[ "${RUN_BG:-0}" == "1" ]]; then
    BG_LOG_DIR="${SCRIPT_DIR}/logs/aisbench_3groups/bg"
    mkdir -p "$BG_LOG_DIR"
    if [[ -f "${BG_LOG_DIR}/latest.pid" ]]; then
        OLD_PID="$(cat "${BG_LOG_DIR}/latest.pid" 2>/dev/null || true)"
        if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
            echo "错误: 已有一个后台实例在运行 (PID $OLD_PID)" >&2
            echo "       确认要放弃它再跑: kill -- -$OLD_PID" >&2
            exit 1
        fi
    fi
    BG_LOG="${BG_LOG_DIR}/bg_$(date +%Y%m%d_%H%M%S).log"
    SCRIPT_PATH="$(readlink -f "$0")"
    nohup env RUN_BG=0 bash "$SCRIPT_PATH" "$@" </dev/null >"$BG_LOG" 2>&1 &
    BG_PID=$!
    echo "$BG_PID" > "${BG_LOG_DIR}/latest.pid"
    echo "========================================================"
    echo "已转入后台运行（SSH 断开/登出均不影响）"
    echo "  PID      : $BG_PID"
    echo "  总日志   : $BG_LOG"
    echo "  实时查看 : tail -f $BG_LOG"
    echo "  停止测试 : kill -- -$BG_PID   # 杀整个进程组（含正在跑的 ais_bench）"
    echo "========================================================"
    exit 0
fi

# ---- preflight 1: 组合长度与 max-model-len ----
for test in "${GROUP_SPECS[@]}"; do
    read -r TOTAL_INPUT_LEN MAX_TOKENS _ <<< "$test"
    if (( TOTAL_INPUT_LEN + MAX_TOKENS > MAX_MODEL_LEN )); then
        echo "错误: 输入 ${TOTAL_INPUT_LEN} + 输出 ${MAX_TOKENS} 超过 MAX_MODEL_LEN=${MAX_MODEL_LEN}" >&2
        exit 1
    fi
    if (( TOTAL_INPUT_LEN + MAX_TOKENS > MAX_MODEL_LEN * 9 / 10 )); then
        echo "WARNING: 输入 ${TOTAL_INPUT_LEN} + 输出 ${MAX_TOKENS} 已超过 MAX_MODEL_LEN 的 90%，" \
             "tokenid 解码膨胀可能导致请求被服务端拒绝" >&2
    fi
done

# ---- preflight 2: 服务可达性与模型名校验 ----
CURL_ARGS=(-s --max-time 10)
[[ -n "$API_KEY" ]] && CURL_ARGS+=(-H "Authorization: Bearer ${API_KEY}")
echo ">>> 预检: GET $MODELS_URL  ($NET_DESC)"
if ! curl "${CURL_ARGS[@]}" "$MODELS_URL" -o /dev/null; then
    echo "错误: 无法访问 vLLM 服务 ($MODELS_URL)" >&2
    echo "       请确认容器已启动且服务就绪（docker logs -f <container>，看到 Uvicorn running 即就绪）" >&2
    exit 1
fi
if ! curl "${CURL_ARGS[@]}" "$MODELS_URL" | grep -Fq "\"$SERVED_NAME\""; then
    echo "错误: 服务在运行，但模型列表中未找到 '$SERVED_NAME'" >&2
    echo "       请确认 vLLM 的 --served-model-name 与 SERVED_NAME 一致" >&2
    exit 1
fi
echo ">>> 预检通过: 模型 '$SERVED_NAME' 就绪"

if ! command -v ais_bench >/dev/null 2>&1; then
    echo "错误: 找不到 ais_bench 命令（请先激活 AISBench 环境）" >&2
    exit 1
fi

# ---- output layout ----
MODE_TAG="cache${CACHE_MODE}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${SCRIPT_DIR}/logs/aisbench_3groups/${MODE_TAG}/${RUN_TS}"
RESULT_ROOT="${SCRIPT_DIR}/outputs/aisbench_3groups/${MODE_TAG}/${RUN_TS}"
mkdir -p "$LOG_ROOT" "$RESULT_ROOT"

echo "========================================================"
echo "缓存模式        : ${CACHE_MODE}%"
echo "模型路径        : $MODEL_PATH"
echo "API 模型名      : $SERVED_NAME"
echo "服务连接        : $NET_DESC"
echo "request-rate    : $REQUEST_RATE（0 = 不限速，并发由 batch-size 控制）"
echo "retry           : $RETRY"
echo "api-key         : $([[ -n "$API_KEY" ]] && echo '<已设置>' || echo '<未设置>')"
echo "模型配置        : $MODEL_CFG"
echo "数据集配置      : $DATASET_CFG（长度/前缀由 CLI 覆盖）"
echo "预热请求数      : $WARMUP_NUM（每个并发档单独预热，不计入指标）"
echo "任务组合数      : ${#GROUP_SPECS[@]}（每组内部再按并发阶梯展开）"
echo "日志目录        : $LOG_ROOT"
echo "结果目录        : $RESULT_ROOT"
echo "========================================================"

failures=0
for test in "${GROUP_SPECS[@]}"; do
    # 解析「总输入长度 输出长度 并发阶梯...」，避免 set -u 下访问不存在的数组下标
    read -r -a GROUP_FIELDS <<< "$test"
    TOTAL_INPUT_LEN="${GROUP_FIELDS[0]:-}"
    MAX_TOKENS="${GROUP_FIELDS[1]:-}"
    CONCURRENCIES=("${GROUP_FIELDS[@]:2}")
    if [[ -z "$TOTAL_INPUT_LEN" || -z "$MAX_TOKENS" || ${#CONCURRENCIES[@]} -eq 0 ]]; then
        echo "错误: GROUP_SPECS 格式非法: '$test'（需要「总输入长度 输出长度 并发阶梯...」）" >&2
        failures=$((failures + 1))
        continue
    fi

    if [[ "$CACHE_MODE" == "80" ]]; then
        # 公共前缀 + 随机后缀 = 总输入长度
        PREFIX_LEN=$(( (TOTAL_INPUT_LEN * 80 + 50) / 100 ))
    else
        PREFIX_LEN=0
    fi
    SUFFIX_LEN=$(( TOTAL_INPUT_LEN - PREFIX_LEN ))

    # 生成的请求总数 = 最大并发档（>= 每档 --num-prompts，避免并发窗口跑不满告警）
    REQUEST_COUNT=0
    for bs in "${CONCURRENCIES[@]}"; do
        (( bs > REQUEST_COUNT )) && REQUEST_COUNT="$bs"
    done

    GROUP_TAG="in${TOTAL_INPUT_LEN}_out${MAX_TOKENS}"
    LOG="${LOG_ROOT}/${GROUP_TAG}.log"
    WORK_DIR="${RESULT_ROOT}/${GROUP_TAG}"
    mkdir -p "$WORK_DIR"

    echo
    echo ">>> [$(date '+%F %T')] 开始: input=${TOTAL_INPUT_LEN}, output=${MAX_TOKENS}"
    echo "prefix   : $PREFIX_LEN"
    echo "suffix   : $SUFFIX_LEN"
    echo "并发阶梯 : ${CONCURRENCIES[*]}"
    echo "请求条数 : ${CONCURRENCIES[*]}（与并发阶梯一一对应）"
    echo "request-count : $REQUEST_COUNT"
    echo "工作目录 : $WORK_DIR"

    # 一条命令展开该组全部并发档：
    #   --batch-size / --num-prompts 取值个数必须一致，按位置一一对应
    #   --request-count 覆盖数据集 RequestCount（>= 最大并发档）
    #   输入长度/公共前缀由 dataset CLI 覆盖，无需手改配置文件
    ais_bench \
        --models "$MODEL_CFG" \
        --datasets "$DATASET_CFG" \
        --mode perf \
        --work-dir "$WORK_DIR" \
        --path "$MODEL_PATH" \
        --model-name "$SERVED_NAME" \
        "${NET_ARGS[@]}" \
        --batch-size "${CONCURRENCIES[@]}" \
        --num-prompts "${CONCURRENCIES[@]}" \
        --request-count "$REQUEST_COUNT" \
        --tokenid-config-request-size "$TOTAL_INPUT_LEN" \
        --tokenid-config-prefix-len "$PREFIX_LEN" \
        --num-warmups "$WARMUP_NUM" \
        --max-out-len "$MAX_TOKENS" \
        --generation-kwargs "$GEN_KWARGS" \
        "${TRUST_ARGS[@]+"${TRUST_ARGS[@]}"}" \
        2>&1 | tee "$LOG"
    status=${PIPESTATUS[0]}
    if (( status != 0 )); then
        echo "FAILED (exit $status): input=${TOTAL_INPUT_LEN}, output=${MAX_TOKENS}; see $LOG" >&2
        failures=$((failures + 1))
    fi
    echo "<<< [$(date '+%F %T')] 完成: input=${TOTAL_INPUT_LEN}, output=${MAX_TOKENS}"
done

echo
echo "========================================================"
if (( failures == 0 )); then
    echo "全部测试完成！"
else
    echo "$failures 组测试失败，请检查对应日志: $LOG_ROOT" >&2
fi
echo "日志目录    : $LOG_ROOT"
echo "结果目录    : $RESULT_ROOT"
echo "产物（每组各一份）: combos_manifest.json / batch_perf_summary.csv / batch_perf_report.html"
echo "定位方式    : find \"$RESULT_ROOT\" -name 'batch_perf_report.html'"
echo "========================================================"
(( failures == 0 )) || exit 1
