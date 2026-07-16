#!/bin/bash

# 默认值
DEFAULT_DATASET="aa_lcr_llmjudge"
DEFAULT_MODEL="vllm_api_general_chat"
DEFAULT_WORK_DIR="outputs/default"

# 从命令行参数获取，支持：./script.sh <dataset> <model> <work_dir>
DATASET="${1:-$DEFAULT_DATASET}"
MODEL="${2:-$DEFAULT_MODEL}"
WORK_DIR_BASE="${3:-$DEFAULT_WORK_DIR}"

# 要监控和启动的命令
CMD="ais_bench --models $MODEL --datasets $DATASET --num-prompts 1 --work-dir $WORK_DIR_BASE"

# 总共运行次数
MAX_RETRIES=3

# 日志文件
LOG_FILE="ais_bench_${DATASET}.log"

echo "========================================"
echo "监控脚本启动"
echo "命令: $CMD"
echo "总共运行次数: $MAX_RETRIES"
echo "stdout 日志: $LOG_FILE"
echo "work_dir 基路径: $WORK_DIR_BASE"
echo "========================================"

for run in $(seq 1 $MAX_RETRIES); do
    echo ""
    echo "========================================"
    echo "第 $run/$MAX_RETRIES 次运行"
    echo "========================================"

    # 记录启动时间标记，用于查找新生成的 progress.log
    START_MARKER=$(mktemp)

    # 启动进程
    nohup $CMD > "$LOG_FILE" 2>&1 &
    BENCH_PID=$!
    echo "进程启动成功，PID: $BENCH_PID"

    # 等待 progress.log 出现（最多等待 60 秒）
    PROGRESS_LOG=""
    echo "正在查找 progress.log ..."
    for i in $(seq 1 60); do
        PROGRESS_LOG=$(find "$WORK_DIR_BASE" -name "progress.log" -newer "$START_MARKER" 2>/dev/null | tail -1)
        if [ -n "$PROGRESS_LOG" ] && [ -s "$PROGRESS_LOG" ]; then
            echo "找到 progress.log: $PROGRESS_LOG"
            break
        fi
        sleep 1
    done

    # 清理标记文件
    rm -f "$START_MARKER"

    if [ -z "$PROGRESS_LOG" ] || [ ! -f "$PROGRESS_LOG" ]; then
        echo "未找到 progress.log，仅显示 stdout 日志..."
    fi

    # 监控循环：每 10 秒刷新，直到进程退出
    while kill -0 $BENCH_PID 2>/dev/null; do
        clear
        echo "========================================"
        echo "第 $run/$MAX_RETRIES 次运行 (PID: $BENCH_PID)"
        echo "刷新时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "下次刷新: 10 秒后"
        echo "========================================"

        # 显示 progress.log
        if [ -n "$PROGRESS_LOG" ] && [ -f "$PROGRESS_LOG" ]; then
            echo ""
            echo "--- 任务进度表 (progress.log, 每10秒更新) ---"
            cat "$PROGRESS_LOG"
        fi

        # 显示 stdout 日志最后 10 行
        echo ""
        echo "--- 最近日志 (最后 10 行) ---"
        tail -10 "$LOG_FILE"

        echo ""
        echo "按 Ctrl+C 停止查看（进程继续运行，等待 10 秒后刷新）..."

        # 等待 10 秒，Ctrl+C 可中断
        sleep 10 &
        wait $! 2>/dev/null || true
    done

    # 等待进程完全退出
    wait $BENCH_PID 2>/dev/null

    # 进程结束后：输出 progress.log 最后 50 行，然后删除
    if [ -n "$PROGRESS_LOG" ] && [ -f "$PROGRESS_LOG" ]; then
        echo ""
        echo "--- progress.log 最终状态 (最后 50 行) ---"
        tail -n 50 "$PROGRESS_LOG"
        echo "--- 删除 progress.log: $PROGRESS_LOG ---"
        rm -f "$PROGRESS_LOG"
    fi

    echo "第 $run/$MAX_RETRIES 次运行已完成"
done

echo ""
echo "========================================"
echo "全部 $MAX_RETRIES 次运行已完成"
echo "========================================"
