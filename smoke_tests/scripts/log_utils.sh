#!/bin/bash

# 日志管理模块
# 特性：同时输出到终端和日志文件，基于文件描述符的可靠开关控制

# 全局变量
declare -g LOG_FILE=""
declare -g TEE_PID=""
declare -g PIPE_PATH=""
declare -g LOG_CAPTURE_ENABLED=1  # 默认启用日志捕获
declare -g LOG_FD=0
declare -g ORIG_STDOUT_FD        # 原始标准输出的文件描述符
declare -g ORIG_STDERR_FD        # 原始标准错误的文件描述符
declare -g LOG_STATUS=-1

# 初始化日志系统 (必须首先调用)
init_logging() {
    local log_dir="${1:-logs}"
    mkdir -p "$log_dir"
    
    # 获取脚本文件名（不含扩展名）
    local script_file=$(basename "$0")
    script_file="${script_file%.*}"
    
    LOG_FILE="$log_dir/${script_file}.log"
    
    # 设置原始标准输出和错误描述符
    ORIG_STDOUT_FD=3
    ORIG_STDERR_FD=4

    # 使用 eval 安全开启保存的原始描述符
    if [[ -n "$ORIG_STDOUT_FD" ]]; then
        eval "exec $ORIG_STDOUT_FD>&1"
    fi
    if [[ -n "$ORIG_STDERR_FD" ]]; then
        eval "exec $ORIG_STDERR_FD>&2"
    fi
    
    # 创建临时管道
    PIPE_PATH=$(mktemp -u)
    mkfifo "$PIPE_PATH"
    
    # 启动日志写入进程（使用unbuffer保持tty行为）
    if command -v unbuffer >/dev/null; then
        unbuffer tee -a "$LOG_FILE" < "$PIPE_PATH" >&$ORIG_STDOUT_FD & 
    else
        tee -a "$LOG_FILE" < "$PIPE_PATH" >&$ORIG_STDOUT_FD & 
    fi
    TEE_PID=$!
    
    # 创建新的文件描述符用于日志捕获
    exec {LOG_FD}>"$PIPE_PATH"
    
    # 重定向所有输出到管道
    exec >&${LOG_FD} 2>&1
    
    echo -e "[NOTE] 日志系统已初始化 => $LOG_FILE"
    LOG_STATUS=0

    return 0
}

# 启用日志捕获
enable_log_capture() {
    if [ $LOG_CAPTURE_ENABLED -eq 0 ]; then
        if [ -n "$LOG_FD" ] && { >&${LOG_FD}; } 2>/dev/null; then
            exec >&${LOG_FD} 2>&1
            LOG_CAPTURE_ENABLED=1
            echo -e "[NOTE] 日志捕获已启用"
        else
            echo -e "[WARNING] 无法启用日志捕获 - 日志文件描述符无效" >&2
        fi
    fi
}

# 禁用日志捕获
disable_log_capture() {
    if [ $LOG_CAPTURE_ENABLED -eq 1 ]; then
        if [ -n "$ORIG_STDOUT_FD" ] && [ -n "$ORIG_STDERR_FD" ] && \
           { >&${ORIG_STDOUT_FD}; } 2>/dev/null && { >&${ORIG_STDERR_FD}; } 2>/dev/null; then
            echo -e "[NOTE] 日志捕获已禁用"
            exec 1>&${ORIG_STDOUT_FD} 2>&${ORIG_STDERR_FD}
            LOG_CAPTURE_ENABLED=0
        else
            echo -e "[WARNING] 无法禁用日志捕获 - 原始描述符无效" >&2
            # 回退到/dev/tty
            exec 1>/dev/tty 2>&1
            LOG_CAPTURE_ENABLED=0
        fi
    fi
}

# 清理资源
cleanup_logging() {

    if [ "$LOG_STATUS" = -1 ]; then
        echo "[NOTE] 日志未初始化，跳过清理"
        return 0
    elif [ "$LOG_STATUS" = 1 ]; then
        echo "[NOTE] 日志资源已清理，跳过清理"
        return 0
    fi

    # 确保日志捕获已启用以便记录清理信息
    if [ $LOG_CAPTURE_ENABLED -eq 0 ]; then
        enable_log_capture
    fi
    
    echo -e "[NOTE] 日志文件已保存至: $LOG_FILE"
    # 等待日志缓存写入
    sleep 0.3
    
    # 关闭日志文件描述符
    if [ -n "$LOG_FD" ] && { >&${LOG_FD}; } 2>/dev/null; then
        exec {LOG_FD}>&-
        echo -e "[NOTE] 已关闭日志文件描述符"
    else
        echo -e "[WARNING] 日志文件描述符无效或已关闭" >&2
    fi
    
    # 恢复原始输出（带安全检查）
    if [ -n "$ORIG_STDOUT_FD" ] && { >&${ORIG_STDOUT_FD}; } 2>/dev/null; then
        eval "exec 1>&${ORIG_STDOUT_FD}"
        eval "exec ${ORIG_STDOUT_FD}>&-"
        echo -e "[NOTE] 已恢复标准输出"
    else
        echo -e "[WARNING] 原始标准输出描述符无效，回退到终端" >&2
        exec 1>/dev/tty
    fi
    
    if [ -n "$ORIG_STDERR_FD" ] && { >&${ORIG_STDERR_FD}; } 2>/dev/null; then
        eval "exec 2>&${ORIG_STDERR_FD}"
        eval "exec ${ORIG_STDERR_FD}>&-"
        echo -e "[NOTE] 已恢复标准错误"
    else
        echo -e "[WARNING] 原始标准错误描述符无效，回退到终端" >&2
        exec 2>/dev/tty
    fi
    
    # 清理管道
    if [ -n "$PIPE_PATH" ] && [ -e "$PIPE_PATH" ]; then
        rm -f "$PIPE_PATH" && echo -e "[NOTE] 已删除管道文件" || \
        echo -e "[WARNING] 无法删除管道文件 $PIPE_PATH" >&2
    fi
    
    # 清理后台进程
    if [ -n "$TEE_PID" ]; then
        if kill -0 $TEE_PID 2>/dev/null; then
            kill $TEE_PID 2>/dev/null
            wait $TEE_PID 2>/dev/null
            echo -e "[NOTE] 已停止日志写入进程"
        else
            echo -e "[NOTE] 日志写入进程已终止" >&2
        fi
    fi

    echo -e "[SUCCESS] 日志资源清理完成"
    LOG_STATUS=1

    return 0
}

# 注册退出时的清理钩子
trap_log_cleanup() {
    cleanup_logging
}