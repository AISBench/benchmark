#!/bin/bash
# 目录清理脚本 - Bash封装器
# 功能：简化Python目录清理脚本的调用
# 用法：./cleanup_helper.sh [选项]

# 定义默认值
DEFAULT_PATH="../RunWorkspace/"
DEFAULT_MAX_COUNT=20
DEFAULT_DAYS=30
DRY_RUN=""
SCRIPT_NAME="directory_cleaner.py"
SCRIPT_PATH=$(realpath `dirname $0`)
LOG_FILE="cleanup.log"

# 显示使用说明
show_help() {
    echo "目录清理工具 - Bash封装器"
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -p, --path      目标目录路径 (默认: $DEFAULT_PATH)"
    echo "  -c, --count    最大子目录数量 (默认: $DEFAULT_MAX_COUNT)"
    echo "  -d, --days     保留天数阈值 (默认: $DEFAULT_DAYS)"
    echo "  -dr, --dry-run 模拟运行模式 (预览但不执行实际删除)"
    echo "  -l, --log      日志文件路径 (默认: $LOG_FILE)"
    echo "  -h, --help     显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -p /data/backups -c 15 -d 45"
    echo "  $0 --path /tmp/cache --count 10 --dry-run"
}

# 准备环境
prepare_env() {
    export PATH=$PATH:/usr/local/bin/:/usr/local/sbin/
    source $HOME/.bashrc

    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate aisbench_smoke
}

# 检查Python脚本是否存在
check_python_script() {
    if [[ ! -f "$SCRIPT_NAME" ]]; then
        echo "错误: Python清理脚本 $SCRIPT_NAME 未找到"
        echo "请确保此脚本与Python脚本在 $SCRIPT_PATH 目录下"
        exit 1
    fi
}

# 检查Python版本
check_python_version() {
    if ! python3 -c "import sys; sys.exit(0) if sys.version_info >= (3, 6) else sys.exit(1)"; then
        echo "错误: 需要 Python 3.6 或更高版本"
        echo "当前Python版本: $(python3 --version 2>&1)"
        exit 1
    fi
}

# 日志函数
log() {
    local message="$1"
    local log_to_file="${2:-true}"
    
    # 显示到控制台
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message"
    
    # 如果需要，记录到文件
    if [[ "$log_to_file" == "true" && -n "$LOG_FILE" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" >> "$LOG_FILE"
    fi
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -p|--path)
                TARGET_PATH="$2"
                shift 2
                ;;
            -c|--count)
                MAX_COUNT="$2"
                shift 2
                ;;
            -d|--days)
                DAYS="$2"
                shift 2
                ;;
            -dr|--dry-run)
                DRY_RUN="--dry-run"
                shift
                ;;
            -l|--log)
                LOG_FILE="$2"
                if [ -n "$LOG_FILE" ]; then
                    # 检查文件是否存在
                    if [ -f "$LOG_FILE" ]; then
                        # 转化为绝对路径
                        LOG_FILE=$(realpath "$LOG_FILE")
                    else
                        echo "错误：日志文件 $LOG_FILE 不存在。"
                        exit 1
                    fi
                fi
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 主函数
main() {
    # 初始化变量
    parse_arguments "$@"
    
    # 初始化环境
    prepare_env

    # 设置默认值
    local target_path="${TARGET_PATH:-$DEFAULT_PATH}"
    local max_count="${MAX_COUNT:-$DEFAULT_MAX_COUNT}"
    local days="${DAYS:-$DEFAULT_DAYS}"
    
    # 检查依赖
    cd $SCRIPT_PATH
    check_python_script
    check_python_version
    
    # 记录开始信息
    log "开始目录清理任务"
    log "目标目录: $target_path"
    log "最大数量: $max_count"
    log "保留天数: $days"
    if [[ -n "$DRY_RUN" ]]; then
        log "运行模式: 模拟运行"
    else
        log "运行模式: 实际执行"
    fi
    
    # 准备Python命令
    local python_cmd="python3 $SCRIPT_NAME --path '$target_path' --max-count $max_count --days $days"
    if [[ -n "$DRY_RUN" ]]; then
        python_cmd+=" --dry-run"
    fi
    
    # 添加日志文件（如果需要）
    if [[ -n "$LOG_FILE" && "$LOG_FILE" != "false" ]]; then
        python_cmd+=" 2>&1 | tee -a '$LOG_FILE'"
    fi
    
    log "执行命令: $python_cmd"
    log ""
    
    # 执行Python脚本
    eval "$python_cmd"
    
    # 检查执行结果
    local status=$?
    if [[ $status -eq 0 ]]; then
        log "目录清理任务成功完成"
    else
        log "目录清理任务失败，错误码: $status"
    fi
    
    # 关闭conda环境
    conda deactivate

    exit $status
}

# 脚本入口
main "$@"