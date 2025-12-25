#!/bin/bash

###################################### run_steps.sh 的 说明显示函数 ####################################################
help_func() {
    # 定义颜色变量
    YELLOW='\033[0;33m'
    GREEN='\033[0;32m'
    BLUE='\033[0;36m'
    MAGENTA='\033[0;35m'
    ORANGE='\033[0;33m'
    RED='\033[0;31m'
    BOLD='\033[1m'
    RESET='\033[0m'
    
    echo "=============================================================================================================="
    echo
    echo
    echo -e "${BOLD}${YELLOW}使用:${RESET}"
    echo -e "  ${GREEN}run_steps.sh [选项]${RESET}"
    echo
    
    echo -e "${BOLD}${YELLOW}选项:${RESET}"
    # 内层脚本参数描述保持一致
    echo -e "  ${BOLD}${GREEN}-p${RESET} ${BOLD}TEST_CASE_PATH${RESET}     测试用例的根目录"
    echo -e "  ${BOLD}${GREEN}-t${RESET} ${BOLD}TEST_CASE_TAGS${RESET}     ${BLUE}选择待测用例的标签（标签用空格分隔）${RESET}"
    echo -e "                        ${BLUE}标签类型：${RESET}"
    echo -e "                          - 只要被包含在用例的 ${MAGENTA}case_type${RESET}、${MAGENTA}case_group${RESET} 或 ${MAGENTA}case_name${RESET} 中即被选中"
    echo -e "  ${BOLD}${GREEN}-l${RESET} ${BOLD}TEST_CASE_LABEL${RESET}    ${BLUE}选择待测用例的特殊值（键值对用空格分隔）${RESET}"
    echo -e "                        ${BLUE}键值格式：${RESET}"
    echo -e "                          - 键值用等号分隔（${BOLD}key${RESET}=${BOLD}value${RESET}）"
    echo -e "                          - 值为可迭代元素时，内部元素用逗号分隔"
    echo -e "  ${BOLD}${GREEN}-d${RESET}                    ${BLUE}启用 debug 模式${RESET}"
    echo -e "  ${BOLD}${GREEN}-r${RESET}                    ${BLUE}重跑错误用例${RESET}"
    
    # 新增参数描述
    echo -e "  ${BOLD}${GREEN}-nc${RESET}                   ${BLUE}不执行克隆操作直接运行${RESET}"
    echo -e "  ${BOLD}${GREEN}-self${RESET} ${BOLD}URL${RESET} [${BOLD}Branch${RESET}]    ${BLUE}使用私有仓库进行测试${RESET}"
    
    # 新增软链接参数描述
    echo -e "  ${BOLD}${GREEN}-l-model${RESET} ${BOLD}PATH${RESET}        ${BLUE}使用软链接方式准备模型${RESET}"
    echo -e "  ${BOLD}${GREEN}-l-repo${RESET} ${BOLD}PATH${RESET}         ${BLUE}使用软链接方式准备代码仓库${RESET}"
    echo
    
    echo -e "${BOLD}${YELLOW}参数说明:${RESET}"
    echo -e "  ${MAGENTA}[]${RESET} : 可选参数"
    echo -e "  ${MAGENTA}()${RESET} : 必选参数"
    echo
    
    echo -e "${BOLD}${YELLOW}详细说明:${RESET}"
    # 相同参数保持原描述
    echo -e "  ${BOLD}${GREEN}-p${RESET} 选项:"
    echo -e "    设置测试用例的根目录路径"
    echo
    echo -e "  ${BOLD}${GREEN}-t${RESET} 选项:"
    echo -e "    选中所有包含指定标签的用例（${RED}逻辑 OR 关系${RESET}）"
    echo -e "    标签可出现在 ${MAGENTA}case_type${RESET}、${MAGENTA}case_group${RESET} 或 ${MAGENTA}case_name${RESET} 中"
    echo -e "    ${RED}示例:${RESET} ${GREEN}-t group1 group2 casename anytags${RESET}"
    echo
    echo -e "  ${BOLD}${GREEN}-l${RESET} 选项:"
    echo -e "    仅选中满足${ORANGE}所有指定键值对${RESET}的用例（${RED}逻辑 AND 关系${RESET}）"
    echo -e "    当值为可迭代元素时（如列表），需要${RED}全等${RESET}才满足条件"
    echo -e "    ${RED}示例:${RESET}"
    echo -e "      ${GREEN}-l rank_size=3 case_group=group1,group2 script=start:run.sh,end:clean.sh${RESET}"
    echo
    echo -e "  ${BOLD}${GREEN}-d${RESET} 选项:"
    echo -e "    启用调试模式，输出详细的执行日志"
    echo
    echo -e "  ${BOLD}${GREEN}-r${RESET} 选项:"
    echo -e "    自动${RED}重新运行失败的测试用例${RESET}"
    echo
    
    # 新增参数详细说明
    echo -e "  ${BOLD}${GREEN}-nc${RESET} 选项:"
    echo -e "    跳过代码克隆步骤，使用现有代码直接运行测试"
    echo
    echo -e "  ${BOLD}${GREEN}-self${RESET} 选项:"
    echo -e "    使用自定义的 Ascend/tools 仓库而非默认仓库"
    echo -e "    ${BOLD}URL${RESET}: ${RED}必填${RESET}，私有仓库地址"
    echo -e "    ${BOLD}Branch${RESET}: ${GREEN}可选${RESET}，默认为 master 分支"
    echo -e "    ${RED}示例:${RESET} ${GREEN}-self https://github.com/user/tools.git dev_branch${RESET}"
    echo
    
    # 新增软链接参数详细说明
    echo -e "  ${BOLD}${GREEN}-l-model${RESET} 选项:"
    echo -e "    使用软链接方式准备模型，避免重复下载大型模型文件"
    echo -e "    ${BOLD}PATH${RESET}: ${RED}必填${RESET}，指向已存在的模型目录路径"
    echo -e "    ${RED}优势:${RESET} 节省下载时间，避免重复存储"
    echo -e "    ${RED}示例:${RESET} ${GREEN}-l-model /data/models/Qwen2.5-7B-Instruct${RESET}"
    echo
    echo -e "  ${BOLD}${GREEN}-l-repo${RESET} 选项:"
    echo -e "    使用软链接方式准备代码仓库，避免重复克隆代码"
    echo -e "    ${BOLD}PATH${RESET}: ${RED}必填${RESET}，指向已存在的代码仓库路径"
    echo -e "    ${RED}优势:${RESET} 节省克隆时间，保留本地修改"
    echo -e "    ${RED}示例:${RESET} ${GREEN}-l-repo ~/projects/benchmark${RESET}"
    echo
    echo
    echo "=============================================================================================================="
}

################################################################################################################

############################################### 文件操作部分 ####################################################
# 函数: ensure_and_enter_dir
# 描述: 确保目标目录及其所有父级目录存在，然后进入该目录
# 参数:
#   $1: target_dir - 目标目录路径（可选，默认值为给定路径）
# 返回值:
#   0: 成功创建目录并进入
#   1: 创建目录失败或无法进入目录
ensure_and_enter_dir() {
    local target_dir="${1:-$PROJECT_PATH}"
    
    # 安全规范: 绝对路径处理前添加双斜杠避免意外根目录操作
    local safe_dir="/${target_dir#/}"
    
    echo "Verifying and entering directory: ${safe_dir}"
    
    # 创建目录结构（-p 确保多级目录创建）
    if ! mkdir -p "${safe_dir}"; then
        echo "[Error] Failed to create directory structure: ${safe_dir}" >&2
        return 1
    fi
    
    # 验证目录可访问性
    if ! [ -d "${safe_dir}" ]; then
        echo "[Error] Directory not accessible after creation: ${safe_dir}" >&2
        return 1
    fi
    
    # 进入目录并验证
    if ! cd "${safe_dir}"; then
        echo "[Error] Failed to enter directory: ${safe_dir}" >&2
        return 1
    fi
    
    # 二次确认当前目录
    if [ "$(pwd)" != "${safe_dir}" ]; then
        echo "[WARNING] Current directory $(pwd) does not match target ${safe_dir}" >&2
        return 1
    fi
    
    echo "[SUCCESS] Successfully entered directory: $(pwd)"
    return 0
}

# 函数：检查目录是否只包含 synthetic/ 文件夹（无其他子目录）
# 参数：
#   $1: 要检查的目标目录路径
# 返回值：
#   0: 目录中只存在 synthetic/ 文件夹（或为空）
#   1: 存在其他子目录
check_datasets_empty() {
    local target_dir="$1"
    local exists_content="$2"
    
    # 1. 检查目录是否存在
    if [ ! -d "$target_dir" ]; then
        echo "[ERROR] 目录 '$target_dir' 不存在"
        return 2
    fi

    # 2. 获取所有直接子目录（不包括隐藏目录）
    local subdirs
    # 使用数组存储结果，正确处理带空格的目录名
    mapfile -t subdirs < <(find "$target_dir" -mindepth 1 -maxdepth 1 -type d -not -path "$target_dir/synthetic" -print 2>/dev/null)
    
    # 3. 检查是否有其他子目录
    if [ ${#subdirs[@]} -gt 0 ]; then
        echo "${exists_content}:"
        printf '  - %s\n' "${subdirs[@]}"
        return 1
    fi
    
    # 4. 检查是否存在 synthetic 文件夹（可选）
    if [ ! -d "$target_dir/synthetic" ]; then
        echo "[WARNING] synthetic/ 文件夹不存在"
    fi
    
    echo "[SUCCESS] 目录检查通过: 仅包含 synthetic/ 或为空"
    return 0
}

# 递归检查多个文件/文件夹是否同时存在
# 参数1: 搜索根路径
# 参数2-N: 要查找的文件/文件夹名（支持多个）
# 返回值: 全部存在时返回0，否则返回1
check_paths() {
    local search_dir="$1"
    shift  # 移除第一个参数，剩余的都是目标
    
    # 验证根目录存在
    if [ ! -d "$search_dir" ]; then
        echo "错误: 搜索目录 '$search_dir' 不存在" >&2
        return 2
    fi
    
    # 处理多个目标文件
    local all_exists=0
    local -a missing_items=()
    
    for target in "$@"; do
        # 使用find递归搜索
        if ! find "$search_dir" -name "$target" -print -quit | grep -q .; then
            missing_items+=("$target")
            all_exists=1
        fi
    done
    
    # 输出缺失项
    if [ ${#missing_items[@]} -gt 0 ]; then
        echo "在 '$search_dir' 中未找到:"
        printf ' - %s\n' "${missing_items[@]}"
    fi
    
    return $all_exists
}

# 路径检查函数
validate_path() {
    local path="$1"
    local description="$2"
    
    if [ -z "$path" ]; then
        echo "Error: $description 路径未指定"
        return 1
    fi
    
    if [ ! -e "$path" ]; then
        echo "Error: $description 路径不存在: $path"
        return 1
    fi
    
    return 0
}

# 快速目录复制函数
fast_copy() {
    local src=$1
    local dest=$2
    local parallel_count=${3:-$(($(nproc) + 1))}  # 默认CPU核心数+1
    local start_time end_time elapsed_time
    
    echo -e "Datasets copy from $src to $dest:"
    
    start_time=$(date +%s.%N)

    # 使用cp命令，实测比 rsync 和 tar 快很多
    find "$src" -mindepth 1 -maxdepth 1 -print0 | \
    xargs -0 -n1 -P "$parallel_count" -I {} cp -R --sparse=always {} "$dest/"

    # 记录结束时间并计算耗时
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo -e "拷贝总耗时: $elapsed_time 秒\n" 
}


# 安全创建或覆盖软链接
create_symlink() {
    local src="$1"
    local dest="$2"
    
    # 如果目标存在且是软链接，安全删除
    if [ -L "$dest" ]; then
        echo "[INFO] 覆盖现有软链接: $dest -> $(readlink $dest)"
        rm -f "$dest"
    fi
    
    # 如果目标存在但不是软链接，报错
    if [ -e "$dest" ]; then
        echo "[ERROR] 无法创建软链接: $dest 已存在且不是软链接, 请不要填写软链接参数或清除 $dest 后再运行"
        return 1
    fi
    
    # 创建新软链接
    ln -s "$src" "$dest" || {
        echo "[ERROR] 创建软链接失败: $dest → $src"
        return 1
    }
    
    echo "[SUCCESS] 软链接创建完成: $dest → $src"
    return 0
}

# 函数：在目标目录创建源目录第一层内容的软链接（仅覆盖同名软链接）
# 参数：<源目录> <目标目录>
create_symlinks_rec() {
    local src_dir="$1"
    local target_dir="$2"
    local start_time end_time elapsed_time

    echo -e "[INFO] 创建软链接: $src_dir → $target_dir"
    
    start_time=$(date +%s.%N)
    # 遍历源目录的第一层内容
    find "$src_dir" -maxdepth 1 -mindepth 1 | while read -r item; do
        local item_name=$(basename "$item")
        local target_link="$target_dir/$item_name"
        
        # 仅当目标存在且是软链接时才删除
        if [ -L "$target_link" ]; then
            echo -e "  覆盖软链接: $item_name"
            rm -f "$target_link"
        fi
        
        # 创建软链接（即使同名文件存在也不覆盖）
        ln -s "$item" "$target_link" 2>/dev/null || 
            echo -e "  [警告] 跳过非软链接同名项: $item_name (手动清理后可重新链接)"
    done
    
    # 记录结束时间并计算耗时
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo -e "拷贝总耗时: $elapsed_time 秒\n" 
}

#######################################################################################################


###################################### 虚拟服务化相关部分 ##############################################

# 检测端口可用性函数
find_available_port() {
    local default_port=${1:-8080}  # 默认起始端口
    local max_port=65535          # 最大端口号
    local port=$default_port
    
    # 端口冲突检测
    is_port_free() {
        # 使用lsof检查端口
        if ! lsof -i :$1 >/dev/null; then
            # 双重检查使用netstat
            ! netstat -tuln | grep -q ":$1 "
        else
            false
        fi
    }
    
    # 查找可用端口
    while [ $port -le $max_port ]; do
        if is_port_free $port; then
            echo $port
            return 0
        fi
        port=$((port + 1))
    done
    
    # 查找失败处理
    echo "[ERROR] 无法找到可用端口（范围：${default_port}-${max_port})" >&2
    return 1
}

# 服务清理全局状态
declare -g SERVICE_STATUS=-1
declare -g SERVICE_STOPPED_CHECK=false

# 启动检查函数
check_service_ready() {
    # 添加首次检查前的短暂延迟（避免立即请求时服务还未开始启动）
    sleep 0.5

    local max_checks=10
    local check_interval=3
    local checks_done=0
    export no_proxy=$no_proxy,$AISBENCH_SMOKE_SERVICE_IP
    while [ $checks_done -lt $max_checks ]; do
        # 1. 检查进程是否存活
        if ! ps -p $VIRTUAL_SERVICE_PID >/dev/null; then
            echo "[ERROR] 进程异常终止"
            return 1
        fi

        # 2. 服务健康检查
        local response=$(curl "http://$AISBENCH_SMOKE_SERVICE_IP:$AISBENCH_SMOKE_SERVICE_PORT/active_connections")
        if [[ "$response" =~ \"active_connections\":\"[0-9]+:[[:space:]]*[0-9]+\" ]]; then
            SERVICE_STATUS=0
            echo "[SUCCESS] 服务已就绪"
            return 0
        fi
        
        echo "等待服务启动($((checks_done+1))/$max_checks)..."
        sleep $check_interval
        checks_done=$((checks_done + 1))
    done
    
    echo "[ERROR] 服务启动超时"
    return 1
}

# 服务终止函数
stop_service() {
    # 防止重复清理
    if [ "$SERVICE_STATUS" = -1 ]; then
        echo "[INFO] 服务未启动，跳过清理"
        return 0
    elif [ "$SERVICE_STATUS" = 1 ]; then
        echo "[INFO] 服务已停止，跳过清理"
        return 0
    fi

    # 检查变量是否定义
    if [ -z "$VIRTUAL_SERVICE_PID" ] || [ -z "$AISBENCH_SMOKE_SERVICE_PORT" ]; then
        echo "[ERROR] PID 或端口未设置!"
        return 1
    fi

    # 优先优雅终止服务进程
    if ps -p $VIRTUAL_SERVICE_PID >/dev/null; then
        echo "尝试优雅终止服务..."
        kill $VIRTUAL_SERVICE_PID
        # 循环等待最多5秒
        timeout=5
        while [ $timeout -gt 0 ] && ps -p $VIRTUAL_SERVICE_PID >/dev/null; do
            sleep 0.5
            timeout=$((timeout - 1))
        done
        # 如果仍未退出，强制终止
        if ps -p $VIRTUAL_SERVICE_PID >/dev/null; then
            echo "强制终止服务..."
            kill -9 $VIRTUAL_SERVICE_PID
            sleep 1
        fi
    fi

    # 释放端口：只处理与服务关联的进程
    if lsof -i :$AISBENCH_SMOKE_SERVICE_PORT >/dev/null; then
        echo "[WARNING] 端口 $AISBENCH_SMOKE_SERVICE_PORT 被占用"
        # 获取占用端口的PID列表
        local port_pids=$(lsof -t -i :$AISBENCH_SMOKE_SERVICE_PORT)
        # 过滤掉已不存在的进程
        for pid in $port_pids; do
            if ps -p $pid >/dev/null; then
                echo "终止进程 $pid"
                kill -9 $pid
            fi
        done
    fi

    # 最终验证
    if ps -p $VIRTUAL_SERVICE_PID >/dev/null; then
        echo "[WARNING] 服务可能未关闭"
        return 2
    elif lsof -i :$AISBENCH_SMOKE_SERVICE_PORT >/dev/null; then
        echo "[ERROR] 端口未释放"
        return 3
    else
        SERVICE_STATUS=1
        echo "[SUCCESS] 服务和端口已清理"
        return 0
    fi
}

# 服务清理确认函数
service_cleanup_check() {
    if [ "$SERVICE_STATUS" = -1 ]; then
        echo "[INFO] 服务未启动，跳过检查"
        return 0
    elif [ "$SERVICE_STOPPED_CHECK" = true ]; then
        echo "[INFO] 服务停止已检查，跳过检查"
        return 0
    fi

    if [ "$SERVICE_STATUS" != 1 ]; then
        echo -e "[WARNING] 服务尚未执行终止"
    fi

    # 检查launch_service.sh进程
    if pgrep -f "launch_service.sh" >/dev/null; then
        echo -e "[WARNING] 仍有launch_service.sh进程在运行"
        echo -e "相关进程详情:"
        ps aux | grep launch_service | grep -v grep
    else
        echo -e "[SUCCESS] 所有launch_service.sh进程已结束"
    fi

    # 检查端口占用情况
    if lsof -i :$AISBENCH_SMOKE_SERVICE_PORT >/dev/null; then
        echo -e "[WARNING] 端口 $AISBENCH_SMOKE_SERVICE_PORT 仍被占用"
        
        # 找出占用端口的进程
        local port_pids=$(lsof -t -i :$AISBENCH_SMOKE_SERVICE_PORT)
        if [ -n "$port_pids" ]; then
            echo -e "[WARNING] 占用端口的进程: $port_pids"
            echo -e "进程详情:"
            ps -p $port_pids -o pid,user,command
        fi
    else
        echo -e "[SUCCESS] 端口 $AISBENCH_SMOKE_SERVICE_PORT 已成功释放"
    fi

    SERVICE_STOPPED_CHECK=true
    return 0
}

# 整合清理函数
trap_service_cleanup() {
    # 保存原始退出码
    local exit_code=$?
    
    # 执行服务清理
    stop_service
    local stop_result=$?
    
    # 执行清理确认
    service_cleanup_check
    local check_result=$?
    
    # 记录清理结果
    if [ $stop_result -eq 0 ] && [ $check_result -eq 0 ]; then
        echo -e "[SUCCESS] 服务完全清理"
    else
        echo -e "[ERROR] 服务清理不完整"
    fi
    
    # 返回原始退出码
    return $exit_code
}

#####################################################################################################


####################################### 退出函数部分 ##################################################
declare -g TRAP_LOCK=false
declare -a TRAP_FUNCTIONS=()

# 注册退出处理函数
# 用法: register_cleanup_functions func1 func2 func3 ...
register_cleanup_functions() {
    # 清空已有函数列表
    TRAP_FUNCTIONS=()
    
    # 添加传入的函数
    for func in "$@"; do
        if declare -f "$func" > /dev/null; then
            TRAP_FUNCTIONS+=("$func")
        else
            echo "[WARNING] 函数 $func 未定义，跳过注册" >&2
        fi
    done
    
    # 注册统一的trap处理器
    trap '_execute_cleanup_functions' EXIT HUP INT TERM
}

# 内部函数：执行所有注册的清理函数
_execute_cleanup_functions() {
    # 防止重复进入
    if $TRAP_LOCK; then
        return
    fi
    TRAP_LOCK=true
    
    local exit_code=${1:-$?}  # 获取退出状态
    
    # 取消其他trap，避免递归
    trap - EXIT HUP INT TERM
    
    # 保存当前信号处理状态
    local old_traps
    old_traps=$(trap -p | grep -E 'INT|TERM' || true)
    
    # 设置忽略中断信号，防止清理函数被打断
    trap '' INT TERM
    
    echo "[INFO] 开始执行清理流程 (禁用中断, 强制清理资源) ..." >&2
    
    # 按注册顺序执行所有清理函数
    for func in "${TRAP_FUNCTIONS[@]}"; do
        echo "[INFO] 正在执行清理函数: $func" >&2
        # 在子shell中执行以防止单个函数失败影响整体流程
        ( $func ) || echo "[WARNING] 清理函数 $func 执行失败" >&2
    done
    
    # 恢复原始信号处理
    if [ -n "$old_traps" ]; then
        eval "$old_traps"
    else
        trap - INT TERM
    fi
    
    echo "[INFO] 清理流程完成" >&2
    exit $exit_code
}

################################################################################################