#!/bin/bash

#######################################################################################################


####################################### 常用 Git 操作 #########################################
# 函数: git_clone_with_retry
# 描述: 尝试多次克隆 Git 仓库，支持自动清理失败残留
# 参数:
#   $1: GIT_PATH - 仓库路径
#   $2: GIT_BRANCH - 克隆的分支
#   $3: REPO_NAME - 目标目录名称
#   $4: MAX_RETRIES - 最大重试次数（可选，默认3）
#   $5: RETRY_DELAY - 重试延迟（秒）（可选，默认5）
# 返回值:
#   0: 克隆成功
#   1: 克隆失败
git_clone_with_retry() {
    # 参数校验
    if [ $# -lt 3 ]; then
        echo "[ERROR] Usage: git_clone_with_retry GIT_PATH GIT_BRANCH REPO_NAME [MAX_RETRIES] [RETRY_DELAY] [ENV_VARS]"
        return 1
    fi

    # 解析参数
    local GIT_PATH="$1"
    local GIT_BRANCH="$2"
    local REPO_NAME="$3"
    local MAX_RETRIES="${4:-3}"     # 默认最大重试3次
    local RETRY_DELAY="${5:-5}"     # 默认延迟5秒
    local ENV_PREFIX="${6:-}"       # 环境变量前缀

    local branch_info=""

    if [ "$GIT_BRANCH" != "null" ]; then
        branch_info="-b $GIT_BRANCH"
    fi

    local clone_success=false

    # 检查必需命令是否存在
    if ! command -v git &> /dev/null; then
        echo "[Error] git command not found"
        return 1
    fi

    echo "Starting clone operation for $REPO_NAME (branch: $GIT_BRANCH)"

    # 重试循环
    for ((i=1; i<=MAX_RETRIES; i++)); do
        echo "Attempt $i/$MAX_RETRIES to clone repository..."

        # 尝试克隆
        local clone_command="$ENV_PREFIX git clone $branch_info $GIT_PATH $REPO_NAME"
        echo $clone_command
        if eval "$clone_command"; then
            clone_success=true
            echo "[SUCCESS] Clone succeeded on attempt $i"
            break
        else
            # 仅在不是最后一次尝试时处理错误
            if [ "$i" -lt "$MAX_RETRIES" ]; then
                echo "[WARNING] Clone failed. Retrying in ${RETRY_DELAY} seconds..."
                sleep "$RETRY_DELAY"

                # 安全清理：检查目录是否存在再删除
                if [ -d "$REPO_NAME" ]; then
                    echo "Cleaning residual files..."
                    rm -rf "$REPO_NAME"
                fi
            fi
        fi
    done

    # 最终状态检查
    if ! $clone_success; then
        echo "[ERROR] Fatal: Failed to clone repository after ${MAX_RETRIES} attempts."
        return 1
    fi
    echo "[SUCCESS] success cloned!"
    return 0
}


# 函数：get_git_info
# 功能：获取指定Git仓库的origin URL和当前分支
# 参数：
#   $1 - 仓库路径（可选，默认为当前目录）
# 返回值：输出Origin URL和当前分支信息，或在错误时返回非零状态码
get_git_info() {
    local target_dir="${1:-.}"  # 使用传入参数或默认当前目录
    local origin_url
    local current_branch
    local git_logs

    # 定义颜色和格式代码
    local RED=$'\e[0;31m'
    local GREEN=$'\e[0;32m'
    local YELLOW=$'\e[1;33m'
    local BLUE=$'\e[0;34m'
    local MAGENTA=$'\e[0;35m'
    local CYAN=$'\e[0;36m'
    local BOLD=$'\e[1m'
    local UNDERLINE=$'\e[4m'
    local RESET=$'\e[0m'  # 重置所有格式

    # 所有颜色代码列表
    local color_codes=(
        "$RED"
        "$GREEN"
        "$YELLOW"
        "$BLUE"
        "$MAGENTA"
        "$CYAN"
        "$BOLD"
        "$UNDERLINE"
        "$RESET"
    )

    # 检查路径是否存在且为目录
    if [ ! -d "$target_dir" ]; then
        echo -e "${RED}错误：路径 '$target_dir' 不存在或不是目录。${RESET}" >&2
        return 1
    fi

    # 检查是否为Git仓库
    if ! git -C "$target_dir" rev-parse --git-dir > /dev/null 2>&1; then
        echo -e "${RED}错误：路径 '$target_dir' 不是一个Git仓库。${RESET}" >&2
        return 1
    fi

    # 获取origin远程URL（如果存在）
    origin_url=$(git -C "$target_dir" remote get-url origin 2>/dev/null)
    if [ $? -ne 0 ]; then
        origin_url="${YELLOW}(未设置origin远程)${RESET}"
    else
        origin_url="${GREEN}$origin_url${RESET}"
    fi

    # 获取当前分支名称
    current_branch=$(git -C "$target_dir" symbolic-ref --short HEAD 2>/dev/null)
    if [ $? -ne 0 ]; then
        # 如果不在分支上（如分离HEAD状态），尝试其他方法
        current_branch=$(git -C "$target_dir" rev-parse --short HEAD 2>/dev/null)
        if [ $? -eq 0 ]; then
            current_branch="${YELLOW}detached-HEAD ($current_branch)${RESET}"
        else
            current_branch="${RED}(无法获取分支信息)${RESET}"
        fi
    else
        current_branch="${GREEN}$current_branch${RESET}"
    fi

    # 获取最近三条提交日志
    git_logs=$(git -C "$target_dir" log --oneline --color=always -3 2>/dev/null)
    if [ $? -ne 0 ]; then
        git_logs="${YELLOW}(无法获取提交日志)${RESET}"
    fi


    # ！！！以下显示打屏部分不使用 echo -e 的原因 ！！！
    # commit中可能包含(\n)等需要避免转义的特殊字符，
    # 而printf不会自动解释反斜杠转义序列，因此内容中的字面量特殊字符（如"\\n"）会被输出为两个字符，而不是换行符。
    # 颜色代码（如\e[0;31m）仍然会被终端正确解释，因为终端会处理输出字符串中的ANSI序列。

    # 获取终端宽度
    local terminal_width=$(tput cols)

    # 更精确的ANSI代码去除函数
    strip_ansi_codes() {
        printf '%s' "$1" | sed -r "s/\x1B\[[0-9;]*[mK]//g"
    }

    # 计算可见字符宽度（考虑多字节字符）
    visible_width() {
        local text=$(strip_ansi_codes "$1")
        # 使用Python计算字符串显示宽度（更准确处理中文等宽字符）
        python3 -c "
import sys
from wcwidth import wcwidth
text = sys.argv[1]
width = 0
for char in text:
    width += max(0, wcwidth(char))
print(width)
" "$text"
    }

    # 格式化行函数
    format_line() {
        local content="$1"
        local visible_len=$(visible_width "$content")
        local max_len=$((terminal_width - 4))  # 边框占4个字符

        # 处理超长内容
        if [ $visible_len -gt $max_len ]; then
            local truncated_output=$(python3 -c "
import sys
from wcwidth import wcwidth

text = sys.argv[1]
max_width = int(sys.argv[2])
red_code = sys.argv[3]
reset_code = sys.argv[4]

# 目标宽度：保留3个字符用于省略号
target_width = max_width - 3
current_width = 0
truncated_chars = []

# 遍历每个字符
for char in text:
    char_width = max(0, wcwidth(char))
    if current_width + char_width > target_width:
        break
    truncated_chars.append(char)
    current_width += char_width

# 构建截断后的字符串
truncated_str = ''.join(truncated_chars)
new_content = truncated_str + red_code + '...' + reset_code

# 计算新内容的可见宽度（省略号宽度假设为3）
new_visible_width = current_width + 3

# 输出格式：内容|宽度
print(f'{new_content}|{new_visible_width}')
" "$content" "$max_len" "$RED" "$RESET")

            # 从Python输出中提取内容和宽度
            content=$(echo "$truncated_output" | cut -d'|' -f1)
            visible_len=$(visible_width "$content")
        fi

        # 计算填充
        local padding_len=$((max_len - visible_len))
        if [ $padding_len -lt 0 ]; then
            padding_len=0
        fi

        local padding=$(printf '%*s' $padding_len '')

        # 使用多次调用printf进行输出，避免解释转义序列：
        #    如果$content或$padding包含%字符，它可能会被解释为格式说明符
        #    为了避免这一点，需要使用%b格式说明符（它解释反斜杠转义，但不会解释%为格式符），或者确保变量被正确转义
        #    但%b会解释反斜杠转义（如\n），而此处需要避免转义特殊字符
        printf "%s" "${BOLD}${CYAN}"
        printf "║ "
        printf "%s" "${RESET}"
        printf "%s" "$content"
        printf "%s" "$padding"
        printf "%s" "${BOLD}${CYAN}"
        printf " ║"
        printf "%s\n" "${RESET}"
    }

    empty_line() {
        local padding=$(printf '%*s' $((terminal_width - 4)) '')
        printf "%s" "${BOLD}${CYAN}"
        printf "║"
        printf "%s" "${RESET}"
        printf " %s " "$padding"  # padding是空格字符串
        printf "%s" "${BOLD}${CYAN}"
        printf "║"
        printf "%s\n" "${RESET}"
    }
    local top_border="${BOLD}${CYAN}╔$(printf '═%.0s' $(seq 1 $((terminal_width - 2))))╗${RESET}"
    local bottom_border="${BOLD}${CYAN}╚$(printf '═%.0s' $(seq 1 $((terminal_width - 2))))╝${RESET}"

    # 输出结果
    printf "%s\n" "$top_border"
    format_line "${BOLD}Git 仓库信息${RESET} ${BLUE}➤${RESET} ${UNDERLINE}$target_dir${RESET}"
    empty_line
    format_line "${BOLD}${MAGENTA}●${RESET} ${BOLD}Origin URL:${RESET}  $origin_url"
    format_line "${BOLD}${MAGENTA}●${RESET} ${BOLD}当前分支:${RESET}    $current_branch"
    empty_line
    format_line "${BOLD}${MAGENTA}●${RESET} ${BOLD}最近三条提交:${RESET}"

    # 处理多行日志输出
    local log_count=0
    while IFS= read -r line && [ $log_count -lt 3 ]; do
        if [ -n "$line" ]; then
            format_line "   ${BLUE}➤${RESET} $line"
            log_count=$((log_count + 1))
        fi
    done <<< "$git_logs"

    # 如果日志少于3条，填充空行
    while [ $log_count -lt 3 ]; do
        empty_line
        log_count=$((log_count + 1))
    done

    empty_line
    printf "%s\n" "$bottom_border"
}


get_git_repo_name() {
    local input_path="${1:-.}"  # 获取输入路径，默认为当前目录
    local target_dir            # 存储目标目录路径

    # 检查路径是否存在
    if [ ! -e "$input_path" ]; then
        echo "Error: The path '$input_path' does not exist." >&2
        return 1
    fi

    # 确定目标目录
    if [ -d "$input_path" ]; then
        # 如果是目录，直接使用
        target_dir="$input_path"
    else
        # 如果是文件，获取其所在目录
        target_dir=$(dirname "$input_path")
    fi

    # 获取Git根目录
    local git_root
    git_root=$(git -C "$target_dir" rev-parse --show-toplevel 2>/dev/null)

    # 检查Git命令是否成功
    if [ $? -ne 0 ]; then
        echo "Error: The path '$input_path' is not within a git repository." >&2
        return 1
    fi

    # 提取并返回仓库名称
    basename "$git_root"
}


#######################################################################################################


###################################### Git 更新检查模块相关部分 #########################################
# 配置要比较的远程仓库和分支
COMPARE_REMOTE=${COMPARE_REMOTE:-"origin"}  # 默认使用origin
COMPARE_BRANCH=${COMPARE_BRANCH:-"master"}  # 默认使用master分支

# 全局变量存储需要清理的临时远程仓库
declare -a TEMP_REMOTES_TO_CLEAN=()

# 函数：检查 Git 项目是否最新
# 参数：
#   $1 - 脚本所在目录（可选，默认为当前脚本的目录）
#   $2 - 远程仓库名称或URL（SSH，可选，默认为 COMPARE_REMOTE）
#   $3 - 分支名称（可选，默认为 COMPARE_BRANCH）
#   $4 - 备选HTTPS远程仓库URL（可选，用于SSH失败时回退）
check_git_updates() {
    local SCRIPT_DIR="${1:-$(realpath $(dirname ${BASH_SOURCE[0]}))}"
    local REMOTE="${2:-$COMPARE_REMOTE}"
    local BRANCH="${3:-$COMPARE_BRANCH}"
    local HTTPS_REMOTE="${4:-}"  # 新增第四个参数：备选HTTPS地址，可选

    local GIT_NAME=$(get_git_repo_name "$SCRIPT_DIR")

    echo -e "[INFO] 检查当前git项目(${GIT_NAME})是否与${REMOTE}/${BRANCH}分支同步..."

    if git -C "$SCRIPT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        # 当前目录是一个Git仓库
        echo -e "[INFO] 当前目录是一个git仓库(${GIT_NAME})，检查更新..."

        local LOCAL_COMMIT
        local REMOTE_COMMIT

        # 检查REMOTE是否是一个已配置的远程名称
        if git -C "$SCRIPT_DIR" remote get-url "$REMOTE" >/dev/null 2>&1; then
            # REMOTE是一个已配置的远程名称
            echo -e "[INFO] 使用已配置的远程仓库: $REMOTE"
            if git -C "$SCRIPT_DIR" fetch "$REMOTE" "$BRANCH"; then
                # 获取本地分支和远程分支的commit hash
                LOCAL_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse HEAD)
                REMOTE_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse "$REMOTE/$BRANCH")
            else
                echo -e "[ERROR] 无法获取远程更新，请检查网络连接或git远程配置。"
                return 1
            fi
        else
            # REMOTE是一个URL，需要特殊处理
            echo -e "[INFO] 使用URL作为远程仓库: $REMOTE"

            # 创建一个临时远程名称
            local TEMP_REMOTE="temp_compare_remote_$(date +%s)"

            # 添加到需要清理的列表
            TEMP_REMOTES_TO_CLEAN+=("$TEMP_REMOTE")

            # 添加临时远程
            if git -C "$SCRIPT_DIR" remote add "$TEMP_REMOTE" "$REMOTE"; then
                echo -e "[INFO] 添加临时远程仓库：$TEMP_REMOTE"
                # 获取远程信息
                if git -C "$SCRIPT_DIR" fetch "$TEMP_REMOTE" "$BRANCH"; then
                    # 获取本地分支和远程分支的commit hash
                    LOCAL_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse HEAD)
                    REMOTE_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse "$TEMP_REMOTE/$BRANCH")

                    # 清理临时远程（正常流程）
                    git -C "$SCRIPT_DIR" remote remove "$TEMP_REMOTE"

                    # 从清理列表中移除
                    TEMP_REMOTES_TO_CLEAN=("${TEMP_REMOTES_TO_CLEAN[@]/$TEMP_REMOTE}")
                else
                    echo -e "[WARNING] SSH fetch失败，尝试HTTPS（如果可用）。"
                    if [ -n "$HTTPS_REMOTE" ]; then
                        local TEMP_HTTPS_REMOTE="temp_https_remote_$(date +%s)"
                        TEMP_REMOTES_TO_CLEAN+=("$TEMP_HTTPS_REMOTE")
                        if git -C "$SCRIPT_DIR" remote add "$TEMP_HTTPS_REMOTE" "$HTTPS_REMOTE"; then
                            echo -e "[INFO] 添加临时HTTPS远程仓库：$TEMP_HTTPS_REMOTE"
                            if git -C "$SCRIPT_DIR" fetch "$TEMP_HTTPS_REMOTE" "$BRANCH"; then
                                LOCAL_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse HEAD)
                                REMOTE_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse "$TEMP_HTTPS_REMOTE/$BRANCH")
                                git -C "$SCRIPT_DIR" remote remove "$TEMP_HTTPS_REMOTE"
                                TEMP_REMOTES_TO_CLEAN=("${TEMP_REMOTES_TO_CLEAN[@]/$TEMP_HTTPS_REMOTE}")
                            else
                                echo -e "[ERROR] HTTPS fetch也失败，请检查网络或URL。"
                                return 1
                            fi
                        else
                            echo -e "[ERROR] 无法添加临时HTTPS远程仓库。"
                            return 1
                        fi
                    else
                        echo -e "[ERROR] SSH fetch失败且未提供HTTPS备选地址。"
                        return 1
                    fi
                fi
            else
                echo -e "[ERROR] 无法添加临时远程仓库。"
                return 1
            fi
        fi

        if ! git -C "$SCRIPT_DIR" merge-base --is-ancestor "$REMOTE_COMMIT" "$LOCAL_COMMIT"; then
            echo -e "[WARNING] 当前git项目(${GIT_NAME})的commit历史中不包含远程最新的commit。本地commit: $LOCAL_COMMIT, 远程commit: $REMOTE_COMMIT"
            echo -e "[WARNING] 建议运行 'git pull $REMOTE $BRANCH' 更新项目后再执行脚本。"

            # 询问用户是否立即更新，直到得到有效输入或超时
            local start_time=$(date +%s)
            local valid_input=false

            while [ $(($(date +%s) - start_time)) -lt 30 ] && [ "$valid_input" = false ]; do
                read -t $((30 - ($(date +%s) - start_time))) -p "是否立即更新项目？([No]/yes): " answer

                # 处理空输入（直接回车或超时）
                if [ -z "$answer" ]; then
                    echo -e "[INFO] 用户选择默认值（不更新），继续执行脚本。"
                    valid_input=true
                    continue
                fi

                case $(echo "$answer" | tr '[:upper:]' '[:lower:]') in
                    "yes")
                        echo -e "[INFO] 正在更新项目..."
                        # 根据REMOTE类型执行不同的更新命令
                        if git -C "$SCRIPT_DIR" remote get-url "$REMOTE" >/dev/null 2>&1; then
                            # REMOTE是一个已配置的远程名称
                            if git -C "$SCRIPT_DIR" pull "$REMOTE" "$BRANCH"; then
                                echo -e "[SUCCESS] 项目更新成功。"
                            else
                                echo -e "[ERROR] 项目更新失败，请手动处理。"
                                return 1
                            fi
                        else
                            # REMOTE是一个URL，需要特殊处理
                            local TEMP_REMOTE="temp_update_remote_$(date +%s)"

                            # 添加到需要清理的列表
                            TEMP_REMOTES_TO_CLEAN+=("$TEMP_REMOTE")

                            if git -C "$SCRIPT_DIR" remote add "$TEMP_REMOTE" "$REMOTE"; then
                                echo -e "[INFO] 添加临时远程仓库：$TEMP_REMOTE"
                                if git -C "$SCRIPT_DIR" pull "$TEMP_REMOTE" "$BRANCH"; then
                                    echo -e "[SUCCESS] 项目更新成功。"

                                    # 清理临时远程（正常流程）
                                    git -C "$SCRIPT_DIR" remote remove "$TEMP_REMOTE"

                                    # 从清理列表中移除
                                    TEMP_REMOTES_TO_CLEAN=("${TEMP_REMOTES_TO_CLEAN[@]/$TEMP_REMOTE}")
                                else
                                    echo -e "[WARNING] SSH pull失败，尝试HTTPS（如果可用）。"
                                    if [ -z "$HTTPS_REMOTE" ]; then
                                        local TEMP_HTTPS_REMOTE="temp_https_update_remote_$(date +%s)"
                                        TEMP_REMOTES_TO_CLEAN+=("$TEMP_HTTPS_REMOTE")
                                        if git -C "$SCRIPT_DIR" remote add "$TEMP_HTTPS_REMOTE" "$HTTPS_REMOTE"; then
                                            echo -e "[INFO] 添加临时HTTPS远程仓库：$TEMP_HTTPS_REMOTE"
                                            if git -C "$SCRIPT_DIR" pull "$TEMP_HTTPS_REMOTE" "$BRANCH"; then
                                                echo -e "[SUCCESS] 项目更新成功。"
                                                git -C "$SCRIPT_DIR" remote remove "$TEMP_HTTPS_REMOTE"
                                                TEMP_REMOTES_TO_CLEAN=("${TEMP_REMOTES_TO_CLEAN[@]/$TEMP_HTTPS_REMOTE}")
                                            else
                                                echo -e "[ERROR] HTTPS pull也失败，请检查网络或URL。"
                                                return 1
                                            fi
                                        else
                                            echo -e "[ERROR] 无法添加临时HTTPS远程仓库。"
                                            return 1
                                        fi
                                    else
                                        echo -e "[ERROR] SSH pull失败且未提供HTTPS备选地址。"
                                        return 1
                                    fi
                                fi
                            else
                                echo -e "[ERROR] 无法添加临时远程仓库。"
                                return 1
                            fi
                        fi
                        valid_input=true
                        ;;
                    "no")
                        echo -e "[INFO] 用户选择不更新，继续执行脚本。"
                        valid_input=true
                        ;;
                    *)
                        echo -e "[ERROR] 无效输入，请输入 'yes' 或 'no'。"
                        echo -e "[INFO] 剩余时间: $((30 - ($(date +%s) - start_time)))秒"
                        ;;
                esac
            done

            # 检查是否超时
            if [ "$valid_input" = false ]; then
                echo -e "[INFO] 输入超时，默认不更新项目，继续执行脚本。"
            fi
        else
            echo -e "[SUCCESS] 当前git项目(${GIT_NAME})已是最新版本。"
        fi
    else
        echo -e "[WARNING] 当前目录($SCRIPT_DIR)不是一个git仓库，跳过版本检查。"
    fi

    return 0
}


# 注册退出时的清理钩子：清理临时远程仓库
trap_git_remote_cleanup() {
    # 保存原始退出码
    local exit_code=$?

    # 如果没有需要清理的远程仓库，直接返回
    if [ ${#TEMP_REMOTES_TO_CLEAN[@]} -eq 0 ]; then
        echo -e "[INFO] 没有需要清理的临时远程仓库"
        return $exit_code
    fi

    echo -e "[INFO] 开始清理临时远程仓库..."

    # 获取脚本目录（假设与上次调用 check_git_updates 相同）
    local script_dir="${1:-$(realpath $(dirname ${BASH_SOURCE[0]}))}"

    # 遍历所有需要清理的临时远程仓库
    local any_failed=0
    for temp_remote in "${TEMP_REMOTES_TO_CLEAN[@]}"; do
        # 直接尝试删除远程仓库
        if git -C "$script_dir" remote remove "$temp_remote" 2>/dev/null; then
            echo -e "[SUCCESS] 已删除临时远程仓库: $temp_remote"
        else
            echo -e "[INFO] 临时远程仓库 $temp_remote 不存在或已被删除"
        fi
    done

    # 清空数组
    TEMP_REMOTES_TO_CLEAN=()

    echo -e "[INFO] 临时远程仓库清理完成"

    # 返回原始退出码
    return $exit_code
}

#########################################################################################################