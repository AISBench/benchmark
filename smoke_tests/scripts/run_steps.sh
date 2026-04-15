#!/bin/bash
umask 0022
declare -i ret_ok=0
declare -i ret_failed=1

# 此AISBench_Smoke当前的中心仓的git地址和分支
COMPARE_REMOTE="https://gh-proxy.test.osinfra.cn/https://github.com/AISBench/benchmark.git"  # 按需进行更改，设置为空时则默认使用 origin
COMPARE_REMOTE_HTTPS="https://github.com/AISBench/benchmark.git"  # 备选HTTPS地址
COMPARE_REMOTE_BRANCH="master"  # 可选，默认使用scripts/git_utils.sh:COMPARE_BRANCH (当前为master)

# 常量路径设置
PROJECT_PATH=$(realpath `dirname $0`/..)
SCRIPTS_PATH=$(realpath `dirname $0`)
export PROJECT_WORKSPACE_PATH=$PROJECT_PATH/RunWorkspace/`date "+%Y-%m-%d_%H-%M-%S"`

MODEL_PATH=$PROJECT_PATH/resource/model
CREDENTIALS_PATH=$PROJECT_PATH/resource/credentials

# 引入其它模块
source $SCRIPTS_PATH/utils_funcs.sh
source $SCRIPTS_PATH/md5_funcs.sh
source $SCRIPTS_PATH/log_utils.sh
source $SCRIPTS_PATH/git_utils.sh

# 数据集路径设置
DATASETS_PATH=$AISBENCH_DATASETS_PATH
if [ -z "$AISBENCH_DATASETS_PATH" ]; then
    DATASETS_PATH=$HOME/smoke_datasets
fi

validate_path "$DATASETS_PATH" "数据集路径" || exit $ret_failed

# 注册程序退出时的清理函数（按执行顺序）
register_cleanup_functions trap_service_cleanup trap_git_remote_cleanup trap_log_cleanup

# --- 初始化日志系统 ---
init_logging $PROJECT_WORKSPACE_PATH # 创建日志目录

# --- 检查 AISBench_Smoke 项目是否最新 ---
check_git_updates "$SCRIPTS_PATH" "$COMPARE_REMOTE" "$COMPARE_REMOTE_BRANCH" "$COMPARE_REMOTE_HTTPS" || exit $ret_failed

# 初始化参数数组和clone标志
declare -a filtered_args=()
clone_flag=1

# 软链接参数的状态标记和路径变量
in_link_model=0
in_link_repo=0

LINK_MODEL_PATH=""
LINK_REPO_PATH=""


if [ $# -eq 0 ]; then
    echo -e "[INFO] 没有提供任何参数"
    help_func
fi

# 入参配置
SELF_REPO=
SELF_BRANCH="master"  # 默认分支
in_self=0              # 状态标志：0-未处理 -self, 1-等待URL, 2-等待分支
# 遍历所有参数进行过滤
for arg in "$@"; do
    if [ "$arg" = "-h" ]; then
        help_func
        exit $ret_ok
    fi

    # 处理软链接参数
    if [ "$arg" = "-l-model" ]; then
        in_link_model=1
        continue
    elif [ "$arg" = "-l-repo" ]; then
        in_link_repo=1
        continue
    fi

    # 捕获软链接参数后的路径值
    if [ $in_link_model -eq 1 ]; then
        LINK_MODEL_PATH="$arg"
        validate_path "$LINK_MODEL_PATH" "模型软链接路径" || exit $ret_failed
        in_link_model=0
        continue
    elif [ $in_link_repo -eq 1 ]; then
        LINK_REPO_PATH="$arg"
        validate_path "$LINK_REPO_PATH" "仓库软链接路径" || exit $ret_failed
        in_link_repo=0
        continue
    fi

    if [ "$arg" = "-nc" ]; then
        clone_flag=0
    else
        # -self 参数的处理逻辑
        if [ "$arg" = "-self" ]; then
            in_self=1  # 标记下一个参数是URL
        elif [ $in_self -eq 1 ]; then
            SELF_REPO="$arg"
            in_self=2  # 标记下一个参数可能是分支
        elif [ $in_self -eq 2 ] && [[ "$arg" != -* ]]; then
            # 如果当前参数不是新选项（-开头），则视为分支名
            SELF_BRANCH="$arg"
            in_self=0
        else
            filtered_args+=("$arg")
        fi
    fi

    # 重置状态（确保不会影响后续非-self参数）
    if [ $in_self -eq 2 ] && [[ "$arg" == -* ]]; then
        in_self=0
    fi
done

# 错误检查：检测到-self但没有提供URL
if [ $in_self -eq 1 ]; then
    echo -e "Error: -self requires a repository URL"
    exit $ret_failed
fi

pip3 install -r $PROJECT_PATH/requirements.txt

# install benchmark
REPO_PATH=${PROJECT_PATH}/../
cd ${REPO_PATH}
pip3 install -e ./ -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --use-pep517
pip3 install -r requirements/extra.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
pip3 install -r requirements/datasets/bfcl_dependencies.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --no-deps

## 数据集放置
target_directory="$REPO_PATH/ais_bench/datasets"

# 验证target_directory路径
validate_path "$target_directory" "数据集路径" || exit $ret_failed

# 创建软链接内容
create_symlinks_rec "$DATASETS_PATH" "$target_directory" || exit $ret_failed
echo -e "[SUCCESS] 软链接创建完成"

tree -C $target_directory -L 1

# 模型词表仓
GIT_PATH=https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git
GIT_BRANCH="null"
REPO_NAME=Qwen2.5-7B-Instruct
ENV_VARS_BEFORE_CLONE="GIT_LFS_SKIP_SMUDGE=1"

ensure_and_enter_dir $MODEL_PATH || exit $ret_failed

if [ $clone_flag -eq 1 ]; then
    # 如果指定了模型软链接
    if [ -n "$LINK_MODEL_PATH" ]; then
        echo -e "[INFO] 使用模型软链接: $LINK_MODEL_PATH"

        # 创建软链接
        create_symlink "$LINK_REPO_PATH" "$MODEL_PATH/$REPO_NAME" || exit $ret_failed

        # 验证软链接路径
        validate_path "$MODEL_PATH/$REPO_NAME" "模型软链接路径" || exit $ret_failed

    else
        # 原始模型下载逻辑
        if [ ! -d "$MODEL_PATH/$REPO_NAME" ] && [ $clone_flag -eq 1 ]; then
            echo -e "模型仓库不存在，开始下载..."
            git_clone_with_retry $GIT_PATH $GIT_BRANCH $REPO_NAME $max_retries $retry_delay $ENV_VARS_BEFORE_CLONE || exit $ret_failed
        fi
    fi
fi

export AISBENCH_SMOKE_MODEL_PATH="$MODEL_PATH/$REPO_NAME"

# ====== 动态端口分配与管理 ======
cd ${REPO_PATH}/tools/infer_serve_simulator || exit $ret_failed
pip3 install -r requirements.txt

export AISBENCH_SMOKE_SERVICE_IP=$(hostname -I | awk '{print $1}')

# 确保配置文件存在
if [ ! -f "config.sh" ]; then
    echo -e "[WARNING] config.sh文件不存在"
    echo -e "创建默认config.sh"
    echo -e "#!/bin/bash" > config.sh
    echo -e "PROCESS_NUM=4" >> config.sh
    echo -e "IP=$AISBENCH_SMOKE_SERVICE_IP" >> config.sh
    echo -e "PORT=8080" >> config.sh
fi

# 更新配置文件中的IP值
if sed -i "s/^IP=.*/IP=$AISBENCH_SMOKE_SERVICE_IP/" config.sh; then
    echo -e "[SUCCESS] 更新config.sh ip成功"
else
    echo -e "[WARNING] 更新config.sh ip失败，手动回写..."
    awk -v new_ip="$AISBENCH_SMOKE_SERVICE_IP" '/^IP=/{print "IP=" new_ip; next}1' config.sh > temp && mv temp config.sh
    chmod +x config.sh
fi


# 获取当前配置端口
current_port=$(grep '^PORT=' config.sh | cut -d'=' -f2 | tr -d '"' 2>/dev/null)
if [ -z "$current_port" ]; then
    echo -e "[WARNING] 无法读取PORT配置，使用默认值"
    current_port=8080
fi

# 查找可用端口
NEW_PORT=$(find_available_port $current_port)
if [ $? -ne 0 ]; then
    echo $NEW_PORT  # 显示错误信息
    exit $ret_failed
fi

echo -e "使用端口：$NEW_PORT (原端口：$current_port)"

# 更新配置文件中的端口值
if sed -i "s/^PORT=.*/PORT=$NEW_PORT/" config.sh; then
    echo -e "[SUCCESS] 更新config.sh成功"
    current_port=$NEW_PORT
else
    echo -e "[WARNING] 更新config.sh失败，手动回写..."
    awk -v new_port="$NEW_PORT" '/^PORT=/{print "PORT=" new_port; next}1' config.sh > temp && mv temp config.sh
    chmod +x config.sh
fi

# ====== 虚拟服务化管理逻辑 ======
echo -e "Starting virtual service simulator..."
cd ${REPO_PATH}/tools/infer_serve_simulator
# 后台启动服务（记录PID）
nohup bash launch_service.sh &> launch_service.log &
VIRTUAL_SERVICE_PID=$!
echo -e "Virtual service PID: $VIRTUAL_SERVICE_PID"

export AISBENCH_SMOKE_SERVICE_PORT=$current_port

# 执行服务化启动检查
if ! check_service_ready; then
    echo -e "[ERROR] 服务启动失败："
    tail -n 20 launch_service.log  # 显示最后20行日志
    stop_service
    service_cleanup_check
    exit $ret_failed
fi

echo -e "[SUCCESS] 虚拟服务化已启动"
echo -e "[NOTE] 开始执行冒烟流程，冒烟日志：$PROJECT_WORKSPACE_PATH/run.log"

# 禁用日志捕获以解决进度条问题
disable_log_capture

# 执行主测试逻辑
cd $PROJECT_PATH
bash scripts/run_aisbench_smoke.sh "${filtered_args[@]}"
ret_value=$?  # 保存测试退出状态

# 重新启用日志捕获
enable_log_capture

# 打印当前验证的repo信息
get_git_info ${REPO_PATH}


# 显式处理服务化关闭和检查
# 主动清理服务（只在服务仍运行时）
if ps -p $VIRTUAL_SERVICE_PID >/dev/null; then
    stop_service
    service_cleanup_check
else
    echo "服务已提前退出，执行清理确认..."
    service_cleanup_check
fi

exit $ret_value
