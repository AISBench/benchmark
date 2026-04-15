#!/bin/bash

# 存储failed-cases的文件
PROJECT_PATH=$(realpath `dirname $0`/..)
WORKSPACE_PATH=$(find $PROJECT_PATH/RunWorkspace -maxdepth 1 -type d ! -name "Cron_Logs"  ! -name "RunWorkspace" -exec stat --format="%Y %n" {} \; | sort -nr | head -n 1 | awk '{print $2}')
FAILED_CASES_FILE_NAME="failed_cases.txt"
FAILED_CASES_FILE="${WORKSPACE_PATH}/${FAILED_CASES_FILE_NAME}"
# 检查失败用例文件是否存在

if [[ -f "$FAILED_CASES_FILE" ]]; then
    # 如果文件存在，读取其内容
    FAILED_CASES=$(cat "$FAILED_CASES_FILE")
else
    # 如果文件不存在，设置一个默认值或者输出警告信息
    echo "[WARNING] ${FAILED_CASES_FILE} file does NOT exist."
    FAILED_CASES=""  # 或者设置为其他默认值
fi

if [ -n "$FAILED_CASES" ]; then
    echo "Failed Cases will be re-executed: $FAILED_CASES"
    cd $PROJECT_PATH
    bash scripts/run_with_env_set.sh -t $FAILED_CASES $*
else
    echo "[INFO] There is not any failed case."
fi 