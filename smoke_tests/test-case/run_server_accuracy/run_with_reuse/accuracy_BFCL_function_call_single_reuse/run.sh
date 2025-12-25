#!/bin/bash
declare -i ret_ok=0
declare -i ret_failed=1

# 每个case通用变量
CUR_DIR=$(dirname $(readlink -f $0))
CASE_NAME=$(basename "$CUR_DIR")
LAST_3_DIRNAME=$(echo  $CUR_DIR | rev | cut -d'/' -f1-3 | rev)
CASE_OUTPUT_PATH=${PROJECT_OUTPUT_PATH}/${LAST_3_DIRNAME} # 项目根路径
AIS_BENCH_CODE_CONFIGS_DIR=${PROJECT_PATH}/../ais_bench/benchmark/configs
CONFIG_DATASET_NAME="BFCL"
OUTPUT_DATASET_NAME="BFCL-v3-simple"
CURR_API="vllm-api-function-call-chat"

SLEEP_BEFORE_INTERRUPT=10

# 清理用例输出
if [ ! -d ${CASE_OUTPUT_PATH} ];then
    mkdir -p ${CASE_OUTPUT_PATH}
fi
rm -rf ${CASE_OUTPUT_PATH}/*

# 复制配置文件
TARGET_DIR="${AIS_BENCH_CODE_CONFIGS_DIR}/datasets/${CONFIG_DATASET_NAME}"
echo "Copying config files to ${TARGET_DIR}..."
cp -r ${CUR_DIR}/ais_bench_configs/* ${AIS_BENCH_CODE_CONFIGS_DIR}/

# 添加外部传入的信息给模型配置文件(具体模型配置文件路径请依据实际情况修改, 模型或数据集配置文件统一命名成CASE_NAME)
{
    echo ""
    echo "models[0]['host_ip'] = '${AISBENCH_SMOKE_SERVICE_IP}'"
    echo "models[0]['host_port'] = ${AISBENCH_SMOKE_SERVICE_PORT}"
    echo "models[0]['path'] = '${AISBENCH_SMOKE_MODEL_PATH}'"
} >> "${AIS_BENCH_CODE_CONFIGS_DIR}/models/vllm_api/${CASE_NAME}.py"

# 启动用例
echo -e "\033[1;32m[1/1]\033[0m Test case - ${CASE_NAME}"

# 执行命令
set -o pipefail  # 启用管道整体失败检测
# 启动 ais_bench 进程，并将其输出保存到日志文件
(
  setsid ais_bench --models ${CASE_NAME} \
         --datasets ${CASE_NAME} \
         --work-dir ${CASE_OUTPUT_PATH}  >${CUR_DIR}/tmplog.txt  2>&1
) &

# 获取后台进程的 PID
PID=$!
PGID=$(ps -o pgid= -p $PID | tr -d ' ')

if [ -z "$PGID" ]; then
    echo "Error: PGID not found. Process $PID may have already exited or does not exist."
    exit ret_failed
else
    echo "PGID retrieved: $PGID"
fi

source $CUR_DIR/../utils.sh
WORK_DIR_INFO=$(wait_for_timestamp_dir "$CASE_OUTPUT_PATH" 240) || exit 1
TIMESTAMP="${WORK_DIR_INFO##*/}"
CURR_OUTPUT_PATH=${CASE_OUTPUT_PATH}/${TIMESTAMP}
echo "捕获到目录: $CURR_OUTPUT_PATH"
LOG_INFER_OUTPUT_PATH=${CURR_OUTPUT_PATH}/logs/infer/${CURR_API}/${OUTPUT_DATASET_NAME}.out
TMP_DATA_OUTPUT_PATH=${CURR_OUTPUT_PATH}/predictions/${CURR_API}/tmp
# 添加变量用于判断是否已经进入主轮询
entered_polling=false

# 等待日志创建
wait_time=0
while [ $wait_time -lt 30 ]; do
    if [ -f "$LOG_INFER_OUTPUT_PATH" ]; then
        break
    fi
    sleep 1
    wait_time=$((wait_time + 1))
done

# 如果 30 秒内日志文件仍未创建，退出
if [ ! -f "$LOG_INFER_OUTPUT_PATH" ]; then
    echo "Log file creat time out"
    exit $ret_failed
fi

poll_received_and_signal "$TMP_DATA_OUTPUT_PATH" "$PGID"  INT 5 300

# 等待 ais_bench 进程退出，ctrl + C 退出码为130
wait $PID
if [ $? -eq 0 ]
then
    echo "Run $CASE_NAME ctrl C interruption: Success"
else
    echo "Run $CASE_NAME ctrl C interruption: Failed"
    exit $ret_failed
fi

# 清理残余results和summary
CURR_OUTPUT_PATH=${CASE_OUTPUT_PATH}/${TIMESTAMP}
rm -rf ${CURR_OUTPUT_PATH}/results
rm -rf ${CURR_OUTPUT_PATH}/summary

(
  setsid ais_bench --models ${CASE_NAME} \
         --datasets ${CASE_NAME} \
         --work-dir ${CASE_OUTPUT_PATH} --reuse  >> ${CUR_DIR}/tmplog.txt  2>&1
) &
PID=$!
wait $PID
if [ $? -eq 0 ]
then
    if grep -qE "Found [0-9]+ completed data in cache, run infer task from the last interrupted position" $LOG_INFER_OUTPUT_PATH; then
        echo "Run $CASE_NAME test: Success"
    else
        echo "No tmp data found! Run $CASE_NAME test: Failed"
        exit $ret_failed
    fi
else
    echo "Run $CASE_NAME test: Failed"
    exit $ret_failed
fi


# 验证产物
CURR_OUTPUT_PATH=${CASE_OUTPUT_PATH}/${TIMESTAMP}
LOG_EVAL_OUTPUT_PATH=${CURR_OUTPUT_PATH}/logs/eval/${CURR_API}/${OUTPUT_DATASET_NAME}.out
LOG_INFER_OUTPUT_PATH=${CURR_OUTPUT_PATH}/logs/infer/${CURR_API}/${OUTPUT_DATASET_NAME}.out
PREDICTIONS_OUTPUT_PATH=${CURR_OUTPUT_PATH}/predictions/${CURR_API}/${OUTPUT_DATASET_NAME}.jsonl
RESULTS_OUTPUT_PATH=${CURR_OUTPUT_PATH}/results/${CURR_API}/${OUTPUT_DATASET_NAME}.json
SUMMARY_OUTPUT_COUNT=$(find "${CURR_OUTPUT_PATH}/summary" -maxdepth 1 -type f -name 'summary_*.csv' | wc -l)

if [ ! -f $LOG_EVAL_OUTPUT_PATH ];then
    echo "Can't find $LOG_EVAL_OUTPUT_PATH"
    exit $ret_failed
elif [ ! -f $LOG_INFER_OUTPUT_PATH ];then
    echo "Can't find $LOG_INFER_OUTPUT_PATH"
    exit $ret_failed
elif [ ! -f $PREDICTIONS_OUTPUT_PATH ];then
    echo "Can't find $PREDICTIONS_OUTPUT_PATH"
    exit $ret_failed
elif [ ! -f $RESULTS_OUTPUT_PATH ];then
    echo "Can't find $RESULTS_OUTPUT_PATH"
    exit $ret_failed
elif [ ! "1" = "$SUMMARY_OUTPUT_COUNT" ]; then
    echo "Can't find SUMMARY_OUTPUT"
    exit $ret_failed
fi

exit $ret_ok
