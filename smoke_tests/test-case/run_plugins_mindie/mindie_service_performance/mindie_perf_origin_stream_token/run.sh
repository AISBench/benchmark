#!/bin/bash
declare -i ret_ok=0
declare -i ret_failed=1

# 每个case通用变量
CUR_DIR=$(dirname $(readlink -f $0))
CASE_NAME=$(basename "$CUR_DIR")
LAST_3_DIRNAME=$(echo  $CUR_DIR | rev | cut -d'/' -f1-3 | rev)
CASE_OUTPUT_PATH=${PROJECT_OUTPUT_PATH}/${LAST_3_DIRNAME} # 项目根路径
PLUGINS_CODE_CONFIGS_DIR=${PROJECT_RESOURCE_PATH}/code/benchmark-mindie/mindie_service_examples
DEFAULT_MODEL_NAME="qwen"
OUTPUT_DATASET_NAME="synthetic"
CURR_API="mindie-stream-token"
CONFIG_SCRIPT_NAME="mindie_infer_origin_stream_token.py"

# 清理用例输出
if [ ! -d ${CASE_OUTPUT_PATH} ];then
    mkdir -p ${CASE_OUTPUT_PATH}
fi
rm -rf ${CASE_OUTPUT_PATH}/*

# 复制配置文件
TARGET_CONFIG_FILE=${PLUGINS_CODE_CONFIGS_DIR}/${CONFIG_SCRIPT_NAME}
echo "Copying custom config files from ${TARGET_CONFIG_FILE}..."
cp ${TARGET_CONFIG_FILE} ${CUR_DIR}

# 添加外部传入的信息给模型配置文件(具体模型配置文件路径请依据实际情况修改, 模型或数据集配置文件统一命名成CASE_NAME)
{
    echo ""
    echo "models[0]['host_ip'] = '${AISBENCH_SMOKE_SERVICE_IP}'"
    echo "models[0]['host_port'] = ${AISBENCH_SMOKE_SERVICE_PORT}"
    echo "models[0]['path'] = '${AISBENCH_SMOKE_MODEL_PATH}'"
    echo "work_dir = '${CASE_OUTPUT_PATH}'"
} >> "${CUR_DIR}/${CONFIG_SCRIPT_NAME}"

# 启动用例
echo -e "\033[1;32m[1/1]\033[0m Test case - ${CASE_NAME}"

# 执行命令
set -o pipefail  # 启用管道整体失败检测
ais_bench ${CUR_DIR}/${CONFIG_SCRIPT_NAME} --mode perf 2>&1 | tee ${CUR_DIR}/tmplog.txt
if [ $? -eq 0 ]
then
    echo "Run $CASE_NAME test: Success"
else
    echo "Run $CASE_NAME test: Failed"
    exit $ret_failed
fi

# 验证日志内容中性能结果
ret=$ret_ok
req_metric_list=("E2EL" "TTFT" "TPOT" "ITL" "LastITL" "MaxITL" "InputTokens" "OutputTokens" \
    "OutputTokenThroughput" "GeneratedCharacters")
comm_metric_list=("Benchmark Duration" "Total Requests" "Failed Requests" "Success Requests" "Concurrency" \
    "Max Concurrency" "Request Throughput" "Total Input Tokens" "Prefill TokenThroughput" "Totalgeneratedtokens" \
    "Input Token Throughput" "Output TokenThroughput" "Total Token Throughput" "lpct")
for metric in ${req_metric_list[@]}
do
    grep_ret=$(cat ${CUR_DIR}/tmplog.txt | grep "$metric")
    if [ "${grep_ret}" == "" ];then
        echo "$CASE_NAME test: Check performance table failed, can't find req metric '$metric'"
        ret=$ret_failed
    fi
done
for metric in ${comm_metric_list[@]}
do
    grep_ret=$(cat ${CUR_DIR}/tmplog.txt | grep "$metric")
    if [ "${grep_ret}" == "" ];then
        echo "$CASE_NAME test: Check performance table failed, can't find comm metric '$metric'"
        ret=$ret_failed
    fi
done
if [ $ret != $ret_ok];then
    exit $ret
fi


# 获取时间戳
WORK_DIR_INFO=$(cat ${CUR_DIR}/tmplog.txt | grep 'Current exp folder: ')
TIMESTAMP="${WORK_DIR_INFO##*/}"

# 验证产物
CURR_OUTPUT_PATH=${CASE_OUTPUT_PATH}/${TIMESTAMP}
LOG_OUTPUT_PATH=${CURR_OUTPUT_PATH}/logs/infer/${CURR_API}/${OUTPUT_DATASET_NAME}.out
PERFORMANCES_JSON_OUTPUT_PATH=${CURR_OUTPUT_PATH}/performances/${CURR_API}/${OUTPUT_DATASET_NAME}.json
PERFORMANCES_CSV_OUTPUT_PATH=${CURR_OUTPUT_PATH}/performances/${CURR_API}/${OUTPUT_DATASET_NAME}.csv
PERFORMANCES_HTML_OUTPUT_PATH=${CURR_OUTPUT_PATH}/performances/${CURR_API}/${OUTPUT_DATASET_NAME}_plot.html


if [ ! -f $LOG_OUTPUT_PATH ];then
    echo "Can't find $LOG_OUTPUT_PATH"
    exit $ret_failed
elif [ ! -f $PERFORMANCES_JSON_OUTPUT_PATH ];then
    echo "Can't find $PERFORMANCES_JSON_OUTPUT_PATH"
    exit $ret_failed
elif [ ! -f $PERFORMANCES_CSV_OUTPUT_PATH ];then
    echo "Can't find $PERFORMANCES_CSV_OUTPUT_PATH"
    exit $ret_failed
elif [ ! -f $PERFORMANCES_HTML_OUTPUT_PATH ];then
    echo "Can't find $PERFORMANCES_HTML_OUTPUT_PATH"
    exit $ret_failed
fi


exit $ret_ok
