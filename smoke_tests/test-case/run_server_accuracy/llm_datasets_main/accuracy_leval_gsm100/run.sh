#!/bin/bash
declare -i ret_ok=0
declare -i ret_failed=1

# 每个case通用变量
CUR_DIR=$(dirname $(readlink -f $0))
CASE_NAME=$(basename "$CUR_DIR")
LAST_3_DIRNAME=$(echo  $CUR_DIR | rev | cut -d'/' -f1-3 | rev)
CASE_OUTPUT_PATH=${PROJECT_OUTPUT_PATH}/${LAST_3_DIRNAME} # 项目根路径
AIS_BENCH_CODE_CONFIGS_DIR=${PROJECT_PATH}/../ais_bench/benchmark/configs
CONFIG_DATASET_NAME="leval"
OUTPUT_DATASET_NAME="LEval_gsm100"
CURR_API="vllm-api-general-chat"

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
ais_bench --models ${CASE_NAME} --datasets ${CASE_NAME} --work-dir ${CASE_OUTPUT_PATH} --max-num-workers 3 2>&1 | tee ${CUR_DIR}/tmplog.txt
if [ $? -eq 0 ]
then
    echo "Run $CASE_NAME test: Success"
else
    echo "Run $CASE_NAME test: Failed"
    exit $ret_failed
fi
# 获取时间戳
WORK_DIR_INFO=$(cat ${CUR_DIR}/tmplog.txt | grep 'Current exp folder: ')
TIMESTAMP="${WORK_DIR_INFO##*/}"

# 验证产物
CURR_OUTPUT_PATH=${CASE_OUTPUT_PATH}/${TIMESTAMP}
LOG_EVAL_OUTPUT_PATH=${CURR_OUTPUT_PATH}/logs/eval/${CURR_API}/${OUTPUT_DATASET_NAME}.out
LOG_INFER_OUTPUT_PATH=${CURR_OUTPUT_PATH}/logs/infer/${CURR_API}/${OUTPUT_DATASET_NAME}.out
PREDICTIONS_OUTPUT_PATH=${CURR_OUTPUT_PATH}/predictions/${CURR_API}/${OUTPUT_DATASET_NAME}.jsonl
RESULTS_OUTPUT_PATH=${CURR_OUTPUT_PATH}/results/${CURR_API}/${OUTPUT_DATASET_NAME}.json
SUMMARY_OUTPUT_PATH=${CURR_OUTPUT_PATH}/summary/summary_${TIMESTAMP}.csv

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
elif [ ! -f $SUMMARY_OUTPUT_PATH ];then
    echo "Can't find $SUMMARY_OUTPUT_PATH"
    exit $ret_failed
fi

exit $ret_ok
