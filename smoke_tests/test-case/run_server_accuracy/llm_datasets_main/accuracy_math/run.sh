#!/bin/bash
declare -i ret_ok=0
declare -i ret_failed=1

CUR_DIR=$(dirname $(readlink -f $0))
CASE_NAME=$(basename "$CUR_DIR")
LAST_3_DIRNAME=$(echo  $CUR_DIR | rev | cut -d'/' -f1-3 | rev)
CASE_OUTPUT_PATH=${PROJECT_OUTPUT_PATH}/${LAST_3_DIRNAME}
AIS_BENCH_CODE_CONFIGS_DIR=${PROJECT_PATH}/../ais_bench/benchmark/configs
CONFIG_DATASET_NAME="math"
OUTPUT_DATASET_NAME="math_prm800k_500"
CURR_API="vllm-api-general-chat"

if [ ! -d ${CASE_OUTPUT_PATH} ];then
    mkdir -p ${CASE_OUTPUT_PATH}
fi
rm -rf ${CASE_OUTPUT_PATH}/*

echo "Copying config files..."
cp -r ${CUR_DIR}/ais_bench_configs/* ${AIS_BENCH_CODE_CONFIGS_DIR}/

{
    echo ""
    echo "models[0]['host_ip'] = '${AISBENCH_SMOKE_SERVICE_IP}'"
    echo "models[0]['host_port'] = ${AISBENCH_SMOKE_SERVICE_PORT}"
    echo "models[0]['path'] = '${AISBENCH_SMOKE_MODEL_PATH}'"
} >> "${AIS_BENCH_CODE_CONFIGS_DIR}/models/vllm_api/${CASE_NAME}.py"

echo -e "\033[1;32m[1/1]\033[0m Test case - ${CASE_NAME}"

set -o pipefail
ais_bench --models ${CASE_NAME} --datasets ${CASE_NAME} --work-dir ${CASE_OUTPUT_PATH} 2>&1 | tee ${CUR_DIR}/tmplog.txt
if [ $? -ne 0 ]
then
    echo "Run $CASE_NAME test: Failed"
    exit $ret_failed
fi
echo "Run $CASE_NAME test: Success"

WORK_DIR_INFO=$(cat ${CUR_DIR}/tmplog.txt | grep 'Current exp folder: ')
TIMESTAMP="${WORK_DIR_INFO##*/}"

CURR_OUTPUT_PATH=${CASE_OUTPUT_PATH}/${TIMESTAMP}
LOG_EVAL_OUTPUT_PATH=${CURR_OUTPUT_PATH}/logs/eval/${CURR_API}/${OUTPUT_DATASET_NAME}.out
LOG_INFER_OUTPUT_PATH=${CURR_OUTPUT_PATH}/logs/infer/${CURR_API}/${OUTPUT_DATASET_NAME}.out
PREDICTIONS_OUTPUT_PATH=${CURR_OUTPUT_PATH}/predictions/${CURR_API}/${OUTPUT_DATASET_NAME}.jsonl
RESULTS_OUTPUT_PATH=${CURR_OUTPUT_PATH}/results/${CURR_API}/${OUTPUT_DATASET_NAME}.json
SUMMARY_OUTPUT_PATH=${CURR_OUTPUT_PATH}/summary/summary_${TIMESTAMP}.csv

if [ ! -f "$LOG_EVAL_OUTPUT_PATH" ];then
    echo "Can't find $LOG_EVAL_OUTPUT_PATH"
    exit $ret_failed
fi
if [ ! -f "$LOG_INFER_OUTPUT_PATH" ];then
    echo "Can't find $LOG_INFER_OUTPUT_PATH"
    exit $ret_failed
fi
if [ ! -f "$PREDICTIONS_OUTPUT_PATH" ];then
    echo "Can't find $PREDICTIONS_OUTPUT_PATH"
    exit $ret_failed
fi
if [ ! -f "$RESULTS_OUTPUT_PATH" ];then
    echo "Can't find $RESULTS_OUTPUT_PATH"
    exit $ret_failed
fi
if [ ! -f "$SUMMARY_OUTPUT_PATH" ];then
    echo "Can't find $SUMMARY_OUTPUT_PATH"
    exit $ret_failed
fi

exit $ret_ok
