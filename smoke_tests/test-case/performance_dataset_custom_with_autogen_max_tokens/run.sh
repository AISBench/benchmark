#!bin/bash
declare -i ret_ok=0
declare -i ret_failed=1

# 每个case通用变量
CUR_DIR=$(dirname $(readlink -f $0))
CASE_NAME=$(basename "$CUR_DIR")
LAST_3_DIRNAME=$(echo  $CUR_DIR | rev | cut -d'/' -f1-3 | rev)
CASE_OUTPUT_PATH=${PROJECT_OUTPUT_PATH}/${LAST_3_DIRNAME} # 项目根路径
AIS_BENCH_CODE_CONFIGS_DIR=${PROJECT_PATH}/../ais_bench/benchmark/configs
DATASET_PATH="$CUR_DIR/ais_bench_configs/datasets/test_mcq.jsonl"
META_JSON_PATH="$CUR_DIR/ais_bench_configs/datasets/test_mcq.jsonl.meta.json"
OUTPUT_DATASET_NAME="test_mcq"
CURR_API="vllm-api-stream-chat"

# 清理用例输出
if [ ! -d ${CASE_OUTPUT_PATH} ];then
    mkdir -p ${CASE_OUTPUT_PATH}
fi
rm -rf ${CASE_OUTPUT_PATH}/*

# 在aisbench源码路径中加入case独有的配置文件
MODEL_TARGET_DIR="${AIS_BENCH_CODE_CONFIGS_DIR}/models/vllm_api/"
echo "Copying config files to ${MODEL_TARGET_DIR}..."
cp ${CUR_DIR}/ais_bench_configs/models/vllm_api/${CASE_NAME}.py  ${MODEL_TARGET_DIR}

# 添加外部传入的信息给模型配置文件(具体模型配置文件路径请依据实际情况修改, 模型或数据集配置文件统一命名成CASE_NAME)
{
    echo ""
    echo "models[0]['host_ip'] = '${AISBENCH_SMOKE_SERVICE_IP}'"
    echo "models[0]['host_port'] = ${AISBENCH_SMOKE_SERVICE_PORT}"
    echo "models[0]['path'] = '${AISBENCH_SMOKE_MODEL_PATH}'"
} >> "${AIS_BENCH_CODE_CONFIGS_DIR}/models/vllm_api/${CASE_NAME}.py"

# 启动用例
echo -e "\033[1;32m[1/1]\033[0m Test case - ${CASE_NAME}"

# 先跑出性能结果
set -o pipefail  # 启用管道整体失败检测
ais_bench --models ${CASE_NAME} --custom-dataset-path $DATASET_PATH --custom-dataset-meta-path $META_JSON_PATH --work-dir ${CASE_OUTPUT_PATH} --mode perf 2>&1 | tee ${CUR_DIR}/tmplog.txt

if [ $? -eq 0 ]
then
    echo "Run perf test: Success"
else
    echo "Run perf test: Failed"
    exit $ret_failed
fi
# 获取时间戳
WORK_DIR_INFO=$(cat ${CUR_DIR}/tmplog.txt | grep 'Current exp folder: ')
TIMESTAMP="${WORK_DIR_INFO##*/}"

# 验证产物
CURR_OUTPUT_PATH=${CASE_OUTPUT_PATH}/${TIMESTAMP}
LOG_OUTPUT_PATH=${CURR_OUTPUT_PATH}/logs/infer/${CURR_API}/${OUTPUT_DATASET_NAME}.out
PERFORMANCES_JSON_OUTPUT_PATH=${CURR_OUTPUT_PATH}/performances/${CURR_API}/${OUTPUT_DATASET_NAME}.json
PERFORMANCES_CSV_OUTPUT_PATH=${CURR_OUTPUT_PATH}/performances/${CURR_API}/${OUTPUT_DATASET_NAME}.csv
PERFORMANCES_DETAILS_OUTPUT_PATH=${CURR_OUTPUT_PATH}/performances/${CURR_API}/${OUTPUT_DATASET_NAME}_details.jsonl
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
elif [ ! -f $PERFORMANCES_DETAILS_OUTPUT_PATH ];then
    echo "Can't find $PERFORMANCES_DETAILS_OUTPUT_PATH"
    exit $ret_failed
fi

python3 $CUR_DIR/compare_max_token.py $PERFORMANCES_DETAILS_OUTPUT_PATH
if [ $? -eq 0 ]
then
    echo "compare_max_token length: Success"
else
    echo "compare_max_token length: Failed"
    exit $ret_failed
fi

exit $ret_ok
