#!bin/bash
declare -i ret_ok=0
declare -i ret_failed=1

# 每个case通用变量
CUR_DIR=$(dirname $(readlink -f $0))
CASE_NAME=$(basename "$CUR_DIR")
LAST_3_DIRNAME=$(echo  $CUR_DIR | rev | cut -d'/' -f1-3 | rev)
CASE_OUTPUT_PATH=${PROJECT_OUTPUT_PATH}/${LAST_3_DIRNAME} # 项目根路径
AIS_BENCH_CODE_CONFIGS_DIR=${PROJECT_PATH}/../ais_bench/benchmark/configs

# 清理用例输出
if [ ! -d ${CASE_OUTPUT_PATH} ];then
    mkdir -p ${CASE_OUTPUT_PATH}
fi
rm -rf ${CASE_OUTPUT_PATH}/*

echo "${CASE_OUTPUT_PATH}"
cp -r ${CUR_DIR}/ais_bench_configs/ ${CASE_OUTPUT_PATH}

# 添加外部传入的信息给模型配置文件(具体模型配置文件路径请依据实际情况修改, 模型或数据集配置文件统一命名成CASE_NAME)
{
    echo ""
    echo "models[0]['host_ip'] = '${AISBENCH_SMOKE_SERVICE_IP}'"
    echo "models[0]['host_port'] = ${AISBENCH_SMOKE_SERVICE_PORT}"
    echo "models[0]['path'] = '${AISBENCH_SMOKE_MODEL_PATH}'"
} >> "${CASE_OUTPUT_PATH}/ais_bench_configs/models/vllm_api/${CASE_NAME}.py"

# 启动用例
echo -e "\033[1;32m[1/1]\033[0m Test case - ${CASE_NAME}"

# 先跑出性能结果
set -o pipefail  # 启用管道整体失败检测
if [ -f ${CUR_DIR}/tmplog.txt ];then
    rm ${CUR_DIR}/tmplog.txt
fi
ais_bench --models ${CASE_NAME} --datasets ${CASE_NAME} --mode perf --config-dir ${CASE_OUTPUT_PATH}/ais_bench_configs/ --work-dir ${CASE_OUTPUT_PATH} 2>&1 | tee ${CUR_DIR}/tmplog.txt
if [ $? -eq 0 ]
then
    echo "First run perf test custom config dir : Success"
else
    echo "First run perf test custom config dir : Failed"
    exit $ret_failed
fi
# 获取时间戳
WORK_DIR_INFO=$(cat ${CUR_DIR}/tmplog.txt | grep 'Current exp folder: ')
TIMESTAMP="${WORK_DIR_INFO##*/}"


if [ ! -f ${CASE_OUTPUT_PATH}/${TIMESTAMP}/performances/vllm-api-stream-chat/gsm8k.json ];then
    echo "Can't find gsm8k.json"
    exit $ret_failed
elif [ ! -f ${CASE_OUTPUT_PATH}/${TIMESTAMP}/performances/vllm-api-stream-chat/gsm8k.csv ];then
    echo "Can't find gsm8k.csv"
    exit $ret_failed
elif [ ! -f ${CASE_OUTPUT_PATH}/${TIMESTAMP}/performances/vllm-api-stream-chat/gsm8k_plot.html ];then
    echo "Can't find gsm8k_plot.html"
    exit $ret_failed
fi

exit $ret_ok




