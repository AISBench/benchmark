#!/bin/bash
declare -i ret_ok=0
declare -i ret_failed=1

# 每个case通用变量
CUR_DIR=$(dirname $(readlink -f $0))
CASE_NAME=$(basename "$CUR_DIR")
LAST_3_DIRNAME=$(echo  $CUR_DIR | rev | cut -d'/' -f1-3 | rev)
CASE_OUTPUT_PATH=${PROJECT_OUTPUT_PATH}/${LAST_3_DIRNAME} # 项目根路径
AIS_BENCH_CODE_CONFIGS_DIR=${PROJECT_PATH}/../ais_bench/benchmark/configs
RUN_LOG_0=${CUR_DIR}/tmplog_warmup_0.txt
RUN_LOG_4=${CUR_DIR}/tmplog_warmup_4.txt

# 清理用例输出
if [ ! -d ${CASE_OUTPUT_PATH} ];then
    mkdir -p ${CASE_OUTPUT_PATH}
fi
rm -rf ${CASE_OUTPUT_PATH}/*

# 在aisbench源码路径中加入case独有的配置文件
cp -r ${CUR_DIR}/ais_bench_configs/* ${AIS_BENCH_CODE_CONFIGS_DIR}/

# 添加外部传入的信息给模型配置文件
{
    echo ""
    echo "models[0]['host_ip'] = '${AISBENCH_SMOKE_SERVICE_IP}'"
    echo "models[0]['host_port'] = ${AISBENCH_SMOKE_SERVICE_PORT}"
    echo "models[0]['path'] = '${AISBENCH_SMOKE_MODEL_PATH}'"
} >> "${AIS_BENCH_CODE_CONFIGS_DIR}/models/vllm_api/${CASE_NAME}.py"

echo -e "\033[1;32m[1/2]\033[0m Test case - ${CASE_NAME} (--num-warmups 0)"
if [ -f ${RUN_LOG_0} ];then
    rm ${RUN_LOG_0}
fi

# --debug 让推理子进程输出直接打印到当前 stdout（tmplog），
# warmup 汇总日志仅在 debug 下打印，因此这里用 debug 校验端到端行为
set -o pipefail
ais_bench --models ${CASE_NAME} --datasets ${CASE_NAME} --work-dir ${CASE_OUTPUT_PATH}/warmup_0 \
    --mode perf --debug --num-warmups 0 --num-prompts 4 2>&1 | tee ${RUN_LOG_0}
if [ $? -ne 0 ]
then
    echo "Perf run with --num-warmups 0: Failed"
    exit $ret_failed
fi

if ! grep -q "Warmup size is 0, skip" ${RUN_LOG_0}
then
    echo "Can't find 'Warmup size is 0, skip' in ${RUN_LOG_0}"
    exit $ret_failed
fi

echo -e "\033[1;32m[2/2]\033[0m Test case - ${CASE_NAME} (--num-warmups 4)"
if [ -f ${RUN_LOG_4} ];then
    rm ${RUN_LOG_4}
fi

ais_bench --models ${CASE_NAME} --datasets ${CASE_NAME} --work-dir ${CASE_OUTPUT_PATH}/warmup_4 \
    --mode perf --debug --num-warmups 4 --num-prompts 4 2>&1 | tee ${RUN_LOG_4}
if [ $? -ne 0 ]
then
    echo "Perf run with --num-warmups 4: Failed"
    exit $ret_failed
fi

if ! grep -q "Warmup finished" ${RUN_LOG_4}
then
    echo "Can't find 'Warmup finished' in ${RUN_LOG_4}"
    exit $ret_failed
fi

if ! grep -q "Total Count: 4" ${RUN_LOG_4}
then
    echo "Can't find 'Total Count: 4' in ${RUN_LOG_4}"
    exit $ret_failed
fi

if grep -q "Traceback" ${RUN_LOG_0} ${RUN_LOG_4}
then
    echo "Traceback found in warmup logs"
    exit $ret_failed
fi

exit $ret_ok
