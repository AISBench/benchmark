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

# 在aisbench源码路径中加入case独有的配置文件
cp -r ${CUR_DIR}/ais_bench_configs/*  ${AIS_BENCH_CODE_CONFIGS_DIR}/

# 添加外部传入的信息给模型配置文件(具体模型配置文件路径请依据实际情况修改, 模型或数据集配置文件统一命名成CASE_NAME)
{
    echo ""
    echo "models[0]['host_ip'] = '${AISBENCH_SMOKE_SERVICE_IP}'"
    echo "models[0]['host_port'] = ${AISBENCH_SMOKE_SERVICE_PORT}"
    echo "models[0]['path'] = '${AISBENCH_SMOKE_MODEL_PATH}'"
} >> "${AIS_BENCH_CODE_CONFIGS_DIR}/models/vllm_api/${CASE_NAME}.py"

# 启动用例
# chat 配置文件
set -o pipefail  # 启用管道整体失败检测
echo -e "\033[1;32m[1/1]\033[0m Test case - ${CASE_NAME}"

dataset_abbr_list=(cevaldataset mmludataset) # 所有数据集配置文件的list
if [ -f ${CUR_DIR}/tmplog.txt ];then
    rm ${CUR_DIR}/tmplog.txt
fi
ais_bench --models ${CASE_NAME} --datasets ${CASE_NAME}_mmlu ${CASE_NAME}_ceval --merge-ds --work-dir ${CASE_OUTPUT_PATH} --max-num-workers 6 2>&1 | tee ${CUR_DIR}/tmplog.txt
if [ $? -eq 0 ]
then
    echo "Run multi_datasets_single_file test: Success"
else
    echo "Run multi_datasets_single_file test: Failed"
    exit $ret_failed
fi
# 获取时间戳
WORK_DIR_INFO=$(cat ${CUR_DIR}/tmplog.txt | grep 'Current exp folder: ')
TIMESTAMP="${WORK_DIR_INFO##*/}"

for abbr in ${dataset_abbr_list[@]}
do
    echo "${abbr%%dataset}"
    if [ ! -f ${CASE_OUTPUT_PATH}/${TIMESTAMP}/logs/infer/vllm-api-general-chat/${abbr}.out ];then
        echo "Failed. Dump file missing: ${CASE_OUTPUT_PATH}/${TIMESTAMP}/logs/infer/vllm-api-general-chat/${abbr}.out"
        exit $ret_failed
    elif [ ! -f ${CASE_OUTPUT_PATH}/${TIMESTAMP}/summary/summary_${TIMESTAMP}.csv ];then
        echo "Failed. Dump file missing: ${CASE_OUTPUT_PATH}/${TIMESTAMP}/summary/summary_${TIMESTAMP}.csv"
        exit $ret_failed
    elif [ ! -f ${CASE_OUTPUT_PATH}/${TIMESTAMP}/summary/summary_${TIMESTAMP}.csv ];then
        echo "Failed. Dump file missing: ${CASE_OUTPUT_PATH}/${TIMESTAMP}/summary/summary_${TIMESTAMP}.txt"
        exit $ret_failed
    elif [ ! -f ${CASE_OUTPUT_PATH}/${TIMESTAMP}/summary/summary_${TIMESTAMP}.csv ];then
        echo "Failed. Dump file missing: ${CASE_OUTPUT_PATH}/${TIMESTAMP}/summary/summary_${TIMESTAMP}.md"
        exit $ret_failed
    fi
    if [ "$abbr" = "cevaldataset" ]; then
    data_name_list=( "ceval-computer_network" "ceval-operating_system" "ceval-computer_architecture" )
    elif [ "$abbr" = "mmludataset" ]; then
    data_name_list=( "lukaemon_mmlu_college_biology" "lukaemon_mmlu_college_chemistry" "lukaemon_mmlu_college_computer_science" )
    else
    echo "⚠️  Unknown abbr: $abbr" >&2
    exit "$ret_failed"
    fi

    for data_name in "${data_name_list[@]:0:3}"; do
    PREDICTIONS_OUTPUT_PATH="${CASE_OUTPUT_PATH}/${TIMESTAMP}/predictions/vllm-api-general-chat/${data_name}.jsonl"
    RESULTS_OUTPUT_PATH="${CASE_OUTPUT_PATH}/${TIMESTAMP}/results/vllm-api-general-chat/${data_name}.json"
    EVAL_OUTPUT_PATH="${CASE_OUTPUT_PATH}/${TIMESTAMP}/logs/eval/vllm-api-general-chat/${data_name}.out"
    if [ ! -f "$PREDICTIONS_OUTPUT_PATH" ]; then
        echo "Can't find $PREDICTIONS_OUTPUT_PATH" >&2
        exit "$ret_failed"
    elif [ ! -f "$RESULTS_OUTPUT_PATH" ]; then
        echo "Can't find $RESULTS_OUTPUT_PATH" >&2
        exit "$ret_failed"
    elif [ ! -f "$EVAL_OUTPUT_PATH" ]; then
        echo "Can't find $EVAL_OUTPUT_PATH" >&2
        exit "$ret_failed"
    fi
    done
done

exit $ret_ok




