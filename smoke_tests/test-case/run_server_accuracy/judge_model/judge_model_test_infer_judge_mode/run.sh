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

# 添加裁判模型ip端口配置
{
    echo ""
    echo "aime2025_datasets[0]['judge_infer_cfg']['judge_model']['host_ip'] = '${AISBENCH_SMOKE_SERVICE_IP}'"
    echo "aime2025_datasets[0]['judge_infer_cfg']['judge_model']['host_port'] = ${AISBENCH_SMOKE_SERVICE_PORT}"
    echo "aime2025_datasets[0]['judge_infer_cfg']['judge_model']['path'] = '${AISBENCH_SMOKE_MODEL_PATH}'"
} >> "${AIS_BENCH_CODE_CONFIGS_DIR}/datasets/aime2025/${CASE_NAME}.py"

# 启动用例
# chat 配置文件
set -o pipefail  # 启用管道整体失败检测
echo -e "\033[1;32m[1/1]\033[0m Test case - ${CASE_NAME}"

if [ -f ${CUR_DIR}/tmplog.txt ];then
    rm ${CUR_DIR}/tmplog.txt
fi

# 同时运行数据集配置
ais_bench --models ${CASE_NAME} --datasets ${CASE_NAME} --work-dir ${CASE_OUTPUT_PATH} --max-num-workers 1 --mode infer_judge 2>&1 | tee ${CUR_DIR}/tmplog.txt

if [ $? -eq 0 ]
then
    echo "Run ${CASE_NAME} test: Success"
else
    echo "Run ${CASE_NAME} test: Failed"
    exit $ret_failed
fi

# 获取时间戳
WORK_DIR_INFO=$(cat ${CUR_DIR}/tmplog.txt | grep 'Current exp folder: ')
TIMESTAMP="${WORK_DIR_INFO##*/}"

# 数据集abbr列表，用于文件检查
dataset_abbr_list=(aime2025_judge)
judge_model_abbr="first"

# 检查每个数据集配置的输出文件
for abbr in ${dataset_abbr_list[@]}
do
    file_patterns=(
        "logs/infer/vllm-api-general-chat/${abbr}.out"
        "logs/infer/vllm-api-general-chat/${abbr}-${judge_model_abbr}.out"
        "predictions/vllm-api-general-chat/${abbr}.jsonl"
        "predictions/vllm-api-general-chat/${abbr}-${judge_model_abbr}.jsonl"
    )

    for pattern in "${file_patterns[@]}"; do
        file_path="${CASE_OUTPUT_PATH}/${TIMESTAMP}/${pattern}"
        if [ ! -f "${file_path}" ]; then
            echo "Failed. Dump file missing: ${file_path}"
            exit $ret_failed
        fi
    done
done

exit $ret_ok




