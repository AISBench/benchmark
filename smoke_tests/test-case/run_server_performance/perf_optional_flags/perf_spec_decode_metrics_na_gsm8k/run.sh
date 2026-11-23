#!/bin/bash
declare -i ret_ok=0
declare -i ret_failed=1

# 每个case通用变量
CUR_DIR=$(dirname $(readlink -f $0))
CASE_NAME=$(basename "$CUR_DIR")
LAST_3_DIRNAME=$(echo  $CUR_DIR | rev | cut -d'/' -f1-3 | rev)
CASE_OUTPUT_PATH=${PROJECT_OUTPUT_PATH}/${LAST_3_DIRNAME} # 项目根路径
AIS_BENCH_CODE_CONFIGS_DIR=${PROJECT_PATH}/../ais_bench/benchmark/configs
TMPLOG=${CUR_DIR}/tmplog.txt

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

echo -e "\033[1;32m[1/1]\033[0m Test case - ${CASE_NAME}"
if [ -f ${TMPLOG} ];then
    rm ${TMPLOG}
fi

# 模拟器 /metrics 默认 metrics.mode=static_counters：前后快照一致，delta=0，
# 预期走 N/A 分支并落盘 performances/spec_decode_<host>_<port>.json
set -o pipefail
ais_bench --models ${CASE_NAME} --datasets ${CASE_NAME} --work-dir ${CASE_OUTPUT_PATH} \
    --mode perf --spec-decode --num-warmups 0 --num-prompts 8 2>&1 | tee ${TMPLOG}
if [ $? -ne 0 ]
then
    echo "Perf run with --spec-decode: Failed"
    exit $ret_failed
fi

# 获取时间戳
WORK_DIR_INFO=$(cat ${TMPLOG} | grep 'Current exp folder: ')
TIMESTAMP="${WORK_DIR_INFO##*/}"
CURR_OUTPUT_PATH=${CASE_OUTPUT_PATH}/${TIMESTAMP}
SPEC_RESULT_PATH=$(ls ${CURR_OUTPUT_PATH}/performances/spec_decode_*.json 2>/dev/null | head -n 1)

if [ -z "${SPEC_RESULT_PATH}" ];then
    echo "Can't find spec decode result under ${CURR_OUTPUT_PATH}/performances"
    exit $ret_failed
fi

python3 - "${SPEC_RESULT_PATH}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as spec_file:
    result = json.load(spec_file)
assert result.get("status") == "na", f"unexpected spec decode status: {result.get('status')!r}"
error = result.get("error") or ""
assert "No spec decode activity detected during benchmark window" in error, \
    f"unexpected spec decode error: {error!r}"
print(f"spec decode N/A result verified: {path}")
PY
if [ $? -ne 0 ]
then
    exit $ret_failed
fi

if grep -q "Traceback" ${TMPLOG}
then
    echo "Traceback found in ${TMPLOG}"
    exit $ret_failed
fi

exit $ret_ok
