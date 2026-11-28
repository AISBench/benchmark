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

# 多档并发 sweep：展开为 {bs8,32} 与 {bs16,64} 两个组合，
# 预期落盘 combos_manifest.json（combos 数=2）与 batch_perf_summary.csv
set -o pipefail
ais_bench --models ${CASE_NAME} --datasets ${CASE_NAME} --work-dir ${CASE_OUTPUT_PATH} \
    --mode perf --num-warmups 0 --batch-size 8 16 --num-prompts 32 64 2>&1 | tee ${TMPLOG}
if [ $? -ne 0 ]
then
    echo "Perf run with multi-value --batch-size/--num-prompts: Failed"
    exit $ret_failed
fi

# 获取时间戳
WORK_DIR_INFO=$(cat ${TMPLOG} | grep 'Current exp folder: ')
TIMESTAMP="${WORK_DIR_INFO##*/}"
CURR_OUTPUT_PATH=${CASE_OUTPUT_PATH}/${TIMESTAMP}
MANIFEST_PATH=${CURR_OUTPUT_PATH}/combos_manifest.json
SUMMARY_CSV_PATH=${CURR_OUTPUT_PATH}/batch_perf_summary.csv

if [ ! -f ${MANIFEST_PATH} ];then
    echo "Can't find ${MANIFEST_PATH}"
    exit $ret_failed
elif [ ! -f ${SUMMARY_CSV_PATH} ];then
    echo "Can't find ${SUMMARY_CSV_PATH}"
    exit $ret_failed
fi

python3 - "${MANIFEST_PATH}" <<'PY'
import json
import os
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as manifest_file:
    manifest = json.load(manifest_file)
combos = manifest.get("combos") or []
assert len(combos) == 2, f"unexpected combo count: {len(combos)}"
batch_sizes = sorted(combo.get("batch_size") for combo in combos)
assert batch_sizes == [8, 16], f"unexpected batch sizes: {batch_sizes}"
model_abbrs = {combo.get("model_abbr") for combo in combos}
expected = {"vllm-api-stream-chat-bs8", "vllm-api-stream-chat-bs16"}
assert model_abbrs == expected, f"unexpected model abbrs: {model_abbrs}"
work_dir = os.path.dirname(path)
for suffix in ("bs8", "bs16"):
    result_path = os.path.join(
        work_dir, "performances", f"vllm-api-stream-chat-{suffix}", "gsm8k.json")
    with open(result_path, encoding="utf-8") as result_file:
        json.load(result_file)
print(f"batch sweep manifest verified: {path}")
PY
if [ $? -ne 0 ]
then
    exit $ret_failed
fi

if ! grep -q -- "-bs8" ${SUMMARY_CSV_PATH} || ! grep -q -- "-bs16" ${SUMMARY_CSV_PATH}
then
    echo "Summary CSV doesn't contain both -bs8/-bs16 combos: ${SUMMARY_CSV_PATH}"
    exit $ret_failed
fi

if grep -q "Traceback" ${TMPLOG}
then
    echo "Traceback found in ${TMPLOG}"
    exit $ret_failed
fi

exit $ret_ok
