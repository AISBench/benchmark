#!/bin/bash

CUR_DIR=$(dirname $(readlink -f $0))
CASE_NAME=$(basename "$CUR_DIR")
LAST_3_DIRNAME=$(echo  $CUR_DIR | rev | cut -d'/' -f1-3 | rev)
CASE_OUTPUT_PATH=${PROJECT_OUTPUT_PATH}/${LAST_3_DIRNAME}
AIS_BENCH_CODE_CONFIGS_DIR=${PROJECT_PATH}/../ais_bench/benchmark/configs

rm -rf ${CASE_OUTPUT_PATH}
rm -f ${CUR_DIR}/tmplog.txt
rm -f ${AIS_BENCH_CODE_CONFIGS_DIR}/datasets/refcocog/${CASE_NAME}.py
rm -f ${AIS_BENCH_CODE_CONFIGS_DIR}/models/vllm_api/${CASE_NAME}.py

exit 0