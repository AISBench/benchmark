#!/bin/bash
declare -i ret_ok=0
declare -i ret_failed=1

# 每个case通用变量
CUR_DIR=$(dirname $(readlink -f $0))
CASE_NAME=$(basename "$CUR_DIR")
AIS_BENCH_CODE_DIR=${PROJECT_RESOURCE_PATH}/code/
BENCHMARK_PLUGIN_DIR=${AIS_BENCH_CODE_DIR}/benchmark-mindie

# 拉取&更新benchmark-mindie代码
cd ${AIS_BENCH_CODE_DIR}
if [ ! -d ${BENCHMARK_PLUGIN_DIR} ];then
    git clone https://gitee.com/aisbench/benchmark-mindie.git
fi
cd ${BENCHMARK_PLUGIN_DIR}
git pull

# 安装插件
pip3 install -e ./
if [ $? -eq 0 ];then
    echo "Run $CASE_NAME test: Success"
else
    echo "Run $CASE_NAME test: Failed"
    cd ${CUR_DIR}
    exit $ret_failed
fi

# 检查安装成功
pkg_name=$(pip3 show mindie_ais_bench_backend | grep "Name: mindie_ais_bench_backend")
if [ "${pkg_name}" != "" ];then
    echo "Check $CASE_NAME test: Success"
else
    echo "Check $CASE_NAME test: Failed"
    exit $ret_failed
fi