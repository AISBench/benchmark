#!/usr/bin/env bash

PROJECT_PATH=$(realpath `dirname $0`/..)
export PROJECT_PATH=$PROJECT_PATH                        # 工程路径
PROJECT_TEST_CASE_PATH=$PROJECT_PATH/test-case/          # 推理用例

bash $PROJECT_PATH/scripts/run.sh -p $PROJECT_TEST_CASE_PATH $*
