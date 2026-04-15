#!/bin/bash
scripts_path=$(realpath `dirname $0`)

# 后台挂起的模式启动冒烟，无需session
bash $scripts_path/run_with_env_set.sh -r -t short