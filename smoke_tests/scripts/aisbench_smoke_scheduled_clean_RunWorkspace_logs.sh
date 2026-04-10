#!/bin/bash
scripts_path=$(realpath `dirname $0`)

# 后台挂起的模式启动冒烟，无需session
bash $scripts_path/RunWorkspace_clean.sh -d 15 -c 50 -l $scripts_path/../RunWorkspace/cleanup.log