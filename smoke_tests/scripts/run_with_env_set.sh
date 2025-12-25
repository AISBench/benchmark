#!/bin/bash
scripts_path=$(realpath `dirname $0`)

# 配置网络
source $HOME/scripts/proxy_set.sh

export PATH=$PATH:/usr/local/bin/:/usr/local/sbin/
source $HOME/.bashrc

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate aisbench_smoke

bash $scripts_path/run_steps.sh $*

conda deactivate