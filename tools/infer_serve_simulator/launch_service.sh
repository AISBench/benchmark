#!bin/bash
. config.sh
CUR_DIR=$(dirname $(readlink -f $0))
export PYTHONPATH=$CUR_DIR:$PYTHONPATH
# gunicorn -w {进程数} -b {IP}:{端口} python脚本名:app --worker-class gevent
gunicorn -w=$PROCESS_NUM -b=$IP:$PORT flask_service:app --worker-class gevent