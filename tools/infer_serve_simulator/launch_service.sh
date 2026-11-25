#!bin/bash
. config.sh
CUR_DIR=$(dirname $(readlink -f $0))
export PYTHONPATH=$CUR_DIR:$PYTHONPATH
# gunicorn -w {进程数} -b {IP}:{端口} python脚本名:app --worker-class gevent
if [ -n "${AISBENCH_SMOKE_PYTHON:-}" ]; then
    exec "$AISBENCH_SMOKE_PYTHON" -m gunicorn -w=$PROCESS_NUM -b=$IP:$PORT flask_service:app --worker-class gevent
else
    exec gunicorn -w=$PROCESS_NUM -b=$IP:$PORT flask_service:app --worker-class gevent
fi