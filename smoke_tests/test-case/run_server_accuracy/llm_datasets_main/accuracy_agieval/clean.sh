#!/bin/bash
CUR_DIR=$(dirname $(readlink -f $0))
[ -f "${CUR_DIR}/tmplog.txt" ] && rm -f "${CUR_DIR}/tmplog.txt"
exit 0
