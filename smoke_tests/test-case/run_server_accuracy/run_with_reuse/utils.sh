#!/bin/bash

# 查询时间戳目录是否被创建函数
wait_for_timestamp_dir() {
  local base_dir="$1"
  local timeout="${2:-60}"  # 最多等待 60 秒
  local interval=1          # 每秒检查一次
  local elapsed=0
  local latest_dir=""

  while [ $elapsed -lt $timeout ]; do
    # 排序后取最新
    latest_dir=$(find "$base_dir" -mindepth 1 -maxdepth 1 -type d \
      | sort -t'_' -k1,1 -k2,2 -r | head -n 1)

    if [ -n "$latest_dir" ]; then
      echo "$latest_dir"
      return 0
    fi

    sleep $interval
    elapsed=$((elapsed + interval))
  done

  echo "Error: Timeout waiting for timestamp directory." >&2
  return 1
}


poll_received_and_signal() {
  local tmp_path="$1"
  local pgid="$2"
  local signal="$3"
  local line_num="${4:-5}"
  local max_wait="${5:-20}"
  local interval="${6:-0.5}"

  local start_time end_time now log_chunk max_received
  start_time=$(date +%s)
  end_time=$(( start_time + max_wait ))

  while true; do
    now=$(date +%s)
    # timeout check
    if (( now >= end_time )); then
      echo "⚠️ Timeout (${max_wait}s)"
      return 1
    fi

    if compgen -G "${tmp_path}/tmp_*.jsonl" > /dev/null; then
      for f in "${tmp_path}"/tmp_*.jsonl; do
        [ -f "$f" ] || continue

        # 获取行数（可能有空格），并把非数字字符清除，保证是整数
        lines=$(wc -l < "$f" 2>/dev/null || echo 0)
        lines=${lines//[^0-9]/}       # 去掉一切非数字字符
        lines=${lines:-0}            # 为空则设为 0

        # 同样保证 line_num 是整数（防止外部传入非数字）
        line_num=${line_num:-0}
        line_num=${line_num//[^0-9]/}
        line_num=${line_num:-0}

        if (( lines >= line_num )); then
          echo "⚠️ Detected ${f} with ${lines} lines (>= ${line_num}); sending SIG${signal} to PGID=${pgid}"
          kill -"${signal}" -"${pgid}"
          return 0
        fi
      done
    fi

    sleep "$interval"
  done
}