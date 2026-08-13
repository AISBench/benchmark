# 推理服务模拟器
## 简介
模拟大模型推理服务，当前支持openai的api（流式&非流式 + chat&非chat），tgi的api（流式&非流式），triton的api（流式&非流式）以及mindie的原生api（流式&非流式）。
## 环境准备
安装相关python依赖：
```shell
pip install -r requirements.txt
```
## 安装工具
clone代码源码使用。
## 配置文件准备
通用配置文件位于`config.sh`，具体配置项如下：
```shell
PROCESS_NUM=4 # 服务进程数，对于高并发场景需要提高进程数才能承载
IP=0.0.0.0 # IP地址
PORT=8080 # 端口号
```

api相关配置文件位于`api/api_config.yaml`，具体配置项如下：
```yaml
general:
  enable_mtp: False # 是否启用mtp场景

stream_latency:
  ttft: 2.0 # seconds
  tpot: 0.01 # seconds

text_latency:
  e2el: 3 # seconds

random_dataset:
  random_content: False # 字符是否随机，不随机默认用'A'构造
  min_tokens: 10 # 最小长度
  tokens_per_chunk: 2 # mtp场景下的chunk大小

# /metrics 端点配置（用于 spec-decode 异常场景测试）
metrics:
  mode: static_counters  # error_http | no_spec | static_counters
  error_code: 503        # mode=error_http 时返回的 HTTP 状态码
  counters:              # mode=static_counters 时返回的固定计数器值
    num_drafts: 100
    num_draft_tokens: 500
    num_accepted_tokens: 450
    accepted_per_pos:    # 每个 draft position 的采纳 token 数
      0: 450
      1: 400
      2: 350
      3: 300
      4: 250
```

### /metrics 端点说明

模拟器额外提供 `/metrics` 端点（Prometheus 格式），用于配合 AISBench 的 `--spec-decode` 功能测试 spec-decode 指标采集的异常场景。通过 `metrics.mode` 配置项控制端点行为：

| mode | 行为 | 触发的 spec-decode 异常分支 |
|------|------|---------------------------|
| `error_http` | 返回指定 HTTP 错误码（默认503） | fetcher: `Metrics endpoint returned HTTP {code}` |
| `no_spec` | 返回200但不含 spec_decode 计数器 | snapshot: `No spec decode metrics found on server` |
| `static_counters` | 返回200 + 固定 spec_decode 计数器（before==after → delta=0） | calculator: `No spec decode activity detected during benchmark window` |

切换模式时修改 `api_config.yaml` 后重启服务即可。

## 支持的API列表
|api类型|endpoint子服务|备注|
|----|-----|----|
|openai chat text|v1/chat/completions||
|openai chat stream|v1/chat/completions||
|openai text|v1/completions||
|openai stream|v1/completions||
|tgi text|generate||
|tgi stream|generate_stream||
|triton text|v2/models/qwen/generate|模型名称为`qwen`|
|triton stream|v2/models/qwen/generate_stream|模型名称为`qwen`|
|mindie origin text|infer||
|mindie origin stream|infer||
|prometheus metrics|metrics|用于 spec-decode 异常场景测试，行为由 `metrics.mode` 配置控制|

## 启动服务
```shell
bash launch_service.sh
```