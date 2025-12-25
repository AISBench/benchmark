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
```

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

## 启动服务
```shell
bash launch_service.sh
```