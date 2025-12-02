# 多轮对话测评指南

## 多轮对话简介
多轮对话是指用户与服务后端之间进行多次交互的对话形式。与单轮对话（用户问一个问题，系统回答一个问题）不同，多轮对话可以包含多个回合，每个回合都依赖于之前的对话内容。这种对话形式更接近人类之间的自然交流。


## 测评能力介绍
当前支持多轮对话数据的服务化性能测评，不同服务化和数据集支持度如下：

服务化支持列表：
+ ✅vLLM
+ ✅MindIE Service
+ ✅SGLang

数据集支持列表：
+ ✅ShareGPT
+ ✅MTBench

## 快速入门
### 使用说明
+ ⚠️`SGLang`服务化需更改API配置文件中client为`OpenAIChatStreamSglangClient`
+ 📚按照轮次作为实际请求数量（例如`2`组对话共有`7`轮，测评结果中则有`7`条请求对应的性能指标）
### 命令含义
以`ShareGPT`多轮对话`vLLM`服务化的`v1/chat`流式接口性能测评场景为例：
```
ais_bench --models vllm_api_stream_chat --datasets sharegpt_gen --debug -m perf
```
其中：
- `--models`指定了模型任务，即`vllm_api_stream_chat`模型任务。

- `--datasets`指定了数据集任务，即`sharegpt_gen`数据集任务。
### 运行命令前置准备
- `--models`: 使用`vllm_api_stream_chat`模型任务，需要准备支持`v1/chat/completions`子服务的推理服务，可以参考🔗 [VLLM启动OpenAI 兼容服务器](https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html#openai-compatible-server)启动推理服务
- `--datasets`: 使用`sharegpt_gen`数据集任务，需要参考🔗 [ShareGPT数据集](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/sharegpt/README.md)准备sharegpt数据集。
### 任务对应配置文件修改
每个模型任务、数据集任务和结果呈现任务都对应一个配置文件，运行命令前需要修改这些配置文件的内容。这些配置文件路径可以通过在原有AISBench命令基础上加上`--search`来查询，例如：
```shell
# 注意search的命令中是否加 "--mode perf" 不影响搜索结果
ais_bench --models vllm_api_stream_chat --datasets sharegpt_gen --mode perf --search
```
> ⚠️ **注意**： 执行带search命令会打印出任务对应的配置文件的绝对路径。

执行查询命令可以得到如下查询结果：
```shell
06/28 11:52:25 - AISBench - INFO - Searching configs...
╒══════════════╤═══════════════════════════════════════╤════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
│ Task Type    │ Task Name                             │ Config File Path                                                                                                               │
╞══════════════╪═══════════════════════════════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
│ --models     │ vllm_api_stream_chat                  │ /your_workspace/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py                                  │
├──────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ --datasets   │ sharegpt_gen                          │ /your_workspace/benchmark/ais_bench/benchmark/configs/datasets/sharegpt/sharegpt_gen.py                                        │
╘══════════════╧═══════════════════════════════════════╧════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

```

- 快速入门中数据集任务配置文件`sharegpt_gen.py`不需要做额外修改，数据集任务配置文件内容介绍可参考📚 [配置开源数据集](../base_tutorials/all_params/datasets.md#配置开源数据集)

模型配置文件`vllm_api_stream_chat.py`中包含了模型运行相关的配置内容，是需要依据实际情况修改的。快速入门中需要修改的内容用注释标明。
```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-chat-stream',
        path="",                       # 指定模型序列化词表文件绝对路径，一般来说就是模型权重文件夹路径
        model="",                      # 指定服务端已加载模型名称，依据实际VLLM推理服务拉取的模型名称配置
        stream=True,                   # 指定为流式接口
        request_rate=0,                # 请求发送频率，每1/request_rate秒发送1个请求给服务端，小于0.1则一次性发送所有请求
        retry=2,
        api_key="",                    # 自定义api_key，默认为空
        host_ip="localhost",           # 指定推理服务的IP
        host_port=8080,                # 指定推理服务的端口
        url="",                        # 自定义url，默认为空
        max_out_len=512,               # 推理服务输出的token的最大数量
        batch_size=1,                  # 请求发送的最大并发数
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=False
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```
### 执行命令
修改好配置文件后，执行命令启动服务化性能评测（⚠️ 第一次执行建议加上`--debug`，可以将具体日志打屏，如果有请求推理服务过程中的报错更方便处理）：
```bash
# 命令行加上--debug
ais_bench --models vllm_api_stream_chat --datasets sharegpt_gen -m perf --debug
```
### 查看性能结果
性能结果打屏示例如下：

```bash
06/05 20:22:24 - AISBench - INFO - Performance Results of task: vllm-api-chat-stream/sharegptdataset:

╒══════════════════════════╤═════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════╕
│ Performance Parameters   │ Stage   │ Average          │ Min              │ Max              │ Median           │ P75              │ P90              │ P99              │  N   │
╞══════════════════════════╪═════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════╡
│ E2EL                     │ total   │ 2048.2945  ms    │ 1729.7498 ms     │ 3450.96 ms       │ 2491.8789 ms     │ 2750.85 ms       │ 3184.9186 ms     │ 3424.4354 ms     │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ TTFT                     │ total   │ 50.332 ms        │ 50.6244 ms       │ 52.0585 ms       │ 50.3237 ms       │ 50.5872 ms       │ 50.7566 ms       │ 50 .0551 ms      │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ TPOT                     │ total   │ 10.6965 ms       │ 10.061 ms        │ 10.8805 ms       │ 10.7495 ms       │ 10.7818 ms       │ 10.808 ms        │ 10.8582 ms       │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ ITL                      │ total   │ 10.6965 ms       │ 7.3583 ms        │ 13.7707 ms       │ 10.7513 ms       │ 10.8009 ms       │ 10.8358 ms       │ 10.9322 ms       │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ InputTokens              │ total   │ 1512.5           │ 1481.0           │ 1566.0           │ 1511.5           │ 1520.25          │ 1536.6           │ 1563.06          │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ OutputTokens             │ total   │ 287.375          │ 200.0            │ 407.0            │ 280.0            │ 322.75           │ 374.8            │ 403.78           │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ OutputTokenThroughput    │ total   │ 115.9216 token/s │ 107.6555 token/s │ 116.5352 token/s │ 117.6448 token/s │ 118.2426 token/s │ 118.3765 token/s │ 118.6388 token/s │ 8    │
╘══════════════════════════╧═════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════╛
╒══════════════════════════╤═════════╤════════════════════╕
│ Common Metric            │ Stage   │ Value              │
╞══════════════════════════╪═════════╪════════════════════╡
│ Benchmark Duration       │ total   │ 19897.8505 ms      │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Requests           │ total   │ 8                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Failed Requests          │ total   │ 0                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Success Requests         │ total   │ 8                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Concurrency              │ total   │ 0.9972             │
├──────────────────────────┼─────────┼────────────────────┤
│ Max Concurrency          │ total   │ 1                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Request Throughput       │ total   │ 0.4021 req/s       │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Input Tokens       │ total   │ 12100              │
├──────────────────────────┼─────────┼────────────────────┤
│ Prefill Token Throughput │ total   │ 17014.3123 token/s │
├──────────────────────────┼─────────┼────────────────────┤
│ Total generated tokens   │ total   │ 2299               │
├──────────────────────────┼─────────┼────────────────────┤
│ Input Token Throughput   │ total   │ 608.7438 token/s   │
├──────────────────────────┼─────────┼────────────────────┤
│ Output Token Throughput  │ total   │ 115.7835 token/s   │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Token Throughput   │ total   │ 723.5273 token/s   │
╘══════════════════════════╧═════════╧════════════════════╛

06/05 20:22:24 - AISBench - INFO - Performance Result files locate in outputs/default/20250605_202220/performances/vllm-api-chat-stream.

```
💡 具体性能参数的含义请参考📚 [性能测评结果说明](../base_tutorials/results_intro/performance_metric.md)

### 性能细节查看
执行AISBench命令后，任务执行更多细节最终会落盘在默认的输出路径，这个输出路径在运行中的打屏日志中有提示，例如：
```shell
06/28 15:13:26 - AISBench - INFO - Current exp folder: outputs/default/20250628_151326
```
这段日志说明任务执行的细节落盘在执行命令的路径下的`outputs/default/20250628_151326`中。
命令执行结束后`outputs/default/20250628_151326`中的任务执行的细节如下所示：
```shell
20250628_151326           # 每次实验基于时间戳生成的唯一目录
├── configs               # 自动存储的所有已转储配置文件
├── logs                  # 执行过程中日志，命令中如果加--debug，不会有过程日志落盘（都直接打印出来了）
│   └── performance/      # 推理阶段的日志文件
└── performance           # 性能测评结果
│    └── vllm-api-chat-stream/          # “服务化模型配置”名称，对应模型任务配置文件中models的 abbr参数
│         ├── sharegptdataset.csv          # 单次请求性能输出（CSV），与性能结果打屏中的Performance Parameters表格一致
│         ├── sharegptdataset.json         # 端到端性能输出（JSON），与性能结果打屏中的Common Metric表格一致
│         ├── sharegptdataset_details.h5 # 完整打点中的ITL数据
│         ├── sharegptdataset_details.json # 完整打点明细
│         └── sharegptdataset_plot.html    # 请求并发可视化报告（HTML）
```
💡其中 `sharegptdataset_plot.html`这个请求并发可视化报告建议使用Chrome或者Edge等浏览器打开，可以看到每个请求的时延以及每个时刻client端感知的服务时间并发数：
> ⚠️ **注意**： 多轮对话场景下，上半图中会将每组对话中的多轮请求拼成一条线，因此纵坐标的实际含义为多轮对话数据组的索引。
  ![full_plot_example.img](../img/request_concurrency/full_plot_example.png)
具体html中的图标如何查看请参考📚 [性能测试可视化并发图使用说明](../base_tutorials/results_intro/performance_visualization.md)