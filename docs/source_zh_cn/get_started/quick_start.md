# 快速入门
## 命令含义
AISBench命令执行的单个或多个评测任务是由模型任务（单个或多个）、数据集任务（单个或多个）和结果呈现任务（单个）的组合定义的，AISBench的其他命令行则规定了评测任务的场景（精度评测场景、性能评测场景等）。以如下AISBench命令为例：
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --summarizer example
```
此命令没有指定其他命令行，默认是一个精度评测场景的任务，其中：
- `--models`指定了模型任务，即`vllm_api_general_chat`模型任务。

- `--datasets`指定了数据集任务，即`demo_gsm8k_gen_4_shot_cot_chat_prompt`数据集任务。

- `--summarizer`指定了结果呈现任务，即`example`结果呈现任务(不指定`--summarizer`精度评测场景默认使用`example`任务)，一般使用默认，不需要在命令行中指定，后续命令不指定。

## 任务含义查询(可选)
所选模型任务`vllm_api_general_chat`、数据集任务`demo_gsm8k_gen_4_shot_cot_chat_prompt`和结果呈现任务`example`的具体信息(简介，使用约束等)可以分别从如下链接中查询含义：
- `--models`: 📚 [服务化推理后端](../base_tutorials/all_params/models.md#服务化推理后端)

- `--datasets`: 📚 [开源数据集](../base_tutorials/all_params/datasets.md#开源数据集) → 📚 [详细介绍](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/demo/README.md)

- `--summarizer`: 📚 [结果汇总任务](../base_tutorials/all_params/summarizer.md)

## 运行命令前置准备
- `--models`: 使用`vllm_api_general_chat`模型任务，需要准备支持`v1/chat/completions`子服务的推理服务，可以参考🔗 [VLLM启动OpenAI 兼容服务器](https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html#openai-compatible-server)启动推理服务
- `--datasets`: 使用`demo_gsm8k_gen_4_shot_cot_chat_prompt`数据集任务，需要准备gsm8k数据集，可以从🔗 [opencompass
提供的gsm8k数据集压缩包](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip)下载。将解压后的`gsm8k/`文件夹部署到AISBench评测工具根路径下的`ais_bench/datasets`文件夹下。

## 任务对应配置文件修改
每个模型任务、数据集任务和结果呈现任务都对应一个配置文件，运行命令前需要修改这些配置文件的内容。这些配置文件路径可以通过在原有AISBench命令基础上加上`--search`来查询，例如：
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --search
```
> ⚠️ **注意**： 执行带search命令会打印出任务对应的配置文件的绝对路径。

执行查询命令可以得到如下查询结果：
```shell
╒══════════════╤═══════════════════════════════════════╤════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
│ Task Type    │ Task Name                             │ Config File Path                                                                                                               │
╞══════════════╪═══════════════════════════════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
│ --models     │ vllm_api_general_chat                 │ /your_workspace/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py                                 │
├──────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ --datasets   │ demo_gsm8k_gen_4_shot_cot_chat_prompt │ /your_workspace/benchmark/ais_bench/benchmark/configs/datasets/demo/demo_gsm8k_gen_4_shot_cot_chat_prompt.py                   │
╘══════════════╧═══════════════════════════════════════╧════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

```

- 快速入门中数据集任务配置文件`demo_gsm8k_gen_4_shot_cot_chat_prompt.py`不需要做额外修改，数据集任务配置文件内容介绍可参考📚 [配置开源数据集](../base_tutorials/all_params/datasets.md#配置开源数据集)

模型配置文件`vllm_api_general_chat.py`中包含了模型运行相关的配置内容，是需要依据实际情况修改的。快速入门中需要修改的内容用注释标明。
```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="",                    # 指定模型序列化词表文件绝对路径（精度测试场景一般不需要配置）
        model="",        # 指定服务端已加载模型名称，依据实际VLLM推理服务拉取的模型名称配置（配置成空字符串会自动获取）
        stream=False,
        request_rate=0,           # 请求发送频率，每1/request_rate秒发送1个请求给服务端，小于0.1则一次性发送所有请求
        retry=2,                  # 每个请求最大重试次数
        headers={"Content-Type": "application/json"}, # 自定义请求头，默认{"Content-Type": "application/json"}
        host_ip="localhost",      # 指定推理服务的IP
        host_port=8080,           # 指定推理服务的端口
        url="",                     # 自定义访问推理服务的URL路径(当base url不是http://host_ip:host_port的组合时需要配置，配置后host_ip和host_port将被忽略)
        max_out_len=512,          # 推理服务输出的token的最大数量
        batch_size=1,               # 请求发送的最大并发数
        trust_remote_code=False,    # tokenizer是否信任远程代码，默认False;
        generation_kwargs=dict(   # 模型推理参数，参考VLLM文档配置，AISBench评测工具不做处理，在发送的请求中附带
            temperature=0.01,
            ignore_eos=False,
        )
    )
]
```
## 执行命令
修改好配置文件后，执行命令启动服务化精度评测（⚠️ 第一次执行建议加上`--debug`，可以将具体日志打屏，如果有请求推理服务过程中的报错更方便处理）：
```bash
# 命令行加上--debug，
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug
```
### 执行命令
修改好配置文件后，执行命令启动服务化精度评测：
```bash
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt
```
#### 查看任务执行细节
执行AISBench命令后，正在执行的任务状态会在命令行实时刷新的看板上显示（键盘按"P"键可以停止刷新，用于复制看板信息，再按"P"可以继续刷新），例如：
```
Base path of result&log : outputs/default/20250628_151326
Task Progress Table (Updated at: 2025-11-06 10:08:21)
Page: 1/1  Total 2 rows of data
Press Up/Down arrow to page,  'P' to PAUZE/RESUME screen refresh, 'Ctrl + C' to exit

+----------------------------------+-----------+-------------------------------------------------+-------------+-------------+-------------------------------------------------+------------------------------------------------+
| Task Name                        |   Process | Progress                                        | Time Cost   | Status      | Log Path                                        | Extend Parameters                              |
+==================================+===========+=================================================+=============+=============+=================================================+================================================+
| vllm-api-general-chat/demo_gsm8k |    547141 | [###############               ] 4/8 [0.5 it/s] | 0:00:11     | inferencing | logs/infer/vllm-api-general-chat/demo_gsm8k.out | {'POST': 5, 'RECV': 4, 'FINISH': 4, 'FAIL': 0} |
+----------------------------------+-----------+-------------------------------------------------+-------------+-------------+-------------------------------------------------+------------------------------------------------+

```

任务执行的细节日志会不断落盘在默认的输出路径，这个输出路径在实时刷新的看板上显示，即`Log Path`。`Log Path`（`logs/infer/vllm-api-general-chat/demo_gsm8k.out`）是在`Base path`（`outputs/default/20250628_151326`）下的路径，以上述的看板信息为例，任务执行的详细日志路径为：
```shell
# {Base path}/{Log Path}
outputs/default/20250628_151326/logs/infer/vllm-api-general-chat/demo_gsm8k.out
```

> 💡 如果希望执行过程中将详细日志直接打印，执行命令时可以加上 `--debug`:
`ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug`




`Base path`（`outputs/default/20250628_151326`）下包含了所有任务的执行细节，命令执行结束后所有的执行细节如下：
```shell
20250628_151326/
├── configs # 模型任务、数据集任务和结构呈现任务对应的配置文件合成的一个配置
│   └── 20250628_151326_29317.py
├── logs # 执行过程中日志，命令中如果加--debug，不会有过程日志落盘（都直接打印出来了）
│   ├── eval
│   │   └── vllm-api-general-chat
│   │       └── demo_gsm8k.out # 基于predictions/文件夹下的推理结果的精度评测过程的日志
│   └── infer
│       └── vllm-api-general-chat
│           └── demo_gsm8k.out # 推理过程日志
├── predictions
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # 推理结果（推理服务返回的所有输出）
├── results
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # 精度评测计算的原始分数
└── summary
    ├── summary_20250628_151326.csv # 最终精度分数呈现（表格格式）
    ├── summary_20250628_151326.md # 最终精度分数呈现（markdown格式）
    └── summary_20250628_151326.txt # # 最终精度分数呈现（文本格式）
```
> ⚠️ **注意**： 不同评测场景落盘任务执行细节内容不同，具体请参考具体评测场景的指南。


### 输出结果
因为只有8条数据，会很快跑出结果，结果显示的示例如下
```bash
dataset                 version  metric   mode  vllm_api_general_chat
----------------------- -------- -------- ----- ----------------------
demo_gsm8k              401e4c   accuracy gen                   62.50
```