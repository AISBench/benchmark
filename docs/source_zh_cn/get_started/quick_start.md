# 快速入门

## 运行命令前置准备

- 需要准备支持`v1/chat/completions`子服务的推理服务，可以参考🔗 [VLLM启动OpenAI 兼容服务器](https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html#openai-compatible-server)启动推理服务
- 需要准备gsm8k数据集，可以从🔗 [opencompass
  提供的gsm8k数据集压缩包](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip)下载。将解压后的`gsm8k/`文件夹部署到AISBench评测工具根路径下的`ais_bench/datasets`文件夹下。

## 启动测评（两种方式任选其一）

| ⭐ 推荐：使用自定义配置文件 | 备选：使用命令行参数(原快速入门方式) |
| :--- | :--- |
| 修改一个文件，集中管理所有配置，在任意路径写配置 | 通过 `--models` `--datasets` 参数指定 |
| 一次编写，多次复用 | 每次运行需输入完整命令 |
| 支持 Python 全部语法，灵活扩展 | 仅支持笛卡尔积组合 |

::::{tab-set}
:::{tab-item} ⭐ 推荐：使用自定义配置文件

AISBench 提供了预置的自定义配置文件 [model_api_test_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/model_api_test_zh_cn.py)，将常见的推理服务化测试配置（模型选择、服务地址、端口、生成参数等）集中在一个文件中，无需分别查找和修改多个配置文件。该文件本质上是 Python 脚本，支持所有 Python 语法，你可以自由扩展。

打开 `ais_bench/configs/model_api_test_zh_cn.py`，根据实际情况修改以下配置（如果是`pip3 install ais_bench_benchmark`方式直接安装工具，可以在任意路径自行创建`model_api_test_zh_cn.py`，将以下配置内容写入该文件）：

```python
from mmengine.config import read_base

with read_base():
# 模型任务，选择其中一个，其他模型任务参考：https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/all_params/models.html 获取更多模型任务
    # vllm_api_general 是基础模型，仅支持文本生成
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general
    # vllm_api_general_chat 是对话模型，支持对话
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat
    # vllm_api_stream_chat 是流式对话模型，支持流式对话
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat
    # vllm_api_general_stream 是流式模型，支持流式生成
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as vllm_api_general_stream

# 数据集任务，参考：https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/get_started/datasets.html 获取更多数据集任务
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets

models = vllm_api_general_chat

models[0]["path"] = ""  # 指定模型序列化词表文件的绝对路径（精度测试场景一般不需要配置）
models[0]["model"] = "" # 指定服务端加载的模型名称，根据 VLLM 推理服务实际拉取的模型名称配置（配置为空字符串则自动获取）
models[0]["request_rate"] = 0 # 请求发送频率：每 1/request_rate 秒向服务端发送 1 条请求；小于 0.001 时一次性发送所有请求
models[0]["api_key"] = "" # 自定义 API key，默认为空字符串
models[0]["host_ip"] = "localhost" # 指定推理服务的 IP
models[0]["host_port"] = 8080 # 指定推理服务的端口
models[0]["url"] = "" # 自定义访问推理服务的 URL 路径（当基础 URL 不是 http://host_ip:host_port 的组合时需要配置；配置后 host_ip 和 host_port 将被忽略）
models[0]["max_out_len"] = 512 # 推理服务输出的最大 token 数
models[0]["batch_size"] = 1 # 发送请求的最大并发数
models[0]["trust_remote_code"] = False # tokenizer 是否信任远程代码，默认为 False
models[0]["generation_kwargs"] = dict( # 模型推理参数，参考 VLLM 文档配置；AISBench 评测工具不做处理，直接附加到发送的请求中
    temperature=0.01,
    ignore_eos=False,
)

# datasets[0]["path"] = ais_bench/datasets/gsm8k # 指定数据集目录的绝对路径（精度测试场景需要配置）

work_dir = 'outputs/default/'  # 指定任务结果和日志的保存工作目录（默认为 outputs/default/）

```

> 💡 配置文件中已预置了常用模型类型的导入（`vllm_api_general`、`vllm_api_general_chat`、`vllm_api_stream_chat`、`vllm_api_general_stream`），只需取消/修改注释即可切换。更多自定义配置文件的用法请参考 📚 [自定义配置文件运行AISBench](../advanced_tutorials/run_custom_config.md)。

数据集任务的选取、准备和使用参考如下步骤：

1. 在📚 [开源数据集](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/get_started/datasets.html#id3)内选取数据集任务
2. 进入数据的 📚 [详细介绍/数据集部署](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/demo/README.md#数据集部署)准备数据集
3. 参考📚 [详细介绍/可用数据集任务](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/demo/README.md#可用数据集任务)选取可用数据集任务，并将对应的任务导入方式（例如`from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets`）复制到自定义配置文件中

修改好配置文件后，执行如下命令启动服务化精度评测：

```bash
ais_bench ais_bench/configs/model_api_test_zh_cn.py
```

:::
:::{tab-item} 备选：使用命令行参数

如果你更习惯使用命令行参数方式，AISBench 同样支持通过 `--models`、`--datasets`、`--summarizer` 参数直接指定任务。以下是与上述自定义配置文件方式**执行效果完全相同**的命令行方式。

AISBench命令执行的单个或多个评测任务是由模型任务（单个或多个）、数据集任务（单个或多个）和结果呈现任务（单个）的组合定义的。以如下AISBench命令为例：

```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --summarizer example
```

此命令没有指定其他命令行，默认是一个精度评测场景的任务，其中：

- `--models`指定了模型任务，即`vllm_api_general_chat`模型任务。
- `--datasets`指定了数据集任务，即`demo_gsm8k_gen_4_shot_cot_chat_prompt`数据集任务。
- `--summarizer`指定了结果呈现任务，即`example`结果呈现任务(不指定`--summarizer`精度评测场景默认使用`example`任务)，一般使用默认，不需要在命令行中指定。

多任务测评请参考：📚 精度场景的[多任务测评](../base_tutorials/scenes_intro/accuracy_benchmark.md#多任务测评) 和 性能场景的[多任务测评](../base_tutorials/scenes_intro/performance_benchmark.md#多任务测评)。

如需自行组合测评任务，实现更灵活的测评方式，可参考：📚 [自定义配置文件运行AISBench](../advanced_tutorials/run_custom_config.md#自定义配置文件运行AISBench)。

所选模型任务`vllm_api_general_chat`、数据集任务`demo_gsm8k_gen_4_shot_cot_chat_prompt`和结果呈现任务`example`的具体信息(简介，使用约束等)可以分别从如下链接中查询含义：

- `--models`: 📚 [服务化推理后端](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/models.html#id2)
- `--datasets`: 📚 [开源数据集](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/get_started/datasets.html#id3) → 📚 [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/demo/README.md)
- `--summarizer`: 📚 [结果汇总任务](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/summarizer.html)

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

- 快速入门中数据集任务配置文件`demo_gsm8k_gen_4_shot_cot_chat_prompt.py`不需要做额外修改，数据集任务配置文件内容介绍可参考📚 [配置开源数据集](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/datasets.html#id6)

模型配置文件`vllm_api_general_chat.py`中包含了模型运行相关的配置内容，是需要依据实际情况修改的。快速入门中需要修改的内容用注释标明。

> 💡 **提示**：模型配置中的部分参数（如 `host_ip`、`host_port`、`model`、`url`、`max_out_len`、`generation_kwargs` 等）无需修改配置文件，可直接通过命令行覆盖，例如：
>
> ```bash
> ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --host-ip 127.0.0.1 --host-port 8000
> ```
>
> 命令行显式指定的参数会覆盖本次执行的所有模型配置中对应字段；仅覆盖配置中**已存在的字段**，未指定的参数保持配置文件原值。更多可覆盖参数及覆盖范围说明请参考 📚 [用户配置参数 - API 模型通用覆盖参数](../base_tutorials/all_params/cli_args.md#api-模型通用覆盖参数)。

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
        use_timestamp=False,      # 是否按数据集中 timestamp 调度请求，适用于含 timestamp 的数据集（如 Mooncake Trace）
        retry=2,                  # 每个请求最大重试次数
        api_key="",               # 自定义API key，默认是空字符串
        host_ip="localhost",      # 指定推理服务的IP
        host_port=8080,           # 指定推理服务的端口
        url="",                     # 自定义访问推理服务的URL路径(当base url不是http://host_ip:host_port的组合时需要配置, 配置后host_ip和host_port会被忽略)
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

修改好配置文件后，执行命令启动服务化精度评测：

```bash
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt
```

:::
::::

## 查看任务执行细节

执行AISBench命令后，任务管理界面会在命令行实时刷新显示任务执行状态（键盘按"P"键可以暂停/恢复刷新，用于复制看板信息，再按"P"键可以继续刷新）。任务管理界面支持同时监控多个任务的详细执行状态，包括任务名称、进度、时间成本、状态、日志路径、扩展参数等信息，例如：

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
> `ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug`

`Base path`（`outputs/default/20250628_151326`）下包含了所有任务的执行细节，命令执行结束后所有的执行细节如下：

```shell
20250628_151326/
├── configs # 模型任务、数据集任务和结构呈现任务对应的配置文件合成的一个配置
│   └── 20250628_151326_29317.py
├── logs # 执行过程中日志，命令中如果加--debug，不会有过程日志落盘（都直接打印出来了）
│   ├── eval
│   │   └── vllm-api-general-chat
│   │       └── demo_gsm8k.out # 基于predictions/文件夹下的推理结果的精度评测过程的日志
│   └── infer
│       └── vllm-api-general-chat
│           └── demo_gsm8k.out # 推理过程日志
├── predictions
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # 推理结果（推理服务返回的所有输出）
├── results
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # 精度评测计算的原始分数
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
