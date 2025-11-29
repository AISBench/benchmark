# 模型配置说明
AISBench Benchmark 支持两类模型后端：
- [服务化推理后端](#服务化推理后端)
- [本地模型后端](#本地模型后端)

> ⚠️ 注意： 不能同时指定两种后端。
## 服务化推理后端
AISBench Benchmark 支持多种服务化推理后端，包括 vLLM、SGLang、Triton、MindIE、TGI 等。这些后端通过暴露的 HTTP API 接口接收推理请求并返回结果。（目前不支持 HTTPS 接口）

以在 GPU 上部署的 vLLM 推理服务为例，您可以参考 [vLLM 官方文档](https://docs.vllm.ai/en/stable/getting_started/quickstart.html) 启动服务。

不同服务化后端对应的模型配置如下：
| 模型配置名称| 简介| 使用前提| 接口类型 | 支持的数据集 Prompt 格式 | 配置文件路径|
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| `vllm_api_general` | 通过 vLLM 兼容 OpenAI 的 API 访问推理服务，接口为 `v1/completions`| 基于 vLLM 版本支持 `v1/completions` 子服务| 文本接口 | 字符串格式| [vllm_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)|
| `vllm_api_general_stream`| 流式访问 vLLM 推理服务，接口为 `v1/completions`| 基于 vLLM 版本支持 `v1/completions` 子服务| 流式接口 | 字符串格式| [vllm_api_general_stream.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py) |
| `vllm_api_general_chat`  | 通过 vLLM 兼容 OpenAI 的 API 访问推理服务，接口为 `v1/chat/completions` | 基于 vLLM 版本支持 `v1/chat/completions` 子服务 | 文本接口 | 字符串格式、对话格式、多模态格式 | [vllm_api_general_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py)  |
| `vllm_api_stream_chat`| 流式访问 vLLM 推理服务，接口为 `v1/chat/completions`| 基于 vLLM 版本支持 `v1/chat/completions` 子服务 | 流式接口 | 字符串格式、对话格式、多模态格式 | [vllm_api_stream_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py) |
| `vllm_api_stream_chat_multiturn`| 多轮对话场景的流式访问 vLLM 推理服务，接口为 `v1/chat/completions`| 基于 vLLM 版本支持 `v1/chat/completions` 子服务 | 流式接口 | 对话格式 | [vllm_api_stream_chat_multiturn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat_multiturn.py) |
| `vllm_api_function_call_chat`| function call精度测评场景访问 vLLM 推理服务的API ，接口为 `v1/chat/completions`（只适用于[BFCL](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/BFCL/README.md)测评场景| 基于 vLLM 版本支持 `v1/chat/completions` 子服务 | 文本接口 | 对话格式 | [vllm_api_function_call_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_function_call_chat.py) |
| `vllm_api_old`  | 通过 vLLM 兼容 API 访问推理服务，接口为 `generate`| 基于 vLLM 版本支持 `generate` 子服务| 文本接口 | 字符串格式、多模态格式| [vllm_api_old.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_old.py)|
| `mindie_stream_api_general` | 通过 MindIE 流式 API 访问推理服务，接口为 `infer`| 基于 MindIE 版本支持 `infer` 子服务 | 流式接口 | 字符串格式、多模态格式| [mindie_stream_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/mindie_api/mindie_stream_api_general.py) |
| `triton_api_general`  | 通过 Triton API 访问推理服务，接口为 `v2/models/{model name}/generate`  | 启动支持 Triton API 的推理服务| 文本接口 | 字符串格式、多模态格式| [triton_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/triton_api/triton_api_general.py) |
| `triton_stream_api_general` | 通过 Triton 流式 API 访问推理服务，接口为 `v2/models/{model name}/generate_stream` | 启动支持 Triton API 的推理服务| 流式接口 | 字符串格式、多模态格式 | [triton_stream_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/triton_api/triton_stream_api_general.py) |
| `tgi_api_general`  | 通过 TGI API 访问推理服务，接口为 `generate`| 启动支持 TGI API 的推理服务| 文本接口 | 字符串格式、多模态格式| [tgi_api_general](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/tgi_api/tgi_api_general.py)|
| `tgi_stream_api_general` | 通过 TGI 流式 API 访问推理服务，接口为 `generate_stream`| 启动支持 TGI API 的推理服务| 流式接口 | 字符串格式、多模态格式| [tgi_stream_api_general](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/tgi_api/tgi_stream_api_general.py) |

### 服务化推理后端配置参数说明
服务化推理后端配置文件采用Python语法格式配置，示例如下：
```python
from ais_bench.benchmark.models import VLLMCustomAPI

models = [
    dict(
        attr="service",
        type=VLLMCustomAPI,
        abbr='vllm-api-general',
        path="",                    # 指定模型序列化词表文件绝对路径（精度测试场景一般不需要配置）
        model="",        # 指定服务端已加载模型名称，依据实际VLLM推理服务拉取的模型名称配置（配置成空字符串会自动获取）
        stream=False,    # 是否为流式接口
        request_rate = 0,           # 请求发送频率，每1/request_rate秒发送1个请求给服务端，小于0.1则一次性发送所有请求
        retry = 2,                  # 每个请求最大重试次数
        api_key="",               # 自定义API key，默认是空字符串
        host_ip = "localhost",      # 指定推理服务的IP
        host_port = 8080,           # 指定推理服务的端口
        url="",                     # 自定义访问推理服务的URL路径(当base url不是http://host_ip:host_port的组合时需要配置)
        max_out_len = 512,          # 推理服务输出的token的最大数量
        batch_size=1,               # 请求发送的最大并发数
        trust_remote_code=False,    # tokenizer是否信任远程代码，默认False;
        generation_kwargs = dict(   # 模型推理参数，参考VLLM文档配置，AISBench评测工具不做处理，在发送的请求中附带
            temperature = 0.01,
            ignore_eos=False,
        )
    )
]

```

服务化推理后端可配置参数说明如下：
| 参数名称 | 参数类型 | 配置说明 |
|----------|-----------|-------------|
| `attr` | String | 推理后端类型标识，固定为 `service`（服务化推理）或 `local`（本地模型），不可配置 |
| `type` | Python Class | API 类型类名，由系统自动关联，用户无需手动配置，参考 [服务化推理后端](#服务化推理后端) |
| `abbr` | String | 服务化任务的唯一标识，用于区分不同任务，英文字符与短横线组合，例如：`vllm-api-general-chat` |
| `path` | String | Tokenizer 路径，通常与模型路径相同，使用 `AutoTokenizer.from_pretrained(path)` 加载。指定可访问的本地路径，例如：`/weight/DeepSeek-R1` |
| `model` | String | 服务端可访问的模型名称，必须与服务化部署时指定的名称一致 |
| `model_name` | String | 仅适用于 Triton 服务，拼接为 endpoint 的 URI `/v2/models/{modelname}/{infer、generate、generate_stream}`，应与部署时名称一致 |
| `stream` | Boolean | API模型推理接口类型，默认为False，表示非流式接口，当为True时表示流式接口（具体请参考🔗[服务化推理后端](#服务化推理后端)）|
| `request_rate` | Float | 请求发送速率（单位：秒），每隔 `1/request_rate` 秒发送一个请求；压测场景下表示每秒新增的服务端连接数；若小于 0.1 表示不限制请求发送速率。合法范围：[0, 64000]。当`traffic_cfg`项配置启用时，该项功能可能被覆盖 （具体原因请参考 🔗 [请求速率(RPS)分布控制及可视化说明中的参数解读章节](../../advanced_tutorials/rps_distribution.md#参数解读)）|
| `traffic_cfg` | Dict | 请求发送速率波动控制参数（具体使用说明请参考 🔗 [请求速率(RPS)分布控制及可视化说明](../../advanced_tutorials/rps_distribution.md)），不填写此项默认不启用该功能。 |
| `retry` | Int | 连接服务端失败后的最大重试次数。合法范围：[0, 1000] |
| `api_key` | String | 自定义API key，默认是空字符串。仅支持 `VLLMCustomAPI` 和 `VLLMCustomAPIChat` 模型类型。 |
| `host_ip` | String | 服务端 IP 地址，支持合法 IPv4 或 IPv6，例如：`127.0.0.1` |
| `host_port` | Int | 服务端端口号，应与服务化部署指定的端口一致 |
| `url` | String | 自定义访问推理服务的URL路径(当base url不是http/https://host_ip:host_port的组合时需要配置，配置后host_ip和host_port将被忽略) ，例如当`models`的`type`为`VLLMCustomAPI`时，配置`url`为`https://xxxxxxx/yyyy/`，实际请求访问的URL为`https://xxxxxxx/yyyy/v1/completions`|
| `max_out_len` | Int | 推理响应的最大输出长度，实际长度可能受服务端限制。合法范围：(0, 131072] |
| `batch_size` | Int | 请求的并发批处理大小。合法范围：(0, 64000] |
| `trust_remote_code` | Boolean | tokenizer是否信任远程代码，默认False; |
| `generation_kwargs` | Dict | 推理生成参数配置，依赖具体的服务化后端和接口类型。注意：当前不支持 `best_of` 和 `n` 等多次采样参数，但支持通过`num_return_sequences`参数进行多次独立推理(具体请参考🔗[Text Generation 文档](https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.num_return_sequences)中`num_return_sequences`的作用) |
| `returns_tool_calls` | Bool | 控制函数调用信息的提取方式。当设置为True时，系统从API响应的`tool_calls`字段中提取函数调用信息；当设置为False时，系统从`content`字段中解析函数调用信息 |
| `pred_postprocessor` | Dict | 模型输出结果的后处理配置。用于对原始模型输出进行格式化、清理或转换，以满足特定评估任务的要求 |

**注意事项：**
- `request_rate` 受硬件性能影响，可通过增加  📚 [WORKERS_NUM](./cli_args.md#配置常量文件参数) 提高并发能力。
- `request_rate` 功能可能被`traffic_cfg`项覆盖，具体原因请参考 🔗 [请求速率(RPS)分布控制及可视化说明中的参数解读章节](../../advanced_tutorials/rps_distribution.md#参数解读)。
- `batch_size` 设置过大可能导致 CPU 占用过高，请根据硬件条件合理配置。
- 服务化推理评测 API 默认使用的服务地址为 `localhost:8080`。实际使用时需根据实际部署修改为服务化后端的 IP 和端口。

## 本地模型后端
|模型配置名称|简介|使用前提|支持的prompt格式(字符串格式或对话格式)|对应源码配置文件路径|
| --- | --- | --- | --- | --- |
|`hf_base_model`|HuggingFace Base 模型后端|已安装评测工具基础依赖，需在配置文件中指定 HuggingFace 模型权重路径（当前不支持自动下载）|字符串格式|[hf_base_model](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/hf_models/hf_base_model.py)|
|`hf_chat_model`|	HuggingFace Chat 模型后端|已安装评测工具基础依赖，需在配置文件中指定 HuggingFace 模型权重路径（当前不支持自动下载）|对话格式|[hf_chat_model](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/hf_models/hf_chat_model.py)|
|`hf_qwenvl_model`|	HuggingFace Chat QwenVL模型后端|已安装评测工具基础依赖，需在配置文件中指定 HuggingFace 模型权重路径（当前不支持自动下载）|对话格式|[hf_qwenvl_model](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/hf_models/hf_qwenvl_model.py)|
|`vllm_offline_vl_model`|	vllm Chat QwenVL离线推理模型后端|已安装评测工具基础依赖，需在配置文件中指定模型模型权重路径（当前不支持自动下载）|对话格式|[vllm_offline_vl_model](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_offline_models/vllm_offline_vl_model.py)|

### 本地huggingface模型后端配置参数说明
本地huggingface模型后端配置文件采用Python语法格式配置，示例如下：
```python
from ais_bench.benchmark.models import HuggingFacewithChatTemplate

models = [
    dict(
        attr="local",                       # 后端类型标识
        type=HuggingFacewithChatTemplate,   # 模型类型
        abbr='hf-chat-model',               # 唯一标识
        path='THUDM/chatglm-6b',            # 模型权重路径
        tokenizer_path='THUDM/chatglm-6b',  # Tokenizer 路径
        model_kwargs=dict(                  # 模型加载参数
            device_map="auto",
            trust_remote_code=True
        ),
        max_out_len=512,                    # 最大输出长度
        batch_size=1,                       # 请求并发数
        generation_kwargs=dict(             # 生成参数
            temperature=0.5,
            top_k=10,
            top_p=0.95,
            seed=None,
            repetition_penalty=1.03,
        )
    )
]
```

本地模型推理后端可配置参数说明如下：
| 参数名称 | 参数类型 | 说明与配置 |
|----------|-----------|-------------|
| `attr` | String | 后端类型标识，固定为 `local`（本地模型）或 `service`（服务化推理） |
| `type` | Python Class | 模型类名称，由系统自动关联，用户无需手动配置 |
| `abbr` | String | 本地任务的唯一标识，用于区分多任务。建议使用英文与短横线组合，如：`hf-chat-model` |
| `path` | String | 模型权重路径，需为本地可访问路径。使用 `AutoModel.from_pretrained(path)` 加载 |
| `tokenizer_path` | String | Tokenizer 路径，通常与模型路径一致。使用 `AutoTokenizer.from_pretrained(tokenizer_path)` 加载 |
| `tokenizer_kwargs` | Dict | Tokenizer 加载参数，参考 🔗 [PreTrainedTokenizerBase 文档](https://huggingface.co/docs/transformers/v4.50.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase) |
| `model_kwargs` | Dict | 模型加载参数，参考 🔗 [AutoModel 配置](https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/auto#transformers.AutoConfig.from_pretrained) |
| `generation_kwargs` | Dict | 推理生成参数，参考 🔗 [Text Generation 文档](https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation) |
| `run_cfg` | Dict | 运行配置，包含 `num_gpus`（使用的 GPU 数量）与 `num_procs`（使用的机器进程数） |
| `max_out_len` | Int | 推理生成的最大输出 Token 数量，合法范围：(0, 131072] |
| `batch_size` | Int | 推理请求的批处理大小，合法范围：(0, 64000] |
| `max_seq_len` | Int | 最大输入序列长度，合法范围：(0, 131072] |
| `batch_padding` | Bool | 是否启用批量 padding。设置为 `True` 或 `False` |

### 本地vllm离线推理模型后端配置参数说明
本地vllm离线推理模型后端配置文件采用Python语法格式配置，示例如下：
```python
from ais_bench.benchmark.models import VLLMOfflineVLModel

models = [
    dict(
        attr="local",                    # 后端类型标识
        type=VLLMOfflineVLModel,         # 模型类型
        abbr='vllm-offline-vl-model',    # 唯一标识
        path = "",                       # 模型权重路径
        model_kwargs=dict(               # 模型初始化参数, 可参考 https://docs.vllm.com.cn/en/latest/serving/engine_args.html#
            max_num_seqs=5,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 24},
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        ),
        sample_kwargs=dict(              # 模型推理采样参数, 可参考 https://docs.vllm.ai/en/v0.6.5/dev/sampling_params.html
            temperature=0.0,
            stop_token_ids=None
        ),
        vision_kwargs=dict(              # 多模态输入参数，可参考 https://docs.vllm.ai/en/v0.7.3/getting_started/examples/vision_language.html
            min_pixels=1280 * 28 * 28,
            max_pixels=16384 * 28 * 28,
        ),
        max_out_len=512,                 # 最大输出长度
        batch_size=1,                    # 请求并发数
    )
]
```

本地vllm离线推理模型后端可配置参数说明如下：
| 参数名称 | 参数类型 | 说明与配置 |
|----------|-----------|-------------|
| `attr` | String | 后端类型标识，固定为 `local`（本地模型）或 `service`（服务化推理） |
| `type` | Python Class | 模型类名称，由系统自动关联，用户无需手动配置 |
| `abbr` | String | 本地任务的唯一标识，用于区分多任务。建议使用英文与短横线组合，如：`vllm-offline-vl-model` |
| `path` | String | 模型权重路径，需为本地可访问路径。使用 `vllm.LLM(model=path)` 加载 |
| `model_kwargs` | Dict | 模型加载参数，参考 🔗 [LLM 模型配置](https://docs.vllm.com.cn/en/latest/serving/engine_args.html#) |
| `sample_kwargs` | Dict | 模型推理采样参数，参考 🔗 [sample parameter配置](https://docs.vllm.ai/en/v0.6.5/dev/sampling_params.html) |
| `vision_kwargs` | Dict | 多模态输入参数，参考 🔗 [多模态推理举例](https://docs.vllm.ai/en/v0.7.3/getting_started/examples/vision_language.html) |
| `max_out_len` | Int | 推理生成的最大输出 Token 数量，合法范围：(0, 131072] |
| `batch_size` | Int | 推理请求的批处理大小，合法范围：(0, 64000] |