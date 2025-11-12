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
| `vllm_api_general` | 通过 vLLM 兼容 OpenAI 的 API 访问推理服务，接口为 `v1/completions`| 基于 vLLM 版本支持 `v1/completions` 子服务| 文本接口 | 字符串格式| [vllm_api_general.py](../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)|
| `vllm_api_general_stream`| 流式访问 vLLM 推理服务，接口为 `v1/completions`| 基于 vLLM 版本支持 `v1/completions` 子服务| 流式接口 | 字符串格式| [vllm_api_general_stream.py](../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py) |
| `vllm_api_general_chat`  | 通过 vLLM 兼容 OpenAI 的 API 访问推理服务，接口为 `v1/chat/completions` | 基于 vLLM 版本支持 `v1/chat/completions` 子服务 | 文本接口 | 字符串格式、对话格式、多模态格式 | [vllm_api_general_chat.py](../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py)  |
| `vllm_api_stream_chat`| 流式访问 vLLM 推理服务，接口为 `v1/chat/completions`| 基于 vLLM 版本支持 `v1/chat/completions` 子服务 | 流式接口 | 字符串格式、对话格式、多模态格式 | [vllm_api_stream_chat.py](../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py) |
| `vllm_api_stream_chat_multiturn`| 多轮对话场景的流式访问 vLLM 推理服务，接口为 `v1/chat/completions`| 基于 vLLM 版本支持 `v1/chat/completions` 子服务 | 流式接口 | 对话格式 | [vllm_api_stream_chat_multiturn.py](../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat_multiturn.py) |
| `vllm_api_function_call_chat`| function call精度测评场景访问 vLLM 推理服务的API ，接口为 `v1/chat/completions`（只适用于[BFCL](../../ais_bench/benchmark/configs/datasets/BFCL/README.md)测评场景)| 基于 vLLM 版本支持 `v1/chat/completions` 子服务 | 文本接口 | 对话格式 | [vllm_api_function_call_chat.py](../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_function_call_chat.py) |
| `vllm_api_old`  | 通过 vLLM 兼容 API 访问推理服务，接口为 `generate`| 基于 vLLM 版本支持 `generate` 子服务| 文本接口 | 字符串格式、多模态格式| [vllm_api_old.py](../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_old.py)|
| `mindie_stream_api_general` | 通过 MindIE 流式 API 访问推理服务，接口为 `infer`| 基于 MindIE 版本支持 `infer` 子服务 | 流式接口 | 字符串格式、多模态格式| [mindie_stream_api_general.py](../../ais_bench/benchmark/configs/models/mindie_api/mindie_stream_api_general.py) |
| `triton_api_general`  | 通过 Triton API 访问推理服务，接口为 `v2/models/{model name}/generate`  | 启动支持 Triton API 的推理服务| 文本接口 | 字符串格式、多模态格式| [triton_api_general.py](../../ais_bench/benchmark/configs/models/triton_api/triton_api_general.py) |
| `triton_stream_api_general` | 通过 Triton 流式 API 访问推理服务，接口为 `v2/models/{model name}/generate_stream` | 启动支持 Triton API 的推理服务| 流式接口 | 字符串格式、多模态格式 | [triton_stream_api_general.py](../../ais_bench/benchmark/configs/models/triton_api/triton_stream_api_general.py) |
| `tgi_api_general`  | 通过 TGI API 访问推理服务，接口为 `generate`| 启动支持 TGI API 的推理服务| 文本接口 | 字符串格式、多模态格式| [tgi_api_general](../../ais_bench/benchmark/configs/models/tgi_api/tgi_api_general.py)|
| `tgi_stream_api_general` | 通过 TGI 流式 API 访问推理服务，接口为 `generate_stream`| 启动支持 TGI API 的推理服务| 流式接口 | 字符串格式、多模态格式| [tgi_stream_api_general](../../ais_bench/benchmark/configs/models/tgi_api/tgi_stream_api_general.py) |

### 服务化推理后端配置参数说明
服务化推理后端配置文件采用Python语法格式配置，示例如下：
```python
from ais_bench.benchmark.models import VLLMCustomAPI

models = [
    dict(
        attr="service",             # 后端类型标识
        type=VLLMCustomAPI,         # API 类型
        abbr='vllm-api-general',    # 唯一标识
        path="/weight/DeepSeek-R1", # 模型路径
        model="DeepSeek-R1",        # 模型名称
        request_rate=0,             # 请求速率
        retry=2,                    # 最大重试次数
        host_ip="localhost",        # 推理服务 IP
        host_port=8080,             # 推理服务端口
        max_out_len=512,            # 最大输出长度
        batch_size=1,               # 请求并发数
        generation_kwargs=dict(     # 后处理参数
            temperature=0.5,
            top_k=10,
            top_p=0.95,
            seed=None,
            repetition_penalty=1.03,
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
| `request_rate` | Float | 请求发送速率（单位：秒），每隔 `1/request_rate` 秒发送一个请求；若小于 0.1 则自动合并为批量发送。合法范围：[0, 64000] |
| `retry` | Int | 连接服务端失败后的最大重试次数。合法范围：[0, 1000] |
| `host_ip` | String | 服务端 IP 地址，支持合法 IPv4 或 IPv6，例如：`127.0.0.1` |
| `host_port` | Int | 服务端端口号，应与服务化部署指定的端口一致 |
| `max_out_len` | Int | 推理响应的最大输出长度，实际长度可能受服务端限制。合法范围：(0, 131072] |
| `batch_size` | Int | 请求的并发批处理大小。合法范围：(0, 64000] |
| `generation_kwargs` | Dict | 推理生成参数配置，依赖具体的服务化后端和接口类型。注意：当前不支持 `best_of` 和 `n` 等多次采样参数，但支持通过`num_return_sequences`参数进行多次独立推理(具体请参考🔗[Text Generation 文档](https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.num_return_sequences\(int,)中`num_return_sequences`的作用) |
| `returns_tool_calls` | Bool | 控制函数调用信息的提取方式。当设置为True时，系统从API响应的`tool_calls`字段中提取函数调用信息；当设置为False时，系统从`content`字段中解析函数调用信息 |
| `pred_postprocessor` | Dict | 模型输出结果的后处理配置。用于对原始模型输出进行格式化、清理或转换，以满足特定评估任务的要求 |

**注意事项：**
- `request_rate` 受硬件性能影响，可通过增加  📚 [WORKERS_NUM](./cli_args.md#配置常量文件参数) 提高并发能力。
- `batch_size` 设置过大可能导致 CPU 占用过高，请根据硬件条件合理配置。
- 服务化推理评测 API 默认使用的服务地址为 `localhost:8080`。实际使用时需根据实际部署修改为服务化后端的 IP 和端口。

## 本地模型后端
|模型配置名称|简介|使用前提|支持的prompt格式(字符串格式或对话格式)|对应源码配置文件路径|
| --- | --- | --- | --- | --- |
|`hf_base_model`|HuggingFace Base 模型后端|已安装评测工具基础依赖，需在配置文件中指定 HuggingFace 模型权重路径（当前不支持自动下载）|字符串格式|[hf_base_model](../../ais_bench/benchmark/configs/models/hf_models/hf_base_model.py)|
|`hf_chat_model`|	HuggingFace Chat 模型后端|已安装评测工具基础依赖，需在配置文件中指定 HuggingFace 模型权重路径（当前不支持自动下载）|对话格式|[hf_chat_model](../../ais_bench/benchmark/configs/models/hf_models/hf_chat_model.py)|

### 本地模型后端配置参数说明
本地模型后端配置文件采用Python语法格式配置，示例如下：
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