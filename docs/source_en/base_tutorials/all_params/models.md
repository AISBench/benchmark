# Model Configuration Instructions
AISBench Benchmark supports two types of model backends:
- [Service-Oriented Inference Backend](#service-oriented-inference-backend)
- [Local Model Backend](#local-model-backend)

> ⚠️ **Note**: The two types of backends cannot be specified simultaneously.


## Service-Oriented Inference Backend
AISBench Benchmark supports multiple service-oriented inference backends, including vLLM, SGLang, Triton, MindIE, TGI, etc. These backends receive inference requests and return results through exposed HTTP API interfaces. (HTTPS interfaces are not supported currently.)

Taking the vLLM inference service deployed on GPU as an example, you can refer to the [vLLM Official Documentation](https://docs.vllm.ai/en/stable/getting_started/quickstart.html) to start the service.

The model configurations corresponding to different service-oriented backends are as follows:

| Model Configuration Name | Description | Prerequisites for Use | Interface Type | Supported Dataset Prompt Formats | Configuration File Path |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| `vllm_api_general` | Access the inference service via vLLM's OpenAI-compatible API, with the interface `v1/completions` | The vLLM version used supports the `v1/completions` sub-service | Text Interface | String Format | [vllm_api_general.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py) |
| `vllm_api_general_stream` | Access the vLLM inference service in streaming mode, with the interface `v1/completions` | The vLLM version used supports the `v1/completions` sub-service | Streaming Interface | String Format | [vllm_api_general_stream.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py) |
| `vllm_api_general_chat` | Access the inference service via vLLM's OpenAI-compatible API, with the interface `v1/chat/completions` | The vLLM version used supports the `v1/chat/completions` sub-service | Text Interface | String Format, Dialogue Format, Multimodal Format | [vllm_api_general_chat.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py) |
| `vllm_api_stream_chat` | Access the vLLM inference service in streaming mode, with the interface `v1/chat/completions` | The vLLM version used supports the `v1/chat/completions` sub-service | Streaming Interface | String Format, Dialogue Format, Multimodal Format | [vllm_api_stream_chat.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py) |
| `vllm_api_stream_chat_multiturn` | Access the vLLM inference service in streaming mode for multi-turn dialogue scenarios, with the interface `v1/chat/completions` | The vLLM version used supports the `v1/chat/completions` sub-service | Streaming Interface | Dialogue Format | [vllm_api_stream_chat_multiturn.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat_multiturn.py) |
| `vllm_api_function_call_chat` | API for accessing the vLLM inference service in function call accuracy evaluation scenarios, with the interface `v1/chat/completions` (only applicable to the [BFCL](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/BFCL/README_en.md) evaluation scenario) | The vLLM version used supports the `v1/chat/completions` sub-service | Text Interface | Dialogue Format | [vllm_api_function_call_chat.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_function_call_chat.py) |
| `vllm_api_old` | Access the inference service via vLLM-compatible API, with the interface `generate` | The vLLM version used supports the `generate` sub-service | Text Interface | String Format, Multimodal Format | [vllm_api_old.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_old.py) |
| `mindie_stream_api_general` | Access the inference service via MindIE streaming API, with the interface `infer` | The MindIE version used supports the `infer` sub-service | Streaming Interface | String Format, Multimodal Format | [mindie_stream_api_general.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/mindie_api/mindie_stream_api_general.py) |
| `triton_api_general` | Access the inference service via Triton API, with the interface `v2/models/{model name}/generate` | Start an inference service that supports Triton API | Text Interface | String Format, Multimodal Format | [triton_api_general.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/triton_api/triton_api_general.py) |
| `triton_stream_api_general` | Access the inference service via Triton streaming API, with the interface `v2/models/{model name}/generate_stream` | Start an inference service that supports Triton API | Streaming Interface | String Format, Multimodal Format | [triton_stream_api_general.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/triton_api/triton_stream_api_general.py) |
| `tgi_api_general` | Access the inference service via TGI API, with the interface `generate` | Start an inference service that supports TGI API | Text Interface | String Format, Multimodal Format | [tgi_api_general](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/tgi_api/tgi_api_general.py) |
| `tgi_stream_api_general` | Access the inference service via TGI streaming API, with the interface `generate_stream` | Start an inference service that supports TGI API | Streaming Interface | String Format, Multimodal Format | [tgi_stream_api_general](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/tgi_api/tgi_stream_api_general.py) |


### Parameter Description for Service-Oriented Inference Backend Configuration
The configuration file for the service-oriented inference backend is configured using Python syntax, as shown in the example below:
```python
from ais_bench.benchmark.models import VLLMCustomAPI

models = [
    dict(
        attr="service",             # Backend type identifier
        type=VLLMCustomAPI,         # API type
        abbr='vllm-api-general',    # Unique identifier
        path="/weight/DeepSeek-R1", # Model path
        model="DeepSeek-R1",        # Model name
        request_rate=0,             # Request rate
        retry=2,                    # Maximum number of retries
        host_ip="localhost",        # Inference service IP
        host_port=8080,             # Inference service port
        max_out_len=512,            # Maximum output length
        batch_size=1,               # Request concurrency count
        generation_kwargs=dict(     # Post-processing parameters
            temperature=0.5,
            top_k=10,
            top_p=0.95,
            seed=None,
            repetition_penalty=1.03,
        )
    )
]

```

The description of configurable parameters for the service-oriented inference backend is as follows:

| Parameter Name | Parameter Type | Configuration Description |
|----------|-----------|-------------|
| `attr` | String | Identifier for the inference backend type, fixed as `service` (service-oriented inference) or `local` (local model); cannot be customized |
| `type` | Python Class | Class name of the API type, automatically associated by the system; no manual configuration is required by the user. Refer to [Service-Oriented Inference Backend](#service-oriented-inference-backend) |
| `abbr` | String | Unique identifier for the service-oriented task, used to distinguish different tasks. It consists of English characters and hyphens, e.g., `vllm-api-general-chat` |
| `path` | String | Tokenizer path, usually the same as the model path. The Tokenizer is loaded using `AutoTokenizer.from_pretrained(path)`. Specify an accessible local path, e.g., `/weight/DeepSeek-R1` |
| `model` | String | Name of the model accessible on the server, which must be consistent with the name specified during service-oriented deployment |
| `model_name` | String | Applicable only to Triton services. It is concatenated into the endpoint URI `/v2/models/{modelname}/{infer, generate, generate_stream}` and must be consistent with the name used during deployment |
| `request_rate` | Float | Request sending rate (unit: requests per second). A request is sent every `1/request_rate` seconds; if the value is less than 0.1, requests are automatically merged and sent in batches. Valid range: [0, 64000]. When the `traffic_cfg` item is enabled, this function may be overwritten (for specific reasons, refer to 🔗 [Parameter Interpretation Section in the Description of Request Rate (RPS) Distribution Control and Visualization](../../advanced_tutorials/rps_distribution.md#parameter-interpretation)) |
| `traffic_cfg` | Dict | Parameters for controlling fluctuations in the request sending rate (for detailed usage instructions, refer to 🔗 [Description of Request Rate (RPS) Distribution Control and Visualization](../../advanced_tutorials/rps_distribution.md)). If this item is not filled in, the function is disabled by default |
| `retry` | Int | Maximum number of retries after failing to connect to the server. Valid range: [0, 1000] |
| `host_ip` | String | Server IP address, supporting valid IPv4 or IPv6, e.g., `127.0.0.1` |
| `host_port` | Int | Server port number, which must be consistent with the port specified during service-oriented deployment |
| `max_out_len` | Int | Maximum output length of the inference response; the actual length may be limited by the server. Valid range: (0, 131072] |
| `batch_size` | Int | Batch size for concurrent requests. Valid range: (0, 64000] |
| `generation_kwargs` | Dict | Configuration of inference generation parameters, depending on the specific service-oriented backend and interface type. Note: Currently, multi-sampling parameters such as `best_of` and `n` are not supported, but multiple independent inferences can be performed using the `num_return_sequences` parameter (for details, refer to 🔗 [the role of `num_return_sequences` in the Text Generation Documentation](https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.num_return_sequences\(int,)) |
| `returns_tool_calls` | Bool | Controls the extraction method of function call information. When set to `True`, the system extracts function call information from the `tool_calls` field of the API response; when set to `False`, the system parses function call information from the `content` field |
| `pred_postprocessor` | Dict | Post-processing configuration for model output results. It is used to format, clean, or convert the original model output to meet the requirements of specific evaluation tasks |


**Precautions**:
- `request_rate` is affected by hardware performance. You can increase 📚 [WORKERS_NUM](./cli_args.md#configuration-constant-file-parameters) to improve concurrency capability.
- The function of `request_rate` may be overwritten by the `traffic_cfg` item. For specific reasons, refer to 🔗 [Parameter Interpretation Section in the Description of Request Rate (RPS) Distribution Control and Visualization](../../advanced_tutorials/rps_distribution.md#parameter-interpretation).
- Setting `batch_size` too large may result in high CPU usage. Please configure it reasonably based on hardware conditions.
- The default service address used by the service-oriented inference evaluation API is `localhost:8080`. In actual use, you need to modify it to the IP and port of the service-oriented backend according to the actual deployment.


## Local Model Backend

| Model Configuration Name | Description | Prerequisites for Use | Supported Prompt Formats (String Format or Dialogue Format) | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- |
| `hf_base_model` | HuggingFace Base Model Backend | The basic dependencies of the evaluation tool have been installed; the HuggingFace model weight path must be specified in the configuration file (automatic download is not supported currently) | String Format | [hf_base_model](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/hf_models/hf_base_model.py) |
| `hf_chat_model` | HuggingFace Chat Model Backend | The basic dependencies of the evaluation tool have been installed; the HuggingFace model weight path must be specified in the configuration file (automatic download is not supported currently) | Dialogue Format | [hf_chat_model](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/hf_models/hf_chat_model.py) |


### Parameter Description for Local Model Backend Configuration
The configuration file for the local model backend is configured using Python syntax, as shown in the example below:
```python
from ais_bench.benchmark.models import HuggingFacewithChatTemplate

models = [
    dict(
        attr="local",                       # Backend type identifier
        type=HuggingFacewithChatTemplate,   # Model type
        abbr='hf-chat-model',               # Unique identifier
        path='THUDM/chatglm-6b',            # Model weight path
        tokenizer_path='THUDM/chatglm-6b',  # Tokenizer path
        model_kwargs=dict(                  # Model loading parameters
            device_map="auto",
            trust_remote_code=True
        ),
        max_out_len=512,                    # Maximum output length
        batch_size=1,                       # Request concurrency count
        generation_kwargs=dict(             # Generation parameters
            temperature=0.5,
            top_k=10,
            top_p=0.95,
            seed=None,
            repetition_penalty=1.03,
        )
    )
]
```

The description of configurable parameters for the local model inference backend is as follows:

| Parameter Name | Parameter Type | Description & Configuration |
|----------|-----------|-------------|
| `attr` | String | Identifier for the backend type, fixed as `local` (local model) or `service` (service-oriented inference) |
| `type` | Python Class | Model class name, automatically associated by the system; no manual configuration is required by the user |
| `abbr` | String | Unique identifier for the local task, used to distinguish multiple tasks. It is recommended to use a combination of English characters and hyphens, e.g., `hf-chat-model` |
| `path` | String | Model weight path, which must be an accessible local path. The model is loaded using `AutoModel.from_pretrained(path)` |
| `tokenizer_path` | String | Tokenizer path, usually the same as the model path. The Tokenizer is loaded using `AutoTokenizer.from_pretrained(tokenizer_path)` |
| `tokenizer_kwargs` | Dict | Tokenizer loading parameters. Refer to 🔗 [PreTrainedTokenizerBase Documentation](https://huggingface.co/docs/transformers/v4.50.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase) |
| `model_kwargs` | Dict | Model loading parameters. Refer to 🔗 [AutoModel Configuration](https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/auto#transformers.AutoConfig.from_pretrained) |
| `generation_kwargs` | Dict | Inference generation parameters. Refer to 🔗 [Text Generation Documentation](https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation) |
| `run_cfg` | Dict | Runtime configuration, including `num_gpus` (number of GPUs used) and `num_procs` (number of machine processes used) |
| `max_out_len` | Int | Maximum number of output tokens generated by inference. Valid range: (0, 131072] |
| `batch_size` | Int | Batch size for inference requests. Valid range: (0, 64000] |
| `max_seq_len` | Int | Maximum input sequence length. Valid range: (0, 131072] |
| `batch_padding` | Bool | Whether to enable batch padding. Set to `True` or `False` |