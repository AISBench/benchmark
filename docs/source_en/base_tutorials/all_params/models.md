# Model Configuration Instructions
AISBench Benchmark supports two types of model backends:
- [Service-Oriented Inference Backend](#service-oriented-inference-backend)
- [Local Model Backend](#local-model-backend)

> ⚠️ **Note**: The two types of backends cannot be specified simultaneously.


## Service-Oriented Inference Backend
AISBench Benchmark supports multiple service-oriented inference backends, including vLLM, SGLang, Triton, MindIE, TGI, etc. These backends receive inference requests and return results through exposed HTTP API interfaces. (HTTPS interfaces are not supported currently.)

Taking the vLLM inference service deployed on GPU as an example, you can refer to the [vLLM Official Documentation](https://docs.vllm.ai/en/stable/getting_started/quickstart.html) to start the service.

The model configurations corresponding to different service-oriented backends are as follows:

| Model Configuration Name | Description | Prerequisites for Use | Supported Evaluation Modes | Interface Type | Supported Dataset Prompt Formats | Configuration File Path |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| `vllm_api_general` | Access the inference service via vLLM's OpenAI-compatible API, with the interface `v1/completions` | The vLLM version used supports the `v1/completions` sub-service | Generative Evaluation, PPL Mode Evaluation | Text Interface | String Format | [vllm_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py) |
| `vllm_api_general_stream` | Access the vLLM inference service in streaming mode, with the interface `v1/completions` | The vLLM version used supports the `v1/completions` sub-service | Generative Evaluation | Streaming Interface | String Format | [vllm_api_general_stream.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py) |
| `vllm_api_general_chat` | Access the inference service via vLLM's OpenAI-compatible API, with the interface `v1/chat/completions` | The vLLM version used supports the `v1/chat/completions` sub-service | Generative Evaluation, PPL Mode Evaluation | Text Interface | String Format, Dialogue Format, Multimodal Format | [vllm_api_general_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py) |
| `vllm_api_stream_chat` | Access the vLLM inference service in streaming mode, with the interface `v1/chat/completions` | The vLLM version used supports the `v1/chat/completions` sub-service | Generative Evaluation | Streaming Interface | String Format, Dialogue Format, Multimodal Format | [vllm_api_stream_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py) |
| `vllm_api_stream_chat_multiturn` | Access the vLLM inference service in streaming mode for multi-turn dialogue scenarios, with the interface `v1/chat/completions` | The vLLM version used supports the `v1/chat/completions` sub-service | Generative Evaluation | Streaming Interface | Dialogue Format | [vllm_api_stream_chat_multiturn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat_multiturn.py) |
| `vllm_api_function_call_chat` | API for accessing the vLLM inference service in function call accuracy evaluation scenarios, with the interface `v1/chat/completions` (only applicable to the [BFCL](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/BFCL/README_en.md) evaluation scenario) | The vLLM version used supports the `v1/chat/completions` sub-service | Generative Evaluation | Text Interface | Dialogue Format | [vllm_api_function_call_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_function_call_chat.py) |
| `vllm_api_old` | Access the inference service via vLLM-compatible API, with the interface `generate` | The vLLM version used supports the `generate` sub-service | Generative Evaluation | Text Interface | String Format, Multimodal Format | [vllm_api_old.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_old.py) |
| `mindie_stream_api_general` | Access the inference service via MindIE streaming API, with the interface `infer` | The MindIE version used supports the `infer` sub-service | Generative Evaluation | Streaming Interface | String Format, Multimodal Format | [mindie_stream_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/mindie_api/mindie_stream_api_general.py) |
| `triton_api_general` | Access the inference service via Triton API, with the interface `v2/models/{model name}/generate` | Start an inference service that supports Triton API | Generative Evaluation | Text Interface | String Format, Multimodal Format | [triton_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/triton_api/triton_api_general.py) |
| `triton_stream_api_general` | Access the inference service via Triton streaming API, with the interface `v2/models/{model name}/generate_stream` | Start an inference service that supports Triton API | Generative Evaluation | Streaming Interface | String Format, Multimodal Format | [triton_stream_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/triton_api/triton_stream_api_general.py) |
| `tgi_api_general` | Access the inference service via TGI API, with the interface `generate` | Start an inference service that supports TGI API | Generative Evaluation | Text Interface | String Format, Multimodal Format | [tgi_api_general](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/tgi_api/tgi_api_general.py) |
| `tgi_stream_api_general` | Access the inference service via TGI streaming API, with the interface `generate_stream` | Start an inference service that supports TGI API | Generative Evaluation | Streaming Interface | String Format, Multimodal Format | [tgi_stream_api_general](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/tgi_api/tgi_stream_api_general.py) |


### Parameter Description for Service-Oriented Inference Backend Configuration
The configuration file for the service-oriented inference backend is configured using Python syntax, as shown in the example below:
```python
from ais_bench.benchmark.models import VLLMCustomAPI

models = [
    dict(
        attr="service",
        type=VLLMCustomAPI,
        abbr='vllm-api-general',
        path="",                    # Specify the absolute path to the model serialized vocabulary file (generally not required for accuracy testing scenarios)
        model="",        # Specify the name of the model loaded on the server, configured according to the actual model name pulled by the VLLM inference service (configuring an empty string will automatically retrieve it)
        stream=False,    # Whether it is a streaming interface
        request_rate = 0,           # Request sending frequency: send 1 request to the server every 1/request_rate seconds; if less than 0.1, all requests are sent at once
        retry = 2,                  # Maximum number of retries for each request
        api_key="",                 # Custom API key, default is an empty string
        host_ip = "localhost",      # Specify the IP address of the inference service
        host_port = 8080,           # Specify the port of the inference service
        url="",                     # Custom URL path for accessing the inference service (needs to be configured when the base URL is not a combination of http://host_ip:host_port)
        max_out_len = 512,          # Maximum number of tokens output by the inference service
        batch_size=1,               # Maximum concurrency for request sending
        trust_remote_code=False,    # Whether the tokenizer trusts remote code, default is False;
        generation_kwargs = dict(   # Model inference parameters, configured with reference to the VLLM documentation; the AISBench evaluation tool does not process these parameters and attaches them to the sent requests
            temperature = 0.01,
            ignore_eos=False,
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
| `stream` | Boolean | Whether the inference service is a streaming interface. Required Parameter. |
| `request_rate` | Float | Request sending rate (unit: requests per second). A request is sent every `1/request_rate` seconds; if the value is less than 0.1, requests are automatically merged and sent in batches. Valid range: [0, 64000]. When the `traffic_cfg` item is enabled, this function may be overwritten (for specific reasons, refer to 🔗 [Parameter Interpretation Section in the Description of Request Rate (RPS) Distribution Control and Visualization](../../advanced_tutorials/rps_distribution.md#parameter-interpretation)) |
| `traffic_cfg` | Dict | Parameters for controlling fluctuations in the request sending rate (for detailed usage instructions, refer to 🔗 [Description of Request Rate (RPS) Distribution Control and Visualization](../../advanced_tutorials/rps_distribution.md)). If this item is not filled in, the function is disabled by default |
| `retry` | Int | Maximum number of retries after failing to connect to the server. Valid range: [0, 1000] |
| `api_key` | String | Custom API key, default is an empty string. Only supports the `VLLMCustomAPI` and `VLLMCustomAPIChat` model type. |
| `host_ip` | String | Server IP address, supporting valid IPv4 or IPv6, e.g., `127.0.0.1` |
| `host_port` | Int | Server port number, which must be consistent with the port specified during service-oriented deployment |
| `url` | String | Custom URL path for accessing the inference service (needs to be configured when the base URL is not a combination of http://host_ip:host_port).For example, when `models`'s `type` is `VLLMCustomAPI`, configure `url` as `https://xxxxxxx/yyyy/`, the actual request URL accessed is `https://xxxxxxx/yyyy/v1/completions` |
| `max_out_len` | Int | Maximum output length of the inference response; the actual length may be limited by the server. Valid range: (0, 131072] |
| `batch_size` | Int | Batch size for concurrent requests. Valid range: (0, 64000] |
| `trust_remote_code` | Boolean | Whether the tokenizer trusts remote code, default is `False`|
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
| `hf_base_model` | HuggingFace Base Model Backend | The basic dependencies of the evaluation tool have been installed; the HuggingFace model weight path must be specified in the configuration file (automatic download is not supported currently) | String Format | [hf_base_model](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/hf_models/hf_base_model.py) |
| `hf_chat_model` | HuggingFace Chat Model Backend | The basic dependencies of the evaluation tool have been installed; the HuggingFace model weight path must be specified in the configuration file (automatic download is not supported currently) | Dialogue Format | [hf_chat_model](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/hf_models/hf_chat_model.py) |
|`hf_qwenvl_model`|	HuggingFace Chat QwenVL Model Backend|The basic dependencies of the evaluation tool have been installed; the HuggingFace model weight path must be specified in the configuration file (automatic download is not supported currently)|Dialogue Format|[hf_qwenvl_model](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/hf_models/hf_qwenvl_model.py)|
|`vllm_offline_vl_model`|	vLLM Chat QwenVL Offline Inference Model Backend|The basic dependencies of the evaluation tool have been installed; the model weight path must be specified in the configuration file (automatic download is not supported currently)|Dialogue Format|[vllm_offline_vl_model](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_offline_models/vllm_offline_vl_model.py)|


### Parameter Description for Huggingface Local Model Backend Configuration
The configuration file for the huggingface local model backend is configured using Python syntax, as shown in the example below:
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

The description of configurable parameters for the huggingface local model inference backend is as follows:

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

### Parameter Description for vLLM Offline Inference Model Backend Configuration
The configuration file for the vllm offline inference local model backend is configured using Python syntax, as shown in the example below:
```python
from ais_bench.benchmark.models import VLLMOfflineVLModel

models = [
    dict(
        attr="local",                    # Backend type identifier
        type=VLLMOfflineVLModel,         # Model type
        abbr='vllm-offline-vl-model',    # Unique identifier
        path = "",                       # Model weight path
        model_kwargs=dict(               # LLM init params, refer https://docs.vllm.com.cn/en/latest/serving/engine_args.html#
            max_num_seqs=5,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 24},
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        ),
        sample_kwargs=dict(              # sample params, refer https://docs.vllm.ai/en/v0.6.5/dev/sampling_params.html
            temperature=0.0,
            stop_token_ids=None
        ),
        vision_kwargs=dict(              # multi-modal params, refer https://docs.vllm.ai/en/v0.7.3/getting_started/examples/vision_language.html
            min_pixels=1280 * 28 * 28,
            max_pixels=16384 * 28 * 28,
        ),
        max_out_len=512,                 # Maximum output length
        batch_size=1,                    # Request concurrency count
    )
]
```
The description of configurable parameters for the vllm offline inference local model inference backend is as follows:
| Parameter Name | Parameter Type | Description & Configuration |
|----------|-----------|-------------|
| `attr` | String | Identifier for the backend type, fixed as `local` (local model) or `service` (service-oriented inference) |
| `type` | Python Class | Model class name, automatically associated by the system; no manual configuration is required by the user |
| `abbr` | String | Unique identifier for the local task, used to distinguish multiple tasks. It is recommended to use a combination of English characters and hyphens, e.g., `vllm-offline-vl-model` |
| `path` | String | Model weight path, which must be an accessible local path. The model is loaded using `vllm.LLM(model=path)`  |
| `model_kwargs` | Dict | LLM init params, refer 🔗 [LLM init params](https://docs.vllm.com.cn/en/latest/serving/engine_args.html#) |
| `sample_kwargs` | Dict | LLM sample params, refer 🔗 [sample params](https://docs.vllm.ai/en/v0.6.5/dev/sampling_params.html) |
| `vision_kwargs` | Dict | multi-modal input params，refer 🔗 [multi-modal vllm offline inference](https://docs.vllm.ai/en/v0.7.3/getting_started/examples/vision_language.html) |
| `max_out_len` | Int | Maximum number of output tokens generated by inference. Valid range: (0, 131072] |
| `batch_size` | Int | Batch size for inference requests. Valid range: (0, 64000] |