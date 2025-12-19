# Supporting New Models

Currently, AISBench supports the following model types:

- **Service Models** (executing inference tasks by accessing endpoints provided by service frameworks): vLLM, Triton, TGI, MindIE
- **Local Models** (executing inference tasks by loading local model files): HuggingFace, vllmOfflineVL, HuggingFaceVL

For certain custom service frameworks or inference backends, it is usually necessary to implement custom models to access services or call models. Currently, two methods are supported: adding new API models and local models.

## Adding New API Models

To add a new API-based model, create a new file `my_custom_api.py` in `ais_bench/benchmark/models/api_models`, inherit from `BaseAPIModel`, and implement the corresponding functional interfaces according to usage scenarios. The currently supported extensible interfaces are as follows:

- **(Required) `get_request_body`**: Get the request body, used to construct the request body
- **(Required) `_get_url`**: Get the request URL
- **(Required when model supports non-streaming inference) `parse_text_response`**: Parse text response, called when model parameter `stream` is `False`
- **(Required when model supports streaming inference) `parse_stream_response`**: Parse stream response, called when model parameter `stream` is `True`

**Note**:

- When the model supports performance evaluation, the `parse_stream_response` interface must be implemented

```python
from typing import Dict, Optional, Union

from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.models import BaseAPIModel
from ais_bench.benchmark.models.output import RequestOutput, Output

PromptType = Union[PromptList, str]

class MyCustomAPI(BaseAPIModel):
    is_api: bool = True

    def __init__(
        self,
        path: str = "",             # Model vocabulary path, used to load model vocabulary
        stream: bool = False,       # Whether it is streaming inference
        max_out_len: int = 4096,    # Maximum output length
        retry: int = 2,             # Number of retries on request failure
        api_key: str = "",          # API key
        host_ip: str = "localhost", # Host IP
        host_port: int = 8080,      # Host port
        url: str = "",              # Custom URL
        trust_remote_code: bool = False, # Whether to trust remote code
        generation_kwargs: Optional[Dict] = dict(), # Generation parameters, additional parameters passed to endpoint
    ):
        super().__init__(
            path=path,
            stream=stream,
            max_out_len=max_out_len,
            retry=retry,
            api_key=api_key,
            host_ip=host_ip,
            host_port=host_port,
            url=url,
            generation_kwargs=generation_kwargs,
        )
        self.url = self._get_url()
        ...

    def _get_url(self):
        """Concatenate URI to form complete request URL"""
        ...

    async def get_request_body(self, input: PromptType, max_out_len: int, output: RequestOutput, **args) -> dict:
        """Assemble into dict format request body according to endpoint protocol format, and save debugging information to output"""
        ...

    async def parse_text_response(self, data: Dict, output: RequestOutput):
        """Parse text response returned by server according to endpoint protocol format, save response content to output"""
        ...

    async def parse_stream_response(self, data: Dict, output: RequestOutput):
        """Parse stream response returned by server according to endpoint protocol format, save response content to output"""
        ...

```

It is recommended to add the new API model class to [`__init__.py`](../../../ais_bench/benchmark/models/api_models/__init__.py) for convenient automatic import later.

For detailed implementation, refer to: [VLLMCustomAPIChat](../../../ais_bench/benchmark/models/api_models/vllm_custom_api_chat.py)

To use the custom API model, add the following configuration in the configuration folder `ais_bench/benchmark/configs/models`:

```python
from ais_bench.benchmark.models.api_models.my_custom_api import MyCustomAPI
models = [
    dict(
        attr="service",         # (Required) Flag indicating the model is a service API model
        type=MyCustomAPI,       # (Required) Custom API model class
        abbr='my_custom_api',   # (Required) Unique model identifier
        path="",                # (Optional) Model vocabulary path, used to load model vocabulary, must be configured for performance evaluation
        model="",               # (Optional) Model name, some endpoints need this parameter to access services, can call self._get_service_model_path() to automatically obtain
        stream=False,           # (Optional) Whether it is a streaming interface
        request_rate=0,         # (Optional) Request sending rate, send 1 request to server every 1/request_rate seconds, if less than 0.1 then send all requests at once
        retry=2,                # (Optional) Maximum number of retries per request
        api_key="",             # (Optional) API key
        host_ip="localhost",    # (Optional) Host IP
        host_port=8080,         # (Optional) Host port
        url="",                 # (Optional) Custom URL path
        max_out_len=512,        # (Optional) Maximum number of tokens output by inference service
        batch_size=1,           # (Optional) Maximum concurrency of request sending
        trust_remote_code=False, # (Optional) Whether tokenizer trusts remote code, default False
        generation_kwargs=dict(   # (Optional) Model inference parameters, refer to endpoint documentation for configuration, AISBench evaluation tool does not process, attached in sent requests
            temperature=0.01,
            ignore_eos=False
        ),
    )
]
```

Then execute the command to start service performance evaluation:

```bash
ais_bench --models my_custom_api --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt
```

## Adding New Local Models

To add a new model based on local model files, create a new file `my_custom_model.py` in `ais_bench/benchmark/models/local_models`, inherit from `BaseModel`, and implement the corresponding functional interfaces according to usage scenarios. The currently supported extensible interfaces are as follows:

- **`__init__`**: Initialize model and vocabulary
- **`generate`**: Call the loaded local model to perform generative inference and return inference results

```python
class MyCustomModel(BaseModel):

    def __init__(self,
                 path: str,                # (Required) Model vocabulary path, used to load model vocabulary
                 max_seq_len: int = 2048,  # Maximum sequence length
                 tokenizer_only: bool = False, # Whether to only load vocabulary
                 meta_template: Optional[Dict] = None, # Meta template
                 generation_kwargs: Optional[Dict] = dict(), # Generation parameters
                 sync_rank: bool = False, # Whether to synchronize input
                 **kwargs):
        super().__init__(
            path,
            max_seq_len,
            tokenizer_only,
            meta_template,
            generation_kwargs,
            sync_rank,
        )
        ...

    def generate(self, input: PromptType, max_out_len: int, **kwargs) -> List[str]:
        """Call the loaded local model to perform inference and return inference results"""
        ...
```

It is recommended to add the new local model class to [`__init__.py`](../../../ais_bench/benchmark/models/local_models/__init__.py) for convenient automatic import later.

For detailed implementation, refer to: [HuggingFacewithChatTemplate](../../../ais_bench/benchmark/models/local_models/huggingface_above_v4_33.py)

To use the custom local model, add the following configuration in the configuration folder `ais_bench/benchmark/configs/models/`:

```python
from ais_bench.benchmark.models import MyCustomModel

models = [
    dict(
        attr="local",               # (Required) Backend type identifier, fixed as `local` (local model) or `service` (service inference)
        type=MyCustomModel,         # (Required) Model type, custom model class
        abbr='my_custom_model',     # (Required) Unique model identifier
        path='THUDM/chatglm-6b',    # (Required) Model weight path and vocabulary path
        max_out_len=100,            # (Required) Maximum output token length
        batch_size=1,               # (Required) Batch size for each inference
        max_seq_len=2048,           # (Required) Maximum sequence length
        generation_kwargs=dict(     # (Optional) Generation parameters, refer to model documentation for configuration
            temperature=0.0,
            stop_token_ids=None
        ),
        # ... Other optional parameters for model initialization and inference task configuration
    )
]
```

Then execute the command to start the evaluation task:

```bash
ais_bench --models my_custom_model --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt
```

