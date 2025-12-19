# 支持新模型

目前 AISBench 已经支持的模型类型如下：

- **服务化模型**（通过访问服务化框架提供的 endpoint 执行推理任务）：vLLM、Triton、TGI、MindIE
- **本地模型**（通过加载本地模型文件执行推理任务）：HuggingFace、vllmOfflineVL、HuggingFaceVL

针对某些自定义服务框架或推理后端，通常需要实现自定义模型来实现对服务的访问或模型的调用。目前支持新增 API 模型和本地模型两种方式。

## 新增 API 模型

新增基于 API 的模型，需要在 `ais_bench/benchmark/models/api_models` 下新建 `my_custom_api.py` 文件，继承 `BaseAPIModel`，并根据使用场景实现对应的功能接口。当前支持拓展的接口如下：

- **（必需）`get_request_body`**：获取请求体，用于构建请求体
- **（必需）`_get_url`**：获取请求 URL
- **（模型支持非流式推理时必需）`parse_text_response`**：解析文本响应，当模型参数 `stream` 为 `False` 时调用
- **（模型支持流式推理时必需）`parse_stream_response`**：解析流响应，当模型参数 `stream` 为 `True` 时调用

**注意**：

- 模型支持性能测评时，必须实现 `parse_stream_response` 接口

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
        path: str = "",             # 模型词表路径，用于加载模型词表
        stream: bool = False,       # 是否为流式推理
        max_out_len: int = 4096,    # 最大输出长度
        retry: int = 2,             # 请求失败重试次数
        api_key: str = "",          # API key
        host_ip: str = "localhost", # 主机IP
        host_port: int = 8080,      # 主机端口
        url: str = "",              # 自定义URL
        trust_remote_code: bool = False, # 是否信任远程代码
        generation_kwargs: Optional[Dict] = dict(), # 生成参数，额外传递给endpoint的参数
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
        """拼接 URI，组成完整的请求 URL"""
        ...

    async def get_request_body(self, input: PromptType, max_out_len: int, output: RequestOutput, **args) -> dict:
        """根据 endpoint 的协议格式组装成 dict 格式的请求体，同时将想要保存的调测信息保存到 output"""
        ...

    async def parse_text_response(self, data: Dict, output: RequestOutput):
        """根据 endpoint 的协议格式，解析服务端返回的文本响应，将响应内容保存到 output"""
        ...

    async def parse_stream_response(self, data: Dict, output: RequestOutput):
        """根据 endpoint 的协议格式，解析服务端返回的流式响应，将响应内容保存到 output"""
        ...

```

新增API模型类建议补充到[`__init__.py`](../../../ais_bench/benchmark/models/__init__.py)中，方便后续自动导入。

详细实现可参考：[VLLMCustomAPIChat](../../../ais_bench/benchmark/models/api_models/vllm_custom_api_chat.py)

若要使用自定义新增的 API 模型，需要在配置文件夹 `ais_bench/benchmark/configs/models` 中新增以下配置：

```python
from ais_bench.benchmark.models.api_models.my_custom_api import MyCustomAPI
models = [
    dict(
        attr="service",         #（必需）标志模型为服务化 API 模型
        type=MyCustomAPI,       #（必需）自定义 API 模型类
        abbr='my_custom_api',   #（必需）模型唯一标识
        path="",                #（可选）模型词表路径，用于加载模型词表，性能测评时必须配置
        model="",               #（可选）模型名称，部分 endpoint 需要传入该参数用于服务的访问，可调用 self._get_service_model_path() 自动获取
        stream=False,           #（可选）是否为流式接口
        request_rate=0,         #（可选）请求发送频率，每 1/request_rate 秒发送 1 个请求给服务端，小于 0.1 则一次性发送所有请求
        retry=2,                #（可选）每个请求最大重试次数
        api_key="",             #（可选）API key
        host_ip="localhost",    #（可选）主机 IP
        host_port=8080,         #（可选）主机端口
        url="",                 #（可选）自定义 URL 路径
        max_out_len=512,        #（可选）推理服务输出的 token 的最大数量
        batch_size=1,           #（可选）请求发送的最大并发数
        trust_remote_code=False, #（可选）tokenizer 是否信任远程代码，默认 False
        generation_kwargs=dict(   #（可选）模型推理参数，参考 endpoint 文档配置，AISBench 评测工具不做处理，在发送的请求中附带
            temperature=0.01,
            ignore_eos=False
        ),
    )
]
```

接着执行命令启动服务化性能评测：

```bash
ais_bench --models my_custom_api --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt
```

## 新增本地模型

新增基于本地模型文件的模型，需要在 `ais_bench/benchmark/models/local_models` 下新建 `my_custom_model.py` 文件，继承 `BaseModel`，并根据使用场景实现对应的功能接口。当前支持拓展的接口如下：

- **`__init__`**：初始化模型和词表
- **`generate`**：调用加载好的本地模型执行生成式推理并返回推理结果

```python
class MyCustomModel(BaseModel):

    def __init__(self,
                 path: str,                # （必需）模型词表路径，用于加载模型词表
                 max_seq_len: int = 2048,  # 最大序列长度
                 tokenizer_only: bool = False, # 是否只加载词表
                 meta_template: Optional[Dict] = None, # 元模板
                 generation_kwargs: Optional[Dict] = dict(), # 生成参数
                 sync_rank: bool = False, # 是否同步输入
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
        """调用加载好的本地模型执行推理并返回推理结果"""
        ...
```

新增本地模型类建议补充到[`__init__.py`](../../../ais_bench/benchmark/models/__init__.py)中，方便后续自动导入。

详细实现可参考：[HuggingFacewithChatTemplate](../../../ais_bench/benchmark/models/local_models/huggingface_above_v4_33.py)

若要使用自定义新增的本地模型，需要在配置文件夹 `ais_bench/benchmark/configs/models/` 中新增以下配置：

```python
from ais_bench.benchmark.models import MyCustomModel

models = [
    dict(
        attr="local",               #（必需）后端类型标识，固定为 `local`（本地模型）或 `service`（服务化推理）
        type=MyCustomModel,         #（必需）模型类型，自定义模型类
        abbr='my_custom_model',     #（必需）模型唯一标识
        path='THUDM/chatglm-6b',    #（必需）模型权重路径以及词表路径
        max_out_len=100,            #（必需）最大输出 token 长度
        batch_size=1,               #（必需）每次推理的 batch size
        max_seq_len=2048,           #（必需）最大序列长度
        generation_kwargs=dict(     #（可选）生成参数，参考模型文档配置
            temperature=0.0,
            stop_token_ids=None
        ),
        # ... 其他可选参数，用于模型初始化和推理任务过程中的配置
    )
]
```

接着执行命令启动评测任务：

```bash
ais_bench --models my_custom_model --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt
```
