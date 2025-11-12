from ais_bench.benchmark.models.local_models.base import BaseModel, LMTemplateParser  # noqa: F401
from ais_bench.benchmark.models.api_models.base_api import APITemplateParser, BaseAPIModel  # noqa: F401
from ais_bench.benchmark.models.api_models.vllm_custom_api import VLLMCustomAPI # noqa: F401
from ais_bench.benchmark.models.api_models.vllm_custom_api_chat import VLLMCustomAPIChat # noqa: F401
from ais_bench.benchmark.models.api_models.mindie_stream_api import MindieStreamApi
from ais_bench.benchmark.models.local_models.huggingface import HuggingFace, HuggingFaceCausalLM
from ais_bench.benchmark.models.local_models.huggingface_above_v4_33 import HuggingFaceBaseModel, HuggingFacewithChatTemplate
from ais_bench.benchmark.models.api_models.tgi_api import TGICustomAPI
from ais_bench.benchmark.models.api_models.triton_api import TritonCustomAPI
from ais_bench.benchmark.models.api_models.vllm_function_call_api_chat import VLLMFunctionCallAPIChat
from ais_bench.benchmark.models.local_models.huggingface_vl import HuggingFaceQwen2VLwithChatTemplate