from ais_bench.benchmark.models import VLLMFunctionCallAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMFunctionCallAPIChat,
        abbr="vllm-api-function-call-chat",
        path="",
        model="",
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=8080,
        max_out_len=10240,
        batch_size=1,
        returns_tool_calls=True,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
