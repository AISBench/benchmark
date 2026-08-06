from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import keep_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="",
        model="",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=8005,
        url="",
        max_out_len=2048,
        batch_size=4,
        trust_remote_code=False,
        generation_kwargs=dict(
            enable_thinking=True,
            temperature=0,
            top_k=-1,
            top_p=1.0,
            max_tokens=2048,
            ignore_eos=True,
            repetition_penalty=1.0,
            logprobs=0,
        ),
        pred_postprocessor=dict(type=keep_reasoning_content),
    )
]
