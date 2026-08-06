from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

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
        host_port=8080,
        url="",
        max_out_len=512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=False,
        ),
        # 仅开启 response_anomaly 时需要填写，其他场景无需填写。
        response_anomaly=dict(
            model_name="",  # 填写模型名称，如 Qwen3.6-27B
            top_logprobs=20,
            msprobe_config_path="/path/to/config.yaml",
            msprobe_mtype_path="/path/to/mtype_config.json",
            msprobe_token2category_dir="/path/to/token2category",
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
