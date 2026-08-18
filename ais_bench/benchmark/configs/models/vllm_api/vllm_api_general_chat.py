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
        # 仅开启 response_anomaly 时生效：模型 path 指向本地 tokenizer 目录时，
        # msProbe 配置与词表会自动生成；也可通过 ais_bench-gen-response-anomaly-config
        # 手动生成后在此填写三个路径。
        response_anomaly=dict(
            model_name="",  # 填写模型名称，如 Qwen3.6-27B
            msprobe_config_path="",
            msprobe_mtype_path="",
            msprobe_token2category_dir="",
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
