from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import (
        synthetic_datasets,
    )
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import (
        models as vllm_api_general_stream,
    )
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import (
        models as vllm_api_stream_chat,
    )

datasets = synthetic_datasets  # 指定数据集列表

vllm_api_general_stream[0]["abbr"] = "demo-" + vllm_api_general_stream[0]["abbr"]
vllm_api_stream_chat[0]["abbr"] = "demo-" + vllm_api_stream_chat[0]["abbr"]

models = vllm_api_general_stream + vllm_api_stream_chat # 指定模型列表

work_dir = "outputs/demo_api-vllm-general-chat-perf/"
