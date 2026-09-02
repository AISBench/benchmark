from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.math.math500_gen_0_shot_cot_chat_prompt import math_datasets as math500_gen_0_shot_cot_chat
    from ais_bench.benchmark.configs.datasets.mmlu.mmlu_gen_5_shot_str import mmlu_datasets as mmlu_5_shot_str
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

datasets = gsm8k_0_shot_cot_str + math500_gen_0_shot_cot_chat + mmlu_5_shot_str
models = vllm_api_general + vllm_api_general_chat + vllm_api_stream_chat

work_dir = 'outputs/multi_model_multi_dataset/'
