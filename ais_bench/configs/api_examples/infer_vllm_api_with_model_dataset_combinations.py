from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.math.math500_gen_0_shot_cot_chat_prompt import math_datasets as math500_gen_0_shot_cot_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

models = vllm_api_general + vllm_api_general_chat + vllm_api_stream_chat
datasets = gsm8k_0_shot_cot_str + math500_gen_0_shot_cot_chat

model_dataset_combinations = [
    dict(models=[models[0]], datasets=[datasets[0]]),
    dict(models=[models[1]], datasets=[datasets[1]]),
    dict(models=[models[2]], datasets=[datasets[0], datasets[1]]),
]

work_dir = 'outputs/custom_combinations/'
