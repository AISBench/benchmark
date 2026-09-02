from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.textvqa.textvqa_gen_0_shot_str import textvqa_datasets
    from ais_bench.benchmark.configs.datasets.docvqa.docvqa_gen_0_shot_str import docvqa_datasets
    from ais_bench.benchmark.configs.models.lmm_models.lmm_vllm_api_chat import models as lmm_vllm_api_chat

datasets = textvqa_datasets + docvqa_datasets
models = lmm_vllm_api_chat

work_dir = 'outputs/lmm_multi_dataset/'
