from mmengine.config import read_base
from ais_bench.benchmark.models import HuggingFacewithChatTemplate
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets
    from ais_bench.benchmark.configs.datasets.aime2024.aime2024_gen_0_shot_chat_prompt import aime2024_datasets

datasets = gsm8k_datasets + aime2024_datasets

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='hf-chat-model',
        path='THUDM/chatglm-6b', # 替换为实际的本地模型权重路径
        tokenizer_path='THUDM/chatglm-6b',
        model_kwargs=dict(device_map='auto'),
        tokenizer_kwargs=dict(padding_side='left'),
        generation_kwargs=dict(
            temperature=0.01,
            do_sample=False,
        ),
        max_out_len=512,
        batch_size=1,
        max_seq_len=2048,
        batch_padding=True,
    )
]

work_dir = 'outputs/default/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLApiInferTask),
    ),
)
