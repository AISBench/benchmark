from mmengine.config import read_base
from ais_bench.benchmark.models import HuggingFaceBaseModel
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_chat_prompt import gsm8k_datasets as gsm8k_0_shot_cot_chat
    from ais_bench.benchmark.configs.datasets.math.math500_gen_0_shot_cot_chat_prompt import math_datasets as math500_gen_0_shot_cot_chat

datasets = [*gsm8k_0_shot_cot_chat] + [*math500_gen_0_shot_cot_chat]

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='hf-base-model',
        path='THUDM/chatglm-6b',
        tokenizer_path='THUDM/chatglm-6b',
        model_kwargs=dict(device_map='auto'),
        tokenizer_kwargs=dict(padding_side='left'),
        generation_kwargs=dict(
            temperature=0.5,
            top_k=10,
            top_p=0.95,
            do_sample=True,
            seed=None,
            repetition_penalty=1.03,
        ),
        max_out_len=100,
        batch_size=1,
        max_seq_len=2048,
        batch_padding=True,
    )
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,
        task=dict(type=OpenICLApiInferTask),
    ),
)

work_dir = 'outputs/hf-multi-model-multi-dataset/'
