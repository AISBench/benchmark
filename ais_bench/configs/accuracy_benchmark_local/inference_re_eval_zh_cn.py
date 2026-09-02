from mmengine.config import read_base
from ais_bench.benchmark.models import HuggingFacewithChatTemplate
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask
from ais_bench.benchmark.datasets import gsm8k_postprocess, gsm8k_dataset_postprocess

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets

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

# 关键：替换或修改答案的提取函数实现
datasets[0]['eval_cfg']['pred_postprocessor'] = dict(type=gsm8k_postprocess)
datasets[0]['eval_cfg']['dataset_postprocessor'] = dict(type=gsm8k_dataset_postprocess)

work_dir = 'outputs/default/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLApiInferTask),
    ),
)
