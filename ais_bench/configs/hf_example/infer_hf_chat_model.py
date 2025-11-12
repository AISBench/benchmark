from mmengine.config import read_base
from ais_bench.benchmark.models import HuggingFacewithChatTemplate
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local_api import LocalAPIRunner
from ais_bench.benchmark.tasks import OpenICLInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_chat_prompt import gsm8k_datasets as gsm8k_0_shot_cot_chat

datasets = [ # all_dataset_configs.py中导入了其他数据集配置，可以将gsm8k_0_shot_cot_chat替换为其他一个或多个数据集
    *gsm8k_0_shot_cot_chat,
]

models = [
    dict(
        type=HuggingFacewithChatTemplate, #  transformers >= 4.33.0 用这个，prompt 构造成字符串格式
        abbr='hf-chat-model',
        path='THUDM/chatglm-6b', # path to model dir, current value is just a example
        tokenizer_path='THUDM/chatglm-6b', # path to tokenizer dir, current value is just a example
        model_kwargs=dict( # 模型参数参考 huggingface.co/docs/transformers/v4.50.0/en/model_doc/auto#transformers.AutoModel.from_pretrained
            device_map='auto',
        ),
        tokenizer_kwargs=dict( # tokenizer参数参考 huggingface.co/docs/transformers/v4.50.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase
            padding_side='left',
        ),
        generation_kwargs = dict( # 后处理参数参考huggingface.co/docs/transformers/main_classes/test_generation
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            do_sample = True,
            seed = None,
            repetition_penalty = 1.03,
        ),
        max_out_len=100, # 最大输出token长度
        batch_size=1, # 每次推理batch size
        max_seq_len=2048,
        batch_padding=True,
    )
]


infer = dict(partitioner=dict(type=NaivePartitioner),
             runner=dict(
                 type=LocalAPIRunner,
                 max_num_workers=2,
                 task=dict(type=OpenICLInferTask)), )

work_dir = 'outputs/hf-chat-model/'