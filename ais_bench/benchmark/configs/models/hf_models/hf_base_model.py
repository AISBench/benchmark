from ais_bench.benchmark.models import HuggingFaceBaseModel

models = [
    dict(
        attr="local", # local or service
        type=HuggingFaceBaseModel, #  transformers >= 4.33.0 用这个，prompt 构造成字符串格式
        abbr='hf-base-model',
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
        run_cfg = dict(num_gpus=1, num_procs=1),  # 多卡/多机多卡 参数，使用torchrun拉起任务
        max_out_len=100, # 最大输出token长度
        batch_size=1, # 每次推理的batch size
        max_seq_len=2048,
        batch_padding=True,
    )
]
