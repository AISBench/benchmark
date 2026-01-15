from ais_bench.benchmark.models import MindFormerModel

models = [
    dict(
        attr="local", # local or service
        type=MindFormerModel, #  transformers < 4.33.0 用这个，优先AutoModelForCausalLM.from_pretrained加载模型，失败则用AutoModel.from_pretrained加载
        abbr='mindformer-model',
        path='THUDM/chatglm-6b', # path to model dir, current value is just a example
        checkpoint = 'THUDM/your_checkpoint', # path to checkpoint file, current value is just a example
        yaml_cfg_file = 'THUDM/your.yaml',
        tokenizer_path='THUDM/chatglm-6b', # path to tokenizer dir, current value is just a example
        model_kwargs=dict( # 模型参数参考 huggingface.co/docs/transformers/v4.50.0/en/model_doc/auto#transformers.AutoModel.from_pretrained
            device_map='npu',
        ),
        tokenizer_kwargs=dict( # tokenizer参数参考 huggingface.co/docs/transformers/v4.50.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase
            padding_side='right',
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
        batch_size=2, # 每次推理的batch size
        max_seq_len=2048,
        batch_padding=True,
    )
]