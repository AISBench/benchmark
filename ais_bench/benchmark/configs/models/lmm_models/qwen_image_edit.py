from ais_bench.benchmark.models.local_models.qwen_image_edit_mindie_sd import QwenImageEditModel

models = [
    dict(
        attr="local", # local or service
        type=QwenImageEditModel, # transformers >= 4.33.0 用这个，prompt 是构造成对话格式
        abbr='qwen-image-edit',
        path='/home/yanhe/models/Qwen-Image-Edit-2509/', # path to model dir, current value is just a example
        device_kwargs=dict(
        ),
        infer_kwargs=dict( # 模型参数参考 huggingface.co/docs/transformers/v4.50.0/en/model_doc/auto#transformers.AutoModel.from_pretrained
            num_inference_steps=50,
            num_images_per_prompt=1,
        ),
        run_cfg = dict(num_gpus=1, num_procs=1),  # 多卡/多机多卡 参数，使用torchrun拉起任务
        batch_size=1, # 每次推理的batch size
    )
]