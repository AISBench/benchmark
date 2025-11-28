from ais_bench.benchmark.models import VLLMOfflineVLModel

models = [
    dict(
        attr="local", # local or service
        type=VLLMOfflineVLModel,
        abbr='vllm-offline-vl-model',
        path = "", # vllm model
        model_kwargs=dict(  # Init vllm LLM, refer https://docs.vllm.com.cn/en/latest/serving/engine_args.html#
            max_num_seqs=5,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 24},
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        ),
        sample_kwargs=dict(  # Smpling Parameters, refer https://docs.vllm.ai/en/v0.6.5/dev/sampling_params.html
            temperature=0.0,
            stop_token_ids=None
        ),
        vision_kwargs=dict(
            min_pixels=1280 * 28 * 28,
            max_pixels=16384 * 28 * 28,
        ),
        max_out_len=512,
        batch_size=1,
    )
]