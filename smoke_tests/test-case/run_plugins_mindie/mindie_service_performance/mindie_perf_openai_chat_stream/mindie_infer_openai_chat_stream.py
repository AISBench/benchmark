from mmengine.config import read_base
from mindie_ais_bench_backend.models import VLLMCustomAPIChatStreamLora
from ais_bench.benchmark.summarizers import DefaultPerfSummarizer
from mindie_ais_bench_backend.calculators import MindIEPerfMetricCalculator
from mindie_ais_bench_backend.clients import OpenAIChatStreamClient

with read_base():
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen import synthetic_datasets
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str_perf import gsm8k_datasets
    from ais_bench.benchmark.configs.summarizers.example import summarizer as summarizer_accuracy

datasets = [ # all_dataset_configs.py中导入了其他数据集配置，可以将synthetic_datasets替换为其他一个或多个数据集
    *synthetic_datasets,
]

models = [
    dict(
        attr="service", # model or service
        type=VLLMCustomAPIChatStreamLora,
        abbr='vllm-api-stream-chat',
        model="",
        path="",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost", # 推理服务的IP
        host_port = 1025, # 推理服务的端口
        enable_ssl = False,
        max_out_len = 10, # 最大输出tokens长度
        batch_size=10, # 推理的最大并发数
        custom_client=dict(type=OpenAIChatStreamClient),
        generation_kwargs = dict( # 后处理参数参考vllm的官方文档
            temperature = 0,
            ignore_eos = True,
            adapter_id = [], # Multi LoRA场景下，指定使用的lora-id，若为空list使用base模型名称
            lora_data_map_file = "", # Multi LoRA场景下，指定lora-id与数据集的映射关系(通过json文件建立映射关系)
        )
    )
]

summarizer_perf = dict(
    type=DefaultPerfSummarizer,
    calculator=dict(
        type=MindIEPerfMetricCalculator,
        stats_list=["Average", "Min", "Max", "Median", "P75", "P90", "P99"],
    )
)

summarizer = summarizer_perf # 精度场景设置为 summarizer_accuracy，性能场景设置为 summarizer_perf


work_dir = 'outputs/api-vllm-stream-chat/'
models[0]['host_ip'] = '10.175.119.75'
models[0]['host_port'] = 8081
models[0]['path'] = '/home/z00883268/dev_branch/AISBench_Smoke/resource/model/Qwen2.5-7B-Instruct'
models[0]['model'] = 'qwen'
work_dir = '/home/z00883268/dev_branch/AISBench_Smoke/output//run_plugins_mindie/mindie_service_performance/mindie_perf_openai_chat_stream'
