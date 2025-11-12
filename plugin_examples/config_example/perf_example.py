from mmengine.config import read_base
from ais_bench_plugin_example_pkg.models import ExampleModel # 导入样例中自定义的模型运行类
from ais_bench_plugin_example_pkg.clients import ExampleClient # 导入样例中自定义的请求客户端类
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local_api import LocalAPIRunner
from ais_bench.benchmark.tasks import OpenICLInferTask
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen import synthetic_datasets

datasets = [
    *synthetic_datasets,
]

models = [
    dict(
        attr="service",
        type=ExampleModel, # 使用自定义的模型运行类
        abbr='example-model',
        path="",
        model="",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",
        host_port = 8080,
        max_out_len = 512,
        batch_size=1,
        trust_remote_code=False,
        custom_client=dict(type=ExampleClient), # 使用自定义的请求客户端类
        generation_kwargs = dict(
            ignore_eos=True,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]


infer = dict(partitioner=dict(type=NaivePartitioner),
             runner=dict(
                 type=LocalAPIRunner,
                 max_num_workers=2,
                 task=dict(type=OpenICLInferTask)), )

work_dir = 'outputs/example_model/'
