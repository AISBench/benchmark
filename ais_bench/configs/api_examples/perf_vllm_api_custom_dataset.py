from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer

datasets = [
    dict(
        abbr='my_custom_dataset',
        type=CustomDataset,
        path='/path/to/your/dataset.jsonl',
        reader_cfg=dict(
            input_columns=['question'],
            output_column='answer',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template='{question}',
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=dict(
            evaluator=dict(type=AccEvaluator),
            pred_role='BOT',
        ),
        meta_path='',
    )
]

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-custom-dataset',
        model="",
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=8080,
        max_out_len=512,
        batch_size=1,
        generation_kwargs=dict(temperature=0.5, top_k=10, top_p=0.95),
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

work_dir = 'outputs/custom_dataset/'
