from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import OmniDocBenchDataset, OmniDocBenchEvaluator


metrics_list = {'text_block': {'metric': ['Edit_dist']},
                'display_formula': {'metric': ['Edit_dist']}, 
                'table': {'metric': ['Edit_dist']}, 
                'reading_order': {'metric': ['Edit_dist']}}

omnidocbench_reader_cfg = dict(
    input_columns=['image_url'],
    output_column='answer'
)

omnidocbench_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={"text": "Parse the image",
                                           "image_url": "{image_url}"})
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

omnidocbench_eval_cfg = dict(
    evaluator=dict(type=OmniDocBenchEvaluator, metrics_list=metrics_list)
)

omnidocbench_datasets = [
    dict(
        abbr='omnidocbench',
        type=OmniDocBenchDataset,
        path='ais_bench/datasets/OmniDocBench/OmniDocBench.json', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        image_path='ais_bench/datasets/OmniDocBench/images',
        reader_cfg=omnidocbench_reader_cfg,
        infer_cfg=omnidocbench_infer_cfg,
        eval_cfg=omnidocbench_eval_cfg
    )
]