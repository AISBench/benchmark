from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets.ocrbench_v2 import OCRBenchV2Dataset, OCRBenchV2Evaluator

ocrbench_v2_reader_cfg = dict(
    input_columns=['question', 'image'],
    output_column='answer'
)

ocrbench_v2_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={
                    "text": {"type": "text", "text": "{question}"},
                    "image": {"type": "image_url", "image_url": {"url": "file://{image}"}},
                })
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

ocrbench_v2_eval_cfg = dict(
    evaluator=dict(type=OCRBenchV2Evaluator)
)

ocrbench_v2_datasets = [
    dict(
        abbr='ocrbench_v2',
        type=OCRBenchV2Dataset,
        path='ais_bench/datasets/ocrbench_v2/OCRBench_v2.tsv', # Dataset path. Relative paths are relative to the source root; absolute paths are supported
        reader_cfg=ocrbench_v2_reader_cfg,
        infer_cfg=ocrbench_v2_infer_cfg,
        eval_cfg=ocrbench_v2_eval_cfg
    )
]