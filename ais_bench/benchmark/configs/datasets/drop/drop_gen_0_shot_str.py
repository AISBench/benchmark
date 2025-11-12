from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import DropOpenAIDataset, DropOpenAIEvaluator


drop_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='answers',
    train_split='validation',
    test_split='validation',
)


drop_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}'
        ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, stopping_criteria=['---', 'Passage', 'Question', 'You will be asked']),
    )

drop_eval_cfg = dict(evaluator=dict(type=DropOpenAIEvaluator))

drop_datasets = [
    dict(
        abbr='drop',
        type=DropOpenAIDataset,
        path='ais_bench/datasets/drop_simple_eval/dev.jsonl',  # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=drop_reader_cfg,
        infer_cfg=drop_infer_cfg,
        eval_cfg=drop_eval_cfg)
]
