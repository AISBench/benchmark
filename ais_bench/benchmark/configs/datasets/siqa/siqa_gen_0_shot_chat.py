from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import EDAccEvaluator
from ais_bench.benchmark.datasets import siqaDataset_V2

siqa_reader_cfg = dict(
    input_columns=['context', 'question', 'answerA', 'answerB', 'answerC'],
    output_column='all_labels',
    test_split='validation')

siqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=
                    '{context}\nQuestion: {question}\nA. {answerA}\nB. {answerB}\nC. {answerC}\nAnswer:'
                )
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

siqa_eval_cfg = dict(
    evaluator=dict(type=EDAccEvaluator),
    pred_role='BOT',
)

siqa_datasets = [
    dict(
        abbr='siqa',
        type=siqaDataset_V2,
        path='ais_bench/datasets/siqa',
        reader_cfg=siqa_reader_cfg,
        infer_cfg=siqa_infer_cfg,
        eval_cfg=siqa_eval_cfg)
]
