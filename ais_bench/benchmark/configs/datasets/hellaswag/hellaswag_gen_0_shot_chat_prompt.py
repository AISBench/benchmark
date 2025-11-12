from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import HellaswagDataset_V2
from ais_bench.benchmark.utils.postprocess.text_postprocessors import first_option_postprocess

hellaswag_reader_cfg = dict(
    input_columns=['ctx', 'A', 'B', 'C', 'D'],
    output_column='label',
)

hellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=('{ctx}\nQuestion: Which ending makes the most sense?\n'
                        'A. {A}\nB. {B}\nC. {C}\nD. {D}\n'
                        "You may choose from 'A', 'B', 'C', 'D'.\n"
                        'Answer:'),
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever, ),
    inferencer=dict(type=GenInferencer),
)

hellaswag_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

hellaswag_datasets = [
    dict(
        abbr='hellaswag',
        type=HellaswagDataset_V2,
        path='ais_bench/datasets/hellaswag/hellaswag.jsonl',
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg)
]
