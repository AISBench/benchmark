from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import GPQADataset, GPQAEvaluator
from ais_bench.benchmark.utils.postprocess.text_postprocessors import first_option_postprocess

gpqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer')

gpqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='What is the correct answer to this question: {question}\nChoices:\n'
                                          '(A){A}\n'
                                          '(B){B}\n'
                                          '(C){C}\n'
                                          '(D){D}\n'
                                          'Format your response as follows: "The correct answer is (insert answer here)"'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

gpqa_eval_cfg = dict(evaluator=dict(type=GPQAEvaluator),
                     pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

gpqa_datasets = []
gpqa_subsets = {
    # 'extended': 'gpqa_extended.csv',
    # 'main': 'gpqa_main.csv',
    'diamond': 'gpqa_diamond.csv'
}

for split in list(gpqa_subsets.keys()):
    gpqa_datasets.append(
        dict(
            abbr='GPQA_' + split,
            type=GPQADataset,
            path='ais_bench/datasets/gpqa/',  # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
            name=gpqa_subsets[split],
            reader_cfg=gpqa_reader_cfg,
            infer_cfg=gpqa_infer_cfg,
            eval_cfg=gpqa_eval_cfg)
    )
