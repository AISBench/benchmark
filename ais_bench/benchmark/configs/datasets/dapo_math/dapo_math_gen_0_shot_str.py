from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import DAPOMathDataset, DAPOMathEvaluator, dapo_math_postprocess

# Reader configuration: specify input and output columns
dapo_math_reader_cfg = dict(
    input_columns=['prompt'],  # Input column: prompt content from the dataset
    output_column='answer'     # Output column: ground_truth from reward_model
)

# Inference configuration: 0-shot generation with string prompt
dapo_math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}'  # Use the prompt directly as it already contains the full instruction
    ),
    retriever=dict(type=ZeroRetriever),  # 0-shot: no retrieval
    inferencer=dict(type=GenInferencer)  # Generation-based inference for RL reasoning
)

# Evaluation configuration: accuracy-based evaluation
dapo_math_eval_cfg = dict(
    evaluator=dict(type=DAPOMathEvaluator), pred_postprocessor=dict(type=dapo_math_postprocess)
)

# Dataset configuration
dapo_math_datasets = [
    dict(
        abbr='dapo-math-17k',  # Dataset abbreviation
        type=DAPOMathDataset,  # Dataset class
        path='ais_bench/datasets/dapo-math-17k',
        reader_cfg=dapo_math_reader_cfg,
        infer_cfg=dapo_math_infer_cfg,
        eval_cfg=dapo_math_eval_cfg
    )
]