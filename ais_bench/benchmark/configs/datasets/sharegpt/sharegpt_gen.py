from ais_bench.benchmark.openicl.icl_prompt_template import MultiTurnPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import MultiTurnGenInferencer
from ais_bench.benchmark.datasets import ShareGPTDataset, ShareGPTEvaluator, math_postprocess_v2


sharegpt_reader_cfg = dict(
    input_columns=["question", "answer"],
    output_column="answer"
)


sharegpt_infer_cfg = dict(
    prompt_template=dict(
        type=MultiTurnPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt="{question}"),
                dict(role="BOT", prompt="{answer}"),
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=MultiTurnGenInferencer, infer_mode="every") # Default using "every" mode, Supports: "last", "every", "every_with_gt"
)

sharegpt_eval_cfg = dict(
    evaluator=dict(type=ShareGPTEvaluator)
)

sharegpt_datasets = [
    dict(
        abbr='sharegpt',
        type=ShareGPTDataset,
        disable_shuffle=True,
        path='ais_bench/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=sharegpt_reader_cfg,
        infer_cfg=sharegpt_infer_cfg,
        eval_cfg=sharegpt_eval_cfg
    )
]