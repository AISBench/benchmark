from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer.icl_lmm_gen_inferencer import LMMGenInferencer
from ais_bench.benchmark.datasets.g_edit import GEditDataset, GEditEvaluator


gedit_reader_cfg = dict(
    input_columns=['question', 'image'],
    output_column='task_type'
)


gedit_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={
                    "text": {"type": "text", "text": "{question}"},
                    "image": {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image}"}},
                })
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=LMMGenInferencer)
)

gedit_eval_cfg = dict(
    evaluator=dict(type=GEditEvaluator)
)

gedit_datasets = [
    dict(
        abbr='gedit',
        type=GEditDataset,
        path='ais_bench/datasets/GEdit-Bench', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        split_count=1,
        split_index=0,
        reader_cfg=gedit_reader_cfg,
        infer_cfg=gedit_infer_cfg,
        eval_cfg=gedit_eval_cfg
    )
]
