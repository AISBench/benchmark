from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import Geometry3KDataset, Geometry3KEvaluator

# ── Read the instruction text from the dataset module to keep parity ──
from ais_bench.benchmark.datasets.geometry3k import GEOMETRY3K_INSTRUCTION


geometry3k_reader_cfg = dict(
    input_columns=["question", "image"],
    output_column="answer",
)

geometry3k_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt_mm={
                        "text": {"type": "text", "text": "{question}"},
                        "image": {
                            "type": "image_url",
                            "image_url": {"url": "file://{image}"},
                        },
                    },
                )
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

geometry3k_eval_cfg = dict(
    evaluator=dict(type=Geometry3KEvaluator),
)

geometry3k_datasets = [
    dict(
        abbr="geometry3k",
        type=Geometry3KDataset,
        path="hiyouga/geometry3k",
        split="test",
        reader_cfg=geometry3k_reader_cfg,
        infer_cfg=geometry3k_infer_cfg,
        eval_cfg=geometry3k_eval_cfg,
    )
]
