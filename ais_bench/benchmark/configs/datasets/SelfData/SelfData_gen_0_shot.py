from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.utils.postprocess.text_postprocessors import match_answer_pattern



# 这里写template，最终输入给模型的数据是：template+question+choices，请注意你的数据选项个数，如果选项数量大于2则此处需要修改
QUERY_TEMPLATE = """
===在这里写Template

{question}

A) {A}
B) {B}
""".strip()

# template后的数据list
head_addr_datasets = []

# infra config
head_addr_infer_cfg = dict(
    # infra template
    prompt_template=dict(
        # 自定义template
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=QUERY_TEMPLATE),
            ],
        ),
    ),
    # 0shot
    retriever=dict(type=ZeroRetriever),
    # general infra
    inferencer=dict(type=GenInferencer),
)

# eval config
head_addr_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(
        type=match_answer_pattern,
        # 匹配选项，如果你的数据有ABCD则此处需要修改
        answer_pattern=r'(?i)答案\s*[:：]\s*[\W]*([AB])[\W]*',
    )
)

head_addr_datasets.append(
    dict(
        type=CustomDataset,
        # 这里写你自己的数据路径，建议数据放到ais_bench/datasets下，支持csv或jsonl格式的数据
        path='ais_bench/datasets/SelfData/YourData.csv',
        abbr='YourData_0_shot',
        reader_cfg=dict(
            input_columns=['question', 'A', 'B'],
            output_column='answer',
        ),
        infer_cfg=head_addr_infer_cfg,
        eval_cfg=head_addr_eval_cfg,
    )
)
