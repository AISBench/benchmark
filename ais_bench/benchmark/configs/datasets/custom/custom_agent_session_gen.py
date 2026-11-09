from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import CustomDataset


# 基于 agent_hint 嵌套结构的数据集。
# 注意：input_columns 不需要显式写 session_id ——
# session_id 会在 icl_gen_inferencer.get_data_list 里自动从 agent_hint.session_id 提取。
# agent_hint 本身作为额外字段被透传到 vLLM 请求 body（不进入 prompt 模板）。
agent_reader_cfg = dict(
    input_columns=['question', 'max_out_len'],
    output_column=None,
)


agent_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{question}',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


agent_eval_cfg = dict(
    evaluator=None,
    pred_role='BOT',
)


agent_demo_datasets = [
    dict(
        abbr='session-agent',
        type=CustomDataset,
        path='ais_bench/datasets/agent/example.jsonl',
        reader_cfg=agent_reader_cfg,
        infer_cfg=agent_infer_cfg,
        eval_cfg=agent_eval_cfg,
    )
]
