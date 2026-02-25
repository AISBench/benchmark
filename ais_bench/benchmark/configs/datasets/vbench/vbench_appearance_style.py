from ais_bench.benchmark.datasets import VBenchDataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

vbench_reader_cfg = dict(
    input_columns=['dummy'],
    output_column='dummy',
)

vbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='dummy'),
                dict(role='BOT', prompt='dummy'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

vbench_eval_cfg = dict(
    use_vbench_task=True,
    device='npu',
    load_ckpt_from_local=True,
)

_BASE_PATH = '/data/zhanggaohua/datasets/vbench/CogVideoX-5B-mini'

vbench_standard_datasets = [
    dict(
        abbr='vbench_appearance_style',
        type=VBenchDataset,
        path=_BASE_PATH,
        reader_cfg=vbench_reader_cfg,
        infer_cfg=vbench_infer_cfg,
        eval_cfg=dict(
            **vbench_eval_cfg,
            dimension_list=['appearance_style'],
        ),
    )
]

