from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MooncakeTraceDataset, MooncakeTraceEvaluator
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
import os

mooncake_trace_reader_cfg = dict[str, list[str] | str](
    input_columns=["prompt", "timestamp","max_out_len"],
    output_column="answer"
)

mooncake_trace_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{prompt}"
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

mooncake_trace_eval_cfg = dict(
    evaluator=dict(type=MooncakeTraceEvaluator)
)

# 获取当前配置文件所在目录，然后找到测试数据文件
_current_dir = os.path.dirname(os.path.abspath(__file__))
_test_data_path = os.path.join(_current_dir, '..', 'test_mooncake_trace.jsonl')

mooncake_trace_datasets = [
    dict(
        abbr='mooncake-trace',
        type=MooncakeTraceDataset,
        path=_test_data_path,  # 测试数据文件路径
        generated_prompts_path='',  # 生成的prompt缓存路径，使用相对路径时相对于源码根路径，支持绝对路径
        random_seed=1234,
        fixed_schedule_auto_offset=False,
        fixed_schedule_start_offset=0,
        fixed_schedule_end_offset=-1,
        reader_cfg=mooncake_trace_reader_cfg,
        infer_cfg=mooncake_trace_infer_cfg,
        eval_cfg=mooncake_trace_eval_cfg
    )
]
