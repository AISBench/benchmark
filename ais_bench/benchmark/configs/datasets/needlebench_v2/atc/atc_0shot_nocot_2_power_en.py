from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets.needlebench_v2.atc import NeedleBenchATCDataset
from ais_bench.benchmark.datasets.needlebench_v2.atc import needlebench_atc_postprocess_v2
from ais_bench.benchmark.datasets.needlebench_v2.atc import NeedleBenchATCEvaluator

# ----------------------- Prompt Settings ----------------------- #
needle_num_list = [2, 4, 8, 16, 32, 64, 128, 256, 512]
path = 'ais_bench/datasets/NeedleBench'
file_name = 'names.json'
repeats = 10

# ----------------------- Dataset Settings ----------------------- #

needlebench_datasets = []

needlebench_atc_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

needlebench_atc_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
    ),
)

needlebench_atc_eval_cfg = dict(
    evaluator=dict(type=NeedleBenchATCEvaluator),
    pred_postprocessor=dict(type=needlebench_atc_postprocess_v2),
)

for num_needles in needle_num_list:
    abbr = f'NeedleBenchATCDataset-{num_needles}Needle-EN'
    language = 'English'
    dataset_dict = {
        'abbr': abbr,
        'type': NeedleBenchATCDataset,
        'path': path,
        'file_name': file_name,
        'num_needles': num_needles,
        'language': language,
        'repeats': repeats,
        'reader_cfg': needlebench_atc_reader_cfg,
        'infer_cfg': needlebench_atc_infer_cfg,
        'eval_cfg': needlebench_atc_eval_cfg,
    }
    needlebench_datasets.append(dataset_dict)
