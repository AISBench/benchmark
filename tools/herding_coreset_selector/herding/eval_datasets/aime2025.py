import os
import json

from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.utils.config.build import build_dataset_from_cfg
from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES

from herding.eval_datasets.dataset_base import (
    EvalDatasetBase, reg_eval_dataset,
    CFG_BASE_DATASET_DIR, CFG_CORESET_OUT_DIR,
)

DATASETS_PATH = os.environ.get('DATASET_PATH', f'{CFG_BASE_DATASET_DIR}/aime2025')


def load_aime_data():
    data = []
    filepath = os.path.join(DATASETS_PATH, 'aime2025.jsonl')
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    return data


dataset = load_aime_data()

prompt_template_cfg = dict(
    type=PromptTemplate,
    template='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.',
)

prompt_template = ICL_PROMPT_TEMPLATES.build(prompt_template_cfg)


@reg_eval_dataset('aime2025')
class Aime2025Dataset(EvalDatasetBase):
    def dataset_size(self):
        return len(dataset)

    def dataset_prompts(self):
        for i in dataset:
            yield prompt_template.generate_item(i)

    def save_data_by_indices(self, indices, outpath):
        FILENAME = 'aime2025.jsonl'
        filepath = os.path.join(DATASETS_PATH, FILENAME)
        with open(filepath, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line.strip()) for line in f]

        outpath = os.path.join(CFG_CORESET_OUT_DIR, outpath)
        os.makedirs(outpath, exist_ok=True)

        selected_data = [all_data[idx] for idx in indices]
        output_filepath = os.path.join(outpath, FILENAME)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for item in selected_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        self.save_indices(indices, outpath)
        return outpath
