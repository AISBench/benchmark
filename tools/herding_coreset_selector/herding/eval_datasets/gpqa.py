import os
import csv

from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.datasets import GPQADataset
from ais_bench.benchmark.utils.config.build import build_dataset_from_cfg
from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES

from herding.eval_datasets.dataset_base import (
    EvalDatasetBase, reg_eval_dataset,
    CFG_BASE_DATASET_DIR, CFG_CORESET_OUT_DIR,
)

DATASETS_PATH = os.environ.get('DATASET_PATH', f'{CFG_BASE_DATASET_DIR}/gpqa')

align_prompt = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

dataset_cfg = dict(
    abbr='GPQA_diamond',
    type=GPQADataset,
    path=DATASETS_PATH,
    name='gpqa_diamond.csv',
    reader_cfg=dict(
        input_columns=['question', 'A', 'B', 'C', 'D'],
        output_column='answer',
    ),
)

prompt_template_cfg = dict(
    type=PromptTemplate,
    template=align_prompt,
)

dataset = build_dataset_from_cfg(dataset_cfg).test
prompt_template = ICL_PROMPT_TEMPLATES.build(prompt_template_cfg)


@reg_eval_dataset('gpqa')
class GpqaDataset(EvalDatasetBase):
    def dataset_size(self):
        return len(dataset)

    def dataset_prompts(self):
        for i in dataset:
            yield prompt_template.generate_item(i)

    def save_data_by_indices(self, indices, outpath):
        FILENAME = 'gpqa_diamond.csv'
        filepath = os.path.join(DATASETS_PATH, FILENAME)
        with open(filepath, newline='', encoding='utf-8') as f:
            data = list(csv.reader(f))

        header = [data[0]]
        data = data[1:]

        outpath = os.path.join(CFG_CORESET_OUT_DIR, outpath)
        os.makedirs(outpath, exist_ok=True)

        rearranged_data = [data[idx] for idx in indices]
        output_filepath = os.path.join(outpath, FILENAME)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            for line in (header + rearranged_data):
                writer.writerow(line)

        self.save_indices(indices, outpath)
        return outpath
