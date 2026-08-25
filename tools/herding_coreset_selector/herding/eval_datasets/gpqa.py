import csv
import os

from ais_bench.benchmark.datasets import GPQADataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES
from ais_bench.benchmark.utils.config.build import build_dataset_from_cfg

from herding.eval_datasets.dataset_base import EvalDatasetBase, reg_eval_dataset


FILENAME = "gpqa_diamond.csv"

align_prompt = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

prompt_template_cfg = dict(
    type=PromptTemplate,
    template=align_prompt,
)

prompt_template = ICL_PROMPT_TEMPLATES.build(prompt_template_cfg)


@reg_eval_dataset("gpqa")
class GpqaDataset(EvalDatasetBase):
    def __init__(self, dataset_path, output_dir):
        super().__init__(dataset_path, output_dir)

        dataset_cfg = dict(
            abbr="GPQA_diamond",
            type=GPQADataset,
            path=self.dataset_path,
            name=FILENAME,
            reader_cfg=dict(
                input_columns=["question", "A", "B", "C", "D"],
                output_column="answer",
            ),
        )
        self.dataset = build_dataset_from_cfg(dataset_cfg).test

    def dataset_size(self):
        return len(self.dataset)

    def dataset_prompts(self):
        for item in self.dataset:
            yield prompt_template.generate_item(item)

    def save_data_by_indices(self, indices, outpath):
        filepath = os.path.join(self.dataset_path, FILENAME)
        with open(filepath, newline="", encoding="utf-8") as f:
            data = list(csv.reader(f))

        header = [data[0]]
        data = data[1:]

        output_dir = os.path.join(self.output_dir, outpath)
        os.makedirs(output_dir, exist_ok=True)

        rearranged_data = [data[idx] for idx in indices]
        output_filepath = os.path.join(output_dir, FILENAME)
        with open(output_filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(header + rearranged_data)

        self.save_indices(indices, output_dir)
        return output_dir
