import json
import os

from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES

from herding.eval_datasets.dataset_base import EvalDatasetBase, reg_eval_dataset


FILENAME = "aime2025.jsonl"

prompt_template_cfg = dict(
    type=PromptTemplate,
    template="{question}\nPlease reason step by step, and put your final answer within \\boxed{}.",
)

prompt_template = ICL_PROMPT_TEMPLATES.build(prompt_template_cfg)


@reg_eval_dataset("aime2025")
class Aime2025Dataset(EvalDatasetBase):
    def __init__(self, dataset_path, output_dir):
        super().__init__(dataset_path, output_dir)
        self.dataset = self._load_data()

    def _load_data(self):
        filepath = os.path.join(self.dataset_path, FILENAME)
        with open(filepath, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def dataset_size(self):
        return len(self.dataset)

    def dataset_prompts(self):
        for item in self.dataset:
            yield prompt_template.generate_item(item)

    def save_data_by_indices(self, indices, outpath):
        output_dir = os.path.join(self.output_dir, outpath)
        os.makedirs(output_dir, exist_ok=True)

        selected_data = [self.dataset[idx] for idx in indices]
        output_filepath = os.path.join(output_dir, FILENAME)
        with open(output_filepath, "w", encoding="utf-8") as f:
            for item in selected_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        self.save_indices(indices, output_dir)
        return output_dir
