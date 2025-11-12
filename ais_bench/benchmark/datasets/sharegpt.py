import json
import os
import random

from datasets import Dataset

from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.file import load_tokenizer
from ais_bench.benchmark.utils.logging import get_logger
from ais_bench.benchmark.datasets.base import BaseDataset

MIN_PROMPT_LEN = 4
MIN_OUTPUT_LEN = 4
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@LOAD_DATASET.register_module()
class ShareGPTDataset(BaseDataset):

    @staticmethod
    def load(path, disable_shuffle, **kwargs):
        tokenizer_path = kwargs.get("model_path", None)
        tokenizer = load_tokenizer(tokenizer_path)
        path = get_data_path(path, local_mode=True)
        with open(path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        cnt_turn = 0
        logger = get_logger()
        new_dataset = []
        for data in dataset:
            if len(data["conversations"]) % 2 != 0:
                continue
            if data["conversations"][0]["from"] != "human":
                continue
            chat = {"question": [], "answer": [], "max_out_len": []}
            chat["id"] = data["id"]
            total_len = len(data["conversations"])
            cnt_turn += total_len
            for i in range(0, total_len, 2):
                # One user One Assistant
                chat["question"].append(data["conversations"][i]["value"])
                try:
                    output_len = len(
                        tokenizer.encode(data["conversations"][i + 1]["value"])
                    )
                except:
                    output_len = None
                chat["max_out_len"] = output_len
                chat["answer"].append(data["conversations"][i + 1]["value"])
            new_dataset.append(chat)
        logger.info(
            f"Number of conversations: {len(dataset)}; Number of requests: {cnt_turn // 2}"
        )
        if not disable_shuffle:
            # Shuffle the dataset.
            random.shuffle(new_dataset)

        return Dataset.from_list(new_dataset)


class ShareGPTEvaluator(BaseEvaluator):

    def find_choice(self, result):
        choose_map = {
            "A": "laughter",
            "B": "sigh",
            "C": "cough",
            "D": "throatclearing",
            "E": "sneeze",
            "F": "sniff"
        }
        if result in choose_map.keys():
            return choose_map[result]
        else:
            return ""

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            detail = {"pred": i, "answer": j, "correct": False}
            if len(i) > 1:
                i = self.find_choice(i[0])
            count += 1
            if i == j:
                correct += 1
                detail["correct"] = True
            details.append(detail)
        result = {"accuracy": 100 * correct / count, "details": details}
        return result
