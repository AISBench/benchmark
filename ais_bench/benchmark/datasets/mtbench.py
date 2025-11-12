import json

from datasets import Dataset

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.logging import get_logger

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MTBenchDataset(BaseDataset):
    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        cnt_turn = 0
        logger = get_logger()
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                chat = {"question": [], "answer": []}
                chat['id'] = data['question_id']
                chat["question"] = data["prompt"]
                cnt_turn += len(data["prompt"])
                if data.get("reference", ""):
                    assert len(data["reference"])==len(data["prompt"])
                    answers = data["reference"]
                else:
                    answers = [""] * len(data["prompt"])
                chat["answer"] = answers
                dataset.append(chat)

        logger.info(f"Number of conversations: {len(dataset)}; Number of requests: {cnt_turn}")
        return Dataset.from_list(dataset)

class MTBenchEvaluator(BaseEvaluator):

    def find_choice(self, result):
        choose_map = {
            "A": "laughter",
            "B": "sigh",
            "C": "cough",
            "D": "throatclearing",
            "E": "sneeze",
            "F": "sniff"
        }
        if choose_map.get(result, ""):
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
            detail = {'pred': i, 'answer': j, 'correct': False}
            if len(i) > 1:
                i = self.find_choice(i[0])
            count += 1
            if i == j:
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result