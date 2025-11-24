import json
import os
import pandas as pd
import numpy as np

from datasets import Dataset

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.datasets.utils.datasets import get_data_path, toliststr, process_line, get_content_str
from ais_bench.benchmark.datasets.mmmu import dump_image 

from .base import BaseDataset

logger = AISLogger()
IMAGE_MAP_LEN = 64
ANLS_THRESHOLD= 0.5


@LOAD_DATASET.register_module()
class DocVQADataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        image_root_path = os.path.join(os.path.dirname(path), "DocVQA_images")
        skip_noimg = True
        
        data = pd.read_csv(path, sep='\t')
        if skip_noimg and 'image' in data:
            data = data[~pd.isna(data['image'])]
        # The image field can store the base64 encoded image or another question index (for saving space)
        data['index'] = [str(x) for x in data['index']]
        if 'image' in data:
            data['image'] = [str(x) for x in data['image']]
            image_map = {x: y for x, y in zip(data['index'], data['image'])}
            for k in image_map:
                if len(image_map[k]) <= IMAGE_MAP_LEN:
                    idx = image_map[k]
                    image_map[k] = image_map[idx]

            images = [toliststr(image_map[k]) for k in data['index']]
            data['image'] = [x[0] if len(x) == 1 else x for x in images]
        if 'image_path' in data:
            paths = [toliststr(x) for x in data['image_path']]
            data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]

        if np.all([isinstance(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        sheet_indices = list(range(0, len(data), 1))
        data = data.iloc[sheet_indices]
        dataset = []
        for i in sheet_indices:
            line = data.iloc[i]
            tgt_path = dump_image(line, image_root_path)

            prompt = line['question']
            prompt += '\nAnswer the question using a single word or phrase.'
            # add image info
            msgs = []
            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image_url', image_url=p) for p in tgt_path])
            else:
                msgs = [dict(type='image_url', image_url=tgt_path)]
            msgs.append(dict(type='text', text=prompt))
            # get content str
            content = get_content_str(msgs)
            dataset.append({"content": content, 
                            "answer": line['answer']})
        return Dataset.from_list(dataset)


class DocVQAEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        details = []
        scores = []
        special_characters = ['<|im_end|>']
        for pred, ref in zip(predictions, references):
            if isinstance(ref, list):
                refer = ref
            elif isinstance(eval(ref), list):
                refer = eval(ref)
            else:
                refer = list(ref)
            detail = {'pred': pred, 'answer': refer, 'correct': False}
            for char in special_characters:
                if char in pred:
                    pred = pred.replace(char, '')
            res = process_line(pred, refer, method='anls')
            score = 0.0 if 1 - np.min(res) < ANLS_THRESHOLD else 1 - np.min(res)
            if score > 0:
                detail['correct'] = True
            details.append(detail)
            scores.append(score)
        results = {"ANLS Acc": 100 * sum(scores) / len(scores)}
        return results