import json
import os
import string
import pandas as pd
import numpy as np

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path, toliststr
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.datasets import dump_image, split_MMMU, build_choices, can_infer
from ais_bench.benchmark.utils.prompt import AIS_CONTENT_TAG, AIS_TEXT_START, AIS_IMAGE_START

from .base import BaseDataset

IMAGE_MAP_LEN = 64
logger = AISLogger()

@LOAD_DATASET.register_module()
class MMStarDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        image_root_path = os.path.join(os.path.dirname(path), "MMStar_images")
        logger.info(f"Convert base64 to image and save it in {image_root_path}")
        skip_noimg = True
        
        data = pd.read_csv(path, sep='\t')
        if skip_noimg and 'image' in data:
            data = data[~pd.isna(data['image'])]
        # The image field can store the base64 encoded image or another question index (for saving space)
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

            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            # get text prompt 
            prompt = ''
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            prompt += f'Question: {line["question"]}\n'
            if len(options):
                prompt += options_prompt
                prompt += 'Please select the correct answer from the options above. \n'
            # add image info
            if isinstance(tgt_path, list):
                tgt_path = tgt_path[0]
                
            content = AIS_IMAGE_START + tgt_path + AIS_CONTENT_TAG \
                            + AIS_TEXT_START + prompt + AIS_CONTENT_TAG
            choices = build_choices(line)
            dataset.append({"content": content, 
                            "answer": {'choices': json.dumps(choices),
                                        'answer': line['answer'],
                                        'split': line.get('split'),
                                        'l2-category': line.get('l2-category'),
                                        'category': line.get('category')}})
        return Dataset.from_list(dataset)

class MMStarEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        result = {}
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        details = []
        overall_key = 'Overall'
        for pred, refer in zip(predictions, references):
            detail = {'pred': pred, 'answer': refer, 'correct': False}
            choices = json.loads(refer['choices'])
            infer_res = can_infer(pred, choices)
            
            key_category = refer['category']
            score = 1 if infer_res == refer['answer'] else 0
            if score == 1:
                detail['correct'] = True
            details.append(detail)
            result.setdefault(overall_key, []).append(score)
            result.setdefault(key_category, []).append(score)
        for key in result:
            result[key] = 100 * sum(result[key]) / len(result[key])
        result['details'] = details
        return result
    