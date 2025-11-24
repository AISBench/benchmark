import json
import os
import string
import pandas as pd
import numpy as np
from collections import defaultdict

from datasets import Dataset

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.datasets.utils.datasets import get_data_path, toliststr, get_content_str
from ais_bench.benchmark.datasets import dump_image, split_MMMU, build_choices, can_infer

from .base import BaseDataset

logger = AISLogger()
IMAGE_MAP_LEN = 64
ANSWER_STR_LEN = 7


@LOAD_DATASET.register_module()
class MMMUProOptions10Dataset(BaseDataset):

    @staticmethod
    def load(path, is_cot=False):
        path = get_data_path(path)
        image_root_path = os.path.join(os.path.dirname(path), "MMMU_Pro_options10_images")
        skip_noimg = True
        try:
            data = pd.read_csv(path, sep='\t')
        except:
            raise FileNotFoundError
        if skip_noimg and 'image' in data:
            data = data[~pd.isna(data['image'])]
        # The image field can store the base64 encoded image or another question index (for saving space)
        if 'image' in data:
            data['image'] = [str(x) for x in data['image']]
            image_map = {x: y for x, y in zip(data['index'], data['image'])}
            for k in image_map:
                if len(image_map[k]) <= IMAGE_MAP_LEN:
                    idx = image_map[k]
                    if idx in image_map:
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
                if is_cot:
                    prompt += (
                        "Answer the following multiple-choice question. The last line of your response should be of "
                        "the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of the options. "
                        "Think step by step before answering. "
                    )
                else:
                    prompt += "Answer directly with the option letter from the given choices. "
            # add image info
            msgs = []
            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image_url', image_url=p) for p in tgt_path])
            else:
                msgs = [dict(type='image_url', image_url=tgt_path)]
            msgs.append(dict(type='text', text=prompt))
            # split image text in order
            msgs = split_MMMU(msgs)
            content = get_content_str(msgs)
            choices = build_choices(line)
            dataset.append({"content": content, 
                            "answer": json.dumps([choices, line['answer'], line['category']])})
        return Dataset.from_list(dataset)


@LOAD_DATASET.register_module()
class MMMUProVisionDataset(BaseDataset):

    @staticmethod
    def load(path, is_cot=False):
        path = get_data_path(path)
        image_root_path = os.path.join(os.path.dirname(path), "MMMU_Pro_vision_images")
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

            question = 'Answer the following multiple-choice question in the image. '
            if is_cot:
                question += (
                    "The last line of your response should be of the following format: 'Answer: $LETTER' "
                    "(without quotes) where LETTER is one of the options. Think step by step before answering. "
                )
            else:
                question += "Answer directly with the option letter from the given choices. "
            # add image info
            msgs = []
            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image_url', image_url=tgt_path[0])])
            else:
                msgs = [dict(type='image_url', image_url=tgt_path)]
            msgs.append(dict(type='text', text=question))
            # split image text in order
            msgs = split_MMMU(msgs)
            content = get_content_str(msgs)
            choices = build_choices(line)
            dataset.append({"content": content, 
                            "answer": json.dumps([choices, line['answer'], line['category']])})
        return Dataset.from_list(dataset)


class MMMUProEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        overall_key = 'Overall'
        result, overall = {}, {overall_key:[]}
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        details = []
        special_characters = ['<|im_end|>']
        for pred, refer in zip(predictions, references):
            detail = {'pred': pred, 'answer': refer, 'correct': False}
            for char in special_characters:
                if char in pred:
                    pred = pred.replace(char, '')
            choices, answer, category = json.loads(refer)
            infer_res = can_infer(pred, choices)
            
            score = 1 if infer_res == answer else 0
            if score == 1:
                detail['correct'] = True
            details.append(detail)
            overall[overall_key].append(score)
            result.setdefault(category, []).append(score)
        for key in result:
            result[key] = 100 * sum(result[key]) / len(result[key])
        overall[overall_key] = 100 * sum(overall[overall_key]) / len(overall[overall_key])
        sorted_items = sorted(result.items())
        result = dict(sorted_items)
        overall.update(result)
        overall['details'] = details
        return overall

def cot_postproc(response):
    lines = response.strip().split('\n')
    lines = [x.strip() for x in lines]
    cands = [x for x in lines if x.startswith('Answer:')]
    if len(cands) == 1:
        counter = defaultdict(lambda: 0)
        for ch in cands[0]:
            if ch in string.ascii_uppercase:
                counter[ch] += 1
        if len(counter) == 1:
            return list(counter.keys())[0]
        else:
            return cands[0][ANSWER_STR_LEN:]
    return response
    
class MMMUProCotEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        overall_key = 'Overall'
        result, overall = {}, {overall_key:[]}
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        details = []
        special_characters = ['<|im_end|>']
        for pred, refer in zip(predictions, references):
            detail = {'pred': pred, 'answer': refer, 'correct': False}
            for char in special_characters:
                if char in pred:
                    pred = pred.replace(char, '')
            pred = cot_postproc(pred)
            choices, answer, category = json.loads(refer)
            infer_res = can_infer(pred, choices)
            
            score = 1 if infer_res == answer else 0
            if score == 1:
                detail['correct'] = True
            details.append(detail)
            overall[overall_key].append(score)
            result.setdefault(category, []).append(score)
        for key in result:
            result[key] = 100 * sum(result[key]) / len(result[key])
        overall[overall_key] = 100 * sum(overall[overall_key]) / len(overall[overall_key])
        sorted_items = sorted(result.items())
        result = dict(sorted_items)
        overall.update(result)
        overall['details'] = details
        return overall