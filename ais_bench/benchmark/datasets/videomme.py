import re
import os
import string
import pandas as pd

from datasets import Dataset

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.prompt import AIS_CONTENT_TAG, AIS_TEXT_START, AIS_VIDEO_START

from .base import BaseDataset

logger = AISLogger()
IMAGE_MAP_LEN = 64
FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""


@LOAD_DATASET.register_module()
class VideoMMEDataset(BaseDataset):

    @staticmethod
    def load(path, video_path):
        path = get_data_path(path)
        data = pd.read_parquet(path)
        dataset = []
        for i in range(len(data)):
            line = data.iloc[i]
            video_url = os.path.abspath(os.path.join(video_path, line['videoID'] + '.mp4'))
            line['question'] += '\n' + '\n'.join(line['options'].tolist())
            prompt = 'Question: {}\nAnswer: '.format(line['question'])
            content = AIS_VIDEO_START + video_url + AIS_CONTENT_TAG \
                        + AIS_TEXT_START + FRAMES_TMPL_NOSUB + AIS_CONTENT_TAG \
                            + AIS_TEXT_START + prompt + AIS_CONTENT_TAG
            dataset.append({"content": content, 
                            "answer": {'duration': line['duration'], 
                                        'domain': line['domain'],
                                        'sub_category': line['sub_category'],
                                        'task_type': line['task_type'],
                                        'answer': line['answer']}})
        return Dataset.from_list(dataset)


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is',
        'The correct option is',
        'Best answer:',
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCD]', s):
        return ''
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ''
    return matches[0]


class VideoMMEEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        result = {}
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        details = []
        for pred, refer in zip(predictions, references):
            detail = {'pred': pred, 'answer': refer, 'correct': False}
            ans = refer['answer']
            extracted_char = extract_characters_regex(pred)
            if extracted_char == '':
                extract_pred = 'Z'
                score = int(extract_pred == ans)
            else:
                score = int(extracted_char == ans)
            dur_overall_key = '[duration: ' + refer['duration'] + ']' + '[overall]'
            domain_key = '[duration: ' + refer['duration'] + ']' + '[domain: ' + refer['domain'] + ']'
            category_key = '[duration: ' + refer['duration'] + ']' + '[sub_category: ' + refer['sub_category'] + ']'
            task_type = '[duration: ' + refer['duration'] + ']' + '[task_type: ' + refer['task_type'] + ']'
            overall_key = '[Overall]' + '[overall]'
            overall_domain_key = '[Overall]' + '[domain: ' + refer['domain'] + ']'
            overall_category_key = '[Overall]' + '[sub_category: ' + refer['sub_category'] + ']'
            overall_task_type = '[Overall]' + '[task_type: ' + refer['task_type'] + ']'

            if score == 1:
                detail['correct'] = True
            details.append(detail)
            result.setdefault(dur_overall_key, []).append(score)
            result.setdefault(domain_key, []).append(score)
            result.setdefault(category_key, []).append(score)
            result.setdefault(task_type, []).append(score)
            result.setdefault(overall_key, []).append(score)
            result.setdefault(overall_domain_key, []).append(score)
            result.setdefault(overall_category_key, []).append(score)
            result.setdefault(overall_task_type, []).append(score)
        for key in result:
            result[key] = 100 * sum(result[key]) / len(result[key])
        result['details'] = details
        return result