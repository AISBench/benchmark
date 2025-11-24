import json
import os
import re
import base64
from os import environ

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.prompt import AIS_CONTENT_TAG, AIS_TEXT_START, AIS_IMAGE_START

from .base import BaseDataset


MAX_TARGET_LENGTH = 32768


class VQAEvalMethod:

    def __init__(self):
        self.contractions = {
            'aint': "ain't",
            'arent': "aren't",
            'cant': "can't",
            'couldve': "could've",
            'couldnt': "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            'didnt': "didn't",
            'doesnt': "doesn't",
            'dont': "don't",
            'hadnt': "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            'hasnt': "hasn't",
            'havent': "haven't",
            'hed': "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            'hes': "he's",
            'howd': "how'd",
            'howll': "how'll",
            'hows': "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            'Im': "I'm",
            'Ive': "I've",
            'isnt': "isn't",
            'itd': "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            'itll': "it'll",
            "let's": "let's",
            'maam': "ma'am",
            'mightnt': "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            'mightve': "might've",
            'mustnt': "mustn't",
            'mustve': "must've",
            'neednt': "needn't",
            'notve': "not've",
            'oclock': "o'clock",
            'oughtnt': "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            'shant': "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            'shouldve': "should've",
            'shouldnt': "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": 'somebodyd',
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            'somebodyll': "somebody'll",
            'somebodys': "somebody's",
            'someoned': "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            'someonell': "someone'll",
            'someones': "someone's",
            'somethingd': "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            'somethingll': "something'll",
            'thats': "that's",
            'thered': "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            'therere': "there're",
            'theres': "there's",
            'theyd': "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            'theyll': "they'll",
            'theyre': "they're",
            'theyve': "they've",
            'twas': "'twas",
            'wasnt': "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            'weve': "we've",
            'werent': "weren't",
            'whatll': "what'll",
            'whatre': "what're",
            'whats': "what's",
            'whatve': "what've",
            'whens': "when's",
            'whered': "where'd",
            'wheres': "where's",
            'whereve': "where've",
            'whod': "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            'wholl': "who'll",
            'whos': "who's",
            'whove': "who've",
            'whyll': "why'll",
            'whyre': "why're",
            'whys': "why's",
            'wont': "won't",
            'wouldve': "would've",
            'wouldnt': "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            'yall': "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            'youd': "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            'youll': "you'll",
            'youre': "you're",
            'youve': "you've",
        }

        self.manual_map = {
            'none': '0',
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'ten': '10',
        }

        self.articles = ['a', 'an', 'the']

        self.period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
        self.comma_strip = re.compile(r'(\d)(,)(\d)')
        self.punct = [
            ';',
            r'/',
            '[',
            ']',
            '"',
            '{',
            '}',
            '(',
            ')',
            '=',
            '+',
            '\\',
            '_',
            '-',
            '>',
            '<',
            '@',
            '`',
            ',',
            '?',
            '!',
        ]
        self.special_tokens = ['☞', '☟', '☜', '<unk>', '<|im_end|>']

    def process_punctuation(self, in_text):
        if len(in_text) > MAX_TARGET_LENGTH:
            raise ValueError(
                f"Invalid in_text length, should be no more than {MAX_TARGET_LENGTH} but got {len(in_text)}"
            ) 
        out_text = in_text
        for p in self.punct:
            if (p + ' ' in in_text or ' ' + p
                    in in_text) or (re.search(self.comma_strip, in_text)):
                out_text = out_text.replace(p, '')
            else:
                out_text = out_text.replace(p, ' ')
        out_text = self.period_strip.sub('', out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        if len(in_text) > MAX_TARGET_LENGTH:
            raise ValueError(
                f"Invalid in_text length, should be no more than {MAX_TARGET_LENGTH} but got {len(in_text)}"
            ) 
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.manual_map.setdefault(word, word)
            if word not in self.articles:
                out_text.append(word)
        for word_id, word in enumerate(out_text):
            if word in self.contractions:
                out_text[word_id] = self.contractions[word]

        return ' '.join(out_text)

    def remove_special_characters(self, input_str):
        for token in self.special_tokens:
            input_str = input_str.replace(token, '')
        return input_str


@LOAD_DATASET.register_module()
class TEXTVQADataset(BaseDataset):

    @staticmethod
    def load(path, image_type):
        """
        Load a TextVQA-style dataset that pairs images with question-answer
        annotations.

        Parameters
        ----------
        path : str
            Full path to the **question file** (usually `*_questions.json` or
            similar).  The corresponding ground-truth answer file is expected to
            live in the same directory and to have the suffix
            `_annotations.json`, e.g.
                /foo/bar/train_questions.json   # <-- `path`
                /foo/bar/train_annotations.json # <-- auto-detected
        image_type : str
            How the image should be returned:
            - "image_path"   : keep the original file path (str)
            - "image_base64" : read the image and return it as a base64-encoded
            string (str)

        Returns
        -------
        datasets.Dataset
            A HuggingFace `datasets.Dataset` where every row is a dictionary
            containing (at least) the keys that were present in the question
            file plus:
            - "answer"    : list of dicts, each dict has an "answer" key
                            (ground-truth answers loaded from the annotation file)
            - "image_url" : str, either the original path or the base64 string
                            depending on `image_type`
        """
        path = get_data_path(path, local_mode=True)

        parts = path.split('/')
        path_prefix = '/'.join(parts[:-1])
        path_suffix = parts[-1].split('.')[0] + "_annotations.json"
        textvqa_gt_json_path = os.path.join(path_prefix, path_suffix)
        gt_answer = {}
        with open(textvqa_gt_json_path, 'r', encoding='utf-8') as f:
            vqa_gt_answer = json.load(f)['annotations']
            for item in vqa_gt_answer:
                gt_answer[item.get('question_id')] = item.get('answers')

        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                try:
                    answer = gt_answer[line['question_id']]
                    line['answer'] = answer
                except:
                    raise ValueError("Please check your dataset!")
                # get image_url
                if image_type=="image_path":
                    image_url = line["image"]
                elif image_type=="image_base64":
                    with open(line['image'], 'rb') as f:
                        binary_data = f.read()
                    image_url = base64.b64encode(binary_data).decode("utf-8")
                line["image_url"] = image_url
                line["content"] = AIS_IMAGE_START + image_url + AIS_CONTENT_TAG \
                                    + AIS_TEXT_START + line['question'] + AIS_CONTENT_TAG
                dataset.append(line)

        return Dataset.from_list(dataset)


class TEXTEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        vqa_eval= VQAEvalMethod()
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        answer_ = "answer"
        details = []
        for res_data, gt_answers in zip(predictions, references):
            detail = {'pred': res_data, 'answer': gt_answers, 'correct': False}
            res_data = res_data.replace('\n', ' ')
            res_data = res_data.replace('\t', ' ')
            res_data = res_data.strip()
            res_data = vqa_eval.remove_special_characters(res_data)
            res_data = vqa_eval.process_punctuation(res_data)
            res_data = vqa_eval.process_digit_article(res_data)
            gt_answer_list = [ans[answer_] for ans in gt_answers]
            if len(set(gt_answer_list)) > 1:
                for ans_dic in gt_answers:
                    ans_dic[answer_] = vqa_eval.process_punctuation(ans_dic[answer_])
            gt_acc = []
            for gt_ans in gt_answers:
                other_gt_ans = [item for item in gt_answers if item != gt_ans]
                matched_ans = [item for item in other_gt_ans if item[answer_] == res_data]
                acc = min(1, len(matched_ans) / 3)
                gt_acc.append(acc)
            avg_acc = sum(gt_acc) / len(gt_acc)
            count += 1
            correct += avg_acc
            detail['correct'] = True if avg_acc > 0.5 else False
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result