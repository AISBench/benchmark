import json
import os
import re
import copy
import string
from os import environ
import pandas as pd
import numpy as np

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.datasets.utils.datasets import get_data_path, toliststr, decode_base64_to_image_file, get_content_str

from .base import BaseDataset

logger = AISLogger()
IMAGE_MAP_LEN = 64

def dump_image(line, image_root_path):
    """Extracts and saves image(s) from a data record to the specified root directory.

    This function handles multiple image formats within a single data record (`line`):
    - Base64-encoded images (as string or list) are decoded and saved to disk.
    - If image paths are provided (absolute or relative), it validates their existence.
    - Saved filenames are determined by `image_path` if provided, otherwise by the record's `index`.

    Args:
        line (dict): A dictionary representing a data record. Expected keys include:
            - 'image' (str or list of str, optional): Base64-encoded image(s).
            - 'image_path' (str or list of str, optional): Target filename(s) for saving.
            - 'index' (str or int): Unique identifier used for naming output files.
        image_root_path (str): The root directory where images will be saved or looked up.

    Returns:
        list[str]: A list of absolute file paths pointing to the saved or verified image(s).
    """
    if 'image' in line:
        if isinstance(line['image'], list):
            tgt_path = []
            if 'image_path' in line:
                image_path = line['image_path']
            else:
                index = line['index']
                image_path = [f'{index}_{i}.png' for i in range(len(line['image']))]
            for img, im_name in zip(line['image'], image_path):
                path = os.path.join(image_root_path, im_name)
                if not os.path.exists(path):
                    decode_base64_to_image_file(img, path)
                tgt_path.append(path)

        elif isinstance(line['image'], str) and 'image_path' in line:
            tgt_path = os.path.join(image_root_path, line['image_path'])
            if not os.path.exists(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path = [tgt_path]
        else:
            tgt_path = os.path.join(image_root_path, f"{line['index']}.jpg")
            
            if not os.path.exists(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path = [tgt_path]
    else:
        tgt_path = toliststr(line['image_path'])
        read_ok_flag = [os.path.exists(x) for x in tgt_path]
        # Might be the Relative Path
        if not all(read_ok_flag):
            tgt_path_abs = [os.path.join(image_root_path, x) for x in tgt_path]
            read_ok_flag = [os.path.exists(x) for x in tgt_path_abs]
            tgt_path = tgt_path_abs
    return tgt_path

def split_MMMU(msgs):
    """Splits a mixed media message into separate text and image segments based on MMMU-style placeholders.

    This function processes a list of message parts (each being either text or an image URL) and parses
    a single text segment containing `<image N>` placeholders (where N is a 1-based image index).
    It expands these placeholders into individual image and text segments in the correct order,
    effectively "flattening" the interleaved multimodal content.

    The input `msgs` is expected to contain exactly one text part (with optional `<image N>` tags)
    and one or more image parts. If no `<image N>` tags are found, the original message list is returned.

    Args:
        msgs (list[dict]): A list of message dictionaries. Each dict must have a 'type' key,
            which is either:
            - 'text': with a 'text' key containing a string that may include `<image N>` placeholders.
            - 'image_url': with an 'image_url' key pointing to an image.
            Exactly one 'text' part is allowed.

    Returns:
        list[dict]: A new list of message segments where `<image N>` placeholders have been replaced
        with actual image entries (using 0-based indexing into the original image list), interleaved
        with the surrounding text fragments. Each entry is of type 'text' or 'image_url'.

    Example:
        Input:
            [
                {"type": "text", "text": "What is <image 1> and <image 2>?"},
                {"type": "image_url", "image_url": "url1"},
                {"type": "image_url", "image_url": "url2"}
            ]
        Output:
            [
                {"type": "text", "text": "What is "},
                {"type": "image_url", "image_url": "url1"},
                {"type": "text", "text": " and "},
                {"type": "image_url", "image_url": "url2"},
                {"type": "text", "text": "?"}
            ]

    Note:
        - Image indices in `<image N>` are 1-based in the text but map to 0-based indices in the `images` list.
        - The function assumes well-formed input (e.g., `<image N>` tags are properly formatted).
    """
    text, images = None, []
    for s in msgs:
        if s['type'] == 'image_url':
            images.append(s['image_url'])
        elif s['type'] == 'text':
            text = s['text']
    text_segs = text.split('<image ')
    if len(text_segs) == 1:
        return msgs

    segs = [dict(type="text", text=text_segs[0])]
    for i, seg in enumerate(text_segs):
        if i == 0:
            continue
        if not seg[0].isdigit() or seg[1] != '>':
            logger.warning("Invalid text seg, please check it!")
        image_idx = int(seg[0]) - 1
        segs.append(dict(type="image_url", image_url=images[image_idx]))
        segs.append(dict(type="text", text=seg[2:]))
    return segs

def build_choices(item):
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and (not pd.isna(item[ch])):
            ret[ch] = item[ch]
    return ret

@LOAD_DATASET.register_module()
class MMMUDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        image_root_path = os.path.join(os.path.dirname(path), "MMMU_images")
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
        lt = len(sheet_indices)
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
                            "answer": {'choices': json.dumps(choices), 
                                        'answer': line['answer'],
                                        'split': line['split'] if 'split' in line else None,
                                        'l2-category': line['l2-category'] if 'l2-category' in line else None,
                                        'category': line['category'] if 'category' in line else None}})
        return Dataset.from_list(dataset)


def can_infer_option(answer, choices):
    """Attempts to infer a single valid option from a model-generated answer string.

    This function analyzes the `answer` text to determine if it unambiguously selects
    one of the provided `choices` (typically single-letter options like 'A', 'B', etc.).
    It handles common failure patterns (e.g., API errors, refusal messages) and applies
    heuristic rules to avoid false positives—such as when a choice letter appears
    as part of natural language (e.g., "A" as an article).
    Args:
        answer (str): The raw output from a language or vision-language model.
        choices (dict or iterable): A collection of valid option identifiers (e.g., {'A', 'B', 'C'}).
            Only the keys (or elements) are used for matching.
    Returns:
        str or bool: 
            - The inferred choice (e.g., 'A') if confidence is high.
            - 'Z' if the model explicitly refuses or indicates inability to answer.
            - False if the answer is ambiguous, contains errors, or no valid choice is reliably detected.
    """
    verbose = os.environ.get('VERBOSE', 0)
    # Choices is a dictionary
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = copy.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3 and verbose: # three options
                logger.info(f'A might be a quantifier in the string: {answer}.')
                return False
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False

def can_infer_text(answer, choices):
    """Infers a single-choice answer by exact substring matching of normalized choice values.

    This function attempts to determine which option (from a dictionary of labeled choices)
    is selected in a free-form `answer` string. It normalizes both the answer and choice
    values to lowercase and checks if any choice value appears as a substring in the answer.
    To reduce false positives, it also enforces:
      - The answer length is not excessively longer than the total length of all choice texts.
      - Only uppercase ASCII letters (e.g., 'A', 'B', 'C') are allowed as choice keys.
    Args:
        answer (str): The model-generated or user-provided response string.
        choices (dict): A mapping from uppercase letters (e.g., 'A', 'B') to their corresponding
            textual labels (e.g., {'A': 'apple', 'B': 'banana'}). Values are converted to strings
            and normalized to lowercase for comparison.
    Returns:
        str or bool: 
            - The key (e.g., 'A') if exactly one choice value is found as a substring in `answer`.
            - False if zero or multiple matches are found, or if the answer is too long relative
              to the combined length of all choice texts.
    """
    answer = answer.lower()
    if len(answer) > 2 * sum(len(str(v)) for v in choices.values()):
        return False
    for k in choices:
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False

def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)

def sort_key(item):
    key, _ = item
    if key.startswith('[validation]'):
        dataset_order = 0
    elif key.startswith('[dev]'):
        dataset_order = 1
    else:
        dataset_order = 2

    task = key.split(': ', 1)[1]
    task_order = 0 if task == 'Overall' else 1

    return (dataset_order, task_order, task)


class MMMUEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        result = {}
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        details = []
        special_characters = ['<|im_end|>']
        for pred, refer in zip(predictions, references):
            for char in special_characters:
                if char in pred:
                    pred = pred.replace(char, '')
            detail = {'pred': pred, 'answer': refer, 'correct': False}
            choices = json.loads(refer['choices'])
            infer_res = can_infer(pred, choices)
            overall_key = '[' + refer['split'] + ']: Overall'
            key_category = '[' + refer['split'] + ']: ' +  refer['category']
            key_l2_category = '[' + refer['split'] + ']: ' +  refer['l2-category']
            score = 1 if infer_res == refer['answer'] else 0
            if score == 1:
                detail['correct'] = True
            details.append(detail)
            result.setdefault(overall_key, []).append(score)
            result.setdefault(key_category, []).append(score)
            result.setdefault(key_l2_category, []).append(score)
        for key in result:
            result[key] = 100 * sum(result[key]) / len(result[key])
        sorted_items = sorted(result.items(), key=sort_key)
        result = dict(sorted_items)
        result['details'] = details
        return result