import json
import os
import re
import copy
import string
import ast
from pathlib import Path
from os import environ
import pandas as pd
import numpy as np

from datasets import Dataset

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.datasets.utils.datasets import get_data_path, toliststr, decode_base64_to_image_file, get_content_str

from .base import BaseDataset

logger = AISLogger()
IMAGE_MAP_LEN = 64

MMMU_SUBSET_LIST = [
    'Accounting',
    'Agriculture',
    'Architecture_and_Engineering',
    'Art',
    'Art_Theory',
    'Basic_Medical_Science',
    'Biology',
    'Chemistry',
    'Clinical_Medicine',
    'Computer_Science',
    'Design',
    'Diagnostics_and_Laboratory_Medicine',
    'Economics',
    'Electronics',
    'Energy_and_Power',
    'Finance',
    'Geography',
    'History',
    'Literature',
    'Manage',
    'Marketing',
    'Materials',
    'Math',
    'Mechanical_Engineering',
    'Music',
    'Pharmacy',
    'Physics',
    'Psychology',
    'Public_Health',
    'Sociology',
]

MMMU_MULTI_CHOICE_TYPE = 'multiple-choice'
MMMU_OPEN_TYPE = 'open'

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


def _safe_list(value):
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except Exception:
            return [stripped]
    return [value]


def _answer_character(index):
    if index < 26:
        return chr(ord('A') + index)
    return str(index - 25)


def _format_mmmu_choices(choices):
    return '\n'.join(
        f'{_answer_character(index)}) {choice}' for index, choice in enumerate(choices)
    )


def _format_mmmu_letters(choices):
    return ','.join(_answer_character(index) for index in range(len(choices)))


def _build_mmmu_mcq_prompt(question, choices, prompt_template=None):
    if not prompt_template:
        raise ValueError('MMMU multiple-choice prompt template must be provided by dataset config.')
    return prompt_template.format(
        question=question,
        choices=_format_mmmu_choices(choices),
        letters=_format_mmmu_letters(choices),
    )


def _parquet_sort_key(path):
    path_str = str(path).replace('\\', '/')
    subset_order = len(MMMU_SUBSET_LIST)
    for index, subset in enumerate(MMMU_SUBSET_LIST):
        if f'/{subset}/' in path_str or f'/{subset}-' in path_str or f'_{subset}_' in path_str:
            subset_order = index
            break
    return subset_order, path_str


def _find_mmmu_parquet_files(root, split, subset_list=None):
    root = Path(root)
    if root.is_file():
        if root.suffix.lower() != '.parquet':
            raise ValueError(f'MMMU only supports local parquet files now, got: {root}')
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f'MMMU dataset path does not exist: {root}')

    subsets = subset_list or MMMU_SUBSET_LIST
    split_patterns = [
        f'{split}-*.parquet',
        f'{split}.parquet',
        f'*{split}*.parquet',
    ]
    files = []
    for subset in subsets:
        subset_dirs = [
            root / subset,
            root / 'data' / subset,
            root / subset / 'data',
        ]
        for subset_dir in subset_dirs:
            if not subset_dir.is_dir():
                continue
            for pattern in split_patterns:
                files.extend(subset_dir.glob(pattern))

        subset_patterns = [
            f'**/{subset}/{split}-*.parquet',
            f'**/{subset}/{split}.parquet',
            f'**/{subset}/*{split}*.parquet',
            f'**/{split}-{subset}-*.parquet',
            f'**/{split}_{subset}_*.parquet',
            f'**/{subset}-{split}-*.parquet',
            f'**/{subset}_{split}_*.parquet',
        ]
        for pattern in subset_patterns:
            files.extend(root.glob(pattern))

    if not files:
        for pattern in split_patterns:
            files.extend(root.glob(f'**/{pattern}'))
    return sorted(set(files), key=_parquet_sort_key)


def _infer_subject_from_parquet_path(parquet_path):
    path_str = str(parquet_path).replace('\\', '/')
    for subset in MMMU_SUBSET_LIST:
        if f'/{subset}/' in path_str or f'/{subset}-' in path_str or f'_{subset}_' in path_str:
            return subset
    return None


def _load_mmmu_records(path, split='validation', subset_list=None):
    resolved_path = get_data_path(path)
    parquet_files = _find_mmmu_parquet_files(resolved_path, split, subset_list=subset_list)
    if not parquet_files:
        raise FileNotFoundError(
            f'No MMMU parquet files found under {resolved_path}. '
            f'Expected files such as {split}-*.parquet in subject subdirectories.'
        )

    records = []
    for parquet_file in parquet_files:
        data = pd.read_parquet(parquet_file)
        subject = _infer_subject_from_parquet_path(parquet_file)
        if subject and 'subject' not in data.columns:
            data['subject'] = subject
        records.extend(row.to_dict() for _, row in data.iterrows())
    return records, resolved_path, True


def _resolve_mmmu_existing_image_path(image_path, data_root, image_root_path):
    if not image_path:
        return None
    image_path = str(image_path)
    candidates = [image_path]
    if data_root and not os.path.isabs(image_path):
        candidates.append(os.path.join(data_root, image_path))
    if image_root_path and not os.path.isabs(image_path):
        candidates.append(os.path.join(image_root_path, image_path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def _write_mmmu_image_bytes(image_bytes, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if isinstance(image_bytes, list):
        image_bytes = bytes(image_bytes)
    with open(output_path, 'wb') as file:
        file.write(image_bytes)
    return output_path


def _build_mmmu_image_path(record, image_index, image_root_path, suffix='.png'):
    stem = record.get('id', record.get('index', 'sample'))
    stem = re.sub(r'[^0-9A-Za-z_.-]+', '_', str(stem))
    return os.path.join(image_root_path, f'{stem}_{image_index}{suffix}')


def _dump_mmmu_image(candidate, record, image_index, image_root_path, data_root):
    if candidate is None:
        return None
    if isinstance(candidate, float) and pd.isna(candidate):
        return None
    if isinstance(candidate, dict):
        path = _resolve_mmmu_existing_image_path(candidate.get('path'), data_root, image_root_path)
        if path:
            return path
        bytes_data = candidate.get('bytes')
        if bytes_data:
            suffix = Path(candidate.get('path', '')).suffix or '.png'
            return _write_mmmu_image_bytes(
                bytes_data,
                _build_mmmu_image_path(record, image_index, image_root_path, suffix=suffix),
            )
    if isinstance(candidate, (bytes, bytearray)):
        return _write_mmmu_image_bytes(
            candidate,
            _build_mmmu_image_path(record, image_index, image_root_path),
        )
    if hasattr(candidate, 'save'):
        output_path = _build_mmmu_image_path(record, image_index, image_root_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        candidate.save(output_path)
        return output_path
    if isinstance(candidate, str):
        path = _resolve_mmmu_existing_image_path(candidate, data_root, image_root_path)
        if path:
            return path
        output_path = _build_mmmu_image_path(record, image_index, image_root_path)
        try:
            decode_base64_to_image_file(candidate, output_path)
            return output_path
        except Exception:
            logger.debug('Failed to decode MMMU image string as base64; trying next candidate.')
    return None


def _collect_mmmu_images(record, image_root_path, data_root):
    image_map = {}
    for image_index in range(1, 8):
        image = record.get(f'image_{image_index}')
        image_path = _dump_mmmu_image(image, record, image_index, image_root_path, data_root)
        if image_path:
            image_map[image_index] = image_path

    if not image_map:
        candidates = _safe_list(record.get('image'))
        for image_index, image in enumerate(candidates, start=1):
            image_path = _dump_mmmu_image(image, record, image_index, image_root_path, data_root)
            if image_path:
                image_map[image_index] = image_path

    if not image_map:
        candidates = _safe_list(record.get('image_path'))
        for image_index, image in enumerate(candidates, start=1):
            image_path = _dump_mmmu_image(image, record, image_index, image_root_path, data_root)
            if image_path:
                image_map[image_index] = image_path
    return image_map


def _parse_mmmu_text_with_images(text, image_map):
    msgs = []
    pattern = r'<image[_ ](\d+)>'
    last_end = 0
    found_placeholder = False

    for match in re.finditer(pattern, text):
        found_placeholder = True
        if match.start() > last_end:
            text_segment = text[last_end:match.start()]
            if text_segment.strip():
                msgs.append(dict(type='text', text=text_segment))
        image_num = int(match.group(1))
        if image_num in image_map:
            msgs.append(dict(type='image_url', image_url=image_map[image_num]))
        last_end = match.end()

    if last_end < len(text):
        text_segment = text[last_end:]
        if text_segment.strip():
            msgs.append(dict(type='text', text=text_segment))

    if not found_placeholder and image_map:
        msgs = [dict(type='image_url', image_url=image_map[index]) for index in sorted(image_map)] + msgs
    return msgs


def _parse_mmmu_choice_prediction(prediction, num_choices):
    match = re.search(
        r'(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)',
        prediction,
        flags=re.MULTILINE,
    )
    if match is None:
        match = re.search(
            r'(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)',
            prediction,
        )
    if match is None:
        for letter in reversed(prediction):
            if letter.isupper():
                return letter
        return ''

    matched = match.group(1).strip().rstrip('.')
    allowed_options = {_answer_character(index) for index in range(num_choices)}
    return matched if matched in allowed_options else ''


def _extract_mmmu_open_prediction(prediction):
    match = re.search(r'ANSWER:\s*(.*)', prediction)
    if match:
        return match.group(1).strip()
    return prediction.strip()

@LOAD_DATASET.register_module()
class MMMUDataset(BaseDataset):

    @staticmethod
    def load(path='ais_bench/datasets/mmmu',
             split='validation',
             subset_list=None,
             mult_choice_prompt=None,
             open_prompt=None,
             start_text_prompt='',
             end_text_prompt='',
             option_prompt=''):
        records, resolved_path, is_local = _load_mmmu_records(
            path, split=split, subset_list=subset_list
        )
        if is_local:
            base_dir = resolved_path if os.path.isdir(resolved_path) else os.path.dirname(resolved_path)
        else:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'datasets'))
        image_root_path = os.path.join(base_dir, 'MMMU_images')
        os.makedirs(image_root_path, exist_ok=True)
        logger.info(f'Preparing MMMU images under {image_root_path}')
        data_root = resolved_path if is_local and os.path.isdir(resolved_path) else os.path.dirname(resolved_path) if is_local else None

        dataset = []
        for index, record in enumerate(records):
            question = str(record.get('question', '')).strip()
            if not question:
                logger.warning(f'Skipping MMMU record without question at index {index}')
                continue

            question_type = record.get('question_type') or (
                MMMU_MULTI_CHOICE_TYPE if _safe_list(record.get('options')) else MMMU_OPEN_TYPE
            )
            options = _safe_list(record.get('options'))
            if not options:
                options = [
                    record[ch] for ch in string.ascii_uppercase
                    if ch in record and not pd.isna(record[ch])
                ]

            if question_type == MMMU_MULTI_CHOICE_TYPE:
                prompt = _build_mmmu_mcq_prompt(question, options, mult_choice_prompt)
                choices = {
                    _answer_character(item_index): str(option)
                    for item_index, option in enumerate(options)
                }
                answer = {
                    'type': MMMU_MULTI_CHOICE_TYPE,
                    'choices': json.dumps(choices, ensure_ascii=False),
                    'answer': str(record.get('answer', '')).strip(),
                    'split': split,
                    'subject': record.get('subject'),
                    'subfield': record.get('subfield'),
                    'category': record.get('category', record.get('subject')),
                    'l2-category': record.get('l2-category', record.get('subfield')),
                }
            else:
                if not open_prompt:
                    raise ValueError('MMMU open-question prompt template must be provided by dataset config.')
                prompt = open_prompt.format(question=question)
                answer = {
                    'type': MMMU_OPEN_TYPE,
                    'answer': str(record.get('answer', '')).strip(),
                    'split': split,
                    'subject': record.get('subject'),
                    'subfield': record.get('subfield'),
                    'category': record.get('category', record.get('subject')),
                    'l2-category': record.get('l2-category', record.get('subfield')),
                }

            image_map = _collect_mmmu_images(record, image_root_path, data_root)
            msgs = _parse_mmmu_text_with_images(prompt, image_map)
            content = get_content_str(msgs)
            first_image = image_map[min(image_map)] if image_map else ''
            dataset.append({
                'content': content,
                'question': prompt,
                'image': first_image,
                'answer': answer,
            })
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
            refer = refer if isinstance(refer, dict) else {'type': MMMU_OPEN_TYPE, 'answer': refer}
            split = refer.get('split') or 'validation'
            category = refer.get('category') or refer.get('subject') or 'Unknown'
            l2_category = refer.get('l2-category') or refer.get('subfield') or category

            if refer.get('type') == MMMU_MULTI_CHOICE_TYPE:
                choices = json.loads(refer['choices']) if isinstance(refer.get('choices'), str) else refer.get('choices', {})
                parsed_pred = _parse_mmmu_choice_prediction(pred, len(choices))
                score = 1 if parsed_pred == str(refer.get('answer', '')).strip() else 0
            else:
                parsed_pred = _extract_mmmu_open_prediction(pred)
                score = 1 if parsed_pred.strip().lower() == str(refer.get('answer', '')).strip().lower() else 0

            overall_key = f'[{split}]: Overall'
            key_category = f'[{split}]: {category}'
            key_l2_category = f'[{split}]: {l2_category}'
            detail['parsed_pred'] = parsed_pred
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
