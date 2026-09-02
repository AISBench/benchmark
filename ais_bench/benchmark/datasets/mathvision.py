import ast
import json
import os
import re
import string
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.math import (
    MATHEvaluator,
    extract_answer,
    extract_boxed_answer,
    normalize_final_answer,
)
from ais_bench.benchmark.datasets.mmmu import can_infer
from ais_bench.benchmark.datasets.utils.datasets import (
    decode_base64_to_image_file,
    get_content_str,
    get_data_path,
)
from ais_bench.benchmark.utils.logging import AISLogger

from .base import BaseDataset

logger = AISLogger()

SPECIAL_CHARACTERS = ['<|im_end|>']
MCQ_ANSWER_PATTERN = re.compile(r'(?im)^ANSWER\s*:\s*([A-Z])\b')


def _resolve_dataset_path(path):
    if not path:
        return path, False
    if os.path.isabs(path):
        return path, True
    try:
        return get_data_path(path, local_mode=True), True
    except Exception:
        return path, False


def _pick_files(search_root, patterns):
    if not search_root.is_dir():
        return []
    for pattern in patterns:
        matches = sorted(search_root.glob(pattern))
        if matches:
            return [str(match) for match in matches]
    return []


def _infer_local_data_files(root, split):
    search_roots = [root / 'data', root]

    for search_root in search_roots:
        files = _pick_files(
            search_root, [f'{split}-*.parquet', f'{split}.parquet', '*.parquet']
        )
        if files:
            return 'parquet', {split: files if len(files) > 1 else files[0]}

    json_patterns = [
        f'{split}.jsonl',
        f'{split}.json',
        f'{split}*.jsonl',
        f'{split}*.json',
        '*.jsonl',
        '*.json',
    ]
    for search_root in search_roots:
        files = _pick_files(search_root, json_patterns)
        if files:
            return 'json', {split: files if len(files) > 1 else files[0]}

    return None, None


def _load_records(path, split='test'):
    resolved_path, is_local = _resolve_dataset_path(path)
    if is_local:
        root = Path(resolved_path)
        if root.is_file():
            suffix = root.suffix.lower()
            if suffix in ['.json', '.jsonl']:
                dataset = load_dataset('json', data_files={split: str(root)})
            elif suffix == '.parquet':
                dataset = load_dataset('parquet', data_files={split: str(root)})
            else:
                raise ValueError(f'Unsupported local MathVision file type: {root}')
        elif root.is_dir():
            loader_name, data_files = _infer_local_data_files(root, split)
            if loader_name is None:
                try:
                    dataset = load_dataset(str(root), split=split, trust_remote_code=True)
                    return list(dataset), resolved_path, is_local
                except Exception as exc:
                    raise FileNotFoundError(
                        f'Unable to locate local MathVision files under {root}. '
                        'Expected JSON/JSONL/Parquet data, optionally split-sharded.'
                    ) from exc
            dataset = load_dataset(loader_name, data_files=data_files)
        else:
            raise FileNotFoundError(f'MathVision path does not exist: {resolved_path}')
    else:
        dataset = load_dataset(path, split=split)
        return list(dataset), path, is_local

    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]
    return list(dataset), resolved_path, is_local


def _safe_literal(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return []
    if stripped[0] in '[{' and stripped[-1] in ']}':
        try:
            return json.loads(stripped)
        except Exception:
            try:
                return ast.literal_eval(stripped)
            except Exception:
                return value
    return value


def _normalize_options(options):
    options = _safe_literal(options)
    if options is None:
        return []
    if isinstance(options, dict):
        ordered = []
        for key in string.ascii_uppercase:
            value = options.get(key)
            if value is not None and str(value).strip():
                ordered.append(str(value))
        if ordered:
            return ordered
        return [
            str(value) for value in options.values()
            if value is not None and str(value).strip()
        ]
    if isinstance(options, (list, tuple)):
        return [str(option) for option in options if option is not None and str(option).strip()]
    if isinstance(options, str) and options.strip():
        return [options.strip()]
    return []


def _normalize_choice_answer(answer, options):
    answer = str(answer).strip()
    if not answer:
        return answer
    allowed_letters = string.ascii_uppercase[:len(options)]
    upper_answer = answer.upper()
    if upper_answer.isdigit():
        index = int(upper_answer)
        if 0 <= index < len(options):
            return allowed_letters[index]
        if 1 <= index <= len(options):
            return allowed_letters[index - 1]

    match = re.match(r'^([A-Z])(?:[\).:\s].*)?$', upper_answer)
    if match and match.group(1) in allowed_letters:
        return match.group(1)

    for index, option in enumerate(options):
        if answer == str(option).strip():
            return allowed_letters[index]
    return upper_answer


def _build_choices_text(options):
    return '\n'.join(
        f'{string.ascii_uppercase[index]}) {option}' for index, option in enumerate(options)
    )


def _build_mcq_prompt(question, options, prompt_template=None):
    letters = ','.join(string.ascii_uppercase[:len(options)])
    choices_text = _build_choices_text(options)
    if not prompt_template:
        return f'{question}\n\n{choices_text}'
    return prompt_template.format(question=question, choices=choices_text, letters=letters)


def _build_output_image_path(record, index, image_root_path, suffix='.png'):
    stem = record.get('id', record.get('index', index))
    stem = re.sub(r'[^0-9A-Za-z_.-]+', '_', str(stem))
    return os.path.join(image_root_path, f'{stem}{suffix}')


def _resolve_existing_image_path(image_path, data_root, image_root_path):
    if not image_path:
        return None
    candidates = []
    if isinstance(image_path, (list, tuple)):
        for item in image_path:
            resolved = _resolve_existing_image_path(item, data_root, image_root_path)
            if resolved:
                return resolved
        return None

    image_path = str(image_path)
    candidates.append(image_path)
    if data_root and not os.path.isabs(image_path):
        candidates.append(os.path.join(data_root, image_path))
    if image_root_path and not os.path.isabs(image_path):
        candidates.append(os.path.join(image_root_path, image_path))

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def _write_image_bytes(image_bytes, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if isinstance(image_bytes, list):
        image_bytes = bytes(image_bytes)
    with open(output_path, 'wb') as file:
        file.write(image_bytes)
    return output_path


def _dump_image(record, index, image_root_path, data_root):
    image_path = _resolve_existing_image_path(record.get('image_path'), data_root, image_root_path)
    if image_path:
        return image_path

    image_field = record.get('image')
    decoded_image = record.get('decoded_image')
    candidates = [image_field, decoded_image]

    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, dict):
            bytes_data = candidate.get('bytes')
            path = _resolve_existing_image_path(candidate.get('path'), data_root, image_root_path)
            if path:
                return path
            if bytes_data:
                suffix = Path(candidate.get('path', '')).suffix or '.png'
                output_path = _build_output_image_path(record, index, image_root_path, suffix=suffix)
                return _write_image_bytes(bytes_data, output_path)
        elif isinstance(candidate, (bytes, bytearray)):
            output_path = _build_output_image_path(record, index, image_root_path)
            return _write_image_bytes(candidate, output_path)
        elif hasattr(candidate, 'save'):
            output_path = _build_output_image_path(record, index, image_root_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            candidate.save(output_path)
            return output_path
        elif isinstance(candidate, str):
            path = _resolve_existing_image_path(candidate, data_root, image_root_path)
            if path:
                return path
            output_path = _build_output_image_path(record, index, image_root_path)
            try:
                decode_base64_to_image_file(candidate, output_path)
                return output_path
            except Exception:
                logger.debug('Failed to decode MathVision image string as base64; trying next candidate.')
        elif isinstance(candidate, (list, tuple)):
            for item in candidate:
                sub_record = {'id': record.get('id', index), 'image': item}
                path = _dump_image(sub_record, index, image_root_path, data_root)
                if path:
                    return path

    return None


@LOAD_DATASET.register_module()
class MathVisionDataset(BaseDataset):

    @staticmethod
    def load(path='evalscope/MathVision',
             split='test',
             open_prompt_template=None,
             single_answer_cot_template=None,
             subset_list=None):
        records, resolved_path, is_local = _load_records(path, split=split)

        # Normalize subset list
        valid_subsets = ['level 1', 'level 2', 'level 3', 'level 4', 'level 5']
        if subset_list:
            subset_list = [s.lower() if isinstance(s, str) else s for s in subset_list]
        else:
            subset_list = valid_subsets

        if is_local:
            base_dir = resolved_path if os.path.isdir(resolved_path) else os.path.dirname(resolved_path)
        else:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'datasets'))
        image_root_path = os.path.join(base_dir, 'MathVision_images')
        os.makedirs(image_root_path, exist_ok=True)
        logger.info(f'Preparing MathVision images under {image_root_path}')

        data_root = resolved_path if is_local and os.path.isdir(resolved_path) else os.path.dirname(resolved_path) if is_local else None
        dataset = []
        for index, record in enumerate(records):
            question = str(record.get('question', '')).strip()
            if not question:
                logger.warning(f'Skipping MathVision record without question at index {index}')
                continue

            # Filter by subset (level)
            record_level = record.get('level')
            if record_level is not None:
                level_key = f'level {record_level}'.lower()
                if level_key not in subset_list:
                    logger.debug(f'Skipping record at index {index} (level {record_level} not in subset_list)')
                    continue

            image_path = _dump_image(record, index, image_root_path, data_root)
            if not image_path:
                logger.warning(f'Skipping MathVision record without usable image at index {index}')
                continue

            options = _normalize_options(record.get('options', []))
            is_mcq = len(options) > 0
            prompt = _build_mcq_prompt(question, options, single_answer_cot_template) if is_mcq else (
                open_prompt_template.format(question=question) if open_prompt_template else question
            )
            choices_text = _build_choices_text(options) if is_mcq else ''

            answer = record.get('answer', '')
            if is_mcq:
                choices = {
                    string.ascii_uppercase[item_index]: option
                    for item_index, option in enumerate(options)
                }
                answer = {
                    'type': 'mcq',
                    'answer': _normalize_choice_answer(answer, options),
                    'choices': json.dumps(choices, ensure_ascii=False),
                    'level': record.get('level'),
                    'subject': record.get('subject'),
                }
            else:
                answer = {
                    'type': 'open',
                    'answer': str(answer),
                    'level': record.get('level'),
                    'subject': record.get('subject'),
                }

            msgs = [
                dict(type='text', text=prompt),
                dict(type='image_url', image_url=image_path),
            ]
            content = get_content_str(msgs)

            dataset.append({
                'question': question,
                'prompt': prompt,
                'content': content,
                'image': image_path,
                'choices_text': choices_text,
                'question_type': 'mcq' if is_mcq else 'open',
                'answer': answer,
                'level': record.get('level'),
                'subject': record.get('subject'),
                'solution': record.get('solution'),
            })
        return Dataset.from_list(dataset)


class MathVisionEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()
        self.math_evaluator = MATHEvaluator(version='v2')

    @staticmethod
    def _clean_prediction(prediction):
        prediction = str(prediction)
        for character in SPECIAL_CHARACTERS:
            prediction = prediction.replace(character, '')
        return prediction.strip()

    @staticmethod
    def _extract_open_prediction(prediction):
        try:
            boxed = extract_boxed_answer(prediction, strip_double_curly_brace=True)
        except Exception:
            boxed = None
        if boxed:
            try:
                return normalize_final_answer(boxed)
            except Exception:
                return boxed.strip()

        boxed_match = re.findall(r'\boxed\{([^{}]+)\}', prediction)
        if boxed_match:
            boxed = boxed_match[-1]
            try:
                return normalize_final_answer(boxed)
            except Exception:
                return boxed.strip()

        answer = extract_answer(prediction)
        if answer:
            try:
                return normalize_final_answer(answer)
            except Exception:
                return answer.strip()

        for maybe_ans in prediction.split('.'):
            if re.search(r'final answer|answer is', maybe_ans.lower()):
                try:
                    return normalize_final_answer(maybe_ans)
                except Exception:
                    return maybe_ans.strip()

        return prediction.strip()

    @staticmethod
    def _extract_mcq_prediction(prediction, choices):
        match = MCQ_ANSWER_PATTERN.search(prediction)
        if match:
            answer = match.group(1)
            if answer in choices:
                return answer
        return can_infer(prediction, choices)

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}

        details = []
        scores = []
        level_scores = {}
        subject_scores = {}
        for prediction, reference in zip(predictions, references):
            prediction = self._clean_prediction(prediction)
            reference = reference if isinstance(reference, dict) else {'type': 'open', 'answer': reference}
            level = reference.get('level')
            subject = reference.get('subject')
            detail = {'pred': prediction, 'answer': reference, 'correct': False}

            if reference.get('type') == 'mcq':
                choices = json.loads(reference['choices']) if isinstance(reference.get('choices'), str) else reference.get('choices', {})
                parsed_prediction = self._extract_mcq_prediction(prediction, choices)
                score = 1 if parsed_prediction == reference.get('answer') else 0
                detail['parsed_pred'] = parsed_prediction
            else:
                parsed_prediction = self._extract_open_prediction(prediction)
                score = 1 if self.math_evaluator.is_equiv(parsed_prediction, str(reference.get('answer', ''))) else 0
                detail['parsed_pred'] = parsed_prediction

            if score == 1:
                detail['correct'] = True
            details.append(detail)
            scores.append(score)

            if level is not None:
                level_scores.setdefault(f'level {level}', []).append(score)
            if subject:
                subject_scores.setdefault(f'subject: {subject}', []).append(score)

        result = {'Accuracy': 100 * sum(scores) / len(scores) if scores else 0.0}
        for key, values in level_scores.items():
            result[key] = 100 * sum(values) / len(values)
        for key, values in subject_scores.items():
            result[key] = 100 * sum(values) / len(values)
        result['details'] = details
        return result
