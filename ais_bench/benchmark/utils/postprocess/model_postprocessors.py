import re
from functools import partial
from multiprocessing import Pool
from typing import Union

from tqdm import tqdm

from ais_bench.benchmark.registry import TEXT_POSTPROCESSORS

from ais_bench.benchmark.utils.postprocess.postprocessors.naive import NaiveExtractor, format_input_naive
from ais_bench.benchmark.utils.postprocess.postprocessors.xfinder.extractor import Extractor
from ais_bench.benchmark.utils.postprocess.postprocessors.xfinder.xfinder_utils import (DataProcessor,
                                                   convert_to_xfinder_format)


def gen_output_naive(ori_data, extractor):
    extracted_answers = []
    for item in tqdm(ori_data):
        user_input = extractor.prepare_input(item)
        extracted_answer = extractor.gen_output(user_input)
        item['extracted_answer'] = extracted_answer
        extracted_answers.append(extracted_answer)

    return extracted_answers


@TEXT_POSTPROCESSORS.register_module('naive')
def naive_model_postprocess(preds: list,
                            model_name: str,
                            custom_instruction: str,
                            api_url: Union[str, list],
                            num_processes: int = 8,
                            **kwargs) -> list:
    """Postprocess the text extracted by custom model.
    Args:
        preds (list): The question, reference answer and model prediction.
        model_name (str): The name of the model.
        custom_instruction (str): Custom instruction for the dataset.
        url (Union[str, list]): The api url of the model.

    Returns:
        list: The postprocessed answers.
    """

    def _eval_pred(texts, extractor, num_processes):
        ori_data = texts
        extracted_answers = []
        batched_ori_data = []
        # Split data into batches
        num_processes = min(num_processes, len(ori_data))
        batch_size = len(ori_data) // num_processes
        for i in range(0, len(ori_data), batch_size):
            batched_ori_data.append(ori_data[i:i + batch_size])
        with Pool(num_processes) as p:
            results = p.map(partial(gen_output_naive, extractor=extractor),
                            batched_ori_data)
            for result in results:
                extracted_answers.extend(result)
        return extracted_answers

    format_data = format_input_naive(preds)
    assert api_url is not None, 'Please provide the api url.'
    extractor = NaiveExtractor(
        model_name=model_name,
        custom_instruction=custom_instruction,
        url=api_url.split(',') if ',' in api_url else api_url)
    calc_acc_func = partial(_eval_pred,
                            extractor=extractor,
                            num_processes=num_processes)
    extracted_answers = calc_acc_func(format_data)
    return extracted_answers


def gen_output_xfinder(ori_data, extractor):
    ext_cor_pairs = []
    extracted_data = []
    extracted_answers = []
    for item in tqdm(ori_data):
        user_input = extractor.prepare_input(item)
        extracted_answer = extractor.gen_output(user_input)
        ext_cor_pairs.append([
            item['key_answer_type'], item['standard_answer_range'],
            extracted_answer, item['correct_answer']
        ])
        item['xfinder_extracted_answer'] = extracted_answer
        extracted_answers.append(extracted_answer)
        extracted_data.append(item)

    return extracted_answers, ext_cor_pairs, extracted_data


@TEXT_POSTPROCESSORS.register_module('xfinder')
def xfinder_postprocess(preds: list, question_type: str, model_name: str,
                        api_url: Union[str, list], **kwargs) -> list:
    """Postprocess the text extracted by xFinder model.
    Args:
        preds (list): The question, reference answer and model prediction.
        question_type (str): The type of the question.
        url (Union[str, list]): The api url of the xFinder model.


    Returns:
        list: The postprocessed texts.
    """

    def _eval_pred(texts, data_processor, extractor, num_processes=8):
        ori_data = data_processor.read_data(texts)
        extracted_correct_pairs = []
        extracted_data = []
        extracted_answers = []
        batched_ori_data = []
        # Split data into batches
        num_processes = min(num_processes, len(ori_data))
        batch_size = len(ori_data) // num_processes
        for i in range(0, len(ori_data), batch_size):
            batched_ori_data.append(ori_data[i:i + batch_size])
        with Pool(num_processes) as p:
            results = p.map(partial(gen_output_xfinder, extractor=extractor),
                            batched_ori_data)
        for result in results:
            extracted_answers += result[0]
            extracted_correct_pairs += result[1]
            extracted_data += result[2]
        return extracted_answers

    format_data = convert_to_xfinder_format(question_type, preds)
    assert api_url is not None, 'Please provide the api url.'
    data_processor = DataProcessor()
    extractor = Extractor(
        model_name=model_name,
        url=api_url.split(',') if ',' in api_url else api_url)
    calc_acc_func = partial(_eval_pred,
                            data_processor=data_processor,
                            extractor=extractor)
    extracted_answers = calc_acc_func(format_data)
    return extracted_answers


def list_decorator(func):
    """Decorator: make the function able to handle list input"""
    def wrapper(text_or_list, *args, **kwargs):
        if isinstance(text_or_list, list):
            return [func(text, *args, **kwargs) for text in text_or_list]
        else:
            return func(text_or_list, *args, **kwargs)
    return wrapper


@TEXT_POSTPROCESSORS.register_module('extract-non-reasoning-content')
@list_decorator
def extract_non_reasoning_content(
    text: str,
    think_start_token: str = '<think>',
    think_end_token: str = '</think>',
) -> str:
    """Extract content after the last reasoning tag from text.

    When only end token is present, returns content after the end token.
    When both tokens are present, removes all content between start and end tokens.

    Args:
        text (str): Input text containing reasoning tags.
        think_start_token (str, optional): Start token for reasoning section. Defaults to '<think>'.
        think_end_token (str, optional): End token for reasoning section. Defaults to '</think>'.

    Returns:
        str: Processed text after removing reasoning sections.

    Examples:
        >>> # When only end token exists
        >>> text = "This is a test.</think> How are you?"
        >>> extract_non_reasoning_content(text)
        'How are you?'

        >>> # When both tokens exist
        >>> text = "Start<think>reasoning here</think> End"
        >>> extract_non_reasoning_content(text)
        'Start End'
        
        >>> # When input is a list
        >>> texts = ["Start<think>reasoning</think> End", "Test</think> Result"]
        >>> extract_non_reasoning_content(texts)
        ['Start End', 'Result']
    """
    # If text contains only end token, split by end token and take the last part
    if think_start_token not in text and think_end_token in text:
        return text.split(think_end_token)[-1].strip()

    # Original behavior for complete tag pairs
    reasoning_regex = re.compile(rf'{think_start_token}(.*?){think_end_token}',
                                 re.DOTALL)
    non_reasoning_content = reasoning_regex.sub('', text).strip()
    return non_reasoning_content