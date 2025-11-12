import copy
import csv
import json
import os
import math
import numpy as np
from typing import List

from datasets import Dataset

from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator, BaseEvaluator
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path, get_meta_json, get_logger
from ais_bench.benchmark.datasets.utils.datasets import get_sample_data
from ais_bench.benchmark.utils.core.types import check_meta_json_dict, check_output_config_from_meta_json

from .base import BaseDataset


class OptionSimAccEvaluator(BaseEvaluator):
    """
    Circular-option similarity-accuracy evaluator.

    For each question it computes multiple accuracy metrics under different
    'circular_pattern' schemes (original, circular, all_possible).
    'acc_*' : simple accuracy (%)
    'more_j_*': % of items correct in at least j positions
    'perf_*' : % of items fully correct across the whole pattern

    Details of every single prediction are stored in metrics['details'].
    """
    def __init__(self, options) -> None:
        super().__init__()
        if not all((isinstance(i, str) and i.isupper() and len(i) == 1)
                   for i in options):
            raise ValueError(
                f'Each options should be single upper letter, got {options}')

        self.options = options

    def match_any_label(self, pred, test_item):
        from rapidfuzz.distance import Levenshtein as L

        from ais_bench.benchmark.utils.postprocess.text_postprocessors import first_option_postprocess

        pred = pred.strip()
        if any([pred == i for i in self.options]):
            parsed = pred
        else:
            parsed = ''
        if parsed == '':
            parsed = first_option_postprocess(pred,
                                              ''.join(self.options),
                                              cushion=False)
        if parsed == '':
            possible_options = []
            for opt in self.options:
                opt_str = test_item[opt]
                if opt_str is not None and opt_str.lower() in pred.lower():
                    possible_options.append(opt)
            if len(possible_options) == 1:
                parsed = possible_options[0]
        if parsed == '':
            dists = []
            for opt in self.options:
                opt_str = test_item[opt]
                if opt_str is None:
                    continue
                cands = [opt, opt_str, opt + '. ' + opt_str]
                d = min(L.distance(pred, cand) for cand in cands)
                dists.append((d, opt))
            if len(dists) > 0:
                parsed = min(dists)[1]
        return parsed

    def score(self, predictions: List, references: List, test_set) -> dict:
        assert len(predictions) == len(references)

        num_correct, num_total = 0, 0
        details = {}
        for index in range(len(predictions)):
            pred = predictions[index]
            refr = references[index]
            parsed = self.match_any_label(pred, test_set[index])
            num_correct += 1 if parsed == refr else 0
            num_total += 1
            details[str(index)] = {}
            details[str(index)]['pred'] = pred
            details[str(index)]['parsed'] = parsed
            details[str(index)]['refr'] = refr
            details[str(index)]['correct'] = parsed == refr
        return {'accuracy': num_correct / num_total * 100, 'details': details}


@LOAD_DATASET.register_module()
class CustomDataset(BaseDataset):

    @staticmethod
    def load(path, file_name=None, meta_path='', local_mode=False):
        path = get_data_path(path, local_mode=True)
        if file_name is not None:
            path = os.path.join(path, file_name)
        if path.endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8-sig') as f:
                data = [json.loads(line) for line in f]
        elif path.endswith('.csv'):
            with open(path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                header = next(reader)
                data = [dict(zip(header, row)) for row in reader]
        else:
            raise ValueError(f'Unsupported file format: {path}')
        meta_json_conf = get_meta_json(path, meta_path)
        if meta_json_conf:
            if check_output_config_from_meta_json(meta_json_conf):
                max_token_list = get_max_token_list_from_meta_json_file(meta_json_conf['output_config'], len(data))
                for i in range(len(data)):
                    data[i]['max_out_len'] = max_token_list[i] # output_config in meta.json will override max_out_len in custom dataset file
            meta_json_conf = check_meta_json_dict(meta_json_conf)
            sample_mode = meta_json_conf.get('sampling_mode', 'default')
            request_count = meta_json_conf.get('request_count', 0)
            data = get_sample_data(data, sample_mode, int(request_count))
        return Dataset.from_list(data)


def stringfy_types(obj):
    for k, v in obj.items():
        if k == 'type':
            obj[k] = f'{v.__module__}.{v.__name__}'
        elif isinstance(v, dict):
            stringfy_types(v)
    return obj


def make_mcq_gen_config(meta, logger):
    from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
    if meta.get('template', None) is None:
        _human_prompt = 'Question: {question}' + ''.join(
            [f'\n{item}. {{{item}}}' for item in meta['options']])
        human_prompt = meta.get('human_prompt', _human_prompt)
        _bot_prompt = f'Answer: {{{meta["output_column"]}}}'
        bot_prompt = meta.get('bot_prompt', _bot_prompt)
        template = human_prompt + "\n" + bot_prompt
        logger.info(f"Using default prompt: {template}")
    else:
        template = meta['template']

    reader_cfg = dict(
        input_columns=meta['input_columns'],
        output_column=meta['output_column'],
    )
    if 'test_range' in meta:
        reader_cfg['test_range'] = meta['test_range']
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=template,
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    eval_cfg = dict(
        evaluator=dict(type=meta.get('evaluator', OptionSimAccEvaluator),
                       **meta.get('evaluator_kwargs',
                                  {'options': meta['options']})),
        pred_role='BOT',
    )

    dataset = dict(
        abbr=meta['abbr'],
        type=CustomDataset,
        path=meta['path'],
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
        meta_path=meta['meta_path'],
    )
    return dataset


def make_qa_gen_config(meta, logger):
    from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
    if meta.get('template', None) is None:
        human_prompt = meta.get('human_prompt', '{question}')
        if meta['output_column'] is None:
            template = human_prompt,
        else:
            bot_prompt = meta.get('bot_prompt', f'{{{meta["output_column"]}}}')
            template = "Question: "+ human_prompt + "\nAnswer: " + bot_prompt
            logger.info(f"Using default prompt: {template}")
    else:
        template = meta['template']
    reader_cfg = dict(
        input_columns=meta['input_columns'],
        output_column=meta['output_column'],
    )
    if 'test_range' in meta:
        reader_cfg['test_range'] = meta['test_range']
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=template,
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    eval_cfg = dict(
        evaluator=dict(type=meta.get('evaluator', AccEvaluator),
                       **meta.get('evaluator_kwargs', {})),
        pred_role='BOT',
    )

    dataset = dict(
        abbr=meta['abbr'],
        type=CustomDataset,
        path=meta['path'],
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
        meta_path=meta['meta_path'],
    )
    return dataset


def parse_example_dataset(config):
    # config -> .meta.jsonl -> parsed_results
    path = config['path']

    # load sample and get parsed_meta
    parsed_meta = {}
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            data_item = json.loads(f.readline())
    elif path.endswith('.csv'):
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
            data_item = dict(zip(header, row))
    else:
        raise ValueError(f'Unsupported ext: {path}, .jsonl or .csv required')

    parsed_meta['path'] = path

    input_columns = ['question', 'max_out_len']
    if not input_columns:
        raise ValueError(f'Unsupported dataset format: NOT contain "question" in {path}')
    parsed_meta['input_columns'] = input_columns
    output_column = 'answer' if 'answer' in data_item else None
    parsed_meta['output_column'] = output_column

    options = []
    for i in range(26):
        i = chr(ord('A') + i)
        if i in data_item:
            options.append(i)
        else:
            break
    parsed_meta['options'] = options
    abbr = os.path.basename(path).split('.')[0]
    parsed_meta['abbr'] = abbr
    parsed_meta['data_type'] = 'mcq' if len(options) > 1 else 'qa'
    parsed_meta['infer_method'] = 'gen'

    # get meta data path
    parsed_meta['meta_path'] = config.get('meta_path')

    # get config meta
    config_meta = copy.deepcopy(config)

    # merge meta
    meta = {}
    meta.update(parsed_meta)
    meta.update(config_meta)

    return meta


def make_custom_dataset_config(config):
    # considered as a custom dataset
    logger = get_logger()
    meta = parse_example_dataset(config)
    make_config_func = {
        ('mcq', 'gen'): make_mcq_gen_config,
        ('qa', 'gen'): make_qa_gen_config,
    }.get((meta['data_type'], meta['infer_method']), None)
    if make_config_func is None:
        raise ValueError(f'Unsupported dataset data_type: {meta["data_type"]}'
                         f' and infer_method: {meta["infer_method"]}')
    dataset = make_config_func(meta, logger)
    dataset = stringfy_types(dataset)
    return dataset

def get_max_token_list_from_meta_json_file(output_config: dict, prompt_length):
    """Get max_token_list from meta.json file output config.
    Args:
        output_config: 'output_config' in meta.json
        prompt_length: num prompts
    Return:
        max_token_list: max_tokens of prompts
    """
    method = output_config["method"]
    params = output_config["params"]
    logger = get_logger()
    logger.info("Distribution Summary: ")
    if method == "uniform":
        logger.info(f"--uniform distribution with min_value: {params['min_value']}, max_value: {params['max_value']}")
        max_token_list = np.random.randint(int(params["min_value"]), int(params["max_value"]) + 1, prompt_length)
        return [int(token) for token in max_token_list]
    elif method == "percentage":
        max_token_list = []
        show_log_info = []
        for max_tokens, rate in params["percentage_distribute"]:
            max_token_list.extend([max_tokens] * math.floor(rate * prompt_length))
            show_log_info.append([max_tokens, rate * 100, math.floor(rate * prompt_length)])
        if len(max_token_list) < prompt_length:
            max_token_list.extend([params["percentage_distribute"][-1][0]] * (prompt_length - len(max_token_list)))
            show_log_info[-1][2] += prompt_length - len(max_token_list)
        for out_token_len, rate, request_num in show_log_info:
            logger.info("--max_out_token: {},  ratio: {:.1f}%,  request_num: {}".format(out_token_len, rate, request_num))
        return max_token_list
    else:
        raise ValueError(f"Unsupport data distribution types: {method}")