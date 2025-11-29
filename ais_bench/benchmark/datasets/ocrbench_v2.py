import json
import os
import pickle
import importlib
import sys
import string
import ast
import pandas as pd
import numpy as np
from collections import defaultdict, namedtuple
from tqdm import tqdm
import re
import jieba
import nltk
from nltk.translate import meteor_score
from nltk.metrics.scores import f_measure, precision, recall
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from lxml import etree
import html
from apted.helpers import Tree
from apted import APTED, Config
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import ipdb
from collections import deque
import math
from zss import simple_distance, Node
from itertools import product
import zipfile
import shutil
from collections import namedtuple

from datasets import Dataset

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import DSET_CODES
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchDataContentError, ParameterValueError, AISBenchValueError,
    AISBenchRuntimeError, FileOperationError
)
from ais_bench.benchmark.datasets.utils.datasets import get_data_path, toliststr, process_line, get_content_str
from ais_bench.benchmark.datasets.mmmu import dump_image

from .base import BaseDataset

IMAGE_MAP_LEN = 64
ANLS_THRESHOLD= 0.5

logger = AISLogger()

def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=0):
    """
        A parallel version of the map function with a progress bar.

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    else:
        front = []
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)

class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.

class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        if not isinstance(n_jobs, int) or n_jobs < 1:
            raise ParameterValueError(
                DSET_CODES.INVALID_PARAM_VALUE,
                f'n_jobs must be an integer greater than 1, got {n_jobs}'
            )
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath('body/table') and true.xpath('body/table'):
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0

    def batch_evaluate(self, pred_json, true_json):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        samples = true_json.keys()
        if self.n_jobs == 1:
            scores = [self.evaluate(pred_json.get(filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        else:
            #inputs = [{'pred': pred_json.get(filename, ''), 'true': true_json[filename]['html']} for filename in samples]
            inputs = [{'pred': pred_json.get(filename, ''), 'true': true_json[filename]} for filename in samples]
            scores = parallel_process(inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        scores = dict(zip(samples, scores))
        return scores

def split_OCRBenchV2(msgs):
    """Splits a mixed media message into separate text and image segments based on OCRBench_v2-style placeholders.

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
            logger.debug("Invalid text seg, please check it!")
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

def is_nan_value(value):
    if value is None:
        return True
    if isinstance(value, str) and value.lower() == 'nan':
        return True
    try:
        import pandas as pd
        if pd.isna(value):
            return True
    except:
        pass
    return False


def get_value_or_zero(value):
    return 0.0 if value is None else value


def calculate_average(scores_dict):
    averages = {key: sum(values) / len(values) for key, values in scores_dict.items() if len(values) > 0}
    return averages

def contain_chinese_string(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))

def cal_per_metrics(pred, gt):
    metrics = {}

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)

    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    return metrics

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def vqa_evaluation_case_sensitive(predict, answers):
    score = 0
    if isinstance(answers, list):
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            try:
                answer = answers[j].strip().replace("\n"," ")
            except:
                ipdb.set_trace()
            predict = predict.strip().replace("\n"," ")
            if len(answer.split()) < 5:
                if answer in predict:
                    score = 1
            else:
                dist = levenshtein_distance(predict, answer)
                length = max(len(predict), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answers = answers.strip().replace("\n"," ")
        predict = predict.strip().replace("\n"," ")
        if len(answers.split()) < 5:
            if answers in predict:
                score = 1
            else:
                dist = levenshtein_distance(predict, answers)
                length = max(len(predict), len(answers))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    return score

def remove_text_tags(latex_str):
    """
    Removes LaTeX \text{...} tags while keeping their content.

    :param latex_str: A string containing LaTeX expressions
    :return: The processed string with \text{...} tags removed
    """

    pattern = r'\\text\{([^{}]*)\}'

    processed_str = re.sub(pattern, r'\1', latex_str)

    return processed_str

def cn_math_expression_evaluation(predict, answers):
    score = 0

    if len(answers) != 1:
        raise AISBenchDataContentError(
            DSET_CODES.DATA_INVALID_STRUCTURE,
            f"Expected exactly 1 answer, got {len(answers)}"
        )
    answers = [remove_text_tags(answers[0])]
    predict = remove_text_tags(predict)

    if isinstance(answers, list):
        for j in range(len(answers)):
            answer = answers[j].strip().replace("\n"," ").replace(" ","")
            predict = predict.strip().replace("\n"," ").replace(" ","")
            if answer in predict:
                score = 1
    else:
        answers = answers.strip().replace("\n"," ").replace(" ","")
        predict = predict.strip().replace("\n"," ").replace(" ","")
        if answers in predict:
            score = 1
    return score

def cn_vqa_evaluation(predict, answers):
    score = 0
    if isinstance(answers, list):
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            try:
                answer = answers[j].lower().strip().replace("\n"," ").replace(" ", "")
            except:
                ipdb.set_trace()
            if isinstance(predict, (int, float)):
                predict = str(predict)
            predict = predict.lower().strip().replace("\n"," ").replace(" ", "")
            if len(answer.split(",")) < 4:
                if answer in predict:
                    score = 1
            else:
                dist = levenshtein_distance(predict, answer)
                length = max(len(predict), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answers = answers.lower().strip().replace("\n"," ").replace(" ", "")
        predict = predict.lower().strip().replace("\n"," ").replace(" ", "")
        if len(answer.split(",")) < 4:
            if answers in predict:
                score = 1
            else:
                dist = levenshtein_distance(predict, answers)
                length = max(len(predict), len(answers))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    return score

def extract_first_number(string):
    match = re.search(r'\d+', string)
    if match:
        return int(match.group())
    return None

def counting_evaluation(predict, answers, eval_method):
    score = 0

    if isinstance(predict, str):
        predict_processed = predict.lower().strip().replace("\n", " ")
    elif math.isnan(predict):
        return 0
    else:
        predict_processed = int(predict)
    if isinstance(answers, list):
        temp_score = 0
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            answer = answers[j].lower().strip().replace("\n"," ")
            if eval_method == "exact match":
                if answer in predict:
                    score = 1
                else:
                    score = 0
            elif eval_method == "regression":
                predict_number = extract_first_number(predict_processed)
                if predict_number:

                    answer = int(answer)

                    if predict_number <= 0 or predict_number >= 2 * answer:
                        score = 0
                    else:
                        iou = 1 - abs(predict_number - answer) / answer
                        if iou > 0.5:
                            score = iou
                        else:
                            score = 0
                else:
                    score = 0
            if score > temp_score:
                temp_score = score
        score = temp_score

    else:
        answers = answers.lower().strip().replace("\n"," ")
        predict = predict.lower().strip().replace("\n"," ")
        if eval_method == "exact match":
            if answer in predict:
                score = 1
            else:
                score = 0
        elif eval_method == "regression":
            predict = extract_first_number(predict)
            if predict:
                answer = int(answer)
                if predict <= 0 or predict >= 2 * answer:
                    score = 0
                else:
                    iou = 1 - abs(predict - answer) / answer

                    if iou > 0.5:
                        score = iou
                    else:
                        score = 0
            else:
                score = 0
    return score

def math_expression_evaluation(predict, answers):
    score = 0
    if isinstance(answers, list):
        for j in range(len(answers)):
            answer = answers[j].strip().replace("\n"," ").replace(" ","")
            predict = predict.strip().replace("\n"," ").replace(" ","")
            if answer in predict:
                score = 1
    else:
        answers = answers.strip().replace("\n"," ").replace(" ","")
        predict = predict.strip().replace("\n"," ").replace(" ","")
        if answers in predict:
            score = 1
    return score

def wrap_html_table(html_table):
    """
    The TEDS computation from PubTabNet code requires that the input html table should have <html>, <body>, and <table> tags.
    Add them if they are missing.
    """
    html_table = html_table.replace('\n','')
    # add missing <table> tag if missing
    if "<table" in html_table and "</table>" not in html_table:
        html_table = html_table + "</table>"
    elif "<table" not in html_table and "</table>" in html_table:
        html_table = "<table>" + html_table
    elif "<table" not in html_table and "</table>" not in html_table:
        html_table = "<table>" + html_table + "</table>"
    else:
        pass
    # add <body> and <html> tags if missing
    if '<body>' not in html_table:
        html_table = '<body>' + html_table + '</body>'
    if '<html>' not in html_table:
        html_table = '<html>' + html_table + '</html>'
    return html_table

def convert_table_to_html_str(table_row_list=[]):
    """
    Given a list of table rows, build the corresponding html string, which is used to compute the TEDS score.
    We use the official code of PubTabNet to compute TEDS score, it does not consider '<th>' label.
    We also remove unneccessary spaces within a table cell and extra '\n' as they will influence the TEDS score.
    """
    html_table_str = "<html><body><table>" + '\n'
    for data_row in table_row_list:
        html_table_str += "<tr>"
        for cell_str in data_row:
            html_table_str += f"<td>{cell_str}</td>"
        html_table_str += "</tr>"
        html_table_str += '\n'
    html_table_str += "</table></body></html>"
    html_table_str = html_table_str.replace('\n','')
    return html_table_str


def convert_markdown_table_to_html(markdown_table):
    """
    Converts a markdown table to the corresponding html string for TEDS computation.
    """
    # remove extra code block tokens like '```markdown' and '```
    markdown_table = markdown_table.strip('```markdown').strip('```').strip()
    row_str_list = markdown_table.split('\n')
    # extra the first header row and other data rows
    valid_row_str_list = [row_str_list[0]]+row_str_list[2:]
    table_rows = []
    for row_str in valid_row_str_list:
        one_row = []
        for cell in row_str.strip().split('|')[1:-1]:
            if set(cell) != set(' '):
                one_row.append(cell.strip())
            else:
                one_row.append(' ')
        table_rows.append(one_row)
    # build html string based on table rows
    html_str = convert_table_to_html_str(table_rows)
    return html_str

def convert_str_to_multi_dict(predict_str: str):
    """
    Parses the 'predict' string and returns a dictionary.
    Handles nested dictionaries and missing or unparseable content gracefully.

    Parameters:
    - predict_str (str): The prediction string containing the output dict.

    Returns:
    - dict: A dictionary extracted from the predict string.
    """
    # Remove code fences like ```python\n...\n```
    code_fence_pattern = r'```(?:python|json)?\n(.*?)\n```'
    matches = re.findall(code_fence_pattern, predict_str, re.DOTALL | re.IGNORECASE)
    if matches:
        content = max(matches, key=len)
    else:
        content = predict_str.strip()

    def strip_variable_assignment(s):
        variable_assignment_pattern = r'^\s*\w+\s*=\s*'
        return re.sub(variable_assignment_pattern, '', s.strip(), count=1)

    content = strip_variable_assignment(content)

    def remove_comments(s):
        return re.sub(r'#.*', '', s)

    content = remove_comments(content)

    last_brace_pos = content.rfind('}')
    if last_brace_pos != -1:
        content = content[:last_brace_pos+1]

    data = {}
    success = False

    # try parsing with ast.literal_eval
    try:
        data = ast.literal_eval(content)
        if isinstance(data, dict):
            success = True
    except (ValueError, SyntaxError, TypeError):
        pass

    if not success:
        return {}

    def process_data(obj):
        if isinstance(obj, dict):
            return {k: process_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [process_data(elem) for elem in obj]
        else:
            return obj

    data = process_data(data)

    return data

def vqa_evaluation(predict, answers):
    score = 0
    if isinstance(answers, list):
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            try:
                answer = answers[j].lower().strip().replace("\n"," ")
            except:
                ipdb.set_trace()
            if isinstance(predict, (int, float)):
                predict = str(predict)
            predict = predict.lower().strip().replace("\n"," ")
            if len(answer.split()) < 5:
                if answer in predict:
                    score = 1
            else:
                dist = levenshtein_distance(predict, answer)
                length = max(len(predict), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answers = answers.lower().strip().replace("\n"," ")
        predict = predict.lower().strip().replace("\n"," ")
        if len(answers.split()) < 5:
            if answers in predict:
                score = 1
            else:
                dist = levenshtein_distance(predict, answers)
                length = max(len(predict), len(answers))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    return score

def dict_to_html(data):
    html = "<html><body><table>\n"
    for key, value in data.items():
        if not isinstance(value, str):
            value = str(value)
        value_str = ' '.join(value)

        html += f"  <tr><td>{key}</td><td>{value_str}</td></tr>\n"
    html += "</table></body></html>"
    return html

def pre_clean(text):
    text = re.sub(r'<bos>|<eos>|<pad>|<unk>', '', text)
    text = re.sub(r'\s##(\S)', r'\1', text)
    text = re.sub(r'\\\s', r'\\', text)
    text = re.sub(r'\s\*\s\*\s', r'**', text)
    text = re.sub(r'{\s', r'{', text)
    text = re.sub(r'\s}', r'}', text)
    text = re.sub(r'\s}', r'}', text)
    text = re.sub(r'\\begin\s', r'\\begin', text)
    text = re.sub(r'\\end\s', r'\\end', text)
    text = re.sub(r'\\end{table}', r'\\end{table} \n\n', text)
    text = text.replace('\n', ' ')
    text = text.replace('*', ' ')
    text = text.replace('_', ' ')
    return text

def get_tree(input_str):
    tree = (Node('ROOT').addkid(Node('TITLE')))

    lines = input_str.split("\n")
    lines = [pre_clean(line) for line in lines]
    last_title = ''
    for line in lines:
        if line.startswith('#'):
            child = tree.get('ROOT')
            line = line.replace('#', '')
            child.addkid(Node(line))
            last_title = line
        else:
            if last_title == '':
                child = tree.get('TITLE')
                child.addkid(Node(line))
            else:
                child = tree.get(last_title)
                child.addkid(Node(line))
    return tree

def STEDS(pred_tree, ref_tree):
    def my_distance(pred, ref):
        if len(pred.split()) == 0 or len(ref.split()) == 0:
            return 1
        else:
            return 0
    total_distance = simple_distance(pred_tree, ref_tree, label_dist=my_distance)
    num_of_nodes = max(len(list(pred_tree.iter())), len(list(ref_tree.iter())))
    return 1-total_distance/num_of_nodes


def doc_parsing_evaluation(pred, gt):
    score = 0
    if not isinstance(pred, str):
        return 0
    pred_tree = get_tree(pred)
    gt_tree = get_tree(gt)
    score = STEDS(pred_tree, gt_tree)

    return score

def generate_combinations(input_dict):
    """
    Function to generate all possible combinations of values from a dictionary.
    """
    kie_answer = input_dict
    if not isinstance(kie_answer, dict):
        kie_answer = kie_answer.strip('"')
        try:
            kie_answer = json.loads(kie_answer)
        except json.JSONDecodeError:
            try:
                kie_answer = ast.literal_eval(kie_answer)
                if not isinstance(kie_answer, dict):
                    kie_answer = ast.literal_eval(kie_answer)
            except (ValueError, SyntaxError):
                logger.debug(f"Unable to parse 'answers' field: {kie_answer}")
                return {}

        # Ensure the parsed result is a dictionary.
        if not isinstance(kie_answer, dict):
            logger.debug("Parsed 'answers' is still not a dictionary.")
            raise AISBenchDataContentError(
                DSET_CODES.DATA_LABEL_PARSE_ERROR,
                "Input could not be parsed into a dictionary."
            )

        keys = list(kie_answer.keys())

        value_lists = []
        for single_key in keys:
            sinlge_value = kie_answer[single_key]
            if not isinstance(sinlge_value, list):
                sinlge_value = [sinlge_value]
            value_lists.append(sinlge_value)

        # Compute the Cartesian product of the value lists.
        combinations = list(product(*value_lists))

        # Create a dictionary for each combination of values.
        result = [dict(zip(keys, values)) for values in combinations]

        return result

    else:
        keys = list(input_dict.keys())
        value_lists = [input_dict[key] for key in keys]

        # Compute the Cartesian product of the value lists.
        combinations = list(product(*value_lists))

        # Create a dictionary for each combination of values.
        result = [dict(zip(keys, values)) for values in combinations]

        return result

def convert_str_to_dict(predict_str: str):
    """
    Parses the 'predict' string and returns a dictionary.
    Missing or unparseable content is handled gracefully.

    Parameters:
    - predict_str (str): The prediction string containing the output dict.

    Returns:
    - dict: A dictionary extracted from the predict string.
    """
    # Remove code fences like ```python\n...\n```
    code_fence_pattern = r'```(?:python|json)?\n(.*?)\n```'
    match = re.search(code_fence_pattern, predict_str, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
    else:
        content = predict_str.strip()

    data = {}
    success = False

    # try parsing with JSON
    try:
        data = json.loads(content)
        success = True
    except json.JSONDecodeError:
        pass

    # try parsing with ast.literal_eval
    if not success:
        try:
            data = ast.literal_eval(content)
            if isinstance(data, dict):
                success = True
        except (ValueError, SyntaxError):
            pass

    # try parsing with regex
    if not success:
        key_value_pattern = r'["\']?([\w\s]+)["\']?\s*[:=]\s*["\']?([^\n,"\'{}]+)["\']?'
        matches = re.findall(key_value_pattern, content)
        try:
            for key, value in matches:
                data[key.strip()] = value.strip()
        except:
            return {}

    if not data:
        return {}

    try:
        result = {k.strip(): str(v).strip() for k, v in data.items()}
    except:
        return {}
    return result

def compute_f1_score(preds, gts, ignores=[]):
    """Compute the F1-score for KIE task between predicted and ground truth dictionaries.

    Args:
        preds (dict): The predicted key-value pairs.
        gts (dict): The ground truth key-value pairs.
        ignores (list): The list of keys to ignore during evaluation.

    Returns:
        dict: A dictionary where keys are field names and values are their corresponding F1-scores.
    """
    # Optionally remove ignored keys from predictions and ground truths
    keys = set(preds.keys()).union(set(gts.keys())) - set(ignores)
    f1_scores = {}

    for key in keys:
        pred_value = preds.get(key, None)
        gt_value = gts.get(key, None)

        if pred_value:
            pred_value = pred_value.lower().strip().replace("\n"," ").replace(" ", "")
        if gt_value:
            gt_value = gt_value.lower().strip().replace("\n"," ").replace(" ", "")

        if pred_value is None and gt_value is None:
            continue
        elif pred_value is None:
            precision = 0.0
            recall = 0.0
        elif gt_value is None:
            # false positive
            precision = 0.0
            recall = 0.0
        else:
            if pred_value == gt_value:
                # True positive
                precision = 1.0
                recall = 1.0
            else:
                precision = 0.0
                recall = 0.0

        # Compute F1-score
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores[key] = f1_score

    if len(f1_scores) == 0:
        return 0
    average_f1 = sum(f1_scores.values()) / len(f1_scores)

    return average_f1

def calculate_iou(box1, box2):

    try:
        box1 = [int(coordinate) for coordinate in box1]
        box2 = [int(coordinate) for coordinate in box2]
    except:
        return 0

    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def vqa_with_position_evaluation(predict, img_metas):

    score_content, score_bbox = .0, .0
    if "answer" in predict.keys():
        score_content = vqa_evaluation(predict["answer"], img_metas["answers"])
    if "bbox" in predict.keys():
        gt_bbox = img_metas["bbox"]
        try:
            predict_bbox_list = ast.literal_eval(predict["bbox"])
            score_bbox = calculate_iou(predict_bbox_list, gt_bbox)
        except:
            score_bbox = 0
    return 0.5 * score_content + 0.5 * score_bbox

def process_predictions(predict_file):
    teds = TEDS(n_jobs=32)

    res_data_list = []

    for index, data_item in enumerate(tqdm(predict_file)):
        if data_item["type"] == "APP agent en" or data_item["type"] == "ASCII art classification en" or data_item["type"] == "math QA en" \
            or data_item["type"] == "reasoning VQA en" or data_item["type"] == "science QA en" \
            or data_item["type"] == "text recognition en" or data_item["type"] == "document classification en" \
            or data_item["type"] == "cognition VQA en" or data_item["type"] == "diagram QA en":
            if "eval" in data_item.keys():
                if data_item["eval"] == "multiple choice":
                    if not isinstance(data_item["answers"], list):
                        data_item["answers"] = [data_item["answers"]]
                    if len(data_item["answers"]) != 1:
                        raise AISBenchDataContentError(
                            DSET_CODES.DATA_INVALID_STRUCTURE,
                            f"Expected exactly 1 answer for multiple choice evaluation, got {len(data_item['answers'])}"
                        )

                    if not isinstance(data_item["predict"], str):
                        data_item["score"] = 0
                    else:
                        predict = ''.join(c for c in data_item["predict"] if c.isalpha())

                        if predict == data_item["answers"][0]:
                            data_item["score"] = 1
                        else:
                            data_item["score"] = 0
                elif data_item["eval"] == "case sensitive":
                    data_item["score"] = vqa_evaluation_case_sensitive(data_item["predict"], data_item["answers"])
                else:
                    raise ParameterValueError(
                        DSET_CODES.INVALID_PARAM_VALUE,
                        f"No such evaluation method: {data_item.get('eval', 'None')}"
                    )
            else:
                data_item["score"] = vqa_evaluation(data_item["predict"], data_item["answers"])

        elif data_item["type"] == "cognition VQA cn" or data_item["type"] == "reasoning VQA cn":
            if "eval" in data_item.keys():
                if data_item["eval"] == "multiple choice":
                    if len(data_item["answers"]) != 1:
                        raise AISBenchDataContentError(
                            DSET_CODES.DATA_INVALID_STRUCTURE,
                            f"Expected exactly 1 answer for multiple choice evaluation, got {len(data_item['answers'])}"
                        )
                    predict = ''.join(c for c in data_item["predict"] if c.isalpha())

                    if predict == data_item["answers"][0]:
                        data_item["score"] = 1
                    else:
                        data_item["score"] = 0
                elif data_item["eval"] == "case sensitive":
                    data_item["score"] = vqa_evaluation_case_sensitive(data_item["predict"], data_item["answers"])
                else:
                    raise ParameterValueError(
                        DSET_CODES.INVALID_PARAM_VALUE,
                        f"No such evaluation method: {data_item.get('eval', 'None')}"
                    )
            else:
                data_item["score"] = cn_vqa_evaluation(data_item["predict"], data_item["answers"])

        elif data_item["type"] == "handwritten answer extraction cn":
            if "简答" in data_item["question"]:
                ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                data_item["score"] = (
                    get_value_or_zero(ocr_metric["bleu"]) +
                    get_value_or_zero(ocr_metric["meteor"]) +
                    get_value_or_zero(ocr_metric["f_measure"]) +
                    (1 - get_value_or_zero(ocr_metric["edit_dist"]))
                ) / 4
            else:
                if len(data_item["answers"]) != 1:
                    raise AISBenchDataContentError(
                        DSET_CODES.DATA_INVALID_STRUCTURE,
                        f"Expected exactly 1 answer for handwritten answer extraction, got {len(data_item['answers'])}"
                    )
                answer = data_item["answers"][0]
                chars = list(answer)
                if len(answer) > 1:

                    answer_list = [
                        "".join(chars),
                        ".".join(chars),
                        ". ".join(chars),
                        ",".join(chars),
                        ", ".join(chars),
                        "、".join(chars),
                        ";".join(chars),
                        "; ".join(chars),
                        " ".join(chars),
                        "和".join(chars)
                    ]
                    max_score = 0
                    for answer in answer_list:
                        if answer in data_item["predict"]:
                            temp_score = 1
                        else:
                            temp_score = 0
                        if temp_score > max_score:
                            max_score = temp_score
                    data_item["score"] = max_score

                else:
                    if data_item["answers"][0] in data_item["predict"]:
                        data_item["score"] = 1
                    else:
                        data_item["score"] = 0

        elif data_item["type"] == "formula recognition cn":
            if is_nan_value(data_item["predict"]):
                data_item["score"] = 0
            else:
                data_item["score"] = cn_math_expression_evaluation(data_item["predict"], data_item["answers"])

        elif data_item["type"] == "text counting en":
            data_item["score"] = counting_evaluation(data_item["predict"], data_item["answers"], data_item["eval"])

        elif data_item["type"] == "formula recognition en":
            data_item["score"] = math_expression_evaluation(data_item["predict"], data_item["answers"])

        elif data_item["type"] == "table parsing en":
            if type(data_item["answers"])==list and len(data_item["answers"]) == 1:
                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                elif not isinstance(data_item["question"], str):
                    data_item["ignore"] = "True"
                    data_item["score"] = 0

                elif "html" in data_item["question"].lower():
                    no_find = False
                    predict_table = data_item["predict"].replace('\n','')
                    if "<body" in predict_table:
                        predict_table = re.findall('<body.*', predict_table)[0]
                    elif "<table" in predict_table:
                        predict_table = re.findall('<table.*', predict_table)[0]
                    else:
                        no_find = True

                    if no_find:
                        data_item["score"] = 0
                    else:
                        pred_table_html = wrap_html_table(predict_table)
                        gold_table_html = wrap_html_table(data_item["answers"][0])
                        try:
                            data_item["score"] = teds.evaluate(pred_table_html, gold_table_html)
                        except:
                            data_item["score"] = 0

                elif "markdown" in data_item["question"].lower():
                    if not isinstance(data_item["predict"], str):

                        prediction = str(data_item["predict"])
                        pred_table_html = convert_markdown_table_to_html(prediction)
                        gt_table_html = convert_markdown_table_to_html(data_item["answers"][0])
                        data_item["score"] = teds.evaluate(pred_table_html, gt_table_html)

                    else:
                        pred_table_html = convert_markdown_table_to_html(data_item["predict"])
                        gt_table_html = convert_markdown_table_to_html(data_item["answers"][0])
                        data_item["score"] = teds.evaluate(pred_table_html, gt_table_html)
            else:
                raise AISBenchDataContentError(
                    DSET_CODES.DATA_INVALID_STRUCTURE,
                    f"Expected answers to be a list with exactly 1 element for table parsing en, got {type(data_item['answers'])} with length {len(data_item['answers']) if isinstance(data_item['answers'], list) else 'N/A'}"
                )

        elif data_item["type"] == "table parsing cn":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                no_find = False
                predict_table = data_item["predict"].replace('\n','')
                if "<body" in predict_table:
                    predict_table = re.findall('<body.*', predict_table)[0]
                elif "<table" in predict_table:
                    predict_table = re.findall('<table.*', predict_table)[0]
                else:
                    no_find = True

                if no_find:
                    data_item["score"] = 0
                else:
                    pred_table_html = wrap_html_table(predict_table)
                    gold_table_html = wrap_html_table(data_item["answers"][0])
                    try:
                        data_item["score"] = teds.evaluate(pred_table_html, gold_table_html)
                    except Exception as err:
                        data_item["score"] = 0
                        logger.debug(f"TEDS evaluation failed: {err}")

        elif data_item["type"] == "chart parsing en":
            answer = data_item["answers"][0]
            if data_item["predict"]:

                pred_chart_dict = convert_str_to_multi_dict(data_item["predict"])
                if len(pred_chart_dict) == 0:
                    data_item["score"] = 0
                else:
                    pred_chart_html = dict_to_html(pred_chart_dict)
                    gt_chart_html = dict_to_html(answer)
                    data_item["score"] = teds.evaluate(pred_chart_html, gt_chart_html)
            else:
                data_item["score"] = 0

        elif data_item["type"] == "document parsing en":
            if not isinstance(data_item["answers"], list) or len(data_item["answers"]) != 1:
                raise AISBenchDataContentError(
                    DSET_CODES.DATA_INVALID_STRUCTURE,
                    f"Expected answers to be a list with exactly 1 element for document parsing en, got {type(data_item['answers'])} with length {len(data_item['answers']) if isinstance(data_item['answers'], list) else 'N/A'}"
                )
            data_item["score"] = doc_parsing_evaluation(data_item["predict"], data_item["answers"][0])

        elif data_item["type"] == "document parsing cn":
            if not isinstance(data_item["answers"], list) or len(data_item["answers"]) != 1:
                raise AISBenchDataContentError(
                    DSET_CODES.DATA_INVALID_STRUCTURE,
                    f"Expected answers to be a list with exactly 1 element for document parsing cn, got {type(data_item['answers'])} with length {len(data_item['answers']) if isinstance(data_item['answers'], list) else 'N/A'}"
                )
            data_item["score"] = doc_parsing_evaluation(data_item["predict"], data_item["answers"][0])

        elif data_item["type"] == "key information extraction en" or data_item["type"] == "key information mapping en":
            if len(data_item["answers"]) != 1:
                raise AISBenchDataContentError(
                    DSET_CODES.DATA_INVALID_STRUCTURE,
                    f"Expected exactly 1 answer for key information extraction/mapping en, got {len(data_item['answers'])}"
                )
            answers = generate_combinations(data_item["answers"][0])

            if type(answers)==list and len(answers) == 1:
                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                else:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answers[0])
            else:
                max_score = 0
                for answer in answers:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answer)

                    if data_item["score"] > max_score:
                        max_score = data_item["score"]
                data_item["score"] = max_score

        elif data_item["type"] == "key information extraction cn":
            if len(data_item["answers"]) != 1:
                raise AISBenchDataContentError(
                    DSET_CODES.DATA_INVALID_STRUCTURE,
                    f"Expected exactly 1 answer for key information extraction cn, got {len(data_item['answers'])}"
                )
            answers = ast.literal_eval(data_item["answers"][0])
            answers = {k: v if isinstance(v, list) else [v] for k, v in answers.items()}
            answers = generate_combinations(answers)
            if type(answers)==list and len(answers) == 1:
                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                else:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answers[0])
            else:
                max_score = 0
                for answer in answers:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answer)

                    if data_item["score"] > max_score:
                        max_score = data_item["score"]
                data_item["score"] = max_score

        elif data_item["type"] == "VQA with position en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                pred_dict = convert_str_to_dict(data_item["predict"])
                data_item["score"] = vqa_with_position_evaluation(pred_dict, data_item)

        elif data_item["type"] == "text translation cn":
            if len(data_item["predict"]) == 0:
                data_item["score"] = 0
            elif len(data_item["answers"][0]) == 0:
                data_item["score"] = 0
                data_item["ignore"] = "True"
            else:
                ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                data_item["score"] = (ocr_metric["bleu"] + ocr_metric["meteor"] + ocr_metric["f_measure"] + (1 - ocr_metric["edit_dist"])) / 4

        elif data_item["type"] == "fine-grained text recognition en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            elif len(data_item["predict"]) == 0:
                data_item["score"] = 0
            else:
                ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                data_item["score"] = (
                    get_value_or_zero(ocr_metric["bleu"]) +
                    get_value_or_zero(ocr_metric["meteor"]) +
                    get_value_or_zero(ocr_metric["f_measure"]) +
                    (1 - get_value_or_zero(ocr_metric["edit_dist"]))
                ) / 4
        elif data_item["type"] == "full-page OCR en":
            if not data_item["predict"]:
                data_item["score"] = 0
            else:
                ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                data_item["score"] = (
                    get_value_or_zero(ocr_metric["bleu"]) +
                    get_value_or_zero(ocr_metric["meteor"]) +
                    get_value_or_zero(ocr_metric["f_measure"]) +
                    (1 - get_value_or_zero(ocr_metric["edit_dist"]))
                ) / 4

        elif data_item["type"] == "full-page OCR cn":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                if len(data_item["predict"]) == 0:
                    data_item["score"] = 0
                else:
                    ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                    data_item["score"] = (ocr_metric["bleu"] + ocr_metric["meteor"] + ocr_metric["f_measure"] + (1 - ocr_metric["edit_dist"])) / 4

        elif data_item["type"] == "text grounding en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                predict_bbox = extract_coordinates(data_item["predict"])
                if not predict_bbox:
                    data_item["score"] = 0
                else:
                    data_item["score"] = calculate_iou(predict_bbox, data_item["answers"])

        elif data_item["type"] == "text spotting en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                predict_bbox = extract_bounding_boxes_robust(data_item["predict"])
                if not predict_bbox:
                    data_item["score"] = 0
                else:
                    data_item["score"] = spotting_evaluation(predict_bbox, data_item)

        else:
            raise ParameterValueError(
                DSET_CODES.INVALID_PARAM_VALUE,
                "Unknown task type!"
            )

        res_data_list.append(data_item)


    return res_data_list

def zip_folder(source_folder, destination_zip):
    abs_source = os.path.abspath(source_folder)
    abs_destination = os.path.abspath(destination_zip)

    with zipfile.ZipFile(abs_destination, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(abs_source):
            for file in files:
                abs_file_path = os.path.join(root, file)

                relative_path = os.path.relpath(abs_file_path, abs_source)
                zf.write(abs_file_path, relative_path)

def main_evaluation(p,default_evaluation_params_fn,validate_data_fn,evaluate_method_fn,show_result=True,per_sample=True):
    """
    This process validates a method, evaluates it and if it succed generates a ZIP file with a JSON entry for each sample.
    Params:
    p: Dictionary of parmeters with the GT/submission locations. If None is passed, the parameters send by the system are used.
    default_evaluation_params_fn: points to a function that returns a dictionary with the default parameters used for the evaluation
    validate_data_fn: points to a method that validates the corrct format of the submission
    evaluate_method_fn: points to a function that evaluated the submission and return a Dictionary with the results
    """

    if (p == None):
        p = dict([s[1:].split('=') for s in sys.argv[1:]])
        if(len(sys.argv)<3):
            logger.debug("Insufficient arguments provided. Expected format: -g=<gtFile> -s=<submFile> [-o=<outputDir>] [-p=<params>]")

    evalParams = default_evaluation_params_fn()
    if 'p' in p.keys():
        evalParams.update( p['p'] if isinstance(p['p'], dict) else json.loads(p['p']) )

    resDict={'calculated':True,'Message':'','method':'{}','per_sample':'{}'}
    try:
        validate_data_fn(p['g'], p['s'], evalParams)
        evalData = evaluate_method_fn(p['g'], p['s'], evalParams)
        resDict.update(evalData)

    except Exception as e:
        resDict['Message']= str(e)
        resDict['calculated']=False

    if 'o' in p:
        if not os.path.exists(p['o']):
            os.makedirs(p['o'])

        resultsOutputname = p['o'] + '/results.zip'
        outZip = zipfile.ZipFile(resultsOutputname, mode='w', allowZip64=True)

        del resDict['per_sample']
        if 'output_items' in resDict.keys():
            del resDict['output_items']

        outZip.writestr('method.json',json.dumps(resDict))

    if not resDict['calculated']:
        if show_result:
            sys.stderr.write('Error!\n'+ resDict['Message']+'\n\n')
        if 'o' in p:
            outZip.close()
        return resDict

    if 'o' in p:
        if per_sample == True:
            for k,v in evalData['per_sample'].items():
                outZip.writestr( k + '.json',json.dumps(v))

        if 'output_items' in evalData.keys():
            for k, v in evalData['output_items'].items():
                outZip.writestr( k,v)

        outZip.close()

    # if show_result:
    #     #sys.stdout.write("Calculated!")
    #     sys.stdout.write(json.dumps(resDict['method']))

    return resDict

def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
            'IOU_CONSTRAINT' :0.5,
            'AREA_PRECISION_CONSTRAINT' :0.5,
            'WORD_SPOTTING' :False,
            'MIN_LENGTH_CARE_WORD' :3,
            'GT_SAMPLE_NAME_2_ID':'gt_img_([0-9]+).txt',
            'DET_SAMPLE_NAME_2_ID':'res_img_([0-9]+).txt',
            'LTRB':False, #LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
            'CRLF':False, # Lines are delimited by Windows CRLF format
            'CONFIDENCES':False, #Detections must include confidence value. AP will be calculated,
            'SPECIAL_CHARACTERS':'!?.:,*"()·[]/\'',
            'ONLY_REMOVE_FIRST_LAST_CHARACTER' : True
        }

def load_zip_file(file,fileNameRegExp='',allEntries=False):
    """
    Returns an array with the contents (filtered by fileNameRegExp) of a ZIP file.
    The key's are the names or the file or the capturing group definied in the fileNameRegExp
    allEntries validates that all entries in the ZIP file pass the fileNameRegExp
    """
    try:
        archive=zipfile.ZipFile(file, mode='r', allowZip64=True)
    except Exception as e:
        raise FileOperationError(
            DSET_CODES.FILE_READ_ERROR,
            f'Error loading the ZIP archive: {str(e)}'
        )

    pairs = []
    for name in archive.namelist():
        addFile = True
        keyName = name
        if fileNameRegExp!="":
            m = re.match(fileNameRegExp,name)
            if m == None:
                addFile = False
            else:
                if len(m.groups())>0:
                    keyName = m.group(1)

        if addFile:
            pairs.append( [ keyName , archive.read(name)] )
        else:
            if allEntries:
                raise AISBenchDataContentError(
                    DSET_CODES.FILE_FORMAT_ERROR,
                    f'ZIP entry not valid: {name}'
                )

    return dict(pairs)

def validate_point_inside_bounds(x,y,imWidth,imHeight):
    if(x<0 or x>imWidth):
            raise AISBenchDataContentError(
                DSET_CODES.DATA_INVALID_STRUCTURE,
                f"X value ({x}) not valid. Image dimensions: ({imWidth},{imHeight})"
            )
    if(y<0 or y>imHeight):
            raise AISBenchDataContentError(
                DSET_CODES.DATA_INVALID_STRUCTURE,
                f"Y value ({y}) not valid. Image dimensions: ({imWidth},{imHeight})"
            )

def validate_clockwise_points(points):
    """
    Validates that the points are in clockwise order.
    """
    edge = []
    for i in range(len(points)//2):
        edge.append( (int(points[(i+1)*2 % len(points)]) - int(points[i*2])) * (int(points[ ((i+1)*2+1) % len(points)]) + int(points[i*2+1]))   )
    if sum(edge)>0:
        raise AISBenchDataContentError(
            DSET_CODES.DATA_INVALID_STRUCTURE,
            "Points are not clockwise. The coordinates of bounding points have to be given in clockwise order. Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the standard one, with the image origin at the upper left, the X axis extending to the right and Y axis extending downwards."
        )


def get_tl_line_values(line,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0):
    """
    Validate the format of the line. If the line is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values are:
    LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription]
    LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription]
    Returns values from a textline. Points , [Confidences], [Transcriptions]
    """
    confidence = 0.0
    transcription = "";
    points = []

    numPoints = 4;

    if LTRB:

        numPoints = 4;

        if withTranscription and withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',line)
            if m == None :
                m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',line)
                raise AISBenchDataContentError(
                    DSET_CODES.FILE_FORMAT_ERROR,
                    "Format incorrect. Should be: xmin,ymin,xmax,ymax,confidence,transcription"
                )
        elif withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$',line)
            if m == None :
                raise AISBenchDataContentError(
                    DSET_CODES.FILE_FORMAT_ERROR,
                    "Format incorrect. Should be: xmin,ymin,xmax,ymax,confidence"
                )
        elif withTranscription:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,(.*)$',line)
            if m == None :
                raise AISBenchDataContentError(
                    DSET_CODES.FILE_FORMAT_ERROR,
                    "Format incorrect. Should be: xmin,ymin,xmax,ymax,transcription"
                )
        else:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,?\s*$',line)
            if m == None :
                raise AISBenchDataContentError(
                    DSET_CODES.FILE_FORMAT_ERROR,
                    "Format incorrect. Should be: xmin,ymin,xmax,ymax"
                )

        xmin = int(m.group(1))
        ymin = int(m.group(2))
        xmax = int(m.group(3))
        ymax = int(m.group(4))
        if(xmax<xmin):
                raise AISBenchDataContentError(
                    DSET_CODES.DATA_INVALID_STRUCTURE,
                    "Xmax value (%s) not valid (Xmax < Xmin)." %(xmax)
                )
        if(ymax<ymin):
                raise AISBenchDataContentError(
                    DSET_CODES.DATA_INVALID_STRUCTURE,
                    "Ymax value (%s)  not valid (Ymax < Ymin)." %(ymax)
                )

        points = [ float(m.group(i)) for i in range(1, (numPoints+1) ) ]

        if (imWidth>0 and imHeight>0):
            validate_point_inside_bounds(xmin,ymin,imWidth,imHeight);
            validate_point_inside_bounds(xmax,ymax,imWidth,imHeight);

    else:

        numPoints = 8;

        if withTranscription and withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',line)
            if m == None :
                raise AISBenchDataContentError(
                    DSET_CODES.FILE_FORMAT_ERROR,
                    "Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence,transcription"
                )
        elif withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$',line)
            if m == None :
                raise AISBenchDataContentError(
                    DSET_CODES.FILE_FORMAT_ERROR,
                    "Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence"
                )
        elif withTranscription:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,(.*)$',line)
            if m == None :
                raise AISBenchDataContentError(
                    DSET_CODES.FILE_FORMAT_ERROR,
                    "Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,transcription"
                )
        else:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*$',line)
            if m == None :
                raise AISBenchDataContentError(
                    DSET_CODES.FILE_FORMAT_ERROR,
                    "Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4"
                )

        points = [ float(m.group(i)) for i in range(1, (numPoints+1) ) ]

        validate_clockwise_points(points)

        if (imWidth>0 and imHeight>0):
            validate_point_inside_bounds(points[0],points[1],imWidth,imHeight);
            validate_point_inside_bounds(points[2],points[3],imWidth,imHeight);
            validate_point_inside_bounds(points[4],points[5],imWidth,imHeight);
            validate_point_inside_bounds(points[6],points[7],imWidth,imHeight);


    if withConfidence:
        try:
            confidence = float(m.group(numPoints+1))
        except ValueError:
            raise AISBenchDataContentError(
                DSET_CODES.INVALID_DATA_TYPE,
                "Confidence value must be a float"
            )

    if withTranscription:
        posTranscription = numPoints + (2 if withConfidence else 1)
        transcription = m.group(posTranscription)
        m2 = re.match(r'^\s*\"(.*)\"\s*$',transcription)
        if m2 != None : #Transcription with double quotes, we extract the value and replace escaped characters
            transcription = m2.group(1).replace("\\\\", "\\").replace("\\\"", "\"")

    return points,confidence,transcription

def decode_utf8(raw):
    """
    Returns a Unicode object on success, or None on failure
    """
    try:
        return raw.decode('utf-8-sig',errors = 'replace')
    except:
       return None

def validate_tl_line(line,LTRB=True,withTranscription=True,withConfidence=True,imWidth=0,imHeight=0):
    """
    Validate the format of the line. If the line is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values are:
    LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription]
    LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription]
    """
    get_tl_line_values(line,LTRB,withTranscription,withConfidence,imWidth,imHeight)

def validate_lines_in_file(fileName,file_contents,CRLF=True,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0):
    """
    This function validates that all lines of the file calling the Line validation function for each line
    """
    utf8File = decode_utf8(file_contents)
    if (utf8File is None) :
        raise FileOperationError(
            DSET_CODES.FILE_READ_ERROR,
            "The file %s is not UTF-8" %fileName
        )

    lines = utf8File.split( "\r\n" if CRLF else "\n" )
    for line in lines:
        line = line.replace("\r","").replace("\n","")
        if(line != ""):
            try:
                validate_tl_line(line,LTRB,withTranscription,withConfidence,imWidth,imHeight)
            except Exception as e:
                raise AISBenchDataContentError(
                    DSET_CODES.DATA_INVALID_STRUCTURE,
                    "Line in sample not valid. Sample: %s Line: %s Error: %s" %(fileName,line,str(e))
                )

def validate_data(gtFilePath, submFilePath, evaluationParams):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """
    gt = load_zip_file(gtFilePath, evaluationParams['GT_SAMPLE_NAME_2_ID'])

    subm = load_zip_file(submFilePath, evaluationParams['DET_SAMPLE_NAME_2_ID'], True)

    #Validate format of GroundTruth
    for k in gt:
        validate_lines_in_file(k,gt[k],evaluationParams['CRLF'],evaluationParams['LTRB'],True)

    #Validate format of results
    for k in subm:
        if (k in gt) == False :
            raise AISBenchDataContentError(
                DSET_CODES.DATA_MISSING_REQUIRED_FIELD,
                "The sample %s not present in GT" %k
            )

        validate_lines_in_file(k,subm[k],evaluationParams['CRLF'],evaluationParams['LTRB'],True,evaluationParams['CONFIDENCES'])


def spotting_evaluation(prediction_list, img_metas):
    score = 0

    submit_path = ".vlmeval/dataset/utils/Ocrbench_v2/spotting_eval/submit"
    gt_path = ".vlmeval/dataset/utils/Ocrbench_v2/spotting_eval/gt"
    submit_zip_path = ".vlmeval/dataset/utils/Ocrbench_v2/spotting_eval/submit.zip"
    gt_zip_path = ".vlmeval/dataset/utils/Ocrbench_v2/spotting_eval/gt.zip"
    for file_path in [submit_path, gt_path, submit_zip_path, gt_zip_path]:
        if "zip" in file_path:
            if os.path.exists(file_path):
                os.remove(file_path)
        else:
            if os.path.exists(file_path):
                shutil.rmtree(file_path)
            os.makedirs(file_path)

    res_submit_list = []
    for item in prediction_list:
        if len(item) != 5:
            ipdb.set_trace()
        x1, y1, x2, y2, rec = item
        if x1 >= x2 or y1 >= y2:
            continue

        res_submit_list.append(",".join([str(x1),str(y1),str(x2),str(y1),str(x2),str(y2),str(x1),str(y2),rec]))

    res_gt_list = []
    for bbox, rec in zip(img_metas["bbox"], img_metas["content"]):
        x_coords = bbox[0::2]
        y_coords = bbox[1::2]

        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)

        res_gt_list.append(",".join([str(x1),str(y1),str(x2),str(y1),str(x2),str(y2),str(x1),str(y2),rec]))

    if len(res_submit_list) == 0 or len(res_gt_list) == 0:
        return 0

    with open(os.path.join(submit_path,"res_img_0.txt"), "w") as f:
        for item in res_submit_list[:-1]:
            f.write(item + "\n")
        f.write(res_submit_list[-1])

    with open(os.path.join(gt_path,"gt_img_0.txt"), "w") as f:
        for item in res_gt_list[:-1]:
            f.write(item + "\n")
        f.write(res_gt_list[-1])

    zip_folder(submit_path, submit_zip_path)
    zip_folder(gt_path, gt_zip_path)

    command = {
        'g': gt_zip_path,
        's': submit_zip_path,
        'o': './',
        'p': '{"IOU_CONSTRAINT":0.5}'
    }

    # run rrc_evaluation_funcs
    result = main_evaluation(command,default_evaluation_params,validate_data,evaluate_method)
    score = result["method"]["hmean"]
    return score

def evaluation_imports():
    """
    evaluation_imports: Dictionary ( key = module name , value = alias  )  with python modules used in the evaluation.
    """
    return {
            'Polygon':'plg',
            'numpy':'np'
            }

def get_tl_line_values_from_file_contents(content,CRLF=True,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0,sort_by_confidences=True):
    """
    Returns all points, confindences and transcriptions of a file in lists. Valid line formats:
    xmin,ymin,xmax,ymax,[confidence],[transcription]
    x1,y1,x2,y2,x3,y3,x4,y4,[confidence],[transcription]
    """
    pointsList = []
    transcriptionsList = []
    confidencesList = []

    lines = content.split( "\r\n" if CRLF else "\n" )
    for line in lines:
        line = line.replace("\r","").replace("\n","")
        if(line != "") :
            points, confidence, transcription = get_tl_line_values(line,LTRB,withTranscription,withConfidence,imWidth,imHeight);
            pointsList.append(points)
            transcriptionsList.append(transcription)
            confidencesList.append(confidence)

    if withConfidence and len(confidencesList)>0 and sort_by_confidences:
        import numpy as np
        sorted_ind = np.argsort(-np.array(confidencesList))
        confidencesList = [confidencesList[i] for i in sorted_ind]
        pointsList = [pointsList[i] for i in sorted_ind]
        transcriptionsList = [transcriptionsList[i] for i in sorted_ind]

    return pointsList,confidencesList,transcriptionsList

def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """
    for module,alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module)

    def polygon_from_points(points,correctOffset=False):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """

        if correctOffset: #this will substract 1 from the coordinates that correspond to the xmax and ymax
            points[2] -= 1
            points[4] -= 1
            points[5] -= 1
            points[7] -= 1

        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(points[0])
        resBoxes[0,4]=int(points[1])
        resBoxes[0,1]=int(points[2])
        resBoxes[0,5]=int(points[3])
        resBoxes[0,2]=int(points[4])
        resBoxes[0,6]=int(points[5])
        resBoxes[0,3]=int(points[6])
        resBoxes[0,7]=int(points[7])
        pointMat = resBoxes[0].reshape([2,4]).T
        return plg.Polygon( pointMat)

    def rectangle_to_polygon(rect):
        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(rect.xmin)
        resBoxes[0,4]=int(rect.ymax)
        resBoxes[0,1]=int(rect.xmin)
        resBoxes[0,5]=int(rect.ymin)
        resBoxes[0,2]=int(rect.xmax)
        resBoxes[0,6]=int(rect.ymin)
        resBoxes[0,3]=int(rect.xmax)
        resBoxes[0,7]=int(rect.ymax)

        pointMat = resBoxes[0].reshape([2,4]).T

        return plg.Polygon( pointMat)

    def rectangle_to_points(rect):
        points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(rect.xmax), int(rect.ymin), int(rect.xmin), int(rect.ymin)]
        return points

    def get_union(pD,pG):
        areaA = pD.area();
        areaB = pG.area();
        return areaA + areaB - get_intersection(pD, pG);

    def get_intersection_over_union(pD,pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG);
        except:
            return 0

    def get_intersection(pD,pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()

    def compute_ap(confList, matchList,numGtCare):
        correct = 0
        AP = 0
        if len(confList)>0:
            confList = np.array(confList)
            matchList = np.array(matchList)
            sorted_ind = np.argsort(-confList)
            confList = confList[sorted_ind]
            matchList = matchList[sorted_ind]
            for n in range(len(confList)):
                match = matchList[n]
                if match:
                    correct += 1
                    AP += float(correct)/(n + 1)

            if numGtCare>0:
                AP /= numGtCare

        return AP

    def transcription_match(transGt,transDet,specialCharacters='!?.:,*"()·[]/\'',onlyRemoveFirstLastCharacterGT=True):

        if onlyRemoveFirstLastCharacterGT:
            #special characters in GT are allowed only at initial or final position
            if (transGt==transDet):
                return True

            if specialCharacters.find(transGt[0])>-1:
                if transGt[1:]==transDet:
                    return True

            if specialCharacters.find(transGt[-1])>-1:
                if transGt[0:len(transGt)-1]==transDet:
                    return True

            if specialCharacters.find(transGt[0])>-1 and specialCharacters.find(transGt[-1])>-1:
                if transGt[1:len(transGt)-1]==transDet:
                    return True
            return False
        else:
            #Special characters are removed from the begining and the end of both Detection and GroundTruth
            while len(transGt)>0 and specialCharacters.find(transGt[0])>-1:
                transGt = transGt[1:]

            while len(transDet)>0 and specialCharacters.find(transDet[0])>-1:
                transDet = transDet[1:]

            while len(transGt)>0 and specialCharacters.find(transGt[-1])>-1 :
                transGt = transGt[0:len(transGt)-1]

            while len(transDet)>0 and specialCharacters.find(transDet[-1])>-1:
                transDet = transDet[0:len(transDet)-1]

            return transGt == transDet


    def include_in_dictionary(transcription):
        """
        Function used in Word Spotting that finds if the Ground Truth transcription meets the rules to enter into the dictionary. If not, the transcription will be cared as don't care
        """
        #special case 's at final
        if transcription[len(transcription)-2:]=="'s" or transcription[len(transcription)-2:]=="'S":
            transcription = transcription[0:len(transcription)-2]

        #hypens at init or final of the word
        transcription = transcription.strip('-');

        specialCharacters = "'!?.:,*\"()·[]/";
        for character in specialCharacters:
            transcription = transcription.replace(character,' ')

        transcription = transcription.strip()

        if len(transcription) != len(transcription.replace(" ","")) :
            return False;

        if len(transcription) < evaluationParams['MIN_LENGTH_CARE_WORD']:
            return False;

        notAllowed = "×÷·";

        range1 = [ ord(u'a'), ord(u'z') ]
        range2 = [ ord(u'A'), ord(u'Z') ]
        range3 = [ ord(u'À'), ord(u'ƿ') ]
        range4 = [ ord(u'Ǆ'), ord(u'ɿ') ]
        range5 = [ ord(u'Ά'), ord(u'Ͽ') ]
        range6 = [ ord(u'-'), ord(u'-') ]

        for char in transcription :
            charCode = ord(char)
            if(notAllowed.find(char) != -1):
                return False

            valid = ( charCode>=range1[0] and charCode<=range1[1] ) or ( charCode>=range2[0] and charCode<=range2[1] ) or ( charCode>=range3[0] and charCode<=range3[1] ) or ( charCode>=range4[0] and charCode<=range4[1] ) or ( charCode>=range5[0] and charCode<=range5[1] ) or ( charCode>=range6[0] and charCode<=range6[1] )
            if valid == False:
                return False

        return True

    def include_in_dictionary_transcription(transcription):
        """
        Function applied to the Ground Truth transcriptions used in Word Spotting. It removes special characters or terminations
        """
        #special case 's at final
        if transcription[len(transcription)-2:]=="'s" or transcription[len(transcription)-2:]=="'S":
            transcription = transcription[0:len(transcription)-2]

        #hypens at init or final of the word
        transcription = transcription.strip('-');

        specialCharacters = "'!?.:,*\"()·[]/";
        for character in specialCharacters:
            transcription = transcription.replace(character,' ')

        transcription = transcription.strip()

        return transcription

    perSampleMetrics = {}

    matchedSum = 0

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

    gt = load_zip_file(gtFilePath,evaluationParams['GT_SAMPLE_NAME_2_ID'])
    subm = load_zip_file(submFilePath,evaluationParams['DET_SAMPLE_NAME_2_ID'],True)

    numGlobalCareGt = 0;
    numGlobalCareDet = 0;

    arrGlobalConfidences = [];
    arrGlobalMatches = [];

    for resFile in gt:

        gtFile = decode_utf8(gt[resFile])
        if (gtFile is None) :
            raise FileOperationError(
                DSET_CODES.FILE_READ_ERROR,
                "The file %s is not UTF-8" %resFile
            )

        recall = 0
        precision = 0
        hmean = 0
        detCorrect = 0
        iouMat = np.empty([1,1])
        gtPols = []
        detPols = []
        gtTrans = []
        detTrans = []
        gtPolPoints = []
        detPolPoints = []
        gtDontCarePolsNum = [] #Array of Ground Truth Polygons' keys marked as don't Care
        detDontCarePolsNum = [] #Array of Detected Polygons' matched with a don't Care GT
        detMatchedNums = []
        pairs = []

        arrSampleConfidences = [];
        arrSampleMatch = [];
        sampleAP = 0;

        evaluationLog = ""

        pointsList,_,transcriptionsList = get_tl_line_values_from_file_contents(gtFile,evaluationParams['CRLF'],evaluationParams['LTRB'],True,False)
        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"
            if evaluationParams['LTRB']:
                gtRect = Rectangle(*points)
                gtPol = rectangle_to_polygon(gtRect)
            else:
                gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtPolPoints.append(points)

            #On word spotting we will filter some transcriptions with special characters
            if evaluationParams['WORD_SPOTTING'] :
                if dontCare == False :
                    if include_in_dictionary(transcription) == False :
                        dontCare = True
                    else:
                        transcription = include_in_dictionary_transcription(transcription)

            gtTrans.append(transcription)
            if dontCare:
                gtDontCarePolsNum.append( len(gtPols)-1 )

        evaluationLog += "GT polygons: " + str(len(gtPols)) + (" (" + str(len(gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum)>0 else "\n")

        if resFile in subm:

            detFile = decode_utf8(subm[resFile])

            pointsList,confidencesList,transcriptionsList = get_tl_line_values_from_file_contents(detFile,evaluationParams['CRLF'],evaluationParams['LTRB'],True,evaluationParams['CONFIDENCES'])

            for n in range(len(pointsList)):
                points = pointsList[n]
                transcription = transcriptionsList[n]

                if evaluationParams['LTRB']:
                    detRect = Rectangle(*points)
                    detPol = rectangle_to_polygon(detRect)
                else:
                    detPol = polygon_from_points(points)
                detPols.append(detPol)
                detPolPoints.append(points)
                detTrans.append(transcription)

                if len(gtDontCarePolsNum)>0 :
                    for dontCarePol in gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePol]
                        intersected_area = get_intersection(dontCarePol,detPol)
                        pdDimensions = detPol.area()
                        precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                        if (precision > evaluationParams['AREA_PRECISION_CONSTRAINT'] ):
                            detDontCarePolsNum.append( len(detPols)-1 )
                            break

            evaluationLog += "DET polygons: " + str(len(detPols)) + (" (" + str(len(detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum)>0 else "\n")

            if len(gtPols)>0 and len(detPols)>0:
                #Calculate IoU and precision matrixs
                outputShape=[len(gtPols),len(detPols)]
                iouMat = np.empty(outputShape)
                gtRectMat = np.zeros(len(gtPols),np.int8)
                detRectMat = np.zeros(len(detPols),np.int8)
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]
                        iouMat[gtNum,detNum] = get_intersection_over_union(pD,pG)

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum :
                            if iouMat[gtNum,detNum]>evaluationParams['IOU_CONSTRAINT']:
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                #detection matched only if transcription is equal
                                if evaluationParams['WORD_SPOTTING']:
                                    correct = gtTrans[gtNum].upper() == detTrans[detNum].upper()
                                else:
                                    correct = transcription_match(gtTrans[gtNum].upper(),detTrans[detNum].upper(),evaluationParams['SPECIAL_CHARACTERS'],evaluationParams['ONLY_REMOVE_FIRST_LAST_CHARACTER'])==True
                                detCorrect += (1 if correct else 0)
                                if correct:
                                    detMatchedNums.append(detNum)
                                pairs.append({'gt':gtNum,'det':detNum,'correct':correct})
                                evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + " trans. correct: " + str(correct) + "\n"

            if evaluationParams['CONFIDENCES']:
                for detNum in range(len(detPols)):
                    if detNum not in detDontCarePolsNum :
                        #we exclude the don't care detections
                        match = detNum in detMatchedNums

                        arrSampleConfidences.append(confidencesList[detNum])
                        arrSampleMatch.append(match)

                        arrGlobalConfidences.append(confidencesList[detNum]);
                        arrGlobalMatches.append(match);

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare >0 else float(1)
            sampleAP = precision
        else:
            recall = float(detCorrect) / numGtCare
            precision = 0 if numDetCare==0 else float(detCorrect) / numDetCare
            if evaluationParams['CONFIDENCES']:
                sampleAP = compute_ap(arrSampleConfidences, arrSampleMatch, numGtCare )

        hmean = 0 if (precision + recall)==0 else 2.0 * precision * recall / (precision + recall)

        matchedSum += detCorrect
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics[resFile] = {
                                        'precision':precision,
                                        'recall':recall,
                                        'hmean':hmean,
                                        'pairs':pairs,
                                        'AP':sampleAP,
                                        'iouMat':[] if len(detPols)>100 else iouMat.tolist(),
                                        'gtPolPoints':gtPolPoints,
                                        'detPolPoints':detPolPoints,
                                        'gtTrans':gtTrans,
                                        'detTrans':detTrans,
                                        'gtDontCare':gtDontCarePolsNum,
                                        'detDontCare':detDontCarePolsNum,
                                        'evaluationParams': evaluationParams,
                                        'evaluationLog': evaluationLog
                                    }

    # Compute AP
    AP = 0
    if evaluationParams['CONFIDENCES']:
        AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)

    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum)/numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum)/numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)

    methodMetrics = {'precision':methodPrecision, 'recall':methodRecall,'hmean': methodHmean, 'AP': AP  }

    resDict = {'calculated':True,'Message':'','method': methodMetrics,'per_sample': perSampleMetrics}


    return resDict


def extract_bounding_boxes_robust(predict_str):
    """
    Extract coordinates and text content from the given prediction string,
    handling potential format issues.

    Args:
    predict_str (str): Model prediction output as a string.

    Returns:
    list: Extracted data in the format [[x1, y1, x2, y2, text_content], ...].
          Returns None if no valid data is extracted.
    """
    results = []
    seen = set()

    # try parsing with ast.literal_eval
    try:
        data = ast.literal_eval(predict_str)
    except Exception:
        data = None

    if data is not None:
        if isinstance(data, (list, tuple)):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 5:
                    x1_str, y1_str, x2_str, y2_str = item[:4]
                    text_content = item[4]

                    x1_str = str(x1_str).strip()
                    y1_str = str(y1_str).strip()
                    x2_str = str(x2_str).strip()
                    y2_str = str(y2_str).strip()
                    text_content = str(text_content).replace("\n", "").strip().strip('"').strip("'")

                    try:
                        x1 = int(x1_str)
                        y1 = int(y1_str)
                        x2 = int(x2_str)
                        y2 = int(y2_str)

                        if not (0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000):
                            continue

                        key = (x1, y1, x2, y2, text_content)
                        if key in seen:
                            continue

                        seen.add(key)
                        results.append([x1, y1, x2, y2, text_content])
                    except ValueError:
                        continue
    else:
        # try parsing with regular expression

        list_content = predict_str
        items = re.findall(r'[\[\(]\s*([^\[\]\(\)]*?)\s*[\]\)]', list_content)

        if not items:
            return None

        for item in items:
            parts = item.split(',', 4)
            if len(parts) < 5:
                continue

            x1_str, y1_str, x2_str, y2_str, text_content = parts

            x1_str = x1_str.strip()
            y1_str = y1_str.strip()
            x2_str = x2_str.strip()
            y2_str = y2_str.strip()
            text_content = text_content.replace("\n", "").strip().strip('"').strip("'")

            try:
                x1 = int(x1_str)
                y1 = int(y1_str)
                x2 = int(x2_str)
                y2 = int(y2_str)

                if not (0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000):
                    continue

                key = (x1, y1, x2, y2, text_content)
                if key in seen:
                    continue

                seen.add(key)
                results.append([x1, y1, x2, y2, text_content])
            except ValueError:
                continue

    if not results:
        return None

    return results


def extract_coordinates(text):
    # Regex pattern to match coordinates in either (x1, y1, x2, y2) or [x1, y1, x2, y2] format

    pattern = r'[\(\[]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\)\]]'

    matches = list(re.finditer(pattern, text))
    coords_list = []
    coords_set = set()
    for match in matches:

        x1, y1, x2, y2 = map(int, match.groups())

        if all(0 <= n <= 1000 for n in [x1, y1, x2, y2]):
            coords = (x1, y1, x2, y2)

            if coords in coords_set:
                coords_list = [c for c in coords_list if c != coords]

            coords_list.append(coords)
            coords_set.add(coords)
    if coords_list:
        last_coords = coords_list[-1]
        return list(last_coords)
    else:
        return None

@LOAD_DATASET.register_module()
class OCRBenchV2Dataset(BaseDataset):
    @staticmethod
    def load(path):
        path = get_data_path(path)
        image_root_path = os.path.join(os.path.dirname(path), "OCRBench_v2_images")
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
            prompt = line['question']
            # add image info
            msgs = []
            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image_url', image_url=p) for p in tgt_path])
            else:
                msgs = [dict(type='image_url', image_url=tgt_path)]
            msgs.append(dict(type='text', text=prompt))
            # split image text in order
            msgs = split_OCRBenchV2(msgs)
            content = get_content_str(msgs)
            choices = build_choices(line)
            dataset.append({"content": content, 
                            "answer": {'choices': json.dumps(choices), 
                                        'question': line['question'],
                                        'content': line['content'],
                                        'answer': line['answer'],
                                        'split': line['split'] if 'split' in line else None,
                                        'category': line['category'] if 'category' in line else None,
                                        'bbox_raw': line['bbox_raw'] if 'bbox_raw' in line else None,
                                        'evals': line['eval'] if 'eval' in line else None}})
        return Dataset.from_list(dataset)


def ocrbench_v2_aggregate_accuracy(data_list):
    en_text_recognition_list, en_text_detection_list, en_text_spotting_list, en_relationship_extraction_list = [], [], [], []
    en_element_parsing_list, en_mathematical_calculation_list, en_visual_text_understanding_list = [], [], []
    en_knowledge_reasoning_list = []

    cn_text_recognition_list, cn_relationship_extraction_list = [], []
    cn_element_parsing_list, cn_visual_text_understanding_list = [], []
    cn_knowledge_reasoning_list = []

    res_list = []
    for item in data_list:
        if "ignore" in item.keys():
            if item["ignore"] != "True":
                raise AISBenchDataContentError(
                    DSET_CODES.DATA_INVALID_STRUCTURE,
                    "Invalid ignore value: %s, expected 'True'" % item["ignore"]
                )

        elif item["type"] == "text recognition en" or item["type"] == "fine-grained text recognition en" or item["type"] == "full-page OCR en":
            en_text_recognition_list.append(item["score"])

        elif item["type"] == "text grounding en" or item["type"] == "VQA with position en":
            en_text_detection_list.append(item["score"])

        elif item["type"] == "text spotting en":
            en_text_spotting_list.append(item["score"])

        elif item["type"] == "key information extraction en" or item["type"] == "key information mapping en":
            en_relationship_extraction_list.append(item["score"])

        elif item["type"] == "document parsing en" or item["type"] == "chart parsing en" \
        or item["type"] == "table parsing en" or item["type"] == "formula recognition en":
            en_element_parsing_list.append(item["score"])

        elif item["type"] == "math QA en" or item["type"] == "text counting en":
            en_mathematical_calculation_list.append(item["score"])

        elif item["type"] == "document classification en" \
        or item["type"] == "cognition VQA en" or item["type"] == "diagram QA en":
            en_visual_text_understanding_list.append(item["score"])

        elif item["type"] == "reasoning VQA en" or item["type"] == "science QA en" \
        or item["type"] == "APP agent en" or item["type"] == "ASCII art classification en":
            en_knowledge_reasoning_list.append(item["score"])

        elif item["type"] == "full-page OCR cn":
            cn_text_recognition_list.append(item["score"])

        elif item["type"] == "key information extraction cn" or item["type"] == "handwritten answer extraction cn":
            cn_relationship_extraction_list.append(item["score"])

        elif item["type"] == "document parsing cn" or item["type"] == "table parsing cn" or item["type"] == "formula recognition cn":
            cn_element_parsing_list.append(item["score"])

        elif item["type"] == "cognition VQA cn":
            cn_visual_text_understanding_list.append(item["score"])

        elif item["type"] == "reasoning VQA cn" or item["type"] == "text translation cn":
            cn_knowledge_reasoning_list.append(item["score"])

        else:
            raise ParameterValueError(
                DSET_CODES.INVALID_PARAM_VALUE,
                "Unknown task type!"
            )

    en_scores = {
        "en_text_recognition": en_text_recognition_list,
        "en_text_detection": en_text_detection_list,
        "en_text_spotting": en_text_spotting_list,
        "en_relationship_extraction": en_relationship_extraction_list,
        "en_element_parsing": en_element_parsing_list,
        "en_mathematical_calculation": en_mathematical_calculation_list,
        "en_visual_text_understanding": en_visual_text_understanding_list,
        "en_knowledge_reasoning": en_knowledge_reasoning_list
    }

    cn_scores = {
        "cn_text_recognition": cn_text_recognition_list,
        "cn_relationship_extraction": cn_relationship_extraction_list,
        "cn_element_parsing": cn_element_parsing_list,
        "cn_visual_text_understanding": cn_visual_text_understanding_list,
        "cn_knowledge_reasoning": cn_knowledge_reasoning_list
    }

    en_averages = calculate_average(en_scores)
    cn_averages = calculate_average(cn_scores)

    return en_averages,cn_averages


class OCRBenchV2Evaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        details = []
        scores = []
        special_characters = ['<|im_end|>']
        predict_result = []
        for pred, ref in zip(predictions, references):
            refer = ref
            predict = str(pred) if pd.notna(pred) else ''
            answers = ast.literal_eval(refer['answer'])
            category = refer['category']
            questions = refer['question']
            evals = refer['evals']
            bbox_raw = refer['bbox_raw']
            content_raw = refer['content']
            # Process bbox and content fields
            bbox = ast.literal_eval(bbox_raw) if bbox_raw != 'without bbox' and bbox_raw != None else bbox_raw 
            content = ast.literal_eval(content_raw) if content_raw != 'without content' else content_raw
            # Build result dictionary
            result_entry = {
                "type": category,
                "question": questions,
                "predict": predict,
                "answers": answers,
                "bbox": bbox,
                "content": content
            }
            # Add eval field if present
            if evals != 'without eval':
                result_entry["eval"] = evals
            predict_result.append(result_entry)
        res_data_list = process_predictions(predict_result)
        en_scores, cn_scores = ocrbench_v2_aggregate_accuracy(res_data_list)
        final_score_dict = {}
        if len(en_scores) > 0:
            score_en_overall = sum(en_scores.values()) / len(en_scores)
            final_score_dict["English Overall Score"] = score_en_overall
        if len(cn_scores) > 0:
            score_cn_overall = sum(cn_scores.values()) / len(cn_scores)
            final_score_dict["Chinese Overall Score"] = score_cn_overall
        return final_score_dict