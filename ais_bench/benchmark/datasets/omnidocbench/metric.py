# Copyright (c) OpenDataLab https://github.com/opendatalab/OmniDocBench (2025/11/07)
# SPDX-License-Identifier: Apache-2.0
# Part of this document is directly reused from the above warehouse without modification.

import re
from datetime import datetime
import os 
import json
import copy
import time
import logging
import Levenshtein
from tqdm import tqdm
import evaluate
import numpy as np
import random
import shutil
from collections import deque, defaultdict
from apted.helpers import Tree
from apted import APTED, Config
from lxml import etree, html
import pandas as pd
import subprocess
from skimage.measure import ransac
from threading import Timer
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor, as_completed

from ais_bench.benchmark.datasets.omnidocbench.registory import METRIC_REGISTRY


SKIP_PATTERNS = [r'\{', r'\}', r'[\[\]]', r'\\begin\{.*?\}', r'\\end\{.*?\}', r'\^', r'\_', r'\\.*rule.*', r'\\.*line.*', r'\[[\-.0-9]+[epm][xtm]\]']
SKIP_Tokens = ['\\', '\\\\', '\\index', '\\a', '&', '$', '\\multirow', '\\def', '\\raggedright', '\\url', '\\cr', '\\ensuremath', '\\left', '\\right', 
               '\\mathchoice', '\\scriptstyle', '\\displaystyle', '\\qquad', '\\quad', '\\,', '\\!', '~', '\\boldmath']
PHANTOM_Tokens = ['\\fontfamily', '\\vphantom', '\\phantom', '\\rowcolor', '\\ref']
TWO_Tail_Tokens = ['\\frac', '\\binom']
AB_Tail_Tokens = ['\\xrightarrow', '\\xleftarrow', '\\sqrt']        # special token \xxx [] {} 
TWO_Tail_Invisb_Tokens = ['\\overset', '\\underset', '\\stackrel']
ONE_Tail_Tokens = ['\\widetilde', '\\overline', '\\hat', '\\widehat', '\\tilde', '\\Tilde', '\\dot', '\\bar', '\\vec', '\\underline', '\\underbrace', '\\check',
                   '\\breve', '\\Bar', '\\Vec', '\\mathring', '\\ddot']
ONE_Tail_Invisb_Tokens = ['\\boldsymbol', '\\pmb', '\\textbf', '\\mathrm', '\\mathbf', '\\mathbb', '\\mathcal', '\\textmd', '\\texttt', '\\textnormal', 
                          '\\text', '\\textit', '\\textup', '\\mathop', '\\mathbin', '\\smash', '\\operatorname', '\\textrm', '\\mathfrak', '\\emph',
                          '\\textsf', '\\textsc']

tabular_template = r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\usepackage{geometry}
\geometry{a<PaperSize>paper,scale=0.98}
\pagestyle{empty}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amssymb}
\usepackage{upgreek}
\usepackage{amsmath}
\usepackage{xcolor}
\begin{document}
\makeatletter
\renewcommand*{\@textcolor}[3]{%%
  \protect\leavevmode
  \begingroup
    \color#1{#2}#3%%
  \endgroup
}
\makeatother
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""

# Need to configure Source Han Sans SC or other Chinese fonts
formular_template = r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\usepackage{geometry}
\geometry{a<PaperSize>paper,scale=0.98}
\pagestyle{empty}
\usepackage{amsmath}
\usepackage{upgreek}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{xeCJK}
\setCJKmainfont{Source Han Sans SC}
\setCJKsansfont{Source Han Sans SC}
\setCJKmonofont{Source Han Sans SC}
\xeCJKsetup{CJKmath=true}
\begin{document}
\makeatletter
\renewcommand*{\@textcolor}[3]{%%
  \protect\leavevmode
  \begingroup
    \color#1{#2}#3%%
  \endgroup
}
\makeatother
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""

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
            if self.colspan is None or self.rowspan is None:
                result = '"tag": %s, "colspan": None, "rowspan": None, "text": %s' % \
                     (self.tag, self.content)
            else:
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
        return float(Levenshtein.distance(*sequences)) / self.maximum(*sequences)

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
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
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
        for n in list(node):
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
        scores = [self.evaluate(pred_json.get(filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        scores = dict(zip(samples, scores))
        return scores

@METRIC_REGISTRY.register("TEDS")
class call_TEDS():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default', out_dir='./'):
        teds = TEDS(structure_only=False)
        teds_structure_only = TEDS(structure_only=True)
        group_scores = defaultdict(list)
        group_scores_structure_only = defaultdict(list)
        samples = self.samples
        per_table_score = {}
        for sample in samples:
            gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
            pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
            try:
                score = teds.evaluate(pred, gt)
            except:
                score = 0
                print(f'TEDS score error for table {sample["gt_idx"]} in {sample["img_id"]}. The score is set to 0.')
            try:
                score_structure_only = teds_structure_only.evaluate(pred, gt)
            except:
                score_structure_only = 0
                print(f'TEDS_structure_only score error for table {sample["gt_idx"]} in {sample["img_id"]}. The score is set to 0.')
            group_scores['all'].append(score)
            group_scores_structure_only['all'].append(score_structure_only)
            if not sample.get('metric'):
                sample['metric'] = {}
            sample['metric']['TEDS'] = score
            sample['metric']['TEDS_structure_only'] = score_structure_only
            per_table_score[sample['img_id']+'_'+str(sample['gt_idx'])] = {'TEDS': score, 'TEDS_structure_only': score_structure_only}
            for group in group_info:
                select_flag = True
                for k, v in group.items():
                    for gt_attribute in sample['gt_attribute']:   # gt_attribute is a list containing all merged gt attributes
                        if not gt_attribute:   # if no GT attributes, don't include in calculation
                            select_flag = False
                        elif gt_attribute[k] != v:  # if any gt attribute doesn't meet criteria, don't select
                            select_flag = False
                if select_flag:
                    group_scores[str(group)].append(score)
        with open(f'./result/{save_name}_per_table_TEDS.json', 'w', encoding='utf-8') as f:
            json.dump(per_table_score, f, indent=4, ensure_ascii=False)
        result = {}
        for group_name, scores in group_scores.items():
            if len(scores) > 0:
                result[group_name] = sum(scores) / len(scores)    # average of normalized scores at sample level
            else:
                result[group_name] = 'NaN'
                print(f'Warning: Empyty matched samples for {group_name}.')
        
        structure_only_result = {}
        for group_name, scores in group_scores_structure_only.items():
            if len(scores) > 0:
                structure_only_result[group_name] = sum(scores) / len(scores)    # average of normalized scores at sample level
            else:
                structure_only_result[group_name] = 'NaN'
                print(f'Warning: Empyty matched samples for {group_name}.')

        return samples, {'TEDS': result, 'TEDS_structure_only': structure_only_result}


def get_groups(samples, group_info):
    group_samples = defaultdict(list)
    for sample in samples:
        group_samples['all'].append(sample)
        for group in group_info:
            select_flag = True
            for k, v in group.items():
                for gt_attribute in sample['gt_attribute']:   # gt_attribute is a list containing all merged gt attributes
                    if not gt_attribute:   # if no GT attributes, don't include in calculation
                        select_flag = False
                    elif gt_attribute[k] != v:  # if any gt attribute doesn't meet criteria, don't select
                        select_flag = False
            if select_flag:
                group_samples[str(group)].append(sample)
    return group_samples


@METRIC_REGISTRY.register("BLEU")
class call_BLEU():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default', out_dir='./'):
        group_samples = get_groups(self.samples, group_info)
        result = {}
        for group_name, samples in group_samples.items():
            predictions, references = [], []
            for sample in samples:
                gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
                pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
                predictions.append(gt)
                references.append(pred)
            bleu = evaluate.load("bleu", keep_in_memory=True, experiment_id=random.randint(1,1e8))
            bleu_results = bleu.compute(predictions=predictions, references=references)
            result[group_name] = bleu_results["bleu"]
        
        return self.samples, {'BLEU': result}
    
@METRIC_REGISTRY.register("METEOR")
class call_METEOR():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default', out_dir='./'):
        group_samples = get_groups(self.samples, group_info)
        result = {}
        for group_name, samples in group_samples.items():
            predictions, references = [], []
            for sample in samples:
                gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
                pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
                predictions.append(gt)
                references.append(pred)
            meteor = evaluate.load('meteor', keep_in_memory=True, experiment_id=random.randint(1,1e8))
            meteor_results = meteor.compute(predictions=predictions, references=references)
            result[group_name] = meteor_results['meteor']
        
        return self.samples, {'METEOR': result}

@METRIC_REGISTRY.register("Edit_dist")
class call_Edit_dist():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default', out_dir='./'):
        samples = self.samples
        for sample in samples:
            img_name = sample['img_id'] if sample['img_id'].endswith('.jpg') or sample['img_id'].endswith('.png') else '_'.join(sample['img_id'].split('_')[:-1])
            sample['image_name'] = img_name
            gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
            pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
            upper_len = max(len(pred), len(gt))
            sample['upper_len'] = upper_len
            if len(pred) > 0 or len(gt) > 0:
                edit_dist = Levenshtein.distance(pred, gt)
                if not sample.get('metric'):
                    sample['metric'] = {}
                sample['metric']['Edit_dist'] = edit_dist / upper_len
                sample['Edit_num'] = edit_dist

        if isinstance(samples, list):
            saved_samples = samples
        else:
            saved_samples = samples.samples
        
        if not saved_samples:
            return samples, {'Edit_dist': {'ALL_page_avg': 'NaN'}}

        df = pd.DataFrame(saved_samples)
        up_total_avg = df.groupby("image_name").apply(lambda x: x['Edit_num'].sum() / x['upper_len'].sum()) # page level, sum of edits divided by sum of max(gt,pred) lengths for each sample
        all_total_avg = df['Edit_num'].sum() / df['upper_len'].sum()
        per_img_score = up_total_avg.to_dict()
        with open(f'{out_dir}/{save_name}_per_page_edit.json', 'w', encoding='utf-8') as f:  #xieruwenjian 
            json.dump(per_img_score, f, indent=4, ensure_ascii=False)        
        
        edit_whole = df['Edit_num'].sum() / df['upper_len'].sum()
        df['ratio'] = df['Edit_num'] / df['upper_len']
        edit_sample_avg = df['ratio'].mean()
        # edit_sample_avg = df['metric']['Edit_dist'].mean()
        return samples, {'Edit_dist': {'ALL_page_avg': up_total_avg.mean(), 'edit_whole': edit_whole, 'edit_sample_avg': edit_sample_avg}}


def norm_same_token(token):
    special_map = {
        "\\cdot": ".",
        "\\mid": "|",
        "\\to": "\\rightarrow",
        "\\top": "T",
        "\\Tilde": "\\tilde",
        "\\cdots": "\\dots",
        "\\prime": "'",
        "\\ast": "*",
        "\\left<": "\\langle",
        "\\right>": "\\rangle"
    }
    if token in special_map.keys():
        token = special_map[token]
    if token.startswith('\\left') or token.startswith('\\right'):
        token = token.replace("\\left", "").replace("\\right", "")
    if token.startswith('\\big') or token.startswith('\\Big'):
        if "\\" in token[4:]:
            token = "\\"+token[4:].split("\\")[-1]
        else:
            token = token[-1]
    
    if token in ['\\leq', '\\geq']:
        return token[0:-1]
    if token in ['\\lVert', '\\rVert', '\\Vert']:
        return '\\|'
    if token in ['\\lvert', '\\rvert', '\\vert']:
        return '|'
    if token.endswith("rightarrow"):
        return "\\rightarrow"
    if token.endswith("leftarrow"):
        return "\\leftarrow"
    if token.startswith('\\wide'):
        return token.replace("wide", "")
    if token.startswith('\\var'):
        return token.replace("\\var", "")
    return token


class HungarianMatcher:
    def __init__(
        self, 
        cost_token: float = 1,
        cost_position: float = 0.05,
        cost_order: float = 0.15,
    ):
        self.cost_token = cost_token
        self.cost_position = cost_position
        self.cost_order = cost_order
        self.cost = {}
    
    def calculate_token_cost_old(self, box_gt, box_pred):
        token_cost = np.ones((len(box_gt), len(box_pred)))
        for i in range(token_cost.shape[0]):
            box1 = box_gt[i]
            for j in range(token_cost.shape[1]):
                box2 = box_pred[j]
                if box1['token'] == box2['token']:
                    token_cost[i, j] = 0
                elif norm_same_token(box1['token']) == norm_same_token(box2['token']):
                    token_cost[i, j] = 0.05
        return np.array(token_cost)
        
    def calculate_token_cost(self, box_gt, box_pred):
        token2id = {}
        for data in box_gt+box_pred:
            if data['token'] not in token2id:
                token2id[data['token']] = len(token2id)
        num_classes = len(token2id)
        
        token2id_norm = {}
        for data in box_gt+box_pred:
            if norm_same_token(data['token']) not in token2id_norm:
                token2id_norm[norm_same_token(data['token'])] = len(token2id_norm)
        num_classes_norm = len(token2id_norm)
        
        gt_token_array = []
        norm_gt_token_array = []    
        for data in box_gt:
            gt_token_array.append(token2id[data['token']])
            norm_gt_token_array.append(token2id_norm[norm_same_token(data['token'])])
            
        pred_token_logits = []
        norm_pred_token_logits = []
        for data in box_pred:
            logits = [0] * num_classes
            logits[token2id[data['token']]] = 1
            pred_token_logits.append(logits)
            
            logits_norm = [0] * num_classes_norm
            logits_norm[token2id_norm[norm_same_token(data['token'])]] = 1
            norm_pred_token_logits.append(logits_norm)
            
        gt_token_array = np.array(gt_token_array)
        pred_token_logits = np.array(pred_token_logits)
        
        norm_gt_token_array = np.array(norm_gt_token_array)
        norm_pred_token_logits = np.array(norm_pred_token_logits)
        
        token_cost = 1.0 - pred_token_logits[:, gt_token_array]
        norm_token_cost = 1.0 - norm_pred_token_logits[:, norm_gt_token_array]

        token_cost[np.logical_and(token_cost==1, norm_token_cost==0)] = 0.05
        return token_cost.T
        
        
    def box2array(self, box_list, size):
        W, H = size
        box_array = []
        for box in box_list:
            x_min, y_min, x_max, y_max = box['bbox']
            box_array.append([x_min/W, y_min/H, x_max/W, y_max/H])
        return np.array(box_array)
        
    def order2array(self, box_list):
        order_array = []
        for idx, box in enumerate(box_list):
            order_array.append([idx / len(box_list)])
        return np.array(order_array)
    
    def calculate_l1_cost(self, gt_array, pred_array):
        scale = gt_array.shape[-1]
        l1_cost = cdist(gt_array, pred_array, 'minkowski', p=1)
        return l1_cost / scale
        
    def __call__(self, box_gt, box_pred, gt_size, pred_size):
        aa = time.time()
        gt_box_array = self.box2array(box_gt, gt_size)
        pred_box_array = self.box2array(box_pred, pred_size)
        gt_order_array = self.order2array(box_gt)
        pred_order_array = self.order2array(box_pred)

        token_cost = self.calculate_token_cost(box_gt, box_pred)
        position_cost = self.calculate_l1_cost(gt_box_array, pred_box_array)
        order_cost = self.calculate_l1_cost(gt_order_array, pred_order_array)

        self.cost["token"] = token_cost
        self.cost["position"] = position_cost
        self.cost["order"] = order_cost
        
        cost = self.cost_token * token_cost + self.cost_position * position_cost + self.cost_order * order_cost
        cost[np.isnan(cost) | np.isinf(cost)] = 100
        indexes = linear_sum_assignment(cost)
        matched_idxes = []
        for a, b in zip(*indexes):
            matched_idxes.append((a, b))
        
        return matched_idxes


def run_cmd(cmd, timeout_sec=30, temp_dir=None):
    # 设置进程独立的环境变量
    env = os.environ.copy()
    if temp_dir:
        env['TMPDIR'] = temp_dir
        env['TMP'] = temp_dir  
        env['TEMP'] = temp_dir
        env['MAGICK_TMPDIR'] = temp_dir
        env['TEXMFCACHE'] = temp_dir
        env['TEXMFVAR'] = temp_dir
    try:
        proc = subprocess.Popen(cmd, shell=True, env=env)
    except Exception as e:
        logging.warning(e)
    kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        stdout,stderr = proc.communicate()
    finally:
        timer.cancel()
        
def convert_pdf2img(pdf_filename, png_filename, temp_dir=None):
    cmd = "magick -density 200 -quality 100 \"%s\" \"%s\""%(pdf_filename, png_filename)
    run_cmd(cmd, temp_dir=temp_dir)

def crop_image(image_path, pad=8):
    img = Image.open(image_path).convert("L")
    img_data = np.asarray(img, dtype=np.uint8)
    nnz_inds = np.where(img_data!=255)
    if len(nnz_inds[0]) == 0:
        y_min = 0
        y_max = 10
        x_min = 0
        x_max = 10
    else:
        y_min = np.min(nnz_inds[0])
        y_max = np.max(nnz_inds[0])
        x_min = np.min(nnz_inds[1])
        x_max = np.max(nnz_inds[1])
        
    img = Image.open(image_path).convert("RGB").crop((x_min-pad, y_min-pad, x_max+pad, y_max+pad))
    img.save(image_path)
    
def extrac_bbox_from_color_image(image_path, color_list):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    pixels = list(img.getdata())
    
    bbox_list = []
    for target_color in color_list:
        target_pixels = [ i for i, pixel in enumerate(pixels)if pixel == target_color ]
        x_list = []
        y_list = []
        for idx in target_pixels:
            x_list.append(idx % W)
            y_list.append(idx // W)
        try:
            y_min, y_max, x_min, x_max = min(y_list), max(y_list), min(x_list), max(x_list)
            bbox_list.append([x_min-1, y_min-1, x_max+1, y_max+1])

        except:
            bbox_list.append([])
            continue
        
    img = img.convert("L")
    img_bw = img.point(lambda x: 255 if x == 255 else 0, '1')
    img_bw.convert("RGB").save(image_path) 
    return bbox_list

def tokenize_latex(latex_code, latex_type="", middle_file=""):
    if not latex_code:
        return False, latex_code
    if not latex_type:
        latex_type = "tabular" if "tabular" in latex_code else "formula"
    if not middle_file:
        middle_file = "out-" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + ".txt"
    temp_file = middle_file + '.tmp'
    
    if latex_type == "formula":
        with open(temp_file, 'w') as f:
            prepre = latex_code
            # replace split, align with aligned
            prepre = re.sub(r'\\begin{(split|align|alignedat|alignat|eqnarray)\*?}(.+?)\\end{\1\*?}', r'\\begin{aligned}\2\\end{aligned}', prepre, flags=re.S)
            prepre = re.sub(r'\\begin{(smallmatrix)\*?}(.+?)\\end{\1\*?}', r'\\begin{matrix}\2\\end{matrix}', prepre, flags=re.S)
            f.write(prepre)
    
        cmd = r"cat %s | node %s %s > %s " % (temp_file, os.path.join(os.path.dirname(__file__), 'preprocess_formula.js'), 'normalize', middle_file)
        ret = subprocess.call(cmd, shell=True)
        os.remove(temp_file)
        if ret != 0:
            return False, latex_code
        
        operators = '\s?'.join('|'.join(['arccos', 'arcsin', 'arctan', 'arg', 'cos', 'cosh', 'cot', 'coth', 'csc', 'deg', 'det', 'dim', 'exp', 'gcd', 'hom', 'inf',
                                        'injlim', 'ker', 'lg', 'lim', 'liminf', 'limsup', 'ln', 'log', 'max', 'min', 'Pr', 'projlim', 'sec', 'sin', 'sinh', 'sup', 'tan', 'tanh']))
        ops = re.compile(r'\\operatorname {(%s)}' % operators)
        with open(middle_file, 'r') as fin:
            for line in fin:
                tokens = line.strip().split()
                tokens_out = []
                for token in tokens:
                    tokens_out.append(token)
                post = ' '.join(tokens_out)
                # use \sin instead of \operatorname{sin}
                names = ['\\'+x.replace(' ', '') for x in re.findall(ops, post)]
                post = re.sub(ops, lambda match: str(names.pop(0)), post).replace(r'\\ \end{array}', r'\end{array}')
        os.remove(middle_file)
        return True, post
    
    elif latex_type == "tabular":
        latex_code = latex_code.replace("\\\\%", "\\\\ %")
        latex_code = latex_code.replace("\%", "<PERCENTAGE_TOKEN>")
        latex_code = latex_code.split('%')[0]
        latex_code = latex_code.replace("<PERCENTAGE_TOKEN>", "\%")
        if not "\\end{tabular}" in latex_code:
            latex_code += "\\end{tabular}"
        with open(middle_file, 'w') as f:
            f.write(latex_code.replace('\r', ' ').replace('\n', ' '))
        cmd = "perl -pe 's|hskip(.*?)(cm\\|in\\|pt\\|mm\\|em)|hspace{\\1\\2}|g' %s > %s"%(middle_file, temp_file)
        ret = subprocess.call(cmd, shell=True)
        if ret != 0:
            return False, latex_code
        os.remove(middle_file)
        cmd = r"cat %s | node %s %s > %s " % (temp_file, os.path.join(os.path.dirname(__file__), 'preprocess_tabular.js'), 'tokenize', middle_file)
        ret = subprocess.call(cmd, shell=True)
        os.remove(temp_file)
        if ret != 0:
            return False, latex_code
        with open(middle_file, 'r') as fin:
            for line in fin:
                tokens = line.strip().split()
                tokens_out = []
                for token in tokens:
                    tokens_out.append(token)
                post = ' '.join(tokens_out)
        os.remove(middle_file)
        return True, post
    else:
        logging.info(f"latex type{latex_type} unrecognized.")
        return False, latex_code

def remove_trailing_latex(formula):
    pattern = r'(\\(hspace\*?\{[^{}]*?\}|vspace\*?\{[^{}]*?\}|smallskip|medskip|quad|qquad|bigskip|[;,])|\~|\.)*$'
    # Replace the matched pattern with an empty string
    cleaned_formula = re.sub(pattern, '', formula, count=1)
    return cleaned_formula

def find_matching_brace(sequence, start_index, brace=['{', '}']):
    # Finds the index of the matching brace for the one at start_index
    left_brace, right_brace = brace
    depth = 0
    for i, char in enumerate(sequence[start_index:], start=start_index):
        if char == left_brace:
            depth += 1
        elif char == right_brace:
            depth -= 1
            if depth == 0:
                return i
    if depth > 0:
        error_info = "Warning! found no matching brace in sequence !"
        raise ValueError(error_info)
    return -1

def normalize_latex(l, rm_trail=False):
    if "tabular" in l:
        latex_type = "tabular"
    else:
        latex_type = "formula"
        
    if rm_trail:
        l = remove_trailing_latex(l)
    l = l.strip().replace(r'\pmatrix', r'\mypmatrix').replace(r'\matrix', r'\mymatrix')
    
    for item in ['\\raggedright', '\\arraybackslash', '\\lowercase', '\\uppercase']:
        l = l.replace(item, "")
    
        
    pattern = r'\\[hv]space { [.0-9a-z ]+ }'
    old_token = re.findall(pattern, l, re.DOTALL)
    if latex_type == "tabular":
        new_token = ["" for item in old_token]
    else:
        new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)
        
    if latex_type == "tabular":
        l = l.replace("\\begin {tabular}", "\\begin{tabular}")
        l = l.replace("\\end {tabular}", "\\end{tabular}")
        l = l.replace("\\begin {array}", "\\begin{array}")
        l = l.replace("\\end {array}", "\\end{array}")
        l_split = l.split(' ')
        idx = 0
        while idx < len(l_split):
            token = l_split[idx]
            if token == "\\begin{tabular}":
                sub_idx = idx + 1
                end_idx = find_matching_brace(l_split, sub_idx)
                new_token = "".join(l_split[idx: end_idx+1])
                l_split = l_split[0:idx] + [new_token] + l_split[end_idx+1:]
                break
            idx += 1
        l = ' '.join(l_split)
        
        l_split = l.split(' ')
        idx = 0
        while idx < len(l_split):
            token = l_split[idx]
            if token in ["\\cmidrule", "\\cline"]:
                sub_idx = idx + 1
                if l_split[sub_idx] == "(":
                    mid_end = find_matching_brace(l_split, sub_idx, brace=['(', ')'])
                    end_idx = find_matching_brace(l_split, mid_end+1)
                else:
                    end_idx = find_matching_brace(l_split, sub_idx)
                new_token = "".join(l_split[idx: end_idx+1])
                l_split = l_split[0:idx] + [new_token] + l_split[end_idx+1:]
            idx += 1
        l = ' '.join(l_split)
    
    pattern = r'\\begin{array} { [lrc ]+ }'
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace("\\begin{array} ", "<s>").replace(" ", "").replace("<s>", "\\begin{array} ") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)
    
    pattern = r'\\not [<>+=\-]'
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)
    
    l = " "+l+" "
    l = l.replace(" \\ldots ", " . . . ")
    l = l.replace(" \\cdots ", " . . . ")
    l = l.replace(" \\dots ", " . . . ")
    l = l.replace(" \\dotsb ", " . . . ")
    l = l.replace(" \\log ", " \\mathrm { l o g } ")
    l = l.replace(" \\exp ", " \\mathrm { e x p } ")
    l = l.replace(" \\sin ", " \\mathrm { s i n } ")
    l = l.replace(" \\cos ", " \\mathrm { c o s } ")
    l = l.replace(" \\tan ", " \\mathrm { t a n } ")
    l = l.replace(" \\tanh ", " \\mathrm { t a n h } ")
    l = l.replace(" \\cosh ", " \\mathrm { c o s h } ")
    l = l.replace(" \\sinh ", " \\mathrm { s i n h } ")
        
    # ** token such as \big( should be one token
    pattern = r'\\[Bb]ig[g]?[glrm]? [(){}|\[\]] '
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft+" ")
        
    pattern = r'\\[Bb]ig[g]?[glrm]? \\.*? '
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft+" ")
        
    pattern = r'\\operatorname \*'
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = ["\\operatorname" for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)

    l = l.replace("\\lefteqn", "")
    l = l.replace("\\footnote ", "^ ")
    
    pattern = r'\\\' [^{] '
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft+" ")
    
    if latex_type == "tabular":
        pattern = r'\[ [\-.0-9 ]+[exptcm ]+ \]'
        old_token = re.findall(pattern, l, re.DOTALL)
        new_token = [item.replace(" ", "") for item in old_token]
        for bef, aft in zip(old_token, new_token):
            l = l.replace(bef, aft)
    
    # ** \parbox { 3cm } {} shoudle be combined as one token
    pattern = r'\\parbox {[^{]+}'
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)
    
    # ** \raisebox{<lift>}[<height>][<depth>] {} shoudle be combined as one token, \raisebox{-1.5ex}[0pt]
    pattern = r'\\raisebox {[^{]+} [\[\]0-9 exptcm]+{'
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft[0:-1]+" {")
        
    # ** \char shoudle be combined as one token
    pattern = r'{ \\char[0-9\' ]+}'
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, "{ "+aft[1:-1]+" }")
        
    # ** \not xx shoudle be combined as one token
    pattern = r'\\not [\\=\<\>][^ ]+ '
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft+" ")
        
    # ** \specialrule{1pt}{2pt}{2pt}, special lines, shoudle be combined as one token
    pattern = r'\\specialrule {[ .0-9a-z]+} {[ .0-9a-z]+} {[ .0-9a-z]+}'
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)
        
    # ** for easier add color, the original color should be removed, there are two type of color for now: \color[rgb]{0, 1, 0} and \color{red}
    pattern = r'\\colorbox[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } |\\color[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } |\\textcolor[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } |\\cellcolor[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } '
    old_token = re.findall(pattern, l, re.DOTALL)
    for bef in old_token:
        l = l.replace(bef, "")
    
    # ** filling the missing brace [] and {} according to token.
    l_split = l.split(' ')
    idx = 0
    while idx < len(l_split):
        token = l_split[idx]
        if token in ONE_Tail_Tokens + ONE_Tail_Invisb_Tokens:
        # ** normalize tokens such as \hat, fill missing the {}, such as \hat \lambda -> \hat {\lambda}
            sub_idx = idx + 1
            while sub_idx < len(l_split) and l_split[sub_idx] in ONE_Tail_Tokens+ONE_Tail_Invisb_Tokens:
                sub_idx += 1
            new_split = l_split[0:idx]
            for ii in range(idx, sub_idx):
                new_split = new_split + [l_split[ii], "{"]
            if l_split[sub_idx] != "{":
                new_split = new_split + [l_split[sub_idx]] + ["}"]*(sub_idx-idx)
                l_split = new_split + l_split[sub_idx+1:]
            else:
                end_idx = find_matching_brace(l_split, sub_idx)
                new_split = new_split + l_split[sub_idx+1:end_idx] + ["}"]*(sub_idx-idx)
                l_split = new_split + l_split[end_idx+1:]
        elif token in AB_Tail_Tokens:
        # ** normalize special tokens such as \sqrt, fill the missing [] {} in \sqrt [] {}, yet the [] is optional, for example: \sqrt A B -> \sqrt {A} B and \sqrt [A] B -> \sqrt [A] {B}
            if l_split[idx + 1] != "[" and l_split[idx + 1] != "{":
                l_split = l_split[0:idx+1] + ["{"] + [l_split[idx+1]] + ["}"] + l_split[idx+2:]
            else:
                if l_split[idx + 1] == "[":
                    end1 = find_matching_brace(l_split, idx+1, brace=['[', ']'])
                else:
                    end1 = idx
                if l_split[end1 + 1] != "{":
                    l_split = l_split[0:end1+1] + ["{"] + [l_split[end1+1]] + ["}"] + l_split[end1+2:]
        elif token in TWO_Tail_Tokens + TWO_Tail_Invisb_Tokens:
        # ** normalize special tokens such as \frac, add missing brace in \frac {A} {B} for example: \frac {\lambda} 2 -> \frac {\lambda} {2}
            if l_split[idx + 1] != "{":
                l_split = l_split[0:idx+1] + ["{"] + [l_split[idx+1]] + ["}"] + l_split[idx+2:]
            end1 = find_matching_brace(l_split, idx+1)
            if l_split[end1 + 1] != "{":
                l_split = l_split[0:end1+1] + ["{"] + [l_split[end1+1]] + ["}"] + l_split[end1+2:]
            
        idx += 1
    l = ' '.join(l_split)
    
    return l

def token_add_color_RGB(l_split, idx, token_list, brace_color=False):
    """using \mathcolor[RGB]{r,g,b} to render latex. 
    """
    token = l_split[idx]
    if not token:
        next_idx = idx + 1
    elif token in PHANTOM_Tokens:
        # ** special tokens that do not need render, skip it 
        if l_split[idx + 1] == '{':
            brace_end = find_matching_brace(l_split, idx + 1)
        else:
            brace_end = idx + 1
        next_idx = brace_end + 1
    elif token in TWO_Tail_Tokens:
        # ** tokens such as \frac A B, and the token needs render too.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        den_start = num_end + 1
        den_end = find_matching_brace(l_split, den_start)
        color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
        l_split = l_split[:idx] + [color_token+token] + l_split[idx+1: den_end+1] + ["}"] + l_split[den_end+1:]
        token_list.append(token)
        next_idx = idx + 1
    elif token in ONE_Tail_Tokens:
        # ** tokens such as \hat A, and the token needs render too.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
        if token != "\\underbrace" and num_end+1 < len(l_split) and l_split[num_end+1] == "_":
            l_split = l_split[:idx] + ["{"+color_token+token] + l_split[idx+1: num_end+1] + ["}}"] + l_split[num_end+1:]
        else:
            l_split = l_split[:idx] + [color_token+token] + l_split[idx+1: num_end+1] + ["}"] + l_split[num_end+1:]
        token_list.append(token)
        next_idx = idx + 1
    elif token in ONE_Tail_Invisb_Tokens:
        # ** tokens such as \text A B, and the token does not need render.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        sub_idx = num_start+1
        if num_end-num_start == 2:
            color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            token_list.append(l_split[num_start+1])
            l_split = l_split[:num_start+1] + [color_token+l_split[num_start+1]+"}"] + l_split[num_end:]
        else:
            while sub_idx < num_end:
                l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list)
        next_idx = num_end + 1
    elif token in AB_Tail_Tokens:
        # ** special token \xrightarrow, could be \xrightarrow [] {} or \xrightarrow {}, process method are different.
        if l_split[idx+1] == '{':
            num_start = idx + 1
            num_end = find_matching_brace(l_split, num_start)
            color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + [color_token+token] + l_split[idx+1: num_end+1] + ["}"] + l_split[num_end+1:]
            token_list.append(token)
            sub_idx = num_start+1
            while sub_idx < num_end:
                l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list)
            next_idx = num_end + 1
        elif l_split[idx+1] == '[':
            num_start = idx + 1
            num_end = find_matching_brace(l_split, num_start, brace=['[', ']'])
            den_start = num_end + 1
            den_end = find_matching_brace(l_split, den_start)
            color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + [color_token+token] + l_split[idx+1: den_end+1] + ["}"] + l_split[den_end+1:]
            token_list.append(token)
            sub_idx = num_start + 1
            while sub_idx < num_end:
                l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list, brace_color=True)
            sub_idx = den_start + 1
            while sub_idx < den_end:
                l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list)
            next_idx = den_end + 1
    elif token in ["\\multicolumn", "\\multirow"]:
        # ** tokens with three {}, such as \multicolumn {} {} {}, the text in third {} need be rendered.
        first_start = idx + 1
        first_end = find_matching_brace(l_split, first_start)
        second_start = first_end + 1
        second_end = find_matching_brace(l_split, second_start)
        third_start = second_end + 1
        third_end = find_matching_brace(l_split, third_start)
        
        sub_idx = third_start+1
        while sub_idx < third_end:
            l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list)
        next_idx = third_end + 1
    elif token in SKIP_Tokens+TWO_Tail_Invisb_Tokens or any(re.match(pattern, token) for pattern in SKIP_PATTERNS):
        # ** tokens no need render, just skip
        # TODO special case :[], could be single, or in \sqrt[]{}.
        if (token == "[" and l_split[idx-1]!="\\sqrt") or (token == "]" and idx>=3 and l_split[idx-3]!="\\sqrt"):
            color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + [color_token + l_split[idx] + "}"] + l_split[idx+1:]
            token_list.append(token)
            next_idx = idx + 1
        else:
            next_idx = idx + 1
    else:
        # ** nomal token
        if brace_color or (idx > 1 and l_split[idx-1] == "_"):
            color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + ["{" + color_token + l_split[idx] + "}}"] + l_split[idx+1:]
            token_list.append(token)
            next_idx = idx + 1
        else:
            color_token = "\\mathcolor[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + [color_token + l_split[idx] + "}"] + l_split[idx+1:]
            token_list.append(token)
            next_idx = idx + 1
    return l_split, next_idx, token_list

def latex2bbox_color(input_arg):
    latex, basename, output_path, temp_dir, total_color_list = input_arg
    template = tabular_template if "tabular" in latex else formular_template
    basename = basename.replace('.jpg', '')
    output_bbox_path = os.path.join(output_path, 'bbox', basename+'.jsonl')
    output_vis_path = os.path.join(output_path, 'vis', basename+'.png')
    output_base_path = os.path.join(output_path, 'vis', basename+'_base.png')
    
    if os.path.exists(output_bbox_path) and os.path.exists(output_vis_path) and os.path.exists(output_base_path):
        return
    
    try:
        latex = latex.replace("\n", " ")
        latex = latex.replace("\%", "<PERCENTAGETOKEN>")
        ret, new_latex = tokenize_latex(latex, middle_file=os.path.join(temp_dir, basename+'.txt'))
        if not(ret and new_latex):
            log = f"ERROR, Tokenize latex failed: {basename}."
            logging.info(log)
            new_latex = latex
        new_latex = new_latex.replace("< P E R C E N T A G E T O K E N >", "\%")
        latex = normalize_latex(new_latex)
        token_list = []
        l_split = latex.strip().split(' ')
        color_list = total_color_list[0:len(l_split)]
        idx = 0
        while idx < len(l_split):
            l_split, idx, token_list = token_add_color_RGB(l_split, idx, token_list)

        rgb_latex = " ".join(l_split)
        for idx, color in enumerate(color_list):
            R, G, B = color
            rgb_latex = rgb_latex.replace(f"<color_{idx}>", f"{R},{G},{B}")

        if len(token_list) > 1300:
            paper_size = 3
        elif len(token_list) > 600:
            paper_size = 4
        else:
            paper_size = 5
        final_latex = formular_template.replace("<PaperSize>", str(paper_size)) % rgb_latex
        
    except Exception as e:
        log = f"ERROR, Preprocess latex failed: {basename}; {e}."
        logging.info(log)
        return
    
    pre_name = output_path.replace('/', '_').replace('.','_') + '_' + basename
    tex_filename = os.path.join(temp_dir, pre_name+'.tex')
    log_filename = os.path.join(temp_dir, pre_name+'.log')
    aux_filename = os.path.join(temp_dir, pre_name+'.aux')
    try:
        with open(tex_filename, "w") as w: 
            w.write(final_latex)
    except Exception as e:
        logging.info(f"write error: {e}")
    # run_cmd(f"pdflatex -interaction=nonstopmode -output-directory={temp_dir} {tex_filename} >/dev/null")
    run_cmd(f"xelatex -interaction=nonstopmode -output-directory={temp_dir} \"{tex_filename}\" >/dev/null", temp_dir=temp_dir)
    try:
        os.remove(tex_filename)
        os.remove(log_filename)
        os.remove(aux_filename)
    except Exception as e:
        logging.info(e)
        pass
    pdf_filename = tex_filename[:-4]+'.pdf'
    if not os.path.exists(pdf_filename):
        log = f"ERROR, Compile pdf failed: {pdf_filename}"
        logging.info(log)
    else:
        convert_pdf2img(pdf_filename, output_base_path)
        os.remove(pdf_filename)
        
        crop_image(output_base_path)
        bbox_list = extrac_bbox_from_color_image(output_base_path, color_list)
        vis = Image.open(output_base_path)
        draw = ImageDraw.Draw(vis)

        with open(output_bbox_path, 'w', encoding='utf-8') as f:
            for token, box in zip(token_list, bbox_list):
                item = {
                    "bbox": box,
                    "token": token
                }
                f.write(json.dumps(item, ensure_ascii=False)+'\n')

                if not box:
                    continue
                x_min, y_min, x_max, y_max = box
                draw.rectangle([x_min, y_min, x_max, y_max], fill=None, outline=(0,250,0), width=1)
                try:
                    draw.text((x_min, y_min), token, (250,0,0))
                except:
                    pass
            
        vis.save(output_vis_path)


class SimpleAffineTransform:
    """
    simple affine transform, only translation and scale.
    """
    def __init__(self, translation=(0, 0), scale=1.0):
        self.translation = np.array(translation)
        self.scale = scale

    def estimate(self, src, dst):
        src_center = np.mean(src, axis=0)
        dst_center = np.mean(dst, axis=0)
        self.translation = dst_center - src_center

        src_dists = np.linalg.norm(src - src_center, axis=1)
        dst_dists = np.linalg.norm(dst - dst_center, axis=1)
        self.scale = np.mean(dst_dists) / (np.mean(src_dists) + 1e-10)

    def inverse(self):
        inverse_transform = SimpleAffineTransform(-self.translation, 1.0/self.scale)
        return inverse_transform

    def __call__(self, coords):
        return self.scale * (coords - np.mean(coords, axis=0)) + np.mean(coords, axis=0) + self.translation

    def residuals(self, src, dst):
        return np.sqrt(np.sum((self(src) - dst) ** 2, axis=1))


class CDM:
    # Evaluation parameters
    max_iter = 3
    min_samples = 3
    residual_threshold = 25
    max_trials = 50
    max_colors = 5800

    def __init__(self, output_root="./result"):
        """
        Initialize the LaTeX formula evaluator.
        Character Detection Matching
        Args:
            output_root (str): Root directory for saving intermediate and final results
        """
        self.output_root = output_root
        self.matcher = HungarianMatcher()
        
    @staticmethod
    def gen_color_list(num=10, gap=15):
        """Generate a list of distinct colors for visualization"""
        num += 1
        single_num = 255 // gap + 1
        max_num = single_num ** 3
        num = min(num, max_num)
        color_list = []
        for idx in range(num):
            R = idx // single_num**2
            GB = idx % single_num**2
            G = GB // single_num
            B = GB % single_num
            color_list.append((R*gap, G*gap, B*gap))
        return color_list[1:]
    
    @staticmethod
    def update_inliers(ori_inliers, sub_inliers):
        """Update inliers status based on new RANSAC results"""
        inliers = np.copy(ori_inliers)
        sub_idx = -1
        for idx in range(len(ori_inliers)):
            if ori_inliers[idx] == False:
                sub_idx += 1
                if sub_inliers[sub_idx] == True:
                    inliers[idx] = True
        return inliers
    
    def _prepare_directories(self, img_id):
        """Create necessary directories for output"""
        os.makedirs(os.path.join(self.output_root, 'gt', 'bbox'), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, 'gt', 'vis'), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, 'pred', 'bbox'), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, 'pred', 'vis'), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, 'vis_match'), exist_ok=True)
    
    def _generate_bboxes(self, gt_latex, pred_latex, img_id):
        """Generate bounding boxes for both GT and prediction"""
        total_color_list = self.gen_color_list(num=max_colors)
        
        for subset, latex in zip(['gt', 'pred'], [gt_latex, pred_latex]):
            output_path = os.path.join(self.output_root, subset)
            temp_dir = os.path.join(self.output_root, f'temp_dir_{subset}_{img_id}')
            os.makedirs(temp_dir, exist_ok=True)
            latex2bbox_color((latex, img_id, output_path, temp_dir, total_color_list))
            shutil.rmtree(temp_dir)
    
    def _load_bboxes(self, img_id):
        """Load generated bounding boxes from files"""
        gt_box_path = os.path.join(self.output_root, 'gt', 'bbox', f"{img_id}.jsonl")
        pred_box_path = os.path.join(self.output_root, 'pred', 'bbox', f"{img_id}.jsonl")
        def load_from_json(path):     
            with open(path, 'r') as f:
                return [json.loads(line) for line in f if json.loads(line)['bbox']]
        box_pred = load_from_json(pred_box_path)
        box_gt = load_from_json(gt_box_path)
            
        return box_gt, box_pred
    
    def _load_images(self, img_id):
        """Load visualization images"""
        gt_img_path = os.path.join(self.output_root, 'gt', 'vis', f"{img_id}_base.png")
        pred_img_path = os.path.join(self.output_root, 'pred', 'vis', f"{img_id}_base.png")
        return Image.open(gt_img_path), Image.open(pred_img_path)
    
    def _match_boxes(self, box_gt, box_pred, img_gt, img_pred):
        """Perform box matching using Hungarian algorithm and RANSAC"""
        matched_idxes = self.matcher(box_gt, box_pred, img_gt.size, img_pred.size)
        
        # Prepare matching points
        src, dst = [], []
        for (idx1, idx2) in matched_idxes:
            x1min, y1min, x1max, y1max = box_gt[idx1]['bbox']
            x2min, y2min, x2max, y2max = box_pred[idx2]['bbox']
            src.append([float((y1min+y1max)/2), float((x1min+x1max)/2)])
            dst.append([float((y2min+y2max)/2), float((x2min+x2max)/2)])
        
        src = np.array(src)
        dst = np.array(dst)
        
        # Apply RANSAC filtering
        if src.shape[0] <= min_samples:
            inliers = np.array([True for _ in matched_idxes])
        else:
            inliers = np.array([False for _ in matched_idxes])
            for i in range(max_iter):
                if src[inliers==False].shape[0] <= min_samples:
                    break
                model, inliers_1 = ransac(
                    (src[inliers==False], dst[inliers==False]), 
                    SimpleAffineTransform, 
                    min_samples=min_samples, 
                    residual_threshold=residual_threshold, 
                    max_trials=max_trials, 
                    random_state=42
                )
                if inliers_1 is not None and inliers_1.any():
                    inliers = self.update_inliers(inliers, inliers_1)
                else:
                    break
                if len(inliers[inliers==True]) >= len(matched_idxes):
                    break
        
        # Filter token mismatches
        for idx, (a,b) in enumerate(matched_idxes):
            if inliers[idx] == True and self.matcher.cost['token'][a, b] == 1:
                inliers[idx] = False
                
        return matched_idxes, inliers
    
    def _calculate_metrics(self, box_gt, box_pred, inliers):
        """Calculate evaluation metrics"""
        final_match_num = len(inliers[inliers==True])
        recall = round(final_match_num/len(box_gt), 3)
        precision = round(final_match_num/len(box_pred), 3)
        F1_score = round(2*final_match_num/(len(box_gt)+len(box_pred)), 3)
        return recall, precision, F1_score
    
    def _visualize_matches(self, img_gt, img_pred, box_gt, box_pred, matched_idxes, inliers, img_id):
        """Generate and save visualization of matches"""
        gap = 5
        W1, H1 = img_gt.size
        W2, H2 = img_pred.size
        H = H1 + H2 + gap
        W = max(W1, W2)

        # Create base visualization
        vis_img = Image.new('RGB', (W, H), (255, 255, 255))
        vis_img.paste(img_gt, (0, 0))
        vis_img.paste(Image.new('RGB', (W, gap), (120, 120, 120)), (0, H1))
        vis_img.paste(img_pred, (0, H1+gap))
        
        # Create match visualization
        match_img = vis_img.copy()
        match_draw = ImageDraw.Draw(match_img)

        gt_matched_idx = {a: flag for (a,b), flag in zip(matched_idxes, inliers)}
        pred_matched_idx = {b: flag for (a,b), flag in zip(matched_idxes, inliers)}
        
        # Draw GT boxes
        for idx, box in enumerate(box_gt):
            color = "green" if idx in gt_matched_idx and gt_matched_idx[idx]==True else "red"
            x_min, y_min, x_max, y_max = box['bbox']
            match_draw.rectangle([x_min-1, y_min-1, x_max+1, y_max+1], fill=None, outline=color, width=2)
        
        # Draw prediction boxes
        for idx, box in enumerate(box_pred):
            color = "green" if idx in pred_matched_idx and pred_matched_idx[idx]==True else "red"
            x_min, y_min, x_max, y_max = box['bbox']
            match_draw.rectangle([x_min-1, y_min-1+H1+gap, x_max+1, y_max+1+H1+gap], fill=None, outline=color, width=2)
        
        # Save visualizations
        vis_img.save(os.path.join(self.output_root, 'vis_match', f"{img_id}_base.png"))
        match_img.save(os.path.join(self.output_root, 'vis_match', f"{img_id}.png"))
    
    def evaluate(self, gt_latex, pred_latex, img_id):
        """
        Evaluate a single LaTeX formula pair (ground truth vs prediction)
        
        Args:
            gt_latex (str): Ground truth LaTeX formula
            pred_latex (str): Predicted LaTeX formula
            img_id (str): Unique identifier for this evaluation
            
        Returns:
            dict: Evaluation metrics (recall, precision, F1_score)
        """
        
        try:
            self._prepare_directories(img_id)
            self._generate_bboxes(gt_latex, pred_latex, img_id)
            box_gt, box_pred = self._load_bboxes(img_id)
            img_gt, img_pred = self._load_images(img_id)
            matched_idxes, inliers = self._match_boxes(box_gt, box_pred, img_gt, img_pred)
        except:
            return {"recall": 0, "precision": 0, "F1_score": 0}

        recall, precision, F1_score = self._calculate_metrics(box_gt, box_pred, inliers)
        self._visualize_matches(img_gt, img_pred, box_gt, box_pred, matched_idxes, inliers, img_id)
        
        return {
            "recall": recall,
            "precision": precision,
            "F1_score": F1_score,
        }

def _process_single_cdm_sample(args):
    """Worker function to process a single CDM sample"""
    idx, sample, output_root, group_info = args
    
    # Create a new CDM instance for this worker to avoid thread safety issues
    cal_cdm = CDM(output_root=output_root)
    
    # Prepare sample data
    sample_copy = copy.deepcopy(sample)
    sample_copy['img_id_cdm'] = str(idx)
    sample_copy['gt'] = sample_copy['gt'].lstrip("$$").rstrip("$$").strip()
    sample_copy['gt'] = sample_copy['gt'].lstrip("$").rstrip("$").strip()
    sample_copy['pred'] = sample_copy['pred'].split("```latex")[-1].split("```")[0]
    sample_copy['pred'] = sample_copy['pred'].lstrip("$$").rstrip("$$").strip()
    sample_copy['pred'] = sample_copy['pred'].lstrip("$").rstrip("$").strip()
    
    # Calculate CDM score
    cdm_score = cal_cdm.evaluate(sample_copy['gt'], sample_copy['pred'], sample_copy['img_id_cdm'])["F1_score"]
    
    # Add metric to sample
    if not sample_copy.get('metric'):
        sample_copy['metric'] = {}
    sample_copy['metric']['CDM'] = cdm_score
    
    # Check which groups this sample belongs to
    matched_groups = []
    for group in group_info:
        select_flag = True
        for k, v in group.items():
            for gt_attribute in sample_copy['gt_attribute']:
                if not gt_attribute:
                    select_flag = False
                elif gt_attribute[k] != v:
                    select_flag = False
        if select_flag:
            matched_groups.append(str(group))
    
    return {
        'sample': sample_copy,
        'cdm_score': cdm_score,
        'sample_key': sample_copy['img_id'] + '_' + str(sample_copy['gt_idx']),
        'matched_groups': matched_groups,
        'original_index': idx
    }


@METRIC_REGISTRY.register("CDM")
class call_CDM():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default', out_dir='./', max_workers=32):
        max_workers = min(max_workers, (os.cpu_count() or 4))
        group_scores = defaultdict(list)
        output_root = f"result/{save_name}/CDM"
        
        if isinstance(self.samples, list):
            original_samples = self.samples
        else:
            original_samples = self.samples.samples
        
        # Prepare arguments for concurrent processing
        worker_args = []
        for idx, sample in enumerate(original_samples):
            worker_args.append((idx, sample, output_root, group_info))
        
        # Use concurrent execution
        per_sample_score = {}
        cdm_samples = []
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(_process_single_cdm_sample, args): args[0] for args in worker_args}
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    idx = future_to_idx[future]
                    print(f'Sample {idx} generated an exception: {exc}')
                    # Create a default result for failed samples
                    sample_copy = copy.deepcopy(original_samples[idx])
                    sample_copy['img_id_cdm'] = str(idx)
                    if not sample_copy.get('metric'):
                        sample_copy['metric'] = {}
                    sample_copy['metric']['CDM'] = 0.0
                    results.append({
                        'sample': sample_copy,
                        'cdm_score': 0.0,
                        'sample_key': sample_copy['img_id'] + '_' + str(sample_copy['gt_idx']),
                        'matched_groups': [],
                        'original_index': idx
                    })
        
        # Sort results by original index to maintain order
        results.sort(key=lambda x: x['original_index'])
        
        # Process results
        for result in results:
            sample = result['sample']
            cdm_score = result['cdm_score']
            sample_key = result['sample_key']
            matched_groups = result['matched_groups']
            
            cdm_samples.append(sample)
            per_sample_score[sample_key] = cdm_score
            group_scores['all'].append(cdm_score)
            
            # Add scores to matched groups
            for group_name in matched_groups:
                group_scores[group_name].append(cdm_score)

        # Save results to files
        with open(f'./result/{save_name}_per_sample_CDM.json', 'w', encoding='utf-8') as f:
            json.dump(per_sample_score, f, indent=4, ensure_ascii=False)

        with open(f'result/{save_name}_result.json', 'w', encoding='utf-8') as f:
            json.dump(cdm_samples, f, indent=4, ensure_ascii=False)

        # Calculate final results
        result = {}
        for group_name, scores in group_scores.items():
            if len(scores) > 0:
                result[group_name] = sum(scores) / len(scores)    # average of normalized scores at sample level
            else:
                result[group_name] = 'NaN'
                logging.info(f'Warning: Empty matched samples for {group_name}.')
        
        return cdm_samples, {'CDM': result}


@METRIC_REGISTRY.register("CDM_plain")
class call_CDM_plain():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default', out_dir='./'):
        if isinstance(self.samples, list):
            cdm_samples = copy.deepcopy(self.samples)
        else:
            cdm_samples = copy.deepcopy(self.samples.samples)
        for idx, sample in enumerate(cdm_samples):
            sample['img_name'] = sample['img_id']
            sample['img_id'] = str(idx)
            sample['gt'] = sample['gt'].lstrip("$$").rstrip("$$").strip()
            sample['pred'] = sample['pred'].split("```latex")[-1].split("```")[0]
            sample['pred'] = sample['pred'].lstrip("$$").rstrip("$$").strip()

        with open(f'result/{save_name}_formula.json', 'w', encoding='utf-8') as f:
            json.dump(cdm_samples, f, indent=4, ensure_ascii=False)
        return self.samples, False