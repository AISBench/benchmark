# Copyright (c) OpenDataLab https://github.com/opendatalab/OmniDocBench (2025/11/07)
# SPDX-License-Identifier: Apache-2.0
# Part of this document is directly reused from the above warehouse without modification.

import os
from collections import defaultdict
import json
from tqdm import tqdm
import time
import logging
import sys
import shutil
from func_timeout import FunctionTimedOut, func_timeout
import traceback
import Levenshtein

from ais_bench.benchmark.datasets.omnidocbench.utils import (match_gt2pred_simple,
                                                            match_gt2pred_quick,
                                                            match_gt2pred_no_split,
                                                            md_tex_filter,
                                                            clean_string,
                                                            normalized_table)

CATEGORY_LIST = ['text_block', 'title', 'code_txt', 'code_txt_caption', 'reference', 'equation_caption',
                'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 
                'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_isolated']
TABLE_LIST = ['html_table','latex_table','md2html_table']
GT_CATEGORY_LIST = ['text_block', 'title', 'code_txt', 'code_txt_caption', 'reference', 'equation_caption',
            'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption',
            'header', 'footer', 'page_footnote', 'page_number']

def read_md_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return content 

class End2EndDataset():
    def __init__(self, predictions, gt_samples):
        self.match_method ='quick_match'
        filtered_types = None
        filtered_gt_samples = []
        if filtered_types:
            for gt_sample in gt_samples:
                select_flag = True
                for k, v in filtered_types.items():
                    if gt_sample["page_info"]["page_attribute"][k] != v:
                        select_flag = False
                if select_flag:
                    filtered_gt_samples.append(gt_sample)
        else:
            filtered_gt_samples = gt_samples

        self.samples = self.get_matched_elements(filtered_gt_samples, predictions)
    
    def __getitem__(self, cat_name, idx):
        return self.samples[cat_name][idx]
    
    # Match the results of GT and prediction, 
    # call the process_get_matched_elements function for matching processing, 
    # and finally organize the matching results into a dictionary to return
    def get_matched_elements(self, gt_samples, predictions):
        plain_text_match = []
        display_formula_match = []
        html_table_match = []
        latex_table_match = []
        order_match = []
        save_time = time.time()
        process_bar = tqdm(range(len(gt_samples)), ascii=True, ncols=140)
        for i in process_bar:
            sample = gt_samples[i]
            pred_content = predictions[i]
            img_name = os.path.basename(sample["page_info"]["image_path"])
            
            # For matching a single sample, based on different element types (such as text blocks, displayed formulas, tables, etc.), 
            # the specified matching method is used to match the ground truth (gt) with the prediction results, and the matching result is returned.
            result = self.process_get_matched_elements(sample, pred_content, img_name, save_time) # Don't use timeout logic

            [plain_text_match_clean, formated_display_formula, latex_table_match_s, html_table_match_s, order_match_single] = result
            if order_match_single:
                order_match.append(order_match_single)
            if plain_text_match_clean:
                plain_text_match.extend(plain_text_match_clean)
            if formated_display_formula:
                display_formula_match.extend(formated_display_formula)
            if latex_table_match_s:
                latex_table_match.extend(latex_table_match_s)
            if html_table_match_s:
                html_table_match.extend(html_table_match_s)

        display_formula_match_clean,display_formula_match_others = [],[]
        for item in display_formula_match:
            pred_category_type = item.get("pred_category_type",None)
            if pred_category_type not in ['equation_inline','equation_isolated', '']:
                from pylatexenc.latex2text import LatexNodes2Text
                gt = item.get('gt',None)
                norm_gt = item.get('norm_gt',None)
                ## latex2unicode
                item['gt'] = LatexNodes2Text().latex_to_text(gt)
                item['norm_gt'] = clean_string(item['gt'])
                display_formula_match_others.append(item)
            else:
                display_formula_match_clean.append(item)
        display_formula_match = display_formula_match_clean
        if display_formula_match_others and plain_text_match:
            plain_text_match.extend(display_formula_match_others)
            
        #  Merge LaTeX into HTML, total 428.
        if latex_table_match:
            latex_to_html = []
            for latex_table in latex_table_match:
                for k,v in latex_table.items():
                    if 'pred' in k:
                        latex_table[k] = ""
                latex_table['edit'] = 1
                latex_to_html.append(latex_table)
            html_table_match.extend(latex_to_html)      
        
        if len(latex_table_match) > len(html_table_match): # Assume model won't randomly output both latex and html, but will choose one
            table_match = latex_table_match
            table_format = 'latex'
        else:
            table_match = html_table_match
            table_format = 'html'
        
        matched_samples_all = {
            'text_block': RecognitionEnd2EndBaseDataset(plain_text_match),
            'display_formula':  RecognitionEnd2EndBaseDataset(display_formula_match),
            'table': RecognitionEnd2EndTableDataset(table_match, table_format),
            'reading_order': RecognitionEnd2EndBaseDataset(order_match)
        }
      
        return matched_samples_all
    
    # Extract the table from gt and match it with the table from pred.
    # For the unmatched pred_table, remove the HTML format and then mix and match it.
    def process_get_matched_elements(self, sample, pred_content, img_name, save_time):
        if self.match_method == 'simple_match':   # add match choice
            match_gt2pred = match_gt2pred_simple
        elif self.match_method == 'quick_match':
            match_gt2pred = match_gt2pred_quick
        elif self.match_method == 'no_split':
            match_gt2pred = match_gt2pred_no_split
        else:
            logging.info('Invalid match method name. The quick_match will be used.')
            match_gt2pred = match_gt2pred_quick

        pred_dataset = md_tex_filter(pred_content)
        gt_page_elements = self.get_page_elements(sample)

        gt_mix,pred_dataset_mix = [],[]
        for category in pred_dataset:
            if category not in CATEGORY_LIST:
                pred_dataset_mix.extend(pred_dataset[category])
        gt_mix = self.get_page_elements_list(gt_page_elements, CATEGORY_LIST)
        if gt_mix:
            gt_mix = self.get_sorted_text_list(gt_mix)

        display_formula_match_s = []
        plain_text_match_clean = []
        latex_table_match_s = []
        html_table_match_s = []
        order_match_single = []

        if gt_page_elements.get('table'):
            gt_table = self.get_sorted_text_list(gt_page_elements['table'])
            latex_table_len = len(pred_dataset['latex_table']) if pred_dataset['latex_table'] else 0
            html_table_len = len(pred_dataset['html_table']) if pred_dataset['html_table'] else 0
            if latex_table_len == html_table_len and latex_table_len == 0:
                html_table_match_s,unmatch_table_pred = match_gt2pred_simple(gt_table, [], 'html_table', img_name) # Don't consider truncated merging for tables
                html_table_match_s = [x for x in html_table_match_s if x['gt_idx'] != [""]]  # Remove extra preds
            elif latex_table_len > html_table_len:
                latex_table_match_s,unmatch_table_pred = match_gt2pred_simple(gt_table, pred_dataset['latex_table'], 'latex_table', img_name) # Don't consider truncated merging for tables
                latex_table_match_s = [x for x in latex_table_match_s if x['gt_idx'] != [""]]  # Remove extra preds                
            else:
                html_table_match_s,unmatch_table_pred = match_gt2pred_simple(gt_table, pred_dataset['html_table'], 'html_table', img_name) # Don't consider truncated merging for tables
                html_table_match_s = [x for x in html_table_match_s if x['gt_idx'] != [""]]  # Remove extra preds

            if unmatch_table_pred:
                pred_dataset_mix.extend(unmatch_table_pred)

        try:
            time_outs = 30
            match = func_timeout(time_outs, match_gt2pred, args=(gt_mix, pred_dataset_mix, 'text_all', img_name))
        except FunctionTimedOut as e1:
            match,_ = match_gt2pred_simple(gt_mix, pred_dataset_mix, 'text_all', img_name)
        except Exception as e:
            logging.error(f"{traceback.format_exc()},error info:{e}")
            sys.exit()
        
        plain_text_match_s = []
        for item in match:
            gt_category = item.get('gt_category_type',None)
            if gt_category in GT_CATEGORY_LIST:
                plain_text_match_s.append(item)
            elif gt_category == 'equation_isolated':
                display_formula_match_s.append(item)

        display_formula_match_s = [x for x in display_formula_match_s if x['gt_idx'] != [""]]

        if not plain_text_match_s:
            pass
        else:
            # Categories that need to be ignored for text
            plain_text_match_clean = self.filtered_out_ignore(plain_text_match_s, ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm',
                                                                                    'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption'])
        order_match_s = plain_text_match_clean
        if order_match_s:
            order_match_single = self.get_order_paired(order_match_s, img_name)


        return [plain_text_match_clean, display_formula_match_s, latex_table_match_s, html_table_match_s, order_match_single]    
    
    # Match elements to handle text truncation issues, merge the truncated text blocks, and store the elements by category in a dictionary
    def get_page_elements(self, selected_annos):
        
        saved_element_dict = defaultdict(list) # save elements
        related_truncated = []
        truncated_all = {}
        for relation in selected_annos["extra"]["relation"]:   # Handle truncated text issues
            if relation["relation_type"] == 'truncated':
                truncated_all[relation["source_anno_id"]] = ""
                truncated_all[relation["target_anno_id"]] = ""
                exist_flag = False
                for merge_list in related_truncated:
                    if relation["source_anno_id"] in merge_list or relation["target_anno_id"] in merge_list:  # Consider cases where three text blocks may need to be merged
                        merge_list.append(relation["source_anno_id"])
                        merge_list.append(relation["target_anno_id"])
                        exist_flag = True
                if not exist_flag:
                    related_truncated.append([relation["source_anno_id"], relation["target_anno_id"]])       
        
        for item in selected_annos['layout_dets']:
            if item['anno_id'] not in truncated_all.keys():
                saved_element_dict[item["category_type"]].append(item)
            else:
                truncated_all[item['anno_id']] = item
        
        for merge_list in related_truncated:
            text_block_list = [truncated_all[key] for key in merge_list]
            sorted_block = sorted(text_block_list, key=lambda x: x['order'])
            text = ""
            for block in sorted_block:
                text += block['text']
            merged_block = {
                "category_type": sorted_block[0]["category_type"], # Directly use information from the first block
                "order": sorted_block[0]["order"],
                "anno_id": sorted_block[0]["anno_id"],   
                "text": text,
                "merge_list": sorted_block
            }
            saved_element_dict[sorted_block[0]["category_type"]].append(merged_block)

        return saved_element_dict
    
    # Extract elements from gt_page_elements based on the category list category_list and merge them into a single list
    def get_page_elements_list(self, gt_page_elements, category_list):
        element_list = []
        for category_type in category_list:
            if gt_page_elements.get(category_type):
                element_list.extend(gt_page_elements[category_type])
        return element_list

    # Sort the list of elements based on the `order` field and return the sorted list.
    def get_sorted_text_list(self, selected_annos):
        # txt_type: text, latex, html
        text_list = []
        for item in selected_annos:
            if item.get('order'):
                order = item['order']
            else:
                order = 0
            text_list.append((order, item))
        sorted_text_list = sorted(text_list, key=lambda x: x[0])
        return [_[1] for _ in sorted_text_list]
    
    # Filter out the elements of gt_category_type in ignore_category_stist from the element list items.
    def filtered_out_ignore(self, items, ignore_category_list):
        filted_items = []
        for item in items:
            if item['gt_category_type'] not in ignore_category_list:
                filted_items.append(item)
        return filted_items
    
    # Calculate the edit distance between the predicted results and the reading order of ground truth values,
    # and return a dictionary containing relevant information.
    def get_order_paired(self, order_match_s, img_name):
        matched = [(item['gt_position'], item['pred_position']) for item in order_match_s if (item['gt_position'] != [""] and item['pred_position'] != "")]
        gt_idx_all = [item['gt_position'] for item in order_match_s if (item['gt_position'] != [""])]
        read_order_pred = [i[0] for i in sorted(matched, key=lambda x: x[1])]  # Sort by pred idx to get Pred ordered GT_idx
        read_order_gt = sum(gt_idx_all, []) # Convert to one-dimensional list
        read_order_gt = [x for x in read_order_gt if x]  # For truncated merges, some discarded classes may be merged in, remove them when calculating edit distance
        gt = sorted(read_order_gt) # Sort by all GT idx to get GT ordered GT_idx
        pred = sum(read_order_pred, [])
        pred = [x for x in pred if x]
        if len(pred) > 0 or len(gt) > 0:
            edit = Levenshtein.distance(gt, pred)/ max(len(pred), len(gt))
            return {
                'gt': gt,  
                'pred': pred,
                'img_id': img_name,
                'edit': edit
            }
        else:
            return {}  # If both GT and pred are empty for the page, return empty


class RecognitionEnd2EndBaseDataset():
    def __init__(self, samples):
        img_id = 0
        for sample in samples:
            if not sample.get('img_id'):
                sample['img_id'] = img_id
            img_id += 1
        self.samples = samples
    def __getitem__(self, idx):
        return self.samples[idx]


class RecognitionTableDataset():
    def __init__(self, cfg_task):
        gt_file = cfg_task['dataset']['ground_truth']['data_path']
        pred_file = cfg_task['dataset']['prediction']['data_path']
        self.pred_table_format = cfg_task['dataset']['prediction'].get('table_format', 'html')

        references, predictions = self.load_data(gt_file), self.load_data(pred_file)
        self.samples = self.normalize_data(references, predictions)

    def normalize_data(self, references, predictions):
        if self.pred_table_format == 'latex2html':
            os.makedirs('./temp', exist_ok=True)

        samples = []
        ref_keys = list(references.keys())

        for img in tqdm(ref_keys, total=len(ref_keys), ncols=140, ascii=True, desc='Normalizing data'):
            if self.pred_table_format == 'html':
                r = references[img]['html']
                p = predictions[img]['html']
            elif self.pred_table_format == 'latex':
                r = references[img]['latex']
                p = predictions[img]['latex']
            else:
                raise ValueError(f'Invalid table format: {self.pred_table_format}')

            img_id = references[img]["page_image_name"]
            p = normalized_table(p, self.pred_table_format)
            r = normalized_table(r, self.pred_table_format)
            samples.append({
                'gt': p,
                'pred': r,
                'img_id': img_id,
                'gt_attribute': [references[img]['attribute']],
            })
        
        if self.pred_table_format == 'latex2html':
            shutil.rmtree('./temp')
        return samples

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def load_data(self, data_path):
        result_dict = {}
        with open(data_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            result_dict[sample["image_path"]] = sample
        
        return result_dict


class RecognitionEnd2EndTableDataset(RecognitionTableDataset):
    def __init__(self, samples, table_format):
        self.pred_table_format = table_format
        self.samples = self.normalize_data(samples)

    def normalize_data(self, samples):
        img_id = 0

        for sample in samples:
            p = sample['pred']
            r = sample['gt']
            p = normalized_table(p, self.pred_table_format)
            r = normalized_table(r)
            sample['norm_gt'] = r
            sample['norm_pred'] = p
            sample['img_id'] = sample['img_id'] if sample.get('img_id') else img_id
            img_id += 1

        return samples