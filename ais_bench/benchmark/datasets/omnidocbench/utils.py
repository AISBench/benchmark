# Copyright (c) OpenDataLab https://github.com/opendatalab/OmniDocBench (2025/11/07)
# SPDX-License-Identifier: Apache-2.0
# Part of this document is directly reused from the above warehouse without modification.

import re
import os
import sys
import shutil
from copy import deepcopy
import copy
from typing import List, Dict, Any
from collections import defaultdict, Counter
import html
import uuid
import unicodedata
import subprocess
import pandas as pd
from scipy.optimize import linear_sum_assignment
import Levenshtein
import numpy as np
from bs4 import BeautifulSoup
from Levenshtein import distance as Levenshtein_distance


inline_reg = re.compile(
    r'\$(.*?)\$|'
    r'\\\((.*?)\\\)',
)
ARRAY_RE = re.compile(
    r'\\begin\{array\}\{(?P<spec>[^}]*)\}(?P<body>.*?)\\end\{array\}',
    re.S
)

# img
img_pattern = r'!\[.*?\]\(.*?\)'

# table 
table_reg = re.compile(
    r'\\begin{table\*?}(.*?)\\end{table\*?}|'
    r'\\begin{tabular\*?}(.*?)\\end{tabular\*?}',
    re.DOTALL 
)
md_table_reg = re.compile(
    r'\|\s*.*?\s*\|\n', 
    re.DOTALL)
html_table_reg = re.compile(
    r'(<table.*?</table>)',
    re.DOTALL
)

# title
title_reg = re.compile(
    r'^\s*#.*$', 
    re.MULTILINE)

# code block
code_block_reg = re.compile(
    r'```(\w+)\n(.*?)```',
    re.DOTALL
)

display_reg = re.compile(
    # r'\\begin{equation\*?}(.*?)\\end{equation\*?}|'
    # r'\\begin{align\*?}(.*?)\\end{align\*?}|'
    # r'\\begin{gather\*?}(.*?)\\end{gather\*?}|'
    # r'\\begin{array\*?}(.*?)\\end{array\*?}|'
    r'\$\$(.*?)\$\$|'
    r'\\\[(.*?)\\\]|'
    r'\$(.*?)\$|'
    r'\\\((.*?)\\\)',  
    re.DOTALL
)


def remove_markdown_fences(content):
    content = re.sub(r'^```markdown\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'^```html\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'^```latex\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'```\n?$', '', content, flags=re.MULTILINE)
    return content

# Standardize all consecutive characters
def replace_repeated_chars(input_str):
    input_str = re.sub(r'_{4,}', '____', input_str) # Replace more than 4 consecutive underscores with 4 underscores
    input_str = re.sub(r' {4,}', '    ', input_str)   # Replace more than 4 consecutive spaces with 4 spaces
    return input_str
    # return re.sub(r'([^a-zA-Z0-9])\1{10,}', r'\1\1\1\1', input_str) # For other consecutive symbols (except numbers and letters), replace more than 10 occurrences with 4

def extract_tabular(text):
    begin_pattern = r'\\begin{tabular}'
    end_pattern = r'\\end{tabular}'

    tabulars = []
    positions = []
    current_pos = 0
    stack = []
    
    while current_pos < len(text):
        begin_match = re.search(begin_pattern, text[current_pos:])
        end_match = re.search(end_pattern, text[current_pos:])
        
        if not begin_match and not end_match:
            break
            
        if begin_match and (not end_match or begin_match.start() < end_match.start()):
            stack.append(current_pos + begin_match.start())
            current_pos += begin_match.start() + len(end_pattern)
        elif end_match:
            if stack:
                start_pos = stack.pop()
                if not stack:
                    end_pos = current_pos + end_match.start() + len(end_pattern)
                    tabular_code = text[start_pos:end_pos]
                    tabulars.append(tabular_code)
                    positions.append((start_pos, end_pos))
            current_pos += end_match.start() + len(end_pattern)
        else:
            current_pos += 1
    
    if stack:
        new_start = stack[0] + len(begin_pattern)
        new_tabulars, new_positions = extract_tabular(text[new_start:])
        new_positions = [(start + new_start, end + new_start) for start, end in new_positions]
        tabulars.extend(new_tabulars)
        positions.extend(new_positions)

    return tabulars, positions

def extract_tex_table(content):
    tables = []
    tables_positions = []

    pattern = r'\\begin{table}(.*?)\\end{table}'
    for match in re.finditer(pattern, content, re.DOTALL):
        start_pos = match.start()
        end_pos = match.end()
        table_content = match.group(0)
        tables.append(table_content)
        tables_positions.append((start_pos, end_pos))
        content = content[:start_pos] + ' '*(end_pos-start_pos) + content[end_pos:]

    tabulars, tabular_positions = extract_tabular(content)
    all_tables = tables + tabulars
    all_positions = tables_positions + tabular_positions

    all_result = sorted([[pos, table]for pos, table in zip(all_positions, all_tables)], key=lambda x: x[0][0])
    all_tables = [x[1] for x in all_result]
    all_positions = [x[0] for x in all_result]

    return all_tables, all_positions

def extract_html_table(text):
    begin_pattern = r'<table(?:[^>]*)>'
    end_pattern = r'</table>'

    tabulars = []
    positions = []
    current_pos = 0
    stack = []
    
    while current_pos < len(text):
        begin_match = re.search(begin_pattern, text[current_pos:])
        end_match = re.search(end_pattern, text[current_pos:])
        
        if not begin_match and not end_match:
            break
            
        if begin_match and (not end_match or begin_match.start() < end_match.start()):
            stack.append(current_pos + begin_match.start())
            current_pos += begin_match.start() + len(end_pattern)
        elif end_match:
            if stack:
                start_pos = stack.pop()
                if not stack:
                    end_pos = current_pos + end_match.start() + len(end_pattern)
                    tabular_code = text[start_pos:end_pos]
                    tabulars.append(tabular_code)
                    positions.append((start_pos, end_pos))
            current_pos += end_match.start() + len(end_pattern)
        else:
            current_pos += 1
    
    if stack:
        new_start = stack[0] + len(begin_pattern)
        new_tabulars, new_positions = extract_html_table(text[new_start:])
        new_positions = [(start + new_start, end + new_start) for start, end in new_positions]
        tabulars.extend(new_tabulars)
        positions.extend(new_positions)

    return tabulars, positions

# The function is from https://github.com/intsig-textin/markdown_tester
def markdown_to_html(markdown_table):
    rows = [row.strip() for row in markdown_table.strip().split('\n')]
    num_columns = len(rows[0].split('|')) - 2

    html_table = '<table>\n  <thead>\n    <tr>\n'

    header_cells = [cell.strip() for cell in rows[0].split('|')[1:-1]]
    for cell in header_cells:
        html_table += f'      <th>{cell}</th>\n'
    html_table += '    </tr>\n  </thead>\n  <tbody>\n'

    for row in rows[2:]:
        cells = [cell.strip() for cell in row.split('|')[1:-1]]
        html_table += '    <tr>\n'
        for cell in cells:
            html_table += f'      <td>{cell}</td>\n'
        html_table += '    </tr>\n'

    html_table += '  </tbody>\n</table>\n'
    return html_table

def convert_table(input_str):
    # Replace <table>
    output_str = input_str.replace("<table>", "<table border=\"1\" >")

    # Replace <td>
    output_str = output_str.replace("<td>", "<td colspan=\"1\" rowspan=\"1\">")

    return output_str

def find_md_table_mode(line):
    if re.search(r'-*?:',line) or re.search(r'---',line) or re.search(r':-*?',line):
        return True
    return False

def delete_table_and_body(input_list):
    res = []
    for line in input_list:
        if not re.search(r'</?t(able|head|body)>',line):
            res.append(line)
    return res

def merge_tables(input_str):
    # Delete HTML comments
    input_str = re.sub(r'<!--[\s\S]*?-->', '', input_str)

    # Use regex to find each <table> block
    table_blocks = re.findall(r'<table>[\s\S]*?</table>', input_str)

    # Process each <table> block, replace <th> with <td>
    output_lines = []
    for block in table_blocks:
        block_lines = block.split('\n')
        for i, line in enumerate(block_lines):
            if '<th>' in line:
                block_lines[i] = line.replace('<th>', '<td>').replace('</th>', '</td>')
        final_tr = delete_table_and_body(block_lines)
        if len(final_tr) > 2:
            output_lines.extend(final_tr)  # Ignore <table> and </table> tags, keep only table content

    # Rejoin the processed strings
    merged_output = '<table>\n{}\n</table>'.format('\n'.join(output_lines))

    return "\n\n" + merged_output + "\n\n"

def replace_table_with_placeholder(input_string):
    lines = input_string.split('\n')
    output_lines = []

    in_table_block = False
    temp_block = ""
    last_line = ""

    for idx, line in enumerate(lines):
        if "<table>" in line:
            in_table_block = True
            temp_block += last_line
        elif in_table_block:
            if not find_md_table_mode(last_line) and "</thead>" not in last_line:
                temp_block += "\n" + last_line
            if "</table>" in last_line:
                if "<table>" not in line:
                    in_table_block = False
                    output_lines.append(merge_tables(temp_block))
                    temp_block = ""
        else:
            output_lines.append(last_line)

        last_line = line
    if last_line:
        if in_table_block or "</table>" in last_line:
            temp_block += "\n" + last_line
            output_lines.append(merge_tables(temp_block))
        else:
            output_lines.append(last_line)

    return '\n'.join(output_lines)


def convert_markdown_to_html(markdown_content):
    # Define a regex pattern to find Markdown tables with newlines
    markdown_content = markdown_content.replace('\r', '')+'\n'
    pattern = re.compile(r'\|\s*.*?\s*\|\n', re.DOTALL)

    # Find all matches in the Markdown content
    matches = pattern.findall(markdown_content)

    for match in matches:
        html_table = markdown_to_html(match)
        markdown_content = markdown_content.replace(match, html_table, 1)  # Only replace the first occurrence

    res_html = convert_table(replace_table_with_placeholder(markdown_content))

    return res_html

def md_tex_filter(content):
    '''
    Input: 1 page md or tex content - String
    Output: text, display, inline, table, title, code - list
    '''
    content = re.sub(img_pattern, '', content)  # remove image
    content = remove_markdown_fences(content)   # remove markdown fences
    content = replace_repeated_chars(content) # replace all consecutive characters
    content = content.replace('<html>', '').replace('</html>', '').replace('<body>', '').replace('</body>', '')
    pred_all = []
    latex_table_array, table_positions = extract_tex_table(content)
    for latex_table, position in zip(latex_table_array, table_positions):
        position = [position[0], position[0]+len(latex_table)]
        pred_all.append({
            'category_type': 'latex_table',
            'position': position,
            'content': latex_table
        })
        content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace latex table with space

    # extract html table  
    html_table_array, table_positions = extract_html_table(content)
    for html_table, position in zip(html_table_array, table_positions):
        position = [position[0], position[0]+len(html_table)]
        pred_all.append({
            'category_type': 'html_table',
            'position': position,
            'content': html_table
        })
        content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace html table with space

    # extract interline formula
    display_matches = display_reg.finditer(content)
    content_copy = content
    for match in display_matches:
        matched = match.group(0)
        if matched:
            # single_line = ''.join(matched.split())
            single_line = ' '.join(matched.strip().split('\n'))
            position = [match.start(), match.end()]
            # replace $$ with \[\]
            dollar_pattern = re.compile(r'\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)', re.DOTALL)
            sub_match = dollar_pattern.search(single_line)
            if sub_match is None:
                # pass
                content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]
                pred_all.append({
                    'category_type': 'equation_isolated',
                    'position': position,
                    'content': single_line
                })
            elif sub_match.group(1):
                single_line = re.sub(dollar_pattern, r'\\[\1\\]', single_line)
                content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace equation with space
                pred_all.append({
                    'category_type': 'equation_isolated',
                    'position': position,
                    'content': single_line
                })
            else:
                single_line = re.sub(dollar_pattern, r'\\[\2\3\\]', single_line)
                pred_all.append({
                    'category_type': 'equation_isolated',
                    'position': position,
                    'content': single_line,
                    'fine_category_type': 'equation_inline'
                })

    md_table_mathces = md_table_reg.findall(content+'\n')
    if len(md_table_mathces) >= 2:
        content = convert_markdown_to_html(content)
        html_table_matches = html_table_reg.finditer(content)
        if html_table_matches:
            for match in html_table_matches:
                matched = match.group(0)
                position = [match.start(), match.end()]
                content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace md table with space
                pred_all.append({
                    'category_type': 'html_table',
                    'position': position,
                    'content': matched.strip(),
                    'fine_category_type': 'md2html_table'
                })

    # extract code blocks
    code_matches = code_block_reg.finditer(content)
    if code_matches:
        for match in code_matches:
            position = [match.start(), match.end()]
            language = match.group(1)
            code = match.group(2).strip()
            content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace code block with space
            pred_all.append({
                'category_type': 'text_all',
                'position': position,
                'content': code,
                'language': language,
                'fine_category_type': 'code'
            })

    # Remove latex style
    content = re.sub(r'\\title\{(.*?)\}', r'\1', content)
    content = re.sub(r'\\title\s*\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)
    content = re.sub(r'\\text\s*\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)
    content = re.sub(r'\\section\*?\{(.*?)\}', r'\1', content)
    content = re.sub(r'\\section\*?\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)

    # extract texts
    res = content.split('\n\n')
    if len(res) == 1:
        res = content.split('\n')  # some models do not use double newlines, so use single newlines to split

    content_position = 0
    for text in res:
        position = [content_position, content_position+len(text)]
        content_position += len(text)
        text = text.strip()
        text = text.strip('\n')
        text = '\n'.join([_.strip() for _ in text.split('\n') if _.strip()])   # avoid some single newline content with many spaces

        if text:  # Check if the stripped text is not empty
            if text.startswith('<table') and text.endswith('</table>'):
                pred_all.append({
                    'category_type': 'html_table',
                    'position': position,
                    'content': text,
                })
            elif text.startswith('$') and text.endswith('$'):
                if text.replace('$', '').strip():
                    pred_all.append({
                        'category_type': 'equation_isolated',
                        'position': position,
                        'content': text.strip(),
                    })
            else:
                text = text.strip()
                if text:
                    pred_all.append({
                        'category_type': 'text_all',
                        'position': position,
                        'content': text,
                        'fine_category_type': 'text_block'
                    })

    pred_dataset = defaultdict(list)
    pred_all = sorted(pred_all, key=lambda x: x['position'][0])
    for item in pred_all:
        pred_dataset[item['category_type']].append(item)
    return pred_dataset

def normalized_formula(text):
    # Normalize math formulas before matching
    filter_list = ['\\mathbf', '\\mathrm', '\\mathnormal', '\\mathit', '\\mathbb', '\\mathcal', '\\mathscr', '\\mathfrak', '\\mathsf', '\\mathtt', 
                   '\\textbf', '\\text', '\\boldmath', '\\boldsymbol', '\\operatorname', '\\bm',
                   '\\symbfit', '\\mathbfcal', '\\symbf', '\\scriptscriptstyle', '\\notag',
                   '\\setlength', '\\coloneqq', '\\space', '\\thickspace', '\\thinspace', '\\medspace', '\\nobreakspace', '\\negmedspace',
                   '\\quad', '\\qquad', '\\enspace', '\\substackw', ' ', '$$', '\\left', '\\right', '\\displaystyle', '\\text']
    
    # delimiter_filter
    text = text.strip().strip('$').strip('\n')
    pattern = re.compile(r"\\\[(.+?)(?<!\\)\\\]")
    match = pattern.search(text)

    if match:
        text = match.group(1).strip()
    
    tag_pattern = re.compile(r"\\tag\{.*?\}")
    text = tag_pattern.sub('', text)
    hspace_pattern = re.compile(r"\\hspace\{.*?\}")
    text = hspace_pattern.sub('', text)
    begin_pattern = re.compile(r"\\begin\{.*?\}")
    text = begin_pattern.sub('', text)
    end_pattern = re.compile(r"\\end\{.*?\}")
    text = end_pattern.sub('', text)
    col_sep = re.compile(r"\\arraycolsep.*?\}")
    text = col_sep.sub('', text)
    text = text.strip('.')
    
    for filter_text in filter_list:
        text = text.replace(filter_text, '')
    text = text.lower()
    return text

def textblock2unicode(text):
    from pylatexenc.latex2text import LatexNodes2Text
    inline_matches = inline_reg.finditer(text)
    removal_positions = []
    for match in inline_matches:
        position = [match.start(), match.end()]
        content = match.group(1) if match.group(1) is not None else match.group(2)
        # Remove escape characters \
        clean_content = re.sub(r'\\([\\_&%^])', '', content)

        try:
            if any(char in clean_content for char in r'\^_'):
                if clean_content.endswith('\\'):
                    clean_content += ' '
                # inline_array.append(match.group(0))
                unicode_content = LatexNodes2Text().latex_to_text(clean_content)
                removal_positions.append((position[0], position[1], unicode_content))
        except:
            continue
    
    # Remove inline formulas from original text
    for start, end, unicode_content in sorted(removal_positions, reverse=True):
        text = text[:start] + unicode_content.strip() + text[end:]

    return text

# Text OCR quality check processing:
def clean_string(input_string):
    # Use regex to keep Chinese characters, English letters and numbers
    # input_string = input_string.replace('\\t', '').replace('\\n', '').replace('\t', '').replace('\n', '').replace('/t', '').replace('/n', '')
    input_string = input_string.replace('\\t', '').replace('\\n', '').replace('\t', '').replace('\n', '').replace('/t', '').replace('/n', '')
    cleaned_string = re.sub(r'[^\w\u4e00-\u9fff]', '', input_string)   # 只保留中英文和数字
    return cleaned_string

def get_pred_category_type(pred_idx, pred_items):
    try:
        if pred_items[pred_idx].get('fine_category_type'):
            pred_pred_category_type = pred_items[pred_idx]['fine_category_type']
        else:
            pred_pred_category_type = pred_items[pred_idx]['category_type']
        return pred_pred_category_type
    except:
        return ""

def compute_edit_distance_matrix_new(gt_lines, matched_lines):
    try:
        distance_matrix = np.zeros((len(gt_lines), len(matched_lines)))
        for i, gt_line in enumerate(gt_lines):
            for j, matched_line in enumerate(matched_lines):
                if len(gt_line) == 0 and len(matched_line) == 0:
                    distance_matrix[i][j] = 0  
                else:
                    distance_matrix[i][j] = Levenshtein.distance(gt_line, matched_line) / max(len(matched_line), len(gt_line))
        return distance_matrix
    except ZeroDivisionError:
        raise  


## mix match
def get_gt_pred_lines(gt_mix,pred_dataset_mix,line_type):

    norm_html_lines,gt_lines,pred_lines,norm_gt_lines,norm_pred_lines,gt_cat_list = [],[],[],[],[],[]
    if line_type in ['html_table','latex_table']:
        for item in gt_mix:
            if item.get('fine_category_type'):
                gt_cat_list.append(item['fine_category_type'])
            else:
                gt_cat_list.append(item['category_type'])
            if item.get('content'):
                gt_lines.append(str(item['content']))
                norm_html_lines.append(str(item['content']))
            elif line_type == 'text':
                gt_lines.append(str(item['text']))
            elif line_type == 'html_table':
                gt_lines.append(str(item['html']))
            elif line_type == 'formula':
                gt_lines.append(str(item['latex']))
            elif line_type == 'latex_table':
                gt_lines.append(str(item['latex']))
                norm_html_lines.append(str(item['html']))
        
        pred_lines = [str(item['content']) for item in pred_dataset_mix]
        if line_type == 'formula':
            norm_gt_lines = [normalized_formula(_) for _ in gt_lines]
            norm_pred_lines = [normalized_formula(_) for _ in pred_lines]
        elif line_type == 'text':
            norm_gt_lines = [clean_string(textblock2unicode(_)) for _ in gt_lines]
            norm_pred_lines = [clean_string(textblock2unicode(_)) for _ in pred_lines]
        else:
            norm_gt_lines = gt_lines
            norm_pred_lines = pred_lines
        if line_type == 'latex_table':
            gt_lines = norm_html_lines

    else:
        for item in pred_dataset_mix:
            # text
            if item['category_type'] == 'text_all':
                pred_lines.append(str(item['content']))
                norm_pred_lines.append(clean_string(textblock2unicode(str(item['content']))))
            # formula
            elif  item['category_type']=='equation_isolated':
                pred_lines.append(str(item['content']))
                norm_pred_lines.append(normalized_formula(str(item['content'])))
            # table
            else:
                pred_lines.append(str(item['content']))
                norm_pred_lines.append(str(item['content']))
        
        for item in gt_mix:
            if item.get('content'):
                gt_lines.append(str(item['content']))
                if item['category_type'] == 'text_all':              
                    norm_gt_lines.append(clean_string(textblock2unicode(str(item['content']))))
                else:
                   norm_gt_lines.append(item['content'])
                
                norm_html_lines.append(str(item['content']))

                if item.get('fine_category_type'):
                    gt_cat_list.append(item['fine_category_type'])
                else:
                    gt_cat_list.append(item['category_type'])
            # text      
            elif item['category_type'] in ['text_block', 'title', 'code_txt', 'code_txt_caption', 'reference', 'equation_caption','figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption','header', 'footer', 'page_footnote', 'page_number']:
                gt_lines.append(str(item['text']))
                norm_gt_lines.append(clean_string(textblock2unicode(str(item['text']))))

                if item.get('fine_category_type'):
                    gt_cat_list.append(item['fine_category_type'])
                else:
                    gt_cat_list.append(item['category_type'])

            # formula
            elif item['category_type'] == 'equation_isolated':
                gt_lines.append(str(item['latex']))
                norm_gt_lines.append(normalized_formula(str(item['latex'])))

                if item.get('fine_category_type'):
                    gt_cat_list.append(item['fine_category_type'])
                else:
                    gt_cat_list.append(item['category_type'])


    filtered_lists = [(a, b, c) for a, b, c in zip(gt_lines, norm_gt_lines, gt_cat_list) if a and b]

    # decompress to three lists
    if filtered_lists:
        gt_lines_c, norm_gt_lines_c, gt_cat_list_c = zip(*filtered_lists)

        # convert to lists
        gt_lines_c = list(gt_lines_c)
        norm_gt_lines_c = list(norm_gt_lines_c)
        gt_cat_list_c = list(gt_cat_list_c)
    else:
        gt_lines_c = []
        norm_gt_lines_c = []
        gt_cat_list_c = []

    # pred's empty values
    filtered_lists = [(a, b) for a, b in zip(pred_lines, norm_pred_lines) if a and b]

    # decompress to two lists
    if filtered_lists:
        pred_lines_c, norm_pred_lines_c = zip(*filtered_lists)

        # convert to lists
        pred_lines_c = list(pred_lines_c)
        norm_pred_lines_c = list(norm_pred_lines_c)
    else:
        pred_lines_c = []
        norm_pred_lines_c = []

    return gt_lines_c, norm_gt_lines_c, gt_cat_list_c, pred_lines_c, norm_pred_lines_c, gt_mix, pred_dataset_mix


def match_gt2pred_simple(gt_items, pred_items, line_type, img_name):

    gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines, gt_items, pred_items = get_gt_pred_lines(gt_items, pred_items,line_type)
    match_list = []

    if not norm_gt_lines: # not matched pred should be concatenate
        pred_idx_list = range(len(norm_pred_lines))
        match_list.append({
            'gt_idx': [""],
            'gt': "",
            'pred_idx': pred_idx_list,
            'pred': ''.join(pred_lines[_] for _ in pred_idx_list), 
            'gt_position': [""],
            'pred_position': pred_items[pred_idx_list[0]]['position'][0],  # get the first pred's position
            'norm_gt': "",
            'norm_pred': ''.join(norm_pred_lines[_] for _ in pred_idx_list),
            'gt_category_type': "",
            'pred_category_type': get_pred_category_type(pred_idx_list[0], pred_items), # get the first pred's category
            'gt_attribute': [{}],
            'edit': 1,
            'img_id': img_name
        })
        return match_list,None
    elif not norm_pred_lines: # not matched gt should be separated
        for gt_idx in range(len(norm_gt_lines)):
            match_list.append({
                'gt_idx': [gt_idx],
                'gt': gt_lines[gt_idx],
                'pred_idx': [""],
                'pred': "",
                'gt_position': [gt_items[gt_idx].get('order') if gt_items[gt_idx].get('order') else gt_items[gt_idx].get('position', [""])[0]],
                'pred_position': "",
                'norm_gt': norm_gt_lines[gt_idx],
                'norm_pred': "",
                'gt_category_type': gt_cat_list[gt_idx],
                'pred_category_type': "",
                'gt_attribute': [gt_items[gt_idx].get("attribute", {})],
                'edit': 1,
                'img_id': img_name
            })
        return match_list,None
    
    cost_matrix = compute_edit_distance_matrix_new(norm_gt_lines, norm_pred_lines)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    
    for gt_idx in range(len(norm_gt_lines)):
        if gt_idx in row_ind:
            row_i = list(row_ind).index(gt_idx)
            pred_idx = int(col_ind[row_i])
            pred_line = pred_lines[pred_idx]
            norm_pred_line = norm_pred_lines[pred_idx]
            edit = cost_matrix[gt_idx][pred_idx]
        else:
            pred_idx = ""
            pred_line = ""
            norm_pred_line = ""
            edit = 1
        

        match_list.append({
            'gt_idx': [gt_idx],
            'gt': gt_lines[gt_idx],
            'norm_gt': norm_gt_lines[gt_idx],
            'gt_category_type': gt_cat_list[gt_idx],
            'gt_position': [gt_items[gt_idx].get('order') if gt_items[gt_idx].get('order') else gt_items[gt_idx].get('position', [""])[0]],
            'gt_attribute': [gt_items[gt_idx].get("attribute", {})],
            'pred_idx': [pred_idx],
            'pred': pred_line,
            'norm_pred': norm_pred_line,
            'pred_category_type': get_pred_category_type(pred_idx, pred_items) if pred_idx else "",
            'pred_position': pred_items[pred_idx]['position'][0] if pred_idx else "",
            'edit': edit,
            'img_id': img_name
        })
    
    pred_idx_list = [pred_idx for pred_idx in range(len(norm_pred_lines)) if pred_idx not in col_ind] # get not matched preds
    if pred_idx_list:
        if line_type in ['html_table', 'latex_table']:
            unmatch_table_pred = []
            for i in pred_idx_list:
                original_item = pred_items[i]
                soup = BeautifulSoup(original_item.get('content'),'html.parser')
                text_block = [re.sub(r'\$\\cdot\$','',item.string).strip() for item in soup.findAll('td') if item.string]
                for concatenate_text in text_block:
                    new_item = deepcopy(original_item)
                    new_item['content'] = concatenate_text
                    new_item['category_type'] = 'text_all'  
                    unmatch_table_pred.append(new_item)
            return match_list, unmatch_table_pred  
        
        else:
            match_list.append({
                'gt_idx': [""],
                'gt': "",
                'pred_idx': pred_idx_list,
                'pred': ''.join(pred_lines[_] for _ in pred_idx_list), 
                'gt_position': [""],
                'pred_position': pred_items[pred_idx_list[0]]['position'][0],  # get the first pred's position
                'norm_gt': "",
                'norm_pred': ''.join(norm_pred_lines[_] for _ in pred_idx_list),
                'gt_category_type': "",
                'pred_category_type': get_pred_category_type(pred_idx_list[0], pred_items), # get the first pred's category
                'gt_attribute': [{}],
                'edit': 1,
                'img_id': img_name
            })
    return match_list,None


def match_gt2pred_no_split(gt_items, pred_items, line_type, img_name):
    # directly concatenate gt and pred by position
    gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines = get_gt_pred_lines(gt_items, pred_items, line_type)
    gt_line_with_position = []
    for gt_line, norm_gt_line, gt_item in zip(gt_lines, norm_gt_lines, gt_items):
        gt_position = gt_item['order'] if gt_item.get('order') else gt_item.get('position', [""])[0]
        if gt_position:
            gt_line_with_position.append((gt_position, gt_line, norm_gt_line))
    sorted_gt_lines = sorted(gt_line_with_position, key=lambda x: x[0])
    gt = '\n\n'.join([_[1] for _ in sorted_gt_lines])
    norm_gt = '\n\n'.join([_[2] for _ in sorted_gt_lines])
    pred_line_with_position = [(pred_item['position'], pred_line, pred_norm_line) for pred_line, pred_norm_line, pred_item in zip(pred_lines, norm_pred_lines, pred_items)]
    sorted_pred_lines = sorted(pred_line_with_position, key=lambda x: x[0])
    pred = '\n\n'.join([_[1] for _ in sorted_pred_lines])
    norm_pred = '\n\n'.join([_[2] for _ in sorted_pred_lines])
    # edit = Levenshtein.distance(norm_gt, norm_pred)/max(len(norm_gt), len(norm_pred))
    if norm_gt or norm_pred:
        return [{
                'gt_idx': [0],
                'gt': gt,
                'norm_gt': norm_gt,
                'gt_category_type': "text_merge",
                'gt_position': [""],
                'gt_attribute': [{}],
                'pred_idx': [0],
                'pred': pred,
                'norm_pred': norm_pred,
                'pred_category_type': "text_merge",
                'pred_position': "",
                # 'edit': edit,
                'img_id': img_name
            }]
    else:
        return []

def is_all_l(spec: str) -> bool:
    spec = re.sub(r'\s+|\|', '', spec)
    spec = re.sub(r'@{[^}]*}', '', spec)
    spec = re.sub(r'!{[^}]*}', '', spec)
    return bool(spec) and len(spec) == 1 and spec in {'l', 'c', 'r'}

def split_gt_equation_arrays(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split GT dictionary entries with \ \ begin {array}... \ \ end {array}.
        -Only applicable to category_type=='Equation_isolvated 'and latex containing an array.
        -Extract a new entry for each line of formula:
    *Update 'latex'
    *If there is a linew_ith_stans, synchronously replace its internal latex
    *The 'order' consists of 7-->7.1, 7.2
    """
    output = []

    for item in data:
        # Only process dictionaries that meet the conditions
        if (item.get("category_type") == "equation_isolated" and
                "\\begin{array" in item.get("latex", "")):

            # Extract the internal content of the array
            match = ARRAY_RE.search(item["latex"])
            if match:

                spec = match.group("spec")
                if not is_all_l(spec):
                    # If there are mixed elements such as r/c/p {...} in the column, keep the original entry directly
                    output.append(item)
                    continue
                
                body  = match.group("body")
                # Split by LaTeX line separator
                lines = [ln.strip() for ln in re.split(r'\\\\', body) if ln.strip()]

                base_order = float(item["order"])  # 7 -> 7.0，Compatible with float/int

                for idx, line in enumerate(lines, start=1):
                    new_item = deepcopy(item)
                    new_item["latex"] = f"\\[{line}\\]"
                    new_item["order"] = round(base_order + idx / 10, 1)
                    output.append(new_item)
                continue
        # Do not modify other situations
        output.append(item)
    return output

def sort_by_position_skip_inline(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort by position [0] from small to large first;
    If fine_category_type=='Equation_inline ', then place it uniformly at the end,
    And maintain their relative order in the original list (stable sorting).
    """
    # Enumerate preserves the original sequential index for stability when equation_inline is in parallel
    return sorted(
        enumerate(items),
        key=lambda pair: (
            pair[1].get('fine_category_type') == 'equation_inline',  # False < True
            pair[1]['position'][0],                                   # pos start
            pair[0]                                                   # original id
        )
    )

def _wrap(line: str) -> str:
    """Re package single line formula \ \ [... \ \]"""
    return f"\\[{line.strip()}\\]"

def split_equation_arrays(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processing entries with category_type=='Equation_isolvated 'and containing \ \ begin {array}...:
    *Split multi line formula
    *Repackaging content
    ** * Recalculate position/positions**
    """
    out: List[Dict[str, Any]] = []

    for item in data:
        if (item.get("category_type").lower() == "equation_isolated" and
                "\\begin{array" in item.get("content", "")):

            content = item["content"]
            m = ARRAY_RE.search(content)
            if not m:
                out.append(item)
                continue

            if not is_all_l(m.group('spec')):
                out.append(item)
                continue
            
            body = m.group('body')
            lines = [ln.strip() for ln in re.split(r'\\\\', body) if ln.strip()]

            # Global starting character index
            pos_key = "position" if "position" in item else "positions"
            global_start = item[pos_key][0]
            body_start_in_content = m.start('body')
            search_from = 0  # cur pos in body 
            for ln in lines:
                # Find the offset of the current row in the body
                idx_in_body = body.find(ln, search_from)
                if idx_in_body == -1:
                    # Unlikely to happen; Conservative treatment
                    idx_in_body = search_from
                search_from = idx_in_body + len(ln)  # update cur pos

                # calculate global index
                line_start_global = global_start + body_start_in_content + idx_in_body
                line_end_global   = line_start_global + len(ln) - 1

                new_item = deepcopy(item)
                new_item["content"] = _wrap(ln)
                new_item[pos_key]   = [line_start_global, line_end_global]

                out.append(new_item)
            continue
        out.append(item)

    return out

def merge_matches(matches, matching_dict):
    final_matches = {}
    processed_gt_indices = set() 

    for gt_idx, match_info in matches.items():
        pred_indices = match_info['pred_indices']
        edit_distance = match_info['edit_distance']

        pred_key = tuple(sorted(pred_indices))

        if pred_key in final_matches:
            if gt_idx not in processed_gt_indices:
                final_matches[pred_key]['gt_indices'].append(gt_idx)
                processed_gt_indices.add(gt_idx)
        else:
            final_matches[pred_key] = {
                'gt_indices': [gt_idx],
                'edit_distance': edit_distance
            }
            processed_gt_indices.add(gt_idx)

    for pred_idx, gt_indices in matching_dict.items():
        pred_key = (pred_idx,) if not isinstance(pred_idx, (list, tuple)) else tuple(sorted(pred_idx))

        if pred_key in final_matches:
            for gt_idx in gt_indices:
                if gt_idx not in processed_gt_indices:
                    final_matches[pred_key]['gt_indices'].append(gt_idx)
                    processed_gt_indices.add(gt_idx)
        else:
            final_matches[pred_key] = {
                'gt_indices': [gt_idx for gt_idx in gt_indices if gt_idx not in processed_gt_indices],
                'edit_distance': None
            }
            processed_gt_indices.update(final_matches[pred_key]['gt_indices'])

    return final_matches


def recalculate_edit_distances(final_matches, gt_lens_dict, norm_gt_lines, norm_pred_lines):
    for pred_key, info in final_matches.items():
        gt_indices = sorted(set(info['gt_indices']))

        if not gt_indices:
            info['edit_distance'] = 1
            continue

        if len(gt_indices) > 1:
            merged_gt_content = ''.join(norm_gt_lines[gt_idx] for gt_idx in gt_indices)
            pred_content = norm_pred_lines[pred_key[0]] if isinstance(pred_key[0], int) else ''

            try:
                edit_distance = Levenshtein_distance(merged_gt_content, pred_content)
                normalized_edit_distance = edit_distance / max(1, len(merged_gt_content), len(pred_content))
            except ZeroDivisionError:
                normalized_edit_distance = 1

            info['edit_distance'] = normalized_edit_distance
        else:
            gt_idx = gt_indices[0]
            pred_content = ' '.join(norm_pred_lines[pred_idx] for pred_idx in pred_key if isinstance(pred_idx, int))

            try:
                edit_distance = Levenshtein_distance(norm_gt_lines[gt_idx], pred_content)
                normalized_edit_distance = edit_distance / max(len(norm_gt_lines[gt_idx]), len(pred_content))
            except ZeroDivisionError:
                normalized_edit_distance = 1

            info['edit_distance'] = normalized_edit_distance
            info['pred_content'] = pred_content

def convert_final_matches(final_matches, norm_gt_lines, norm_pred_lines):
    converted_results = []

    all_gt_indices = set(range(len(norm_gt_lines)))
    all_pred_indices = set(range(len(norm_pred_lines)))

    for pred_key, info in final_matches.items():
        pred_content = ' '.join(norm_pred_lines[pred_idx] for pred_idx in pred_key if isinstance(pred_idx, int))
        
        for gt_idx in sorted(set(info['gt_indices'])):
            result_entry = {
                'gt_idx': int(gt_idx),
                'gt': norm_gt_lines[gt_idx],
                'pred_idx': list(pred_key),
                'pred': pred_content,
                'edit': info['edit_distance']
            }
            converted_results.append(result_entry)
    
    matched_gt_indices = set().union(*[set(info['gt_indices']) for info in final_matches.values()])
    unmatched_gt_indices = all_gt_indices - matched_gt_indices
    matched_pred_indices = set(idx for pred_key in final_matches.keys() for idx in pred_key if isinstance(idx, int))
    unmatched_pred_indices = all_pred_indices - matched_pred_indices

    if unmatched_pred_indices:
        if unmatched_gt_indices:
            distance_matrix = [
                # [Levenshtein_distance(norm_gt_lines[gt_idx], norm_pred_lines[pred_idx]) for pred_idx in unmatched_pred_indices]
                [Levenshtein_distance(norm_gt_lines[gt_idx], norm_pred_lines[pred_idx])/max(len(norm_gt_lines[gt_idx]), len(norm_pred_lines[pred_idx])) for pred_idx in unmatched_pred_indices]
                for gt_idx in unmatched_gt_indices
            ]

            row_ind, col_ind = linear_sum_assignment(distance_matrix)

            for i, j in zip(row_ind, col_ind):
                gt_idx = list(unmatched_gt_indices)[i]
                pred_idx = list(unmatched_pred_indices)[j]
                result_entry = {
                    'gt_idx': int(gt_idx),
                    'gt': norm_gt_lines[gt_idx],
                    'pred_idx': [pred_idx],
                    'pred': norm_pred_lines[pred_idx],
                    'edit': 1
                }
                converted_results.append(result_entry)

            matched_gt_indices.update(list(unmatched_gt_indices)[i] for i in row_ind)
        else:
            result_entry = {
                'gt_idx': "",
                'gt': '',
                'pred_idx': list(unmatched_pred_indices),
                'pred': ' '.join(norm_pred_lines[pred_idx] for pred_idx in unmatched_pred_indices),
                'edit': 1
            }
            converted_results.append(result_entry)
    else:
        for gt_idx in unmatched_gt_indices:
            result_entry = {
                'gt_idx': int(gt_idx),
                'gt': norm_gt_lines[gt_idx],
                'pred_idx': "",
                'pred': '',
                'edit': 1
            }
            converted_results.append(result_entry)

    return converted_results

def merge_duplicates_add_unmatched(converted_results, norm_gt_lines, norm_pred_lines, gt_lines, pred_lines, all_gt_indices, all_pred_indices):
    merged_results = []
    processed_pred = set()
    processed_gt = set()

    for entry in converted_results:
        pred_idx = tuple(entry['pred_idx']) if isinstance(entry['pred_idx'], list) else (entry['pred_idx'],)
        if pred_idx not in processed_pred and pred_idx != ("",):
            merged_entry = {
                'gt_idx': [entry['gt_idx']],
                'gt': entry['gt'],
                'pred_idx': entry['pred_idx'],
                'pred': entry['pred'],
                'edit': entry['edit']
            }
            for other_entry in converted_results:
                other_pred_idx = tuple(other_entry['pred_idx']) if isinstance(other_entry['pred_idx'], list) else (other_entry['pred_idx'],)
                if other_pred_idx == pred_idx and other_entry is not entry:
                    merged_entry['gt_idx'].append(other_entry['gt_idx'])
                    merged_entry['gt'] += other_entry['gt']
                    processed_gt.add(other_entry['gt_idx'])
            merged_results.append(merged_entry)
            processed_pred.add(pred_idx)
            processed_gt.add(entry['gt_idx'])

    for gt_idx in range(len(norm_gt_lines)):
        if gt_idx not in processed_gt:
            merged_results.append({
                'gt_idx': [gt_idx],
                'gt': gt_lines[gt_idx],
                'pred_idx': [""],
                'pred': "",
                'edit': 1
            })
    return merged_results

def sub_pred_fuzzy_matching(gt, pred):
    
    min_d = float('inf')

    gt_len = len(gt)
    pred_len = len(pred)

    if gt_len >= pred_len and pred_len > 0:
        for i in range(gt_len - pred_len + 1):
            sub = gt[i:i + pred_len]
            dist = Levenshtein_distance(sub, pred)/pred_len
            if dist < min_d:
                min_d = dist
                pos = i

        return min_d
    else:
        return False

def judge_pred_merge(gt_list, pred_list, threshold=0.6):
    if len(pred_list) == 1:
        return False, False

    cur_pred = ' '.join(pred_list[:-1])
    merged_pred = ' '.join(pred_list)
    
    cur_dist = Levenshtein.distance(gt_list[0], cur_pred) / max(len(gt_list[0]), len(cur_pred))
    merged_dist = Levenshtein.distance(gt_list[0], merged_pred) / max(len(gt_list[0]), len(merged_pred))
    
    if merged_dist > cur_dist:
        return False, False

    cur_fuzzy_dists = [sub_pred_fuzzy_matching(gt_list[0], cur_pred) for cur_pred in pred_list[:-1]]
    if any(dist is False or dist > threshold for dist in cur_fuzzy_dists):
        return False, False

    add_fuzzy_dist = sub_pred_fuzzy_matching(gt_list[0], pred_list[-1])
    if add_fuzzy_dist is False:
        return False, False

    merged_pred_flag = add_fuzzy_dist < threshold
    continue_flag = len(merged_pred) <= len(gt_list[0])

    return merged_pred_flag, continue_flag

def get_final_subset(subset_certain, subset_certain_cost):
    if not subset_certain or not subset_certain_cost:
        return []  

    subset_turple = sorted([(a, b) for a, b in zip(subset_certain, subset_certain_cost)], key=lambda x: x[0][0])

    group_list = defaultdict(list)
    group_idx = 0
    group_list[group_idx].append(subset_turple[0])

    for item in subset_turple[1:]:
        overlap_flag = False
        for subset in group_list[group_idx]:
            for idx in item[0]:
                if idx in subset[0]:
                    overlap_flag = True
                    break
            if overlap_flag:
                break
        if overlap_flag:
            group_list[group_idx].append(item)
        else:
            group_idx += 1
            group_list[group_idx].append(item)

    final_subset = []
    for _, group in group_list.items():
        if len(group) == 1: 
            final_subset.append(group[0][0])
        else:
            path_dict = defaultdict(list)
            path_idx = 0
            path_dict[path_idx].append(group[0])
            
            for subset in group[1:]:
                new_path = True
                for path_idx_s, path_items in path_dict.items():
                    is_dup = False
                    is_same = False
                    for path_item in path_items:
                        if path_item[0] == subset[0]:
                            is_dup = True
                            is_same = True
                            if path_item[1] > subset[1]:
                                path_dict[path_idx_s].pop(path_dict[path_idx_s].index(path_item))
                                path_dict[path_idx_s].append(subset)
                        else:
                            for num_1 in path_item[0]:
                                for num_2 in subset[0]:
                                    if num_1 == num_2:
                                        is_dup = True
                    if not is_dup:
                        path_dict[path_idx_s].append(subset)
                        new_path = False
                    if is_same:
                        new_path = False
                if new_path:
                    path_idx = len(path_dict.keys())
                    path_dict[path_idx].append(subset)

            saved_cost = float('inf')
            saved_subset = []  
            for path_idx, path in path_dict.items():
                avg_cost = sum([i[1] for i in path]) / len(path)
                if avg_cost < saved_cost:
                    saved_subset = [i[0] for i in path]
                    saved_cost = avg_cost

            final_subset.extend(saved_subset)

    return final_subset

def merge_lists_with_sublists(main_list, sub_lists):
    main_list_final = list(copy.deepcopy(main_list))
    for sub_list in sub_lists:
        pop_idx = main_list_final.index(sub_list[0])
        for _ in sub_list: 
            main_list_final.pop(pop_idx)
        main_list_final.insert(pop_idx, sub_list) 
    return main_list_final   

def deal_with_truncated(cost_matrix, norm_gt_lines, norm_pred_lines):
    matched_first = np.argwhere(cost_matrix < 0.25)
    masked_gt_idx = [i[0] for i in matched_first]
    unmasked_gt_idx = [i for i in range(cost_matrix.shape[0]) if i not in masked_gt_idx]
    masked_pred_idx = [i[1] for i in matched_first]
    unmasked_pred_idx = [i for i in range(cost_matrix.shape[1]) if i not in masked_pred_idx]

    merges_gt_dict = {}
    merges_pred_dict = {}
    merged_gt_subsets = []

    for gt_idx in unmasked_gt_idx:
        check_merge_subset = []
        merged_dist = []

        for pred_idx in unmasked_pred_idx: 
            step = 1
            merged_pred = [norm_pred_lines[pred_idx]]

            while True:
                if pred_idx + step in masked_pred_idx or pred_idx + step >= len(norm_pred_lines):
                    break
                else:
                    merged_pred.append(norm_pred_lines[pred_idx + step])
                    merged_pred_flag, continue_flag = judge_pred_merge([norm_gt_lines[gt_idx]], merged_pred) 
                    if not merged_pred_flag:
                        break
                    else:
                        step += 1
                    if not continue_flag:
                        break

            check_merge_subset.append(list(range(pred_idx, pred_idx + step)))
            matched_line = ' '.join([norm_pred_lines[i] for i in range(pred_idx, pred_idx + step)])
            dist = Levenshtein_distance(norm_gt_lines[gt_idx], matched_line) / max(len(matched_line), len(norm_gt_lines[gt_idx]))
            merged_dist.append(dist)

        if not merged_dist:
            subset_certain = []
            min_cost_idx = ""
            min_cost = float('inf')
        else:
            min_cost = min(merged_dist)
            min_cost_idx = merged_dist.index(min_cost)
            subset_certain = check_merge_subset[min_cost_idx]

        merges_gt_dict[gt_idx] = {
            'merge_subset': check_merge_subset,
            'merged_cost': merged_dist,
            'min_cost_idx': min_cost_idx,
            'subset_certain': subset_certain,
            'min_cost': min_cost
        }

    subset_certain = [merges_gt_dict[gt_idx]['subset_certain'] for gt_idx in unmasked_gt_idx if merges_gt_dict[gt_idx]['subset_certain']]
    subset_certain_cost = [merges_gt_dict[gt_idx]['min_cost'] for gt_idx in unmasked_gt_idx if merges_gt_dict[gt_idx]['subset_certain']]

    subset_certain_final = get_final_subset(subset_certain, subset_certain_cost)

    if not subset_certain_final:  
        return cost_matrix, norm_pred_lines, range(len(norm_pred_lines))

    final_pred_idx_list = merge_lists_with_sublists(range(len(norm_pred_lines)), subset_certain_final)
    final_norm_pred_lines = [' '.join(norm_pred_lines[idx_list[0]:idx_list[-1]+1]) if isinstance(idx_list, list) else norm_pred_lines[idx_list] for idx_list in final_pred_idx_list]

    new_cost_matrix = compute_edit_distance_matrix_new(norm_gt_lines, final_norm_pred_lines)

    return new_cost_matrix, final_norm_pred_lines, final_pred_idx_list

def cal_final_match(cost_matrix, norm_gt_lines, norm_pred_lines):

    new_cost_matrix, final_norm_pred_lines, final_pred_idx_list = deal_with_truncated(cost_matrix, norm_gt_lines, norm_pred_lines)

    row_ind, col_ind = linear_sum_assignment(new_cost_matrix)

    cost_list = [new_cost_matrix[r][c] for r, c in zip(row_ind, col_ind)]
    matched_col_idx = [final_pred_idx_list[i] for i in col_ind]

    return matched_col_idx, row_ind, cost_list

def initialize_indices(norm_gt_lines, norm_pred_lines):
    gt_lens_dict = {idx: len(gt_line) for idx, gt_line in enumerate(norm_gt_lines)}
    pred_lens_dict = {idx: len(pred_line) for idx, pred_line in enumerate(norm_pred_lines)}
    return gt_lens_dict, pred_lens_dict

def sub_gt_fuzzy_matching(pred, gt):  
    
    min_d = float('inf')  
    pos = ""  
    matched_sub = ""  
    gt_len = len(gt)  
    pred_len = len(pred)  
    
    if pred_len >= gt_len and gt_len > 0:  
        for i in range(pred_len - gt_len + 1):  
            sub = pred[i:i + gt_len]  
            dist = Levenshtein.distance(sub, gt)  /gt_len
            if dist < min_d:  
                min_d = dist  
                pos = i  
                matched_sub = sub  
        return min_d, pos, gt_len, matched_sub  
    else:  
        return 1, "", gt_len, "" 

def process_matches(matched_col_idx, row_ind, cost_list, norm_gt_lines, norm_pred_lines, pred_lines):
    matches = {}
    unmatched_gt_indices = []
    unmatched_pred_indices = []

    for i in range(len(norm_gt_lines)):
        if i in row_ind:
            idx = list(row_ind).index(i)
            pred_idx = matched_col_idx[idx]

            if pred_idx is None or (isinstance(pred_idx, list) and None in pred_idx):
                unmatched_pred_indices.append(pred_idx)
                continue

            if isinstance(pred_idx, list):
                pred_line = ' | '.join(norm_pred_lines[pred_idx[0]:pred_idx[-1]+1])
                ori_pred_line = ' | '.join(pred_lines[pred_idx[0]:pred_idx[-1]+1])
                matched_pred_indices_range = list(range(pred_idx[0], pred_idx[-1]+1))
            else:
                pred_line = norm_pred_lines[pred_idx]
                ori_pred_line = pred_lines[pred_idx]
                matched_pred_indices_range = [pred_idx]

            edit = cost_list[idx]
            threshold  = 0.7

            if edit > threshold:
                unmatched_pred_indices.extend(matched_pred_indices_range)
                unmatched_gt_indices.append(i)
            else:
                matches[i] = {
                    'pred_indices': matched_pred_indices_range,
                    'edit_distance': edit,
                }
                for matched_pred_idx in matched_pred_indices_range:
                    if matched_pred_idx in unmatched_pred_indices:
                        unmatched_pred_indices.remove(matched_pred_idx)
        else:
            unmatched_gt_indices.append(i)

    return matches, unmatched_gt_indices, unmatched_pred_indices

def fuzzy_match_unmatched_items(unmatched_gt_indices, norm_gt_lines, norm_pred_lines):
    matching_dict = {}

    for pred_idx, pred_content in enumerate(norm_pred_lines):
        if isinstance(pred_idx, list):
            continue

        matching_indices = []

        for unmatched_gt_idx in unmatched_gt_indices:
            gt_content = norm_gt_lines[unmatched_gt_idx]
            cur_fuzzy_dist_unmatch, cur_pos, gt_lens, matched_field = sub_gt_fuzzy_matching(pred_content, gt_content)
            threshold = 0.4 
            if cur_fuzzy_dist_unmatch < threshold:
                matching_indices.append(unmatched_gt_idx)

        if matching_indices:
            matching_dict[pred_idx] = matching_indices

    return matching_dict


def match_gt2pred_quick(gt_items, pred_items, line_type, img_name):

    gt_items = split_gt_equation_arrays(gt_items)
    
    pred_items = [pair[1] for pair in sort_by_position_skip_inline(pred_items)]

    pred_items = split_equation_arrays(pred_items)

    gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines, gt_items, pred_items = get_gt_pred_lines(gt_items, pred_items, None)
    all_gt_indices = set(range(len(norm_gt_lines)))  
    all_pred_indices = set(range(len(norm_pred_lines)))  
    
    if not norm_gt_lines:
        match_list = []
        for pred_idx in range(len(norm_pred_lines)):
            match_list.append({
                'gt_idx': [""],
                'gt': "",
                'pred_idx': [pred_idx],
                'pred': pred_lines[pred_idx],
                'gt_position': [""],
                'pred_position': pred_items[pred_idx]['position'][0],
                'norm_gt': "",
                'norm_pred': norm_pred_lines[pred_idx],
                'gt_category_type': "",
                'pred_category_type': get_pred_category_type(pred_idx, pred_items),
                'gt_attribute': [{}],
                'edit': 1,
                'img_id': img_name
            })
        return match_list
    elif not norm_pred_lines:
        match_list = []
        for gt_idx in range(len(norm_gt_lines)):
            match_list.append({
                'gt_idx': [gt_idx],
                'gt': gt_lines[gt_idx],
                'pred_idx': [""],
                'pred': "",
                'gt_position': [gt_items[gt_idx].get('order') if gt_items[gt_idx].get('order') else gt_items[gt_idx].get('position', [""])[0]],
                'pred_position': "",
                'norm_gt': norm_gt_lines[gt_idx],
                'norm_pred': "",
                'gt_category_type': gt_cat_list[gt_idx],
                'pred_category_type': "",
                'gt_attribute': [gt_items[gt_idx].get("attribute", {})],
                'edit': 1,
                'img_id': img_name
            })
        return match_list
    elif len(norm_gt_lines) == 1 and len(norm_pred_lines) == 1:
        edit_distance = Levenshtein_distance(norm_gt_lines[0], norm_pred_lines[0])
        normalized_edit_distance = edit_distance / max(len(norm_gt_lines[0]), len(norm_pred_lines[0]))
        return [{
            'gt_idx': [0],
            'gt': gt_lines[0],
            'pred_idx': [0],
            'pred': pred_lines[0],
            'gt_position': [gt_items[0].get('order') if gt_items[0].get('order') else gt_items[0].get('position', [""])[0]],
            'pred_position': pred_items[0]['position'][0],
            'norm_gt': norm_gt_lines[0],
            'norm_pred': norm_pred_lines[0],
            'gt_category_type': gt_cat_list[0],
            'pred_category_type': get_pred_category_type(0, pred_items),
            'gt_attribute': [gt_items[0].get("attribute", {})],
            'edit': normalized_edit_distance,
            'img_id': img_name
        }]
    
    # match category ignore first
    ignores = ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 
               'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption']
    
    ignore_gt_lines = []
    ignores_ori_gt_lines= []
    ignores_gt_items = []
    ignore_gt_idxs = []
    ignores_gt_cat_list = []
    
    no_ignores_gt_lines = []
    no_ignores_ori_gt_lines = []
    no_ignores_gt_idxs = []
    no_ignores_gt_items = []
    no_ignores_gt_cat_list = []

    for i, line in enumerate(norm_gt_lines):
        if gt_cat_list[i] in ignores:
            ignore_gt_lines.append(line)
            ignores_ori_gt_lines.append(gt_lines[i])
            ignores_gt_items.append(gt_items[i])
            ignore_gt_idxs.append(i)
            ignores_gt_cat_list.append(gt_cat_list[i])
        else:
            no_ignores_gt_lines.append(line)
            no_ignores_ori_gt_lines.append(gt_lines[i])
            no_ignores_gt_items.append(gt_items[i])
            no_ignores_gt_cat_list.append(gt_cat_list[i])
            no_ignores_gt_idxs.append(i)

    ignore_pred_idxs = []
    ignore_pred_lines = []
    ignores_pred_items = []
    ignores_ori_pred_lines = []

    merged_ignore_results = []

    if len(ignore_gt_lines) > 0:
        
        ignore_matches_dict = {}

        ignore_matrix = compute_edit_distance_matrix_new(ignore_gt_lines, norm_pred_lines)
        
        ignores_gt_indices = set(range(len(ignore_gt_lines)))  
        ignores_pred_indices = set(range(len(ignore_pred_lines)))

        ignore_matches = np.argwhere(ignore_matrix < 0.25) 
        if len(ignore_matches) > 0:
            ignore_pred_idxs = [_[1] for _ in ignore_matches]
            ignore_gt_matched_idxs = [ignore_gt_idxs[_[0]] for _ in ignore_matches]
            for i in ignore_pred_idxs:
                ignore_pred_lines.append(norm_pred_lines[i])
                ignores_ori_pred_lines.append(pred_lines[i])
                ignores_pred_items.append(pred_items[i])

                ignores_gt_indices = set(range(len(ignore_gt_lines)))  
                ignores_pred_indices = set(range(len(ignore_pred_lines))) 

            for idx, i in enumerate(ignore_matches):
                ignore_matches_dict[i[0]] = {
                    'pred_indices': [idx],
                    'edit_distance': ignore_matrix[i[0]][i[1]]
                }

        ignore_final_matches = merge_matches(ignore_matches_dict, {})
        recalculate_edit_distances(ignore_final_matches, {}, ignore_gt_lines, ignore_pred_lines)
        converted_ignore_results = convert_final_matches(ignore_final_matches, ignore_gt_lines, ignore_pred_lines)
        merged_ignore_results = merge_duplicates_add_unmatched(converted_ignore_results, ignore_gt_lines, ignore_pred_lines, ignores_ori_gt_lines, ignores_ori_pred_lines, ignores_gt_indices, ignores_pred_indices)

        for entry in merged_ignore_results:
            entry['gt_idx'] = [entry['gt_idx']] if not isinstance(entry['gt_idx'], list) else entry['gt_idx']
            entry['pred_idx'] = [entry['pred_idx']] if not isinstance(entry['pred_idx'], list) else entry['pred_idx']
            entry['gt_position'] = [ignores_gt_items[_].get('order') if ignores_gt_items[_].get('order') else ignores_gt_items[_].get('position', [""])[0] for _ in entry['gt_idx']] if entry['gt_idx'] != [""] else [""]
            entry['pred_position'] = ignores_pred_items[entry['pred_idx'][0]]['position'][0] if entry['pred_idx'] != [""] else "" 
            entry['gt'] = ''.join([ignores_ori_gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [""] else ""
            entry['pred'] = ''.join([ignores_ori_pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [""] else ""
            entry['norm_gt'] = ''.join([ignore_gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [""] else ""
            entry['norm_pred'] = ''.join([ignore_pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [""] else ""

            if entry['gt_idx'] != [""]:
                ignore_type = ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption']
                gt_cagegory_clean = [ignores_gt_cat_list[_] for _ in entry['gt_idx'] if ignores_gt_cat_list[_] not in ignore_type] 
                if gt_cagegory_clean:
                    entry['gt_category_type'] = Counter(gt_cagegory_clean).most_common(1)[0][0] 
                else:
                    entry['gt_category_type'] = Counter([ignores_gt_cat_list[_] for _ in entry['gt_idx']]).most_common(1)[0][0] 
            else:
                entry['gt_category_type'] = ""
                entry['pred_category_type'] = get_pred_category_type(entry['pred_idx'][0], ignores_pred_items) if entry['pred_idx'] != [""] else ""
                if entry['pred_category_type'] == 'equation_inline':
                    merged_ignore_results.remove(entry)
            entry['pred_category_type'] = get_pred_category_type(entry['pred_idx'][0], ignores_pred_items) if entry['pred_idx'] != [""] else ""
            entry['gt_attribute'] = [ignores_gt_items[_].get("attribute", {}) for _ in entry['gt_idx']] if entry['gt_idx'] != [""] else [{}] 
            entry['img_id'] = img_name
        
        for entry in merged_ignore_results:
            if isinstance(entry['gt_idx'], list) and entry['gt_idx'] != [""]:
                gt_idx = []
                for i in entry['gt_idx']:
                    gt_idx.append(ignore_gt_idxs[i])
                entry['gt_idx'] = gt_idx
            if isinstance(entry['pred_idx'], list) and entry['pred_idx'] != [""]:
                pred_idx = []
                for i in entry['pred_idx']:
                    pred_idx.append(int(ignore_pred_idxs[i]))
                entry['pred_idx'] = pred_idx

    no_ignores_pred_lines = []
    no_ignores_ori_pred_lines = []
    no_ignores_pred_indices = []
    no_ignores_pred_items = []
    no_ignore_pred_idxs = []

    for idx, line in enumerate(norm_pred_lines):
        if not idx in ignore_pred_idxs:
            no_ignores_pred_lines.append(line)
            no_ignores_ori_pred_lines.append(pred_lines[idx])
            no_ignores_pred_items.append(pred_items[idx])
            no_ignore_pred_idxs.append(idx)
    
    # initialize new indices for lines without ignore categories
    no_ignores_gt_indices = set(range(len(no_ignores_gt_lines)))  
    no_ignores_pred_indices = set(range(len(no_ignores_pred_lines)))  
    
    # exclude ignore categories
    cost_matrix = compute_edit_distance_matrix_new(no_ignores_gt_lines, no_ignores_pred_lines)

    matched_col_idx, row_ind, cost_list = cal_final_match(cost_matrix, no_ignores_gt_lines, no_ignores_pred_lines)
        
    gt_lens_dict, _ = initialize_indices(no_ignores_gt_lines, no_ignores_pred_lines)
    
    matches, unmatched_gt_indices, unmatched_pred_indices = process_matches(matched_col_idx, row_ind, cost_list, no_ignores_gt_lines, no_ignores_pred_lines, no_ignores_ori_pred_lines)
    
    matching_dict = fuzzy_match_unmatched_items(unmatched_gt_indices, no_ignores_gt_lines, no_ignores_pred_lines)

    final_matches = merge_matches(matches, matching_dict)

    recalculate_edit_distances(final_matches, gt_lens_dict, no_ignores_gt_lines, no_ignores_pred_lines)
    
    converted_results = convert_final_matches(final_matches, no_ignores_gt_lines, no_ignores_pred_lines)
    
    merged_results = merge_duplicates_add_unmatched(converted_results, no_ignores_gt_lines, no_ignores_pred_lines, no_ignores_ori_gt_lines, no_ignores_ori_pred_lines, no_ignores_gt_indices, no_ignores_pred_indices)

    for entry in merged_results:
        if entry['gt_idx'] != [""]:
            ignore_type = ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption']
            gt_cagegory_clean = [no_ignores_gt_cat_list[_] for _ in entry['gt_idx'] if no_ignores_gt_cat_list[_] not in ignore_type] 
            if gt_cagegory_clean:
                entry['gt_category_type'] = Counter(gt_cagegory_clean).most_common(1)[0][0] 
            else:
                entry['gt_category_type'] = Counter([no_ignores_gt_cat_list[_] for _ in entry['gt_idx']]).most_common(1)[0][0] 
        else:
            entry['gt_category_type'] = ""
            entry['pred_category_type'] = get_pred_category_type(entry['pred_idx'][0], no_ignores_pred_items) if entry['pred_idx'] != [""] else ""
            if entry['pred_category_type'] == 'equation_inline':
                merged_results.remove(entry)


        entry['gt_idx'] = [entry['gt_idx']] if not isinstance(entry['gt_idx'], list) else entry['gt_idx']
        entry['pred_idx'] = [entry['pred_idx']] if not isinstance(entry['pred_idx'], list) else entry['pred_idx']
        entry['gt_position'] = [no_ignores_gt_items[_].get('order') if no_ignores_gt_items[_].get('order') else no_ignores_gt_items[_].get('position', [""])[0] for _ in entry['gt_idx']] if entry['gt_idx'] != [""] else [""]
        entry['pred_position'] = no_ignores_pred_items[entry['pred_idx'][0]]['position'][0] if entry['pred_idx'] != [""] else "" 
        #  Multi line formula splicing modification
        if entry['gt_category_type'] == 'equation_isolated' and len(entry['gt_idx']) > 1:
            mutli_formula  = ' \\\\ '.join(['{'+no_ignores_ori_gt_lines[_].strip('$$').strip('\n')+'}' for _ in  entry['gt_idx']]) if entry['gt_idx'] != [""] else ""
            mutli_formula = '\\begin{array}{l} ' + mutli_formula + ' \end{array}'
            entry['gt'] = mutli_formula
        else:
            entry['gt'] = ''.join([no_ignores_ori_gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [""] else ""

        entry['pred_category_type'] = get_pred_category_type(entry['pred_idx'][0], no_ignores_pred_items) if entry['pred_idx'] != [""] else ""
        entry['gt_attribute'] = [no_ignores_gt_items[_].get("attribute", {}) for _ in entry['gt_idx']] if entry['gt_idx'] != [""] else [{}] 
        entry['img_id'] = img_name
        
        #  Multi line formula splicing modification pred
        if 'equation' in entry['pred_category_type'] and len(entry['pred_idx']) > 1:
            mutli_formula  = ' \\\\ '.join(['{'+no_ignores_ori_pred_lines[_].strip('$$').strip('\n')+'}' for _ in  entry['pred_idx']]) if entry['pred_idx'] != [""] else ""
            mutli_formula = '\\begin{array}{l} ' + mutli_formula + ' \end{array}'
            entry['pred'] = mutli_formula
        else:
            entry['pred'] = ''.join([no_ignores_ori_pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [""] else ""

        entry['norm_gt'] = ''.join([no_ignores_gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [""] else ""
        entry['norm_pred'] = ''.join([no_ignores_pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [""] else ""

    for entry in merged_results:
        if isinstance(entry['gt_idx'], list) and entry['gt_idx'] != [""]:
            gt_idx = []
            for i in entry['gt_idx']:
                gt_idx.append(no_ignores_gt_idxs[i])
            entry['gt_idx'] = gt_idx
        if isinstance(entry['pred_idx'], list) and entry['pred_idx'] != [""]:
            pred_idx = []
            for i in entry['pred_idx']:
                pred_idx.append(int(no_ignore_pred_idxs[i]))
            entry['pred_idx'] = pred_idx

    if len(merged_ignore_results) > 0:
        merged_results.extend(merged_ignore_results)

    return merged_results

def normalized_html_table(text):
    def process_table_html(md_i):
        """
        pred_md format edit
        """
        def process_table_html(html_content):
            soup = BeautifulSoup(html_content, 'html.parser')
            th_tags = soup.find_all('th')
            for th in th_tags:
                th.name = 'td'
            thead_tags = soup.find_all('thead')
            for thead in thead_tags:
                thead.unwrap()  # Unwrap() will remove the tag but retain its content
            math_tags = soup.find_all('math')
            for math_tag in math_tags:
                alttext = math_tag.get('alttext', '')
                alttext = f'${alttext}$'
                if alttext:
                    math_tag.replace_with(alttext)
            span_tags = soup.find_all('span')
            for span in span_tags:
                span.unwrap()
            return str(soup)

        table_res=''
        table_res_no_space=''
        if '<table' in md_i.replace(" ","").replace("'",'"'):
            md_i = process_table_html(md_i)
            table_res = html.unescape(md_i).replace('\n', '')
            table_res = unicodedata.normalize('NFKC', table_res).strip()
            pattern = r'<table\b[^>]*>(.*)</table>'
            tables = re.findall(pattern, table_res, re.DOTALL | re.IGNORECASE)
            table_res = ''.join(tables)
            table_res = re.sub('( style=".*?")', "", table_res)
            table_res = re.sub('( height=".*?")', "", table_res)
            table_res = re.sub('( width=".*?")', "", table_res)
            table_res = re.sub('( align=".*?")', "", table_res)
            table_res = re.sub('( class=".*?")', "", table_res)
            table_res = re.sub('</?tbody>',"",table_res)
            
            table_res = re.sub(r'\s+', " ", table_res)
            table_res_no_space = '<html><body><table border="1" >' + table_res.replace(' ','') + '</table></body></html>'
            table_res_no_space = re.sub('colspan="', ' colspan="', table_res_no_space)
            table_res_no_space = re.sub('rowspan="', ' rowspan="', table_res_no_space)
            table_res_no_space = re.sub('border="', ' border="', table_res_no_space)

            table_res = '<html><body><table border="1" >' + table_res + '</table></body></html>'

        return table_res, table_res_no_space
    
    def clean_table(input_str,flag=True):
        if flag:
            input_str = input_str.replace('<sup>', '').replace('</sup>', '')
            input_str = input_str.replace('<sub>', '').replace('</sub>', '')
            input_str = input_str.replace('<span>', '').replace('</span>', '')
            input_str = input_str.replace('<div>', '').replace('</div>', '')
            input_str = input_str.replace('<p>', '').replace('</p>', '')
            input_str = input_str.replace('<spandata-span-identity="">', '')
            input_str = re.sub('<colgroup>.*?</colgroup>','',input_str)
        return input_str
    
    norm_text, _ = process_table_html(text)
    norm_text = clean_table(norm_text)
    return norm_text

def normalized_latex_table(text):
    def latex_template(latex_code):  
        template = r'''
        \documentclass[border=20pt]{article}
        \usepackage{subcaption}
        \usepackage{url}
        \usepackage{graphicx}
        \usepackage{caption}
        \usepackage{multirow}
        \usepackage{booktabs}
        \usepackage{color}
        \usepackage{colortbl}
        \usepackage{xcolor,soul,framed}
        \usepackage{fontspec}
        \usepackage{amsmath,amssymb,mathtools,bm,mathrsfs,textcomp}
        \setlength{\parindent}{0pt}''' + \
        r'''
        \begin{document}
        ''' + \
        latex_code + \
        r'''
        \end{document}'''
    
        return template

    def process_table_latex(latex_code):
        SPECIAL_STRINGS= [
            ['\\\\vspace\\{.*?\\}', ''],
            ['\\\\hspace\\{.*?\\}', ''],
            ['\\\\rule\{.*?\\}\\{.*?\\}', ''],
            ['\\\\addlinespace\\[.*?\\]', ''],
            ['\\\\addlinespace', ''],
            ['\\\\renewcommand\\{\\\\arraystretch\\}\\{.*?\\}', ''],
            ['\\\\arraystretch\\{.*?\\}', ''],
            ['\\\\(row|column)?colors?\\{[^}]*\\}(\\{[^}]*\\}){0,2}', ''],
            ['\\\\color\\{.*?\\}', ''],
            ['\\\\textcolor\\{.*?\\}', ''],
            ['\\\\rowcolor(\\[.*?\\])?\\{.*?\\}', ''],
            ['\\\\columncolor(\\[.*?\\])?\\{.*?\\}', ''],
            ['\\\\cellcolor(\\[.*?\\])?\\{.*?\\}', ''],
            ['\\\\colorbox\\{.*?\\}', ''],
            ['\\\\(tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)', ''],
            [r'\s+', ' '],
            ['\\\\centering', ''],
            ['\\\\begin\\{table\\}\\[.*?\\]', '\\\\begin{table}'],
            ['\t', ''],
            ['@{}', ''],
            ['\\\\toprule(\\[.*?\\])?', '\\\\hline'],
            ['\\\\bottomrule(\\[.*?\\])?', '\\\\hline'],
            ['\\\\midrule(\\[.*?\\])?', '\\\\hline'],
            ['p\\{[^}]*\\}', 'l'],
            ['m\\{[^}]*\\}', 'c'],
            ['\\\\scalebox\\{[^}]*\\}\\{([^}]*)\\}', '\\1'],
            ['\\\\textbf\\{([^}]*)\\}', '\\1'],
            ['\\\\textit\\{([^}]*)\\}', '\\1'],
            ['\\\\cmidrule(\\[.*?\\])?\\(.*?\\)\\{([0-9]-[0-9])\\}', '\\\\cline{\\2}'],
            ['\\\\hline', ''],
            [r'\\multicolumn\{1\}\{[^}]*\}\{((?:[^{}]|(?:\{[^{}]*\}))*)\}', r'\1']
        ]
        pattern = r'\\begin\{tabular\}.*\\end\{tabular\}'
        matches = re.findall(pattern, latex_code, re.DOTALL)
        latex_code = ' '.join(matches)

        for special_str in SPECIAL_STRINGS:
            latex_code = re.sub(fr'{special_str[0]}', fr'{special_str[1]}', latex_code)

        return latex_code
    
    def convert_latex_to_html(latex_content, cache_dir='./temp'):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        uuid_str = str(uuid.uuid1())
        with open(f'{cache_dir}/{uuid_str}.tex', 'w') as f:
            f.write(latex_template(latex_content))

        cmd = ['latexmlc', '--quiet', '--nocomments', f'--log={cache_dir}/{uuid_str}.log',
               f'{cache_dir}/{uuid_str}.tex', f'--dest={cache_dir}/{uuid_str}.html']
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(f'{cache_dir}/{uuid_str}.html', 'r') as f:
                html_content = f.read()

            pattern = r'<table\b[^>]*>(.*)</table>'
            tables = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
            tables = [f'<table>{table}</table>' for table in tables]
            html_content = '\n'.join(tables)
        
        except Exception as e:
            html_content = ''
        
        shutil.rmtree(cache_dir)
        return html_content
    
    html_text = convert_latex_to_html(text)
    normlized_tables = normalized_html_table(html_text)
    return normlized_tables

def normalized_table(text, format='html'):
    if format not in ['html', 'latex']:
        raise ValueError('Invalid format: {}'.format(format))
    else:
        return globals()['normalized_{}_table'.format(format)](text)

def show_result(results):
    from tabulate import tabulate
    for metric_name in results.keys():
        print(f'{metric_name}:')
        score_table = [[k,v] for k,v in results[metric_name].items()]
        print(tabulate(score_table))
        print('='*100)

def sort_nested_dict(d):
    # If it's a dictionary, recursively sort it
    if isinstance(d, dict):
        # Sort the current dictionary
        sorted_dict = {k: sort_nested_dict(v) for k, v in sorted(d.items())}
        return sorted_dict
    # If not a dictionary, return directly
    return d

def get_full_labels_results(samples):
    if not samples:
        return {}
    label_group_dict = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        label_list = []
        if not sample.get("gt_attribute"):
            continue
        for anno in sample["gt_attribute"]:
            for k,v in anno.items():
                label_list.append(k+": "+str(v))
        for label_name in list(set(label_list)):  # Currently if there are merged cases, calculate based on the set of all labels involved after merging
            for metric, score in sample['metric'].items():
                label_group_dict[label_name][metric].append(score)

    result = {}
    result['sample_count'] = {}
    for attribute in label_group_dict.keys():
        for metric, scores in label_group_dict[attribute].items():
            mean_score = sum(scores) / len(scores)
            if not result.get(metric):
                result[metric] = {}
            result[metric][attribute] = mean_score
            result['sample_count'][attribute] = len(scores)
    result = sort_nested_dict(result)
    return result

def get_page_split(samples, page_info):   # Page level metric
    if not page_info:
        return {}
    result_list = defaultdict(list)
    for sample in samples:
        img_name = sample['img_id'] if sample['img_id'].endswith('.jpg') or sample['img_id'].endswith('.png') else '_'.join(sample['img_id'].split('_')[:-1])
        page_info_s = page_info[img_name]
        if not sample.get('metric'):
            continue
        for metric, score in sample['metric'].items():
            gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
            pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
            result_list[metric].append({
                'image_name': img_name,
                'metric': metric,
                'attribute': 'ALL',
                'score': score,
                'upper_len': max(len(gt), len(pred))
            })
            for k,v in page_info_s.items():
                if isinstance(v, list): # special issue
                    for special_issue in v:
                        if 'table' not in special_issue:  # Table-related special fields have duplicates
                            result_list[metric].append({
                                'image_name': img_name,
                                'metric': metric,
                                'attribute': special_issue,
                                'score': score,
                                'upper_len': max(len(gt), len(pred))
                            })
                else:
                    result_list[metric].append({
                        'image_name': img_name,
                        'metric': metric,
                        'attribute': k+": "+str(v),
                        'score': score,
                        'upper_len': max(len(gt), len(pred))
                    })
    
    # Page level logic, accumulation is only done within pages, and mean operation is performed between pages
    result = {}
    if result_list.get('Edit_dist'):   # Only Edit-DIst needs to perform page level calculations
        df = pd.DataFrame(result_list['Edit_dist'])
        up_total_avg = df.groupby(["image_name", "attribute"]).apply(lambda x: (x["score"]*x['upper_len']).sum() / x['upper_len'].sum()).groupby('attribute').mean()  # At page level, accumulate edits, denominator is sum of max(gt, pred) from each sample
        # up_total_avg = df.groupby(["attribute"]).apply(lambda x: (x["score"]*x['upper_len']).sum() / x['upper_len'].sum()) # whole_level
        result['Edit_dist'] = up_total_avg.to_dict()
    for metric in result_list.keys():
        if metric == 'Edit_dist':
            continue
        df = pd.DataFrame(result_list[metric])
        page_avg = df.groupby(["image_name", "attribute"]).apply(lambda x: x["score"].mean()).groupby('attribute').mean()
        result[metric] = page_avg.to_dict()

    result = sort_nested_dict(result)
    return result