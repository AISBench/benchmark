"""OneIG 文本评估器 - LLM-as-Judge 模式"""
import os
import shutil
import tempfile

from ais_bench.benchmark.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import ICL_EVALUATORS
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES
from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
    split_image_grid,
    rm_error,
    ONEIG_DTYPE_MAP,
    OneIGJudgeInferencer,
)


@ICL_EVALUATORS.register_module()
class OneIGTextEvaluator(BaseEvaluator):
    """文本评估器 - LLM-as-Judge 模式。

    使用 OneIGJudgeInferencer 进行 Judge 推理，
    对图片进行 OCR 识别，然后与预期文本比较。

    评测逻辑：
    1. 逐切分图 OCR，每张切分图独立计算 ED/CR/WAC
    2. 使用 preprocess_string、clean_and_remove_hallucinations、
       levenshtein_distance、calculate_char_match_ratio
    3. 最终分数 = 1 - min(MAX_EDIT_DISTANCE, ED) * (1 - CR) * (1 - WAC) / MAX_EDIT_DISTANCE

    Args:
        judge_model_path: str - Judge 模型路径
        judge_device: str - Judge 模型运行设备
        judge_dtype: str - Judge 模型数据类型
        judge_batch_size: int - Judge 批量推理批次大小
        judge_use_flash_attention: bool - Judge 是否使用 Flash Attention
        judge_seed: int - Judge 模型随机种子（默认 42）
        mode: str - 语言模式（'EN' 或 'ZH'），影响 MAX_EDIT_DISTANCE
    """

    def __init__(
        self,
        judge_model_path: str = "",
        judge_device: str = "cuda",
        judge_dtype: str = "bfloat16",
        judge_batch_size: int = 8,
        judge_use_flash_attention: bool = True,
        judge_seed: int = 42,
        mode: str = "EN",
        **kwargs
    ):
        super().__init__()
        self.judge_model_path = judge_model_path
        self.judge_device = judge_device
        self.judge_dtype = judge_dtype
        self.judge_batch_size = judge_batch_size
        self.judge_use_flash_attention = judge_use_flash_attention
        self.judge_seed = judge_seed
        self.mode = mode
        self._inferencer = None

    def _ensure_inferencer(self):
        """延迟加载 Judge 推理器"""
        if self._inferencer is None and self.judge_model_path:
            import torch
            dtype = ONEIG_DTYPE_MAP.get(self.judge_dtype, torch.bfloat16)

            self._inferencer = OneIGJudgeInferencer(
                model_path=self.judge_model_path,
                device=self.judge_device,
                dtype=dtype,
                use_flash_attention=self.judge_use_flash_attention,
                batch_size=self.judge_batch_size,
                seed=self.judge_seed,
            )
            self.logger.info("[Text] Judge inferencer loaded successfully")

    @staticmethod
    def _preprocess_string(s):
        """文本预处理

        清理非目标字符，保留中文、英文、数字及常见欧洲语言字符。
        """
        import re
        cleaned = re.sub(
            r"[^\u4e00-\u9fa5a-zA-Z0-9\s"
            r"\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb"
            r"\u00ee\u00ef\u00f4\u00f6\u00f9\u00fb\u00fc\u00e7"
            r"\u00c0\u00c2\u00c4\u00c9\u00c8\u00ca\u00cb"
            r"\u00ce\u00cf\u00d4\u00d6\u00d9\u00db\u00dc\u00c7]",
            '', s)

        # 检查是否包含中文
        if re.search('[\u4e00-\u9fff]', cleaned):
            pattern = re.compile(
                r"[\u4e00-\u9fa5a-zA-Z0-9"
                r"\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb"
                r"\u00ee\u00ef\u00f4\u00f6\u00f9\u00fb\u00fc\u00e7"
                r"\u00c0\u00c2\u00c4\u00c9\u00c8\u00ca\u00cb"
                r"\u00ce\u00cf\u00d4\u00d6\u00d9\u00db\u00dc\u00c7]")
            s = ''.join(pattern.findall(s))
            return s.strip()

        normalized = re.sub(r'\s+', ' ', cleaned)
        return normalized.strip()

    @staticmethod
    def _clean_and_remove_hallucinations(texts):
        """清洗 OCR 幻觉

        移除常见的 OCR 错误关键词。
        """
        keywords_list = ["addCriterion", "No text recognized."]
        for keyword in keywords_list:
            texts = [
                text.replace(keyword, "")
                    .replace(f"\n{keyword}", "")
                    .replace(f"{keyword}\n", "")
                for text in texts
            ]
        return texts

    @staticmethod
    def _levenshtein_distance(s1, s2):
        """编辑距离计算

        使用动态规划计算两个字符串之间的编辑距离。
        """
        import numpy as np
        matrix = np.zeros((len(s1) + 1, len(s2) + 1))

        for i in range(len(s1) + 1):
            matrix[i][0] = i
        for j in range(len(s2) + 1):
            matrix[0][j] = j
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if s1[i - 1] == s2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,
                    matrix[i][j - 1] + 1,
                    matrix[i - 1][j - 1] + cost
                )

        return matrix[len(s1)][len(s2)]

    @staticmethod
    def _calculate_char_match_ratio(text_gt, ocr_str):
        """字符匹配率计算

        计算预期文本与 OCR 结果的字符匹配情况。
        中文按字符匹配，英文按单词匹配。
        """
        import re
        from collections import Counter

        # 检查是否包含中文
        if re.search('[\u4e00-\u9fff]', text_gt):
            gt_counter = Counter(text_gt)
            ocr_counter = Counter(ocr_str)
            total_match_count = sum((gt_counter & ocr_counter).values())
            ratio = total_match_count / len(text_gt) if len(text_gt) > 0 else 0.0
        else:
            words_gt = text_gt.split()
            words_ocr = ocr_str.split()
            gt_counter = Counter(words_gt)
            ocr_counter = Counter(words_ocr)
            total_match_count = sum((gt_counter & ocr_counter).values())
            total_gt_count = len(words_gt)
            ratio = total_match_count / total_gt_count if total_gt_count > 0 else 0.0
        return total_match_count, ratio, sum(gt_counter.values())

    def score(self, predictions, references, test_set=None, **kwargs):
        if test_set is None:
            self.logger.error(UTILS_CODES.UNKNOWN_ERROR, "test_set is required for OneIGTextEvaluator")
            return {'accuracy': 0.0, 'details': []}

        # 加载 Judge 推理器
        self._ensure_inferencer()

        if self._inferencer is None:
            self.logger.error(UTILS_CODES.UNKNOWN_ERROR, "[Text] Judge inferencer not available")
            return {'accuracy': 0.0, 'details': []}

        self.logger.info(f"[Text] Starting evaluation, {len(test_set)} samples, mode={self.mode}")

        # EN: MAX_EDIT_DISTANCE = 100; ZH: MAX_EDIT_DISTANCE = 50
        MAX_EDIT_DISTANCE = 50 if self.mode == 'ZH' else 100

        cache_dir = tempfile.mkdtemp()

        # 全局累积 ED/CR/WAC
        edit_distances = []
        completion_ratios = []
        match_word_counts = []
        gt_word_counts = []

        prompt_scores = {}  # sample_id -> [ED_avg, CR_avg, WAC_avg] or None
        trace_data = {}  # 落盘溯源数据：sample_id -> dict

        try:
            for i, item in enumerate(test_set):
                sample_id = item.get('id', f'sample_{i}')
                image_path = item.get('image_path', '')
                expected_text = item.get('expected_text', '')
                grid_rows = item.get('grid_rows', 2)
                grid_cols = item.get('grid_cols', 2)

                if not image_path or not os.path.exists(image_path):
                    prompt_scores[sample_id] = None
                    continue

                # 切分图片
                split_img_list = split_image_grid(
                    image_path, (grid_rows, grid_cols), cache_dir)

                if len(split_img_list) == 0:
                    prompt_scores[sample_id] = None
                    continue

                # 动态调整 max_new_tokens
                word_count = len(expected_text.split()) if expected_text else 0
                if word_count > 60:
                    max_new_tokens = 256
                else:
                    max_new_tokens = 128

                # 预处理预期文本
                text_gt_preprocessed = self._preprocess_string(expected_text)

                # 逐切分图 OCR
                ocr_results = self._inferencer.infer_ocr(
                    split_img_list, max_new_tokens)

                # 清洗 OCR 幻觉
                text_ocr_list = self._clean_and_remove_hallucinations(
                    ocr_results)

                # 逐切分图计算 ED/CR/WAC
                ED_score = []
                CR_score = []
                WAC_score = []
                ocr_details = []  # 落盘溯源：逐切分图 OCR 详情

                for ocr_idx, text_ocr in enumerate(text_ocr_list):
                    text_ocr_preprocessed = self._preprocess_string(text_ocr)

                    # 计算编辑距离
                    edit_distance = self._levenshtein_distance(
                        text_ocr_preprocessed, text_gt_preprocessed)

                    # 完成率：编辑距离为0时为1，否则为0
                    completion_ratio = 1 if edit_distance == 0 else 0

                    # 计算字符匹配率
                    match_word_count, text_word_accuracy, gt_word_count = \
                        self._calculate_char_match_ratio(
                            text_gt_preprocessed, text_ocr_preprocessed)

                    edit_distances.append(edit_distance)
                    completion_ratios.append(completion_ratio)
                    match_word_counts.append(match_word_count)
                    gt_word_counts.append(gt_word_count)

                    ED_score.append(edit_distance)
                    CR_score.append(completion_ratio)
                    WAC_score.append(text_word_accuracy)

                    ocr_details.append({
                        'grid_idx': ocr_idx,
                        'ocr_raw': ocr_results[ocr_idx],
                        'ocr_cleaned': text_ocr,
                        'ocr_preprocessed': text_ocr_preprocessed,
                        'ED': edit_distance,
                        'CR': completion_ratio,
                        'WAC': text_word_accuracy,
                    })

                # 记录该样本的平均分数
                prompt_scores[sample_id] = [
                    (sum(ED_score) / len(ED_score)).item()
                    if hasattr(sum(ED_score) / len(ED_score), 'item')
                    else sum(ED_score) / len(ED_score),
                    sum(CR_score) / len(CR_score),
                    sum(WAC_score) / len(WAC_score),
                ]

                # 落盘溯源：保存中间数据
                trace_data[sample_id] = {
                    'image_path': image_path,
                    'grid': f"{grid_rows}x{grid_cols}",
                    'num_splits': len(split_img_list),
                    'expected_text': expected_text,
                    'expected_text_preprocessed': text_gt_preprocessed,
                    'ocr_prompt': (
                        "Recognize the text in the image, only reply with the text content, "
                        "but avoid repeating previously mentioned content. "
                        "If no text is recognized, please reply with 'No text recognized'."
                    ),
                    'ocr_details': ocr_details,
                }

        finally:
            shutil.rmtree(cache_dir, onerror=rm_error)

        # 计算整体分数
        if not edit_distances:
            return {'accuracy': 0.0, 'details': []}

        ED = sum(edit_distances) / len(edit_distances)
        CR = sum(completion_ratios) / len(completion_ratios)
        WAC = sum(match_word_counts) / sum(gt_word_counts) if sum(gt_word_counts) > 0 else 0.0

        # text score = 1 - min(MAX_EDIT_DISTANCE, ED) * (1 - CR) * (1 - WAC) / MAX_EDIT_DISTANCE
        text_score = 1 - min(MAX_EDIT_DISTANCE, ED) * (1 - CR) * (1 - WAC) / MAX_EDIT_DISTANCE

        results = []
        for i, (sample_id, score) in enumerate(prompt_scores.items()):
            if score is not None:
                # 从 test_set 中查找对应的 class_item
                class_item = ''
                for item in test_set:
                    if item.get('id') == sample_id:
                        class_item = item.get('class_item', '')
                        break
                results.append({
                    'id': sample_id,
                    'class_item': class_item,
                    'ED': score[0],
                    'CR': score[1],
                    'WAC': score[2],
                    **trace_data.get(sample_id, {})
                })

        self.logger.info(f"[Text] Evaluation complete, score={text_score * 100:.2f}, ED={ED:.2f}, CR={CR:.2f}, WAC={WAC:.2f}")
        return {
            'accuracy': text_score * 100,
            'details': results,
            'ED': ED,
            'CR': CR,
            'WAC': WAC,
        }
