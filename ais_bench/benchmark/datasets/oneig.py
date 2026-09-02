"""
OneIG 数据集模块

OneIG-Bench 是一个面向文生图模型的综合评测基准，包含5个子任务：
- Alignment: 对齐评估
- Text: 文本评估
- Reasoning: 推理评估
- Style: 风格评估
- Diversity: 多样性评估

分类说明：
- EN 模式：5 个分类（Anime_Stylization, Portrait, General_Object, Text_Rendering, Knowledge_Reasoning）
- ZH 模式：6 个分类（额外包含 Multilingualism）
"""

import glob
import json
import os
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.registry import LOAD_DATASET


# 任务目录映射
# Alignment: anime, human, object (EN) / + multilingualism (ZH)
# Text: text
# Reasoning: reasoning
# Style: anime
# Diversity: anime, human, object, text, reasoning (EN) / + multilingualism (ZH)
TASK_CLASS_MAP = {
    'alignment': [
        ("Anime_Stylization", "anime"),
        ("Portrait", "human"),
        ("General_Object", "object")
    ],
    'text': [("Text_Rendering", "text")],
    'reasoning': [("Knowledge_Reasoning", "reasoning")],
    'style': [("Anime_Stylization", "anime")],
    'diversity': [
        ("Anime_Stylization", "anime"),
        ("Portrait", "human"),
        ("General_Object", "object"),
        ("Text_Rendering", "text"),
        ("Knowledge_Reasoning", "reasoning")
    ]
}


@LOAD_DATASET.register_module()
class OneIGDataset(BaseDataset):
    """
    OneIG 数据集类

    支持5个子任务的图片数据集加载，每个任务对应不同的辅助数据。

    Args:
        path: str - 图片根目录
        task_type: str - 任务类型（alignment/text/reasoning/style/diversity）
        model_names: List[str] - 模型名称列表
        image_grids: List[str] - 网格配置列表（格式：'rows,cols'）
        mode: str - 语言模式（'EN' 或 'ZH'）
        aux_data_paths: dict - 辅助数据路径配置

    示例:
        >>> config = dict(
        ...     type=OneIGDataset,
        ...     path="/path/to/oneig/images",
        ...     task_type="alignment",
        ...     model_names=["model_a"],
        ...     image_grids=["2,2"],
        ...     mode="EN",
        ...     aux_data_paths=dict(
        ...         question_dependency_dir="third_party/oneig/alignment/Q_D"
        ...     )
        ... )
    """

    def load(
        self,
        path: str,
        task_type: str = "alignment",
        model_names: Optional[List[str]] = None,
        image_grids: Optional[List[str]] = None,
        mode: str = "EN",
        aux_data_paths: Optional[Dict] = None,
        **kwargs
    ) -> Dataset:
        """
        加载 OneIG 数据集

        Args:
            path: 图片根目录
            task_type: 任务类型
            model_names: 模型名称列表
            image_grids: 网格配置列表
            mode: 语言模式
            aux_data_paths: 辅助数据路径

        Returns:
            Dataset: 处理后的数据集
        """
        self.logger.info(f"Loading OneIG dataset: task_type={task_type}, mode={mode}")

        self.task_type = task_type
        self.mode = mode
        self.model_names = model_names or ["default"]
        self.image_grids = image_grids or ["2,2"]
        if len(self.model_names) != len(self.image_grids):
            raise ValueError(
                f"model_names length ({len(self.model_names)}) must equal "
                f"image_grids length ({len(self.image_grids)})")
        self.aux_data_paths = aux_data_paths or {}

        # 加载辅助数据
        aux_data = self._load_aux_data()

        # 加载图片数据
        data = self._load_image_data(path, aux_data)

        # 数据预处理
        processed_data = self._preprocess_data(data)

        self.logger.info(f"Loaded {len(processed_data)} samples")

        return Dataset.from_list(processed_data)

    def _get_task_class_iter(self) -> List[tuple]:
        """
        根据任务类型获取要遍历的目录列表

        Returns:
            List[tuple]: (class_name, class_item) 列表
        """
        # 从预定义的映射中获取目录列表
        class_iter = TASK_CLASS_MAP.get(self.task_type, [])
        
        # ZH 模式：alignment 和 diversity 需要额外包含 multilingualism
        if self.mode == 'ZH' and self.task_type in ('alignment', 'diversity'):
            class_iter = class_iter + [("Multilingualism", "multilingualism")]
        
        return class_iter

    def _load_aux_data(self) -> Dict:
        """
        加载任务相关的辅助数据

        Returns:
            dict: 辅助数据字典
        """
        aux_data = {}
        suffix = '_zh' if self.mode == 'ZH' else ''

        if self.task_type == 'alignment':
            # 按 class_item 分层存储，避免不同类别 JSON 的 key 冲突覆盖
            # 同时保留原始 JSON 数据，不在 Dataset 层解析嵌套 dict
            # （PyArrow struct schema 推断会对缺失 key 填充 None）
            qd_dir = self.aux_data_paths.get('question_dependency_dir')
            if qd_dir:
                aux_data['alignment_data'] = {}
                class_iter = self._get_task_class_iter()
                for class_name, class_item in class_iter:
                    json_path = os.path.join(qd_dir, f"{class_item}{suffix}.json")
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            loaded = json.load(f)
                            aux_data['alignment_data'][class_item] = loaded
                            self.logger.debug(f"Loaded alignment data for {class_item}: {len(loaded)} entries")

        elif self.task_type == 'reasoning':
            # 加载参考答案JSON
            gt_path = self.aux_data_paths.get('gt_answer_path')
            if gt_path and os.path.exists(gt_path):
                with open(gt_path, 'r', encoding='utf-8') as f:
                    aux_data['gt_answers'] = json.load(f)

        elif self.task_type == 'text':
            # 加载文本内容CSV
            # CSV 列名：id, prompt_en, text_content
            csv_path = self.aux_data_paths.get('text_content_csv_path')
            if csv_path and os.path.exists(csv_path):
                # 指定 dtype=str 避免 id 列被推断为 int64
                df = pd.read_csv(csv_path, dtype=str)
                # 使用 text_content 列
                if 'id' in df.columns and 'text_content' in df.columns:
                    aux_data['text_contents'] = dict(zip(df['id'], df['text_content']))
                else:
                    self.logger.warning(f"Text CSV missing required columns: id, text_content")

        elif self.task_type == 'style':
            # 加载风格标签CSV
            csv_path = self.aux_data_paths.get('style_csv_path')
            self.logger.info(f"[Style] Loading style CSV from: {csv_path}")
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path, dtype=str)
                self.logger.info(f"[Style] CSV loaded, rows={len(df)}, columns={list(df.columns)}")

                if 'id' in df.columns and 'class' in df.columns:
                    style_labels = {}
                    for idx, row in df.iterrows():
                        style_label = row['class']
                        # 格式转换：lower().replace(' ', '_')
                        style_labels[row['id']] = str(style_label).lower().replace(' ', '_')
                    
                    aux_data['style_labels'] = style_labels
                    self.logger.info(f"[Style] Loaded {len(style_labels)} style labels")
            else:
                self.logger.warning(f"[Style] Style CSV path not found: {csv_path}")

        return aux_data
    
    def _is_valid_style_label(self, label) -> bool:
        """
        检查风格标签是否有效

        Args:
            label: 风格标签值

        Returns:
            bool: 是否为有效风格标签
        """
        # 1. 检查 pandas NaN
        if pd.isna(label):
            return False

        # 2. 转换为字符串并检查
        label_str = str(label).strip()

        # 3. 检查空字符串或 "nan"
        if label_str == '' or label_str[:3].lower() == 'nan':
            return False

        return True

    def _load_image_data(self, path: str, aux_data: Dict) -> List[Dict]:
        """
        加载图片数据

        Args:
            path: 图片根目录
            aux_data: 辅助数据

        Returns:
            List[dict]: 图片数据列表
        """
        data = []
        
        # Style 任务调试：打印辅助数据状态
        if self.task_type == 'style':
            style_labels = aux_data.get('style_labels', {})
            self.logger.info(f"[Style] aux_data loaded, style_labels count: {len(style_labels)}")

        # 支持的图片扩展名
        supported_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')

        for model_idx, model_name in enumerate(self.model_names):
            grid_rows, grid_cols = map(
                int,
                self.image_grids[model_idx].split(',')
            )
            self.logger.info(f"[Load] Processing model: {model_name}, grid: {grid_rows}x{grid_cols}")

            # 根据任务类型选择目录
            class_iter = self._get_task_class_iter()

            for class_name, class_item in class_iter:
                img_dir = os.path.join(path, class_item, model_name)
                if not os.path.exists(img_dir):
                    self.logger.info(f"Image directory not found: {img_dir}")
                    continue

                # 匹配所有支持的图片格式
                img_patterns = [
                    os.path.join(img_dir, f'*{ext}')
                    for ext in supported_extensions
                ]

                img_paths = []
                for pattern in img_patterns:
                    img_paths.extend(glob.glob(pattern))

                # 排序图片路径
                img_paths = sorted(img_paths)

                if not img_paths:
                    self.logger.info(f"No images found in: {img_dir}")
                    continue
                
                self.logger.info(f"[Load] Found {len(img_paths)} images in {img_dir}")

                for img_path in img_paths:
                    if not os.path.isfile(img_path):
                        self.logger.info(f"Skipping non-file path: {img_path}")
                        continue

                    img_basename = os.path.basename(img_path)
                    # 从文件名提取样本ID（取前3个字符）
                    sample_id = img_basename[:3]

                    item = {
                        'id': sample_id,
                        'class_item': class_item,
                        'class_name': class_name,
                        'model_name': model_name,
                        'image_path': img_path,
                        'grid_rows': grid_rows,
                        'grid_cols': grid_cols,
                        'mode': self.mode
                    }

                    # 添加任务特定字段
                    task_fields = self._get_task_specific_fields(sample_id, aux_data, class_item)
                    item.update(task_fields)
                    
                    # Style 任务调试：打印样本信息
                    if self.task_type == 'style':
                        self.logger.info(f"[Style] Sample {sample_id}: style_label={task_fields.get('style_label', 'N/A')}")

                    # 检查是否应该跳过该样本
                    if self._should_skip_sample(sample_id, task_fields):
                        continue

                    data.append(item)

        self.logger.info(f"Loaded {len(data)} images from {path}")
        
        # Style 任务调试：打印最终数据统计
        if self.task_type == 'style':
            self.logger.info(f"[Style] Final data count: {len(data)}")
        
        return data

    def _should_skip_sample(self, sample_id: str, task_fields: Dict) -> bool:
        """
        判断是否应该跳过该样本（异常数据检测）
        
        Args:
            sample_id: 样本ID
            task_fields: 任务特定字段
        
        Returns:
            bool: 是否应该跳过该样本
        """
        # Style任务：检测无效风格标签
        if self.task_type == 'style':
            style_label = task_fields.get('style_label', '')
            label_repr = repr(style_label)
            if not self._is_valid_style_label(style_label):
                self.logger.info(f"Skipping style sample: {sample_id}, label={label_repr}")
                return True
        
        # Reasoning任务：检测空参考答案
        if self.task_type == 'reasoning':
            gt_answer = task_fields.get('gt_answer', '')
            if not gt_answer or gt_answer == '':
                self.logger.warning(f"Skipping reasoning sample with empty gt_answer: {sample_id}")
                return True
        
        return False

    def _get_task_specific_fields(
        self,
        sample_id: str,
        aux_data: Dict,
        class_item: str = None
    ) -> Dict:
        """
        根据任务类型获取特定字段

        Args:
            sample_id: 样本ID
            aux_data: 辅助数据
            class_item: 类别标识（alignment 任务需要，用于按类别查找数据）

        Returns:
            dict: 任务特定字段
        """
        fields = {}

        if self.task_type == 'alignment':
            # 按 class_item 查找对应类别的数据，避免跨类别 key 冲突
            # question/dependency 保持 JSON 字符串，不在 Dataset 层解析
            # （PyArrow struct schema 推断会对缺失 key 填充 None）
            alignment_data = aux_data.get('alignment_data', {})
            class_data = alignment_data.get(class_item, {}) if class_item else {}
            item_data = class_data.get(sample_id, {})
            fields['question'] = item_data.get('question', '{}')
            fields['dependency'] = item_data.get('dependency', '{}')

        elif self.task_type == 'reasoning':
            gt_answer = aux_data.get('gt_answers', {}).get(sample_id, '')
            fields['gt_answer'] = gt_answer

        elif self.task_type == 'text':
            fields['expected_text'] = aux_data.get('text_contents', {}).get(sample_id, '')

        elif self.task_type == 'style':
            fields['style_label'] = aux_data.get('style_labels', {}).get(sample_id, '')

        return fields

    def _preprocess_data(self, data: List[Dict]) -> List[Dict]:
        """
        数据预处理

        Args:
            data: 原始数据列表

        Returns:
            List[dict]: 预处理后的数据
        """
        for item in data:
            # 清理字符串字段（统一处理）
            for field in ('gt_answer', 'expected_text', 'style_label'):
                if field in item and isinstance(item[field], str):
                    item[field] = item[field].strip()

        return data

