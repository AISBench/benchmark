"""OneIG 推理评估器 - LLM2CLIP 模式"""
import os
import shutil
import tempfile

from ais_bench.benchmark.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import ICL_EVALUATORS
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES
from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
    split_image_grid,
    rm_error,
    ensure_oneig_path,
)


@ICL_EVALUATORS.register_module()
class OneIGReasoningEvaluator(BaseEvaluator):
    """
    推理评估器 - LLM2CLIP 模式

    使用 LLM2CLIP 计算图片与参考文本的相似度。

    Args:
        llm2clip_cfg: dict - LLM2CLIP模型配置
            - processor_model: CLIP处理器模型路径
            - clip_model: CLIP模型路径
            - llm_model: LLM模型路径
        device: str - 运行设备（'cuda' 或 'cpu'）

    延迟加载机制：
        - 模型在首次调用score()时加载，不在__init__中加载
    """

    def __init__(self, oneig_root='', llm2clip_cfg=None, device="cuda", **kwargs):
        super().__init__()
        self.oneig_root = oneig_root
        self.llm2clip_cfg = llm2clip_cfg or {}
        self.device = device
        self.model = None

    def _ensure_model(self):
        """延迟加载LLM2CLIP模型"""
        if self.model is None:
            try:
                ensure_oneig_path(self.oneig_root)
                from scripts.utils.inference import LLM2CLIP

                # 使用配置传入的本地路径
                self.model = LLM2CLIP(
                    processor_model=self.llm2clip_cfg.get('processor_model', ''),
                    model_name=self.llm2clip_cfg.get('clip_model', ''),
                    llm_model_name=self.llm2clip_cfg.get('llm_model', ''),
                    device=self.device
                )
                self.logger.info("LLM2CLIP model loaded successfully")
            except ImportError as e:
                self.logger.error(UTILS_CODES.DEPENDENCY_MODULE_IMPORT_ERROR, f"Failed to import LLM2CLIP: {e}")
                raise

    def score(self, predictions, references, test_set=None, **kwargs):
        if test_set is None:
            self.logger.error(UTILS_CODES.UNKNOWN_ERROR, "test_set is required for OneIGReasoningEvaluator")
            return {'accuracy': 0.0, 'details': []}

        self._ensure_model()

        self.logger.info(f"[Reasoning] Starting evaluation, {len(test_set)} samples")
        cache_dir = tempfile.mkdtemp()

        results = []

        try:
            for i, item in enumerate(test_set):
                sample_id = item.get('id', f'sample_{i}')
                image_path = item.get('image_path', '')
                gt_answer = item.get('gt_answer', '')
                grid_rows = item.get('grid_rows', 2)
                grid_cols = item.get('grid_cols', 2)
                self.logger.debug(f"[Reasoning] Processing {sample_id}, gt_answer={repr(gt_answer[:80])}")

                # 校验：如果参考答案为空，跳过该样本（防御性检查）
                if not gt_answer:
                    self.logger.warning(f"Empty reference answer for sample {sample_id}, skipping")
                    results.append({
                        'id': sample_id,
                        'class_item': item.get('class_item', ''),
                        'subject': item.get('subject', ''),
                        'model_name': item.get('model_name', ''),
                        'score': None,
                        'image_path': image_path,
                        'grid': f"{grid_rows}x{grid_cols}",
                        'num_splits': 0,
                        'gt_answer': gt_answer,
                        'similarity_details': [],
                    })
                    continue

                if not os.path.exists(image_path):
                    self.logger.warning(f"Image not found: {image_path}")
                    results.append({
                        'id': sample_id,
                        'class_item': item.get('class_item', ''),
                        'subject': item.get('subject', ''),
                        'model_name': item.get('model_name', ''),
                        'score': None,
                        'image_path': image_path,
                        'grid': f"{grid_rows}x{grid_cols}",
                        'num_splits': 0,
                        'gt_answer': gt_answer,
                        'similarity_details': [],
                    })
                    continue

                try:
                    split_img_list = split_image_grid(
                        image_path,
                        (grid_rows, grid_cols),
                        cache_dir
                    )

                    # 批量计算所有切分图片的相似度
                    scores = self.model.text_img_similarity_score(split_img_list, gt_answer)

                    # 过滤 None 值并处理空列表
                    if scores:
                        scores = [x for x in scores if x is not None]
                        avg_score = sum(scores) / len(scores) if scores else None
                    else:
                        avg_score = None

                    # 落盘溯源：保存逐切分图相似度
                    # 注意：LLM2CLIP.text_img_similarity_score() 仅返回相似度分数，
                    # 不返回特征向量，故 img_embed/text_embed 为缺失数据项
                    similarity_details = [
                        {'grid_idx': idx, 'similarity': s}
                        for idx, s in enumerate(scores) if s is not None
                    ] if scores else []
                except Exception as e:
                    self.logger.error(UTILS_CODES.UNKNOWN_ERROR, f"Error processing {sample_id}: {e}")
                    self.logger.error(UTILS_CODES.UNKNOWN_ERROR, f"[Reasoning] Error context: image_path={image_path}, gt_answer={repr(gt_answer[:80])}, grid={grid_rows}x{grid_cols}")
                    avg_score = None
                    split_img_list = []
                    similarity_details = []

                results.append({
                    'id': sample_id,
                    'class_item': item.get('class_item', ''),
                    'subject': item.get('subject', ''),
                    'model_name': item.get('model_name', ''),
                    'score': avg_score,
                    'image_path': image_path,
                    'grid': f"{grid_rows}x{grid_cols}",
                    'num_splits': len(split_img_list),
                    'gt_answer': gt_answer,
                    'similarity_details': similarity_details,
                })
        finally:
            shutil.rmtree(cache_dir, onerror=rm_error)

        # 过滤 None 值后计算平均值
        valid_scores = [r['score'] for r in results if r['score'] is not None]
        overall_score = (
            sum(valid_scores) / len(valid_scores) * 100
            if valid_scores else 0.0
        )

        self.logger.info(f"[Reasoning] Evaluation complete, accuracy={overall_score:.2f}, valid_samples={len(valid_scores)}")
        return {
            'accuracy': overall_score,
            'details': results
        }
