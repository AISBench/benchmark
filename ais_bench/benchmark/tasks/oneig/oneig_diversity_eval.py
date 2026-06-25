"""OneIG 多样性评估器 - DreamSim 模式"""
import os
import shutil
import tempfile

from ais_bench.benchmark.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import ICL_EVALUATORS
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES
from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
    split_image_grid,
    rm_error,
)


@ICL_EVALUATORS.register_module()
class OneIGDiversityEvaluator(BaseEvaluator):
    """
    多样性评估器 - DreamSim 模式

    使用 DreamSim 计算同一模型生成的图片之间的多样性。

    Args:
        device: str - 运行设备（'cuda' 或 'cpu'）
        oneig_root: str - OneIG项目根目录，DreamSim权重默认缓存到 {oneig_root}/models
        dreamsim_path: str - DreamSim模型缓存目录（覆盖oneig_root/models的默认值）

    延迟加载机制：
        - 模型在首次调用score()时加载，不在__init__中加载
    """

    def __init__(self, device="cuda", oneig_root="", dreamsim_path=None, **kwargs):
        super().__init__()
        self.device = device
        self.oneig_root = oneig_root
        self.dreamsim_path = dreamsim_path
        self.dreamsim = None

    def _ensure_model(self):
        """延迟加载DreamSim模型"""
        if self.dreamsim is None:
            try:
                from dreamsim import dreamsim

                # cache_dir 优先级：dreamsim_path > oneig_root/models > ./models
                if self.dreamsim_path and os.path.exists(self.dreamsim_path):
                    cache_dir = self.dreamsim_path
                elif self.oneig_root:
                    cache_dir = os.path.join(self.oneig_root, "models")
                else:
                    cache_dir = "./models"

                self.dreamsim, self.preprocess = dreamsim(
                    pretrained=True,
                    device=self.device,
                    cache_dir=cache_dir
                )
                self.logger.info(f"DreamSim model loaded successfully (cache_dir: {cache_dir})")
            except ImportError as e:
                self.logger.error(UTILS_CODES.DEPENDENCY_MODULE_IMPORT_ERROR, f"Failed to import DreamSim: {e}")
                raise

    def score(self, predictions, references, test_set=None, **kwargs):
        if test_set is None:
            self.logger.error(UTILS_CODES.UNKNOWN_ERROR, "test_set is required for OneIGDiversityEvaluator")
            return {'accuracy': 0.0, 'details': []}

        self._ensure_model()

        self.logger.info(f"[Diversity] Starting evaluation, {len(test_set)} samples")
        cache_dir = tempfile.mkdtemp()

        prompt_scores = {}
        class_scores = {}
        trace_data = {}  # 落盘溯源数据：prompt_id -> dict

        try:
            for i, item in enumerate(test_set):
                sample_id = item.get('id', f'sample_{i}')
                class_item = item.get('class_item', '')
                image_path = item.get('image_path', '')
                grid_rows = item.get('grid_rows', 2)
                grid_cols = item.get('grid_cols', 2)
                self.logger.debug(f"[Diversity] Processing {class_item}_{sample_id}")

                if not os.path.exists(image_path):
                    self.logger.warning(f"Image not found: {image_path}")
                    continue

                try:
                    split_img_list = split_image_grid(
                        image_path,
                        (grid_rows, grid_cols),
                        cache_dir
                    )

                    if len(split_img_list) <= 1:
                        continue

                    score = []
                    pairwise_distances = []  # 落盘溯源：逐对切分图距离
                    from PIL import Image
                    for si in range(len(split_img_list)):
                        for sj in range(si+1, len(split_img_list)):
                            img1 = self.preprocess(Image.open(split_img_list[si])).to(self.device)
                            img2 = self.preprocess(Image.open(split_img_list[sj])).to(self.device)
                            dist = self.dreamsim(img1, img2)
                            dist_val = dist.item()
                            score.append(dist_val)
                            pairwise_distances.append({
                                'grid_i': si,
                                'grid_j': sj,
                                'distance': dist_val,
                            })

                    avg_score = sum(score) / len(score) if score else None

                    prompt_id = f"{class_item}_{sample_id}"
                    prompt_scores[prompt_id] = avg_score

                    if avg_score is not None:
                        if class_item not in class_scores:
                            class_scores[class_item] = []
                        class_scores[class_item].append(avg_score)

                        # 落盘溯源：保存中间数据
                        trace_data[prompt_id] = {
                            'image_path': image_path,
                            'grid': f"{grid_rows}x{grid_cols}",
                            'num_splits': len(split_img_list),
                            'pairwise_distances': pairwise_distances,
                        }

                except Exception as e:
                    self.logger.error(UTILS_CODES.UNKNOWN_ERROR, f"Error processing {sample_id}: {e}")
                    self.logger.error(UTILS_CODES.UNKNOWN_ERROR, f"[Diversity] Error context: image_path={image_path}, class={class_item}, grid={grid_rows}x{grid_cols}")
                    prompt_scores[f"{class_item}_{sample_id}"] = None
        finally:
            shutil.rmtree(cache_dir, onerror=rm_error)

        valid_scores = [s for s in prompt_scores.values() if s is not None]
        overall_score = (
            sum(valid_scores) / len(valid_scores) * 100
            if valid_scores else 0.0
        )

        results = []
        for prompt_id, score in prompt_scores.items():
            if score is not None:
                # prompt_id 格式为 "class_item_sample_id"，提取 class_item
                class_item = prompt_id.split('_')[0] if '_' in prompt_id else ''
                results.append({
                    'id': prompt_id,
                    'class_item': class_item,
                    'score': score,
                    **trace_data.get(prompt_id, {})
                })

        self.logger.info(f"[Diversity] Evaluation complete, accuracy={overall_score:.2f}, valid_samples={len(valid_scores)}, classes={len(class_scores)}")
        return {
            'accuracy': overall_score,
            'details': results,
            'class_scores': class_scores
        }
