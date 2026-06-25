"""OneIG 对齐评估器 - LLM-as-Judge 模式"""
import json
import os
import shutil
import tempfile
from copy import deepcopy

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
class OneIGAlignmentEvaluator(BaseEvaluator):
    """对齐评估器 - LLM-as-Judge 模式。

    使用 OneIGJudgeInferencer 进行 Judge 推理，
    对图片进行 Yes/No 问答，计算对齐准确率。
    支持问题依赖关系（dependency）。

    评测逻辑：
    1. 对每个样本，切分图片后对每个问题调用 infer_semantic(split_img_list, question)
       得到 N 个切分图各自的 Yes/No 回答
    2. 对每个问题，记录每张切分图的分数 score[q_id][img_idx]
    3. 依赖过滤：逐切分图检查，若父问题回答 No，则子问题该切分图分数置 0
    4. 对每张切分图，计算问题平均分；再对所有切分图求平均

    Args:
        judge_model_path: str - Judge 模型路径（本地路径或 HuggingFace 模型名）
        judge_device: str - Judge 模型运行设备
        judge_dtype: str - Judge 模型数据类型（'bfloat16' 或 'float16'）
        judge_batch_size: int - Judge 批量推理批次大小
        judge_use_flash_attention: bool - Judge 是否使用 Flash Attention
        judge_seed: int - Judge 模型随机种子（默认 42）
    """

    def __init__(
        self,
        judge_model_path: str = "",
        judge_device: str = "cuda",
        judge_dtype: str = "bfloat16",
        judge_batch_size: int = 8,
        judge_use_flash_attention: bool = True,
        judge_seed: int = 42,
        **kwargs
    ):
        super().__init__()
        self.judge_model_path = judge_model_path
        self.judge_device = judge_device
        self.judge_dtype = judge_dtype
        self.judge_batch_size = judge_batch_size
        self.judge_use_flash_attention = judge_use_flash_attention
        self.judge_seed = judge_seed
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
            self.logger.info("[Alignment] Judge inferencer loaded successfully")

    def score(self, predictions, references, test_set=None, **kwargs):
        if test_set is None:
            self.logger.error(UTILS_CODES.UNKNOWN_ERROR, "test_set is required for OneIGAlignmentEvaluator")
            return {'accuracy': 0.0, 'details': []}

        # 加载 Judge 推理器
        self._ensure_inferencer()

        if self._inferencer is None:
            self.logger.error(UTILS_CODES.UNKNOWN_ERROR, "[Alignment] Judge inferencer not available")
            return {'accuracy': 0.0, 'details': []}

        self.logger.info(f"[Alignment] Starting evaluation, {len(test_set)} samples")
        cache_dir = tempfile.mkdtemp()

        prompt_scores = {}  # "class_item_sample_id" -> float or None
        # 使用 class_item + sample_id 作为唯一 key，避免跨类别 sample_id 冲突
        trace_data = {}  # 落盘溯源数据：unique_key -> dict

        try:
            for i, item in enumerate(test_set):
                sample_id = item.get('id', f'sample_{i}')
                class_item = item.get('class_item', '')
                unique_key = f'{class_item}_{sample_id}'
                image_path = item.get('image_path', '')
                self.logger.debug(f"[Alignment] Processing {unique_key}")
                questions_raw = item.get('question', '{}')
                dependencies_raw = item.get('dependency', '{}')

                # 从 JSON 字符串解析 question/dependency，并转换键为整数
                # 数据集中保持 JSON 字符串格式，避免 PyArrow struct schema 推断
                # 对缺失 key 填充 None 导致 TypeError
                try:
                    if isinstance(questions_raw, str) and questions_raw:
                        questions = {int(k): v for k, v in json.loads(questions_raw).items()}
                    elif isinstance(questions_raw, dict):
                        questions = {int(k) if str(k).isdigit() else k: v
                                     for k, v in questions_raw.items()}
                    else:
                        questions = {}
                except (json.JSONDecodeError, ValueError):
                    self.logger.info(f"[Alignment] Failed to parse question for {unique_key}: {repr(questions_raw)[:200]}")
                    questions = {}

                try:
                    if isinstance(dependencies_raw, str) and dependencies_raw:
                        dependencies = {int(k): v for k, v in json.loads(dependencies_raw).items()}
                    elif isinstance(dependencies_raw, dict):
                        dependencies = {int(k) if str(k).isdigit() else k: v
                                        for k, v in dependencies_raw.items()}
                    else:
                        dependencies = {}
                except (json.JSONDecodeError, ValueError):
                    self.logger.error(UTILS_CODES.UNKNOWN_ERROR, f"[Alignment] Failed to parse dependency for {unique_key}: {repr(dependencies_raw)[:200]}")
                    dependencies = {}
                grid_rows = item.get('grid_rows', 2)
                grid_cols = item.get('grid_cols', 2)

                # 图片路径校验
                if not image_path or not os.path.exists(image_path):
                    prompt_scores[unique_key] = None
                    continue

                # 切分图片
                split_img_list = split_image_grid(
                    image_path, (grid_rows, grid_cols), cache_dir)

                # 检查切分结果
                if len(split_img_list) == 0:
                    prompt_scores[unique_key] = None
                    continue

                # 对每个问题，对所有切分图推理，得到每张切分图的 Yes/No 回答
                score = {}
                judge_details = []  # 落盘溯源：逐问题 Judge I/O
                for q_id, question in questions.items():
                    batch_answer = self._inferencer.infer_semantic(
                        split_img_list, question)
                    score[q_id] = [float(ans == "Yes") for ans in batch_answer]
                    judge_details.append({
                        'question_id': q_id,
                        'question': question,
                        'judge_prompt': f"{question}. Please answer 'Yes' or 'No' only.",
                        'judge_outputs': [
                            {'grid_idx': idx, 'raw_output': ans,
                             'parsed_answer': ans, 'score': float(ans == "Yes")}
                            for idx, ans in enumerate(batch_answer)
                        ],
                        'dependency': dependencies.get(q_id, [0]),
                        'filtered_scores': None,
                    })

                # 依赖过滤：逐切分图检查
                filter_score = deepcopy(score)
                for img_idx in range(len(split_img_list)):
                    for q_id, parent_ids in dependencies.items():
                        any_parent_answered_no = False
                        for parent_id in parent_ids:
                            # parent_id == 0 表示无依赖，跳过
                            if parent_id == 0:
                                continue
                            try:
                                if score[parent_id][img_idx] == 0:
                                    any_parent_answered_no = True
                                    break
                                else:
                                    continue
                            except (KeyError, IndexError):
                                self.logger.error(
                                    UTILS_CODES.UNKNOWN_ERROR,
                                    f"Parent question {parent_id} not found for {unique_key}")
                        if any_parent_answered_no:
                            filter_score[q_id][img_idx] = 0

                # 落盘溯源：回填依赖过滤后的分数
                for jd in judge_details:
                    jd['filtered_scores'] = list(filter_score.get(jd['question_id'], []))

                # 对每张切分图，计算问题平均分；再对所有切分图求平均
                num_questions = len(filter_score)
                if num_questions == 0:
                    prompt_scores[unique_key] = None
                    continue

                sum_of_filter_score = [0] * len(split_img_list)
                for question_id in range(num_questions):
                    for img_idx in range(len(split_img_list)):
                        sum_of_filter_score[img_idx] += (
                            filter_score[question_id + 1][img_idx]
                            if (question_id + 1) in filter_score
                            else 0
                        )

                sum_of_filter_score = [
                    img_score / num_questions
                    for img_score in sum_of_filter_score
                ]

                # 计算最终分数
                prompt_scores[unique_key] = (
                    sum(sum_of_filter_score) / len(sum_of_filter_score)
                    if sum_of_filter_score else None
                )

                # 落盘溯源：保存中间数据
                trace_data[unique_key] = {
                    'image_path': image_path,
                    'grid': f"{grid_rows}x{grid_cols}",
                    'num_splits': len(split_img_list),
                    'judge_details': judge_details,
                }

        finally:
            shutil.rmtree(cache_dir, onerror=rm_error)

        # 计算整体平均分
        valid_scores = [s for s in prompt_scores.values() if s is not None]
        overall_score = (
            sum(valid_scores) / len(valid_scores) * 100
            if valid_scores else 0.0
        )

        results = []
        for unique_key, score in prompt_scores.items():
            if score is not None:
                # 从 unique_key "class_item_sample_id" 中提取 class_item 和 sample_id
                parts = unique_key.split('_', 1)
                class_item = parts[0] if len(parts) > 0 else ''
                sample_id = parts[1] if len(parts) > 1 else unique_key
                results.append({
                    'id': sample_id,
                    'class_item': class_item,
                    'score': score,
                    **trace_data.get(unique_key, {})
                })

        self.logger.info(f"[Alignment] Evaluation complete, accuracy={overall_score:.2f}, valid_samples={len(valid_scores)}")
        return {
            'accuracy': overall_score,
            'details': results
        }
