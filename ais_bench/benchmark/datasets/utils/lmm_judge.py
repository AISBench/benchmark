import re
import os
import json
import base64
import concurrent.futures
from io import BytesIO
from PIL import Image
from tqdm import tqdm

from ais_bench.benchmark.datasets.needlebench_v2 import origin
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)
from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from ais_bench.benchmark.datasets.base import BaseJDGDataset
from ais_bench.benchmark.utils.file.file import load_jsonl
from ais_bench.benchmark.utils.prompt import AIS_CONTENT_TAG, AIS_TEXT_START, AIS_IMAGE_START
from ais_bench.benchmark.utils.logging.exceptions import AISBenchRuntimeError
from ais_bench.benchmark.utils.logging.error_codes import DSET_CODES
logger = AISLogger()


@TEXT_POSTPROCESSORS.register_module("get_lmm_point_list")
def get_lmm_point_list(pred: str) -> str:
    """从模型回复中提取列表的字符串"""
    match = re.search(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', pred)
    return match.group(0) if match else '[]'


class LMMImgJDGDataset(BaseJDGDataset):
    def _load_from_predictions(self, prediction_path: str):
        """从prediction中拿到对应图片相对路径，将这个路径的图片加载并转换为Base64字符串.

        Args:
            prediction_path (str): The path to the prediction file.

        Returns:
            Dataset: The merged dataset with predictions.
        """
        if not os.path.exists(prediction_path):
            return []

        preds = load_jsonl(prediction_path)
        base_path = os.path.dirname(prediction_path)

        # 定义图片处理函数
        def process_image(index, pred_item):
            # 现在可以使用index来知道pred_item是preds中的第几个
            image_path = os.path.join(base_path, pred_item.get('prediction', ''))
            if image_path and os.path.exists(image_path):
                try:
                    # 加载图片
                    with Image.open(image_path) as img:
                        # 转换为RGB格式
                        img = img.convert('RGB')
                        # 保存到BytesIO
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        # 转换为Base64字符串
                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        # 更新pred中的image字段为Base64字符串
                        pred_item['prediction'] = img_base64
                    self.update_task_state(
                        {
                            "total_count": len(preds),
                            "progress_description": f"Convert prediction images to base64",
                            "finish_count": index,
                        }
                    )
                except Exception as e:
                    raise AISBenchRuntimeError(DSET_CODES.UNKNOWN_ERROR, f"Failed to process image {image_path} at index {index}: {e}")
            return pred_item

        # 使用并行处理加速图片处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # 使用tqdm显示进度
            processed_preds = list(tqdm(
                executor.map(lambda x: process_image(x[0], x[1]), enumerate(preds)),
                total=len(preds),
                desc="Convert prediction images to base64",
                unit="image"
            ))

        processed_preds.sort(key=lambda x: x.get('id', 0))
        return processed_preds


class ImgSCJDGDataset(LMMImgJDGDataset):
    def _modify_dataset_item(self, dataset_item, pred_item):
        for item in dataset_item["content"].split(AIS_CONTENT_TAG):
            if item.startswith(AIS_TEXT_START):
                question = item.replace(AIS_TEXT_START, "")
            elif item.startswith(AIS_IMAGE_START):
                org_image_url = item.replace(AIS_IMAGE_START, "")
        self.logger.debug(f"org_image_url: {org_image_url[:64]} \n pred_image_url: {pred_item['prediction'][:64]}")
        dataset_item["content"] = AIS_TEXT_START + question + AIS_CONTENT_TAG \
            + AIS_IMAGE_START + org_image_url + AIS_CONTENT_TAG \
            + AIS_IMAGE_START + pred_item['prediction'] + AIS_CONTENT_TAG


class ImgPQJDGDataset(LMMImgJDGDataset):
    def _modify_dataset_item(self, dataset_item, pred_item):
        for item in dataset_item["content"].split(AIS_CONTENT_TAG):
            if item.startswith(AIS_TEXT_START):
                question = item.replace(AIS_TEXT_START, "")
            elif item.startswith(AIS_IMAGE_START):
                org_image_url = item.replace(AIS_IMAGE_START, "")
        self.logger.debug(f"org_image_url: {org_image_url[:64]} \n pred_image_url: {pred_item['prediction'][:64]}")
        dataset_item["content"] = AIS_TEXT_START + question + AIS_CONTENT_TAG \
            + AIS_IMAGE_START + pred_item['prediction'] + AIS_CONTENT_TAG


POINT_KEY_LIST_MAP = {
    "SC": ["editing success", "over editing"],
    "PQ": ["naturalness", "artifacts"]
}


@ICL_EVALUATORS.register_module()
class LMMJudgeImageEditEvaluator(BaseEvaluator):
    def __init__(self, metric: str = "SC"):
        self.metric = metric
        self.point_key_list = POINT_KEY_LIST_MAP[metric]
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        # 将get_lmm_point_list获取的字符串格式的list转换成list格式
        if not all(isinstance(pred, str) for pred in predictions):
            return {'error': 'preds must be strings'}
        predictions = [json.loads(get_lmm_point_list(pred)) for pred in predictions]

        total_score = 0
        count = 0
        details = []
        for pred, ref in zip(predictions, references):
            if len(pred) != len(self.point_key_list):
                detail = {'pred': pred, 'org_uuid': ref, 'eval_success': False, 'failed reason': 'prediction format error, length of point list not equal to point key list'}
            else:
                detail = {'pred': {key: score for key, score in zip(self.point_key_list, pred)}, 'org_uuid': ref, 'eval_success': True}
            count += 1
            if detail['eval_success']:
                total_score += min(pred)
            details.append(detail)
        result = {f"{self.metric}": total_score / count if count > 0 else 0.0, 'details': details}
        return result