import re
import os
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
        def process_image(pred_item):
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
                except Exception as e:
                    raise AISBenchRuntimeError(DSET_CODES.UNKNOWN_ERROR, f"Failed to process image {image_path}: {e}")
            return pred_item

        # 使用并行处理加速图片处理
        max_workers = min(8, os.cpu_count())  # 根据CPU核心数调整
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用tqdm显示进度
            processed_preds = list(tqdm(
                executor.map(process_image, preds),
                total=len(preds),
                desc="Processing images",
                unit="image"
            ))

        processed_preds.sort(key=lambda x: x.get('id', 0))
        return processed_preds

    def _modify_dataset_item(self, dataset_item, pred_item):
        for item in dataset_item["content"].split(AIS_CONTENT_TAG):
            if item.startswith(AIS_TEXT_START):
                question = item.replace(AIS_TEXT_START, "")
            elif item.startswith(AIS_IMAGE_START):
                org_image_url = item.replace(AIS_IMAGE_START, "")
        dataset_item["content"] = AIS_TEXT_START + question + AIS_CONTENT_TAG \
            + AIS_IMAGE_START + org_image_url + AIS_CONTENT_TAG \
            + AIS_IMAGE_START + pred_item['prediction'] + AIS_CONTENT_TAG