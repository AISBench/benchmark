import os
import re
import base64
import pandas as pd

from datasets import Dataset

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.datasets.utils.datasets import get_data_path, get_content_str
from ais_bench.benchmark.utils.prompt import AIS_CONTENT_TAG, AIS_TEXT_START, AIS_IMAGE_START

from .base import BaseDataset

# 提示词模板,参考evalscope
OPEN_PROMPT = (
    'Read the picture and solve the following problem step by step.'
    'The last line of your response should be of the form'
    ' "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.\n\n'
    '{question}\n\n'
    'Remember to put your answer on its own line at the end in the form'
    ' "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem,'
    ' and you do not need to use a \\boxed command.'
)

logger = AISLogger()


@LOAD_DATASET.register_module()
class RealworldQADataset(BaseDataset):

    @staticmethod
    def load(path, image_type='image_path'):
        path = get_data_path(path)
        image_root_path = os.path.join(os.path.dirname(path), "RealworldQA_images")
        
        if not os.path.exists(image_root_path):
            os.makedirs(image_root_path, exist_ok=True)
        
        data_dir = os.path.dirname(path)
        parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])

        if not parquet_files:
            raise ValueError(f"No parquet files found in {data_dir}")

        dfs = [pd.read_parquet(os.path.join(data_dir, f)) for f in parquet_files]
        data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(data)} samples from RealworldQA dataset")   
        
        dataset = []
        for idx in range(len(data)):
            row = data.iloc[idx]
            question = row['question']
            answer = row['answer']
            image_data = row['image']
            
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image_bytes = image_data['bytes']
                if image_type == 'image_path':
                    image_path = os.path.join(image_root_path, f"image_{idx}.webp")
                    if not os.path.exists(image_path):
                        with open(image_path, 'wb') as f:
                            f.write(image_bytes)
                    image_url = image_path
                elif image_type == 'image_base64':
                    image_url = base64.b64encode(image_bytes).decode('utf-8')
                else:
                    raise ValueError(f"Unsupported image_type: {image_type}")
            else:
                logger.warning(f"Image data format unexpected at index {idx}, skipping...")
                continue
            
            question = OPEN_PROMPT.format(question=question)
            # 参考eavlscope实现，先放问题，再放图片
            content =  AIS_TEXT_START + question + AIS_CONTENT_TAG + AIS_IMAGE_START + image_url + AIS_CONTENT_TAG

            dataset.append({
                "content": content,
                "answer": answer,
            })
        
        return Dataset.from_list(dataset)


class RealworldQAEvaluator(BaseEvaluator):
    # 给出更多的得分细节
    def score(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references ({len(references)}) "
                f"have different length"
            )

        correct = 0
        details = []

        for pred, ref in zip(predictions, references):
            origin_prediction = pred
            gt_answer = ref.get('answer', '') if isinstance(ref, dict) else ref

            extracted_prediction = self._extract_answer(pred)

            pred_norm = self._normalize_answer(extracted_prediction)
            gt_norm = self._normalize_answer(gt_answer)
            is_correct = pred_norm == gt_norm

            if is_correct:
                correct += 1

            details.append({
                'origin_prediction': origin_prediction,
                'pred': extracted_prediction,
                'answer': gt_answer,
                'normalized_prediction': pred_norm,
                'normalized_answer': gt_norm,
                'correct': is_correct,
            })

        total = len(predictions)
        return {
            'accuracy': 100 * correct / total if total else 0,
            'details': details,
        }

    # 参考eavlscope实现，后处理规则使用强制匹配
    def _extract_answer(self, prediction):

        pattern = r'ANSWER:\s*(.*)'
        match = re.search(pattern, prediction)
        if match:
            return match.group(1).strip()
        return ''

    def _normalize_answer(self, answer):
        return answer.strip().lower()