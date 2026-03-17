import glob
import io
import json
import os
import re

import pandas as pd
from PIL import Image

from datasets import Dataset

from ais_bench.benchmark.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.image_process import pil_to_base64
from ais_bench.benchmark.utils.logging import AISLogger

from ..base import BaseDataset

logger = AISLogger()


def _remove_leading_articles(text: str) -> str:
    cleaned_text = re.sub(r'^(a|an|the)\s+', '', text.strip(), flags=re.IGNORECASE)
    return cleaned_text or text.strip()


def parse_float_sequence_within(input_str: str):
    """Extract the first sequence of four floats inside square brackets."""
    pattern = r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
    match = re.search(pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    return [0.0, 0.0, 0.0, 0.0]  # Default bbox if parsing fails


@TEXT_POSTPROCESSORS.register_module('refcoco_bbox_1000')
def refcoco_bbox_postprocess(text) -> list:
    if not isinstance(text, str):
        raise ValueError('Prediction must be a string')

    stripped_text = text.strip()
    bbox = parse_float_sequence_within(stripped_text)

    logger.debug(f'refcoco_bbox_postprocess: bbox={bbox}')
    return bbox


@LOAD_DATASET.register_module()
class RefCOCODataset(BaseDataset):

    @staticmethod
    def load(path, split, **kwargs):  # pyright: ignore[reportIncompatibleMethodOverride]
        resolved_path = get_data_path(path)
        shard_paths = sorted(glob.glob(os.path.join(resolved_path, f'{split}-*.parquet')))
        if not shard_paths:
            raise FileNotFoundError(
                f'No RefCOCO parquet shards found for split {split} in {resolved_path}'
            )

        logger.info(f'Loading RefCOCO split {split} from {len(shard_paths)} shard(s) in {resolved_path}')
        data = pd.concat([pd.read_parquet(shard_path) for shard_path in shard_paths], ignore_index=True)

        rows = []
        for i in range(len(data)):
            line = data.iloc[i]
            img_field = line['image']
            if not isinstance(img_field, dict) or 'bytes' not in img_field:
                raise ValueError(f'RefCOCO row {i} has invalid image payload: {type(img_field)}')

            pil_img = Image.open(io.BytesIO(img_field['bytes'])).convert('RGB')
            width, height = pil_img.width, pil_img.height
            image_b64 = pil_to_base64(pil_img, format='JPEG')

            x_coord, y_coord, bbox_width, bbox_height = [float(value) for value in line['bbox']]
            pixel_bbox = [x_coord, y_coord, x_coord + bbox_width, y_coord + bbox_height]

            for answer_text in line['answer']:
                ref_sentence = _remove_leading_articles(str(answer_text))
                answer = json.dumps({
                    'question_id': int(line['question_id']),
                    'bbox': pixel_bbox,
                    'image_width': width,
                    'image_height': height,
                })
                rows.append({
                    'ref_sentence': ref_sentence,
                    'image': image_b64,
                    'answer': answer,
                })

        return Dataset.from_list(rows)
