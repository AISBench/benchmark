import glob
import io
import json
import os
import re

import pandas as pd
from PIL import Image

from datasets import Dataset

from ais_bench.benchmark.datasets.utils.datasets import get_content_str
from ais_bench.benchmark.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.logging import AISLogger

from ..base import BaseDataset

logger = AISLogger()

REFCOCO_PROMPT_TEMPLATE = (
    'Locate every object that matches the description "{ref_sentence}" '
    'in the image. Report bbox coordinates in JSON format.'
)


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
    TEMP_IMAGE_STORE_DIR = 'temp_save_images'

    @staticmethod
    def _generate_image_store_dir(resolved_path: str, split: str) -> str:
        image_root_path = os.path.join(
            os.path.dirname(resolved_path),
            RefCOCODataset.TEMP_IMAGE_STORE_DIR,
        )
        return os.path.join(image_root_path, split)

    @staticmethod
    def _load_split_dataframe(resolved_path: str, split: str) -> pd.DataFrame:
        shard_paths = sorted(glob.glob(os.path.join(resolved_path, f'{split}-*.parquet')))
        if not shard_paths:
            raise FileNotFoundError(
                f'No RefCOCO parquet shards found for split {split} in {resolved_path}'
            )

        logger.info(f'Loading RefCOCO split {split} from {len(shard_paths)} shard(s) in {resolved_path}')
        return pd.concat([pd.read_parquet(shard_path) for shard_path in shard_paths], ignore_index=True)

    @staticmethod
    def _persist_image_if_not_exist(image_payload, image_name: str, image_root_dir: str, row_index: int) -> tuple[str, int, int]:
        if not isinstance(image_payload, dict) or 'bytes' not in image_payload:
            raise ValueError(f'RefCOCO row {row_index} has invalid image payload: {type(image_payload)}')

        pil_img = Image.open(io.BytesIO(image_payload['bytes'])).convert('RGB')
        image_path = os.path.join(image_root_dir, image_name)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        if not os.path.exists(image_path):
            pil_img.save(image_path, format='JPEG')
        return image_path, pil_img.width, pil_img.height

    @staticmethod
    def _build_pixel_bbox(raw_bbox) -> list[float]:
        x_coord, y_coord, bbox_width, bbox_height = [float(value) for value in raw_bbox]
        return [x_coord, y_coord, x_coord + bbox_width, y_coord + bbox_height]

    @staticmethod
    def _build_prompt(answer_text) -> str:
        ref_sentence = _remove_leading_articles(str(answer_text))
        return REFCOCO_PROMPT_TEMPLATE.format(ref_sentence=ref_sentence)

    @staticmethod
    def _build_answer_payload(question_id, pixel_bbox: list[float], width: int, height: int) -> str:
        return json.dumps({
            'question_id': int(question_id),
            'bbox': pixel_bbox,
            'image_width': width,
            'image_height': height,
        })

    @staticmethod
    def _build_rows(sample, image_path: str, width: int, height: int, pixel_bbox: list[float]) -> list[dict]:
        rows = []
        reference_answer = RefCOCODataset._build_answer_payload(
            sample['question_id'],
            pixel_bbox,
            width,
            height,
        )

        for answer_text in sample['answer']:
            prompt = RefCOCODataset._build_prompt(answer_text)
            content = get_content_str([
                {'type': 'image_url', 'image_url': image_path},
                {'type': 'text', 'text': prompt},
            ])
            rows.append({
                'content': content,
                'question': prompt,
                'image': image_path,
                'answer': reference_answer,
            })
        return rows

    @staticmethod
    def load(path, split, **kwargs):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Load a RefCOCO split and normalize it into benchmark rows.

        The source data is stored as parquet shards under ``path`` with shard
        names matching ``<split>-*.parquet``. Each source row contains an image
        payload, a ground-truth bounding box in ``[x, y, w, h]`` format, and a
        list of referring expressions. This loader persists each image to
        ``RefCOCO_images/<split>/<file_name>``, converts the bbox to
        ``[x_min, y_min, x_max, y_max]``, and expands the answer list into one
        benchmark row per referring expression.

        Args:
            path: Dataset root containing RefCOCO parquet shards.
            split: Split prefix to load, for example ``val`` or ``testA``.
            **kwargs: Unused extra keyword arguments passed by the dataset
                builder.

        Returns:
            A HuggingFace ``Dataset`` whose rows contain ``content`` for
            multimodal prompting and ``answer`` as the serialized reference
            bbox payload used by evaluation.
        """
        resolved_path = get_data_path(path)
        image_root_dir = RefCOCODataset._generate_image_store_dir(resolved_path, split)
        logger.info(f'Saving RefCOCO images to {image_root_dir}')
        data = RefCOCODataset._load_split_dataframe(resolved_path, split)
        os.makedirs(image_root_dir, exist_ok=True)

        rows = []
        for row_index, (_, sample) in enumerate(data.iterrows()):
            image_path, width, height = RefCOCODataset._persist_image_if_not_exist(
                sample['image'],
                sample['file_name'],
                image_root_dir,
                row_index,
            )
            pixel_bbox = RefCOCODataset._build_pixel_bbox(sample['bbox'])
            rows.extend(
                RefCOCODataset._build_rows(
                    sample,
                    image_path,
                    width,
                    height,
                    pixel_bbox,
                )
            )

        return Dataset.from_list(rows)
