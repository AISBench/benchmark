import glob
import io
import json
import os
import re

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from PIL import Image

from datasets import Dataset

from ais_bench.benchmark.datasets.utils.datasets import get_content_str
from ais_bench.benchmark.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.image_process import pil_to_base64
from ais_bench.benchmark.utils.logging import AISLogger

from ..base import BaseDataset

logger = AISLogger()

IMAGE_PATH_TYPE = 'image_path'
IMAGE_BASE64_TYPE = 'image_base64'

REFCOCO_PROMPT_TEMPLATE = (
    'Locate every object that matches the description "{ref_sentence}" '
    'in the image. Report bbox coordinates in JSON format.'
)

TEMP_IMAGE_STORE_DIR = 'temp_save_images'

def _parse_float_sequence_within(input_str: str) -> list[float]:
    """Extract the first sequence of four floats inside square brackets."""
    pattern = r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
    match = re.search(pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    return [0.0, 0.0, 0.0, 0.0]


def _remove_leading_articles(text: str) -> str:
    cleaned_text = re.sub(r'^(a|an|the)\s+', '', text.strip(), flags=re.IGNORECASE)
    return cleaned_text or text.strip()


@TEXT_POSTPROCESSORS.register_module('refcoco_bbox_1000')
def refcoco_bbox_postprocess(text: str) -> list[float]:
    stripped_text = text.strip()
    bbox = _parse_float_sequence_within(stripped_text)

    logger.debug(f'refcoco_bbox_postprocess: bbox={bbox}')
    return bbox


class ImageResolver(ABC):
    """Strategy interface for converting a PIL image into a transport value."""

    @abstractmethod
    def setup(self, resolved_path: str, split: str) -> None:
        ...

    @abstractmethod
    def resolve(self, pil_img: Image.Image, file_name: str) -> str:
        ...


class PathImageResolver(ImageResolver):
    def setup(self, resolved_path: str, split: str) -> None:
        image_cache_path = os.path.join(
            resolved_path,
            TEMP_IMAGE_STORE_DIR,
            split,
        )
        logger.info(f'Saving RefCOCO images to {image_cache_path}')
        os.makedirs(image_cache_path, exist_ok=True)
        self._cache_dir = image_cache_path

    def resolve(self, pil_img: Image.Image, file_name: str) -> str:
        image_path = os.path.join(self._cache_dir, file_name)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        if not os.path.exists(image_path):
            pil_img.save(image_path, format='JPEG')
        return image_path


class Base64ImageResolver(ImageResolver):
    def setup(self, resolved_path: str, split: str) -> None:
        logger.info(f'Encoding RefCOCO images as base64 for split {split}')

    def resolve(self, pil_img: Image.Image, file_name: str) -> str:
        return pil_to_base64(pil_img, format='JPEG')


IMAGE_RESOLVERS = {
    IMAGE_PATH_TYPE: PathImageResolver,
    IMAGE_BASE64_TYPE: Base64ImageResolver,
}


@LOAD_DATASET.register_module()
class RefCOCODataset(BaseDataset):
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
    def _decode_image_payload(image_payload: Any, row_index: int) -> Image.Image:
        if not isinstance(image_payload, dict) or 'bytes' not in image_payload:
            raise ValueError(f'RefCOCO row {row_index} has invalid image payload: {type(image_payload)}')

        return Image.open(io.BytesIO(image_payload['bytes'])).convert('RGB')

    @staticmethod
    def _build_pixel_bbox(raw_bbox: Any) -> list[float]:
        x_coord, y_coord, bbox_width, bbox_height = [float(value) for value in raw_bbox]
        return [x_coord, y_coord, x_coord + bbox_width, y_coord + bbox_height]

    @staticmethod
    def _build_prompt(answer_text: Any) -> str:
        ref_sentence = _remove_leading_articles(str(answer_text))
        return REFCOCO_PROMPT_TEMPLATE.format(ref_sentence=ref_sentence)

    @staticmethod
    def _build_rows(
        sample: pd.Series,
        image_value: str,
        width: int,
        height: int,
    ) -> list[dict[str, str]]:
        reference_answer = json.dumps({
            'question_id': int(sample['question_id']),
            'bbox': RefCOCODataset._build_pixel_bbox(sample['bbox']),
            'image_width': width,
            'image_height': height,
        })

        rows: list[dict[str, str]] = []
        for answer_text in sample['answer']:
            prompt = RefCOCODataset._build_prompt(answer_text)
            content = get_content_str([
                {'type': 'image_url', 'image_url': image_value},
                {'type': 'text', 'text': prompt},
            ])
            rows.append({
                'content': content,
                'question': prompt,
                'image': image_value,
                'answer': reference_answer,
            })
        return rows

    @staticmethod
    def load(path: str, split: str, **kwargs: Any) -> Dataset:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Load a RefCOCO split and normalize it into benchmark rows.

        The source data is stored as parquet shards under ``path`` with shard
        names matching ``<split>-*.parquet``. Each source row contains an image
        payload, a ground-truth bounding box in ``[x, y, w, h]`` format, and a
        list of referring expressions. This loader can either persist each image
        to a split-specific cache directory or encode it as base64, converts the
        bbox to ``[x_min, y_min, x_max, y_max]``, and expands the answer list
        into one benchmark row per referring expression.

        Args:
            path: Dataset root containing RefCOCO parquet shards.
            split: Split prefix to load, for example ``val`` or ``testA``.
            **kwargs: Extra keyword arguments passed by the dataset builder.
                Supported key: ``image_type`` with values ``IMAGE_PATH_TYPE`` or
                ``IMAGE_BASE64_TYPE``.

        Returns:
            A HuggingFace ``Dataset`` whose rows contain ``content`` for
            multimodal prompting and ``answer`` as the serialized reference
            bbox payload used by evaluation.
        """
        resolved_path = get_data_path(path)
        image_type = kwargs.get('image_type', IMAGE_PATH_TYPE)
        if image_type not in IMAGE_RESOLVERS:
            raise ValueError(
                f'Unsupported image_type: {image_type}. Expected one of {sorted(IMAGE_RESOLVERS)}'
            )
        data = RefCOCODataset._load_split_dataframe(resolved_path, split)
        resolver = IMAGE_RESOLVERS[image_type]()
        resolver.setup(resolved_path, split)

        rows: list[dict[str, str]] = []
        for row_index, (_, sample) in enumerate(data.iterrows()):
            pil_img = RefCOCODataset._decode_image_payload(sample['image'], row_index)
            image_value = resolver.resolve(pil_img, sample['file_name'])

            width, height = pil_img.width, pil_img.height
            sample_rows = RefCOCODataset._build_rows(sample, image_value, width, height)
            rows.extend(sample_rows)

        return Dataset.from_list(rows)
