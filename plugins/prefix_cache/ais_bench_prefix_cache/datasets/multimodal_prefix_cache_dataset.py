from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.prompt import AIS_CONTENT_TAG, AIS_IMAGE_START, AIS_TEXT_START

from ..errors import ArtifactValidationError
@LOAD_DATASET.register_module()
class MultimodalPrefixCacheDataset(BaseDataset):
    """Load compact Base64-reference rows as image-first multimodal prompts."""

    @staticmethod
    def load(path: str, **kwargs) -> Dataset:
        source_path = Path(path).resolve()
        rows = []
        try:
            with source_path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    image_refs = row.get("image_refs")
                    if not isinstance(image_refs, list) or not image_refs:
                        raise ArtifactValidationError(
                            f"invalid multimodal row {line_number}: expected Base64 image references"
                        )
                    content = "".join(
                        AIS_IMAGE_START + image_ref + AIS_CONTENT_TAG
                        for image_ref in image_refs
                    )
                    content += AIS_TEXT_START + row["question"]
                    rows.append(row | {"content": content})
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError(f"invalid JSONL in {source_path}: {exc}") from exc
        return Dataset.from_list(rows)
