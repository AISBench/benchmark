from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.registry import LOAD_DATASET

from ..artifacts import read_jsonl, validate_artifacts
from ..errors import ArtifactValidationError


@LOAD_DATASET.register_module()
class PrefixCacheDataset(BaseDataset):
    @staticmethod
    def load(requests_path: str, full_path: str, manifest_path: str, **kwargs) -> Dataset:
        manifest_file = Path(manifest_path).resolve()
        validate_artifacts(manifest_file)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        requests = read_jsonl(Path(requests_path).resolve())
        full = read_jsonl(Path(full_path).resolve())
        if len(requests) != len(full):
            raise ArtifactValidationError("requests/full row count mismatch")
        mode = manifest["prefix_cache"]["mode"]
        rows = []
        for index, (request, audit) in enumerate(zip(requests, full)):
            if audit["sequence_index"] != index:
                raise ArtifactValidationError(f"route metadata order mismatch at row {index}")
            rows.append({
                "question": request["question"],
                "answer": request["answer"],
                "max_out_len": request["max_tokens"],
                "dp_rank": audit.get("dp_rank"),
                "group_id": audit["group_id"],
                "lane_sequence": audit.get("lane_sequence"),
                "cache_mode": mode,
            })
        return Dataset.from_list(rows)
