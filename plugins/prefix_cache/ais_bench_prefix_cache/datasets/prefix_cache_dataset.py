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
        """把 prepare 阶段生成的 requests/full 工件加载为 HuggingFace Dataset。

        校验 manifest 与两份工件的一致性，再把请求的 question/answer、full 中的
        max_tokens 与审计信息（dp_rank、group_id、lane_sequence、cache_mode）合并到同一行，
        供推理器按 prefix cache 的冷/热路由语义执行。
        """
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
            # 审计行的 sequence_index 必须与行号一致，保证路由元数据对齐。
            if audit["sequence_index"] != index:
                raise ArtifactValidationError(f"route metadata order mismatch at row {index}")
            rows.append({
                "question": request["question"],
                "answer": request["answer"],
                # requests.jsonl 的输出长度字段可省略或改名；在线执行始终使用
                # full.jsonl 中的审计值，保证生成长度配置不丢失。
                "max_out_len": audit["max_tokens"],
                "dp_rank": audit.get("dp_rank"),
                "group_id": audit["group_id"],
                "lane_sequence": audit.get("lane_sequence"),
                "cache_mode": mode,
            })
        return Dataset.from_list(rows)
