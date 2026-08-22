from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import ArtifactValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactPaths:
    full: Path
    requests: Path
    manifest: Path
    analysis: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    logger.info("[artifacts] sha256_file path=%s sha256=%s", path, result)
    return result


def _atomic_text(path: Path, text: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[artifacts] _atomic_text path=%s exists=%s overwrite=%s text_bytes=%d", path, path.exists(), overwrite, len(text.encode("utf-8")))
    if path.exists() and not overwrite:
        raise ArtifactValidationError(f"refusing to overwrite existing artifact: {path}")
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    logger.info("[artifacts] _atomic_text written path=%s", path)


def write_json(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    logger.info("[artifacts] write_json path=%s overwrite=%s keys=%s", path, overwrite, sorted(value))
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", overwrite)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], overwrite: bool) -> int:
    materialized = list(rows)
    logger.info("[artifacts] write_jsonl path=%s overwrite=%s rows=%d", path, overwrite, len(materialized))
    # Preserve insertion order: requests.jsonl has a documented public field
    # order (question, answer, max_tokens).
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in materialized)
    _atomic_text(path, text, overwrite)
    return len(materialized)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"cannot read JSONL {path}: {exc}") from exc
    logger.info("[artifacts] read_jsonl path=%s rows=%d", path, len(rows))
    return rows


def artifact_paths(output_dir: Path, run_id: str) -> ArtifactPaths:
    paths = ArtifactPaths(
        output_dir / f"{run_id}.full.jsonl",
        output_dir / f"{run_id}.requests.jsonl",
        output_dir / f"{run_id}.manifest.json",
        output_dir / f"{run_id}.analysis.json",
    )
    logger.info("[artifacts] artifact_paths output_dir=%s run_id=%s full=%s requests=%s manifest=%s analysis=%s", output_dir, run_id, paths.full, paths.requests, paths.manifest, paths.analysis)
    return paths


def validate_artifacts(manifest_path: Path) -> dict[str, Any]:
    logger.info("[artifacts] validate_artifacts manifest_path=%s", manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"cannot read Manifest {manifest_path}: {exc}") from exc
    base = manifest_path.parent
    files = manifest.get("artifacts", {})
    full_path = base / files["full"]["name"]
    requests_path = base / files["requests"]["name"]
    logger.info("[artifacts] validate_artifacts run_id=%s artifacts=%s", manifest["run_id"], files)
    full_rows = read_jsonl(full_path)
    request_rows = read_jsonl(requests_path)
    logger.info("[artifacts] validate_artifacts full_rows=%d request_rows=%d expected_count=%s", len(full_rows), len(request_rows), manifest["requests"]["count"])
    if len(full_rows) != len(request_rows) or len(full_rows) != manifest["requests"]["count"]:
        raise ArtifactValidationError("artifact row counts do not match")
    for index, (full, request) in enumerate(zip(full_rows, request_rows)):
        if full["sequence_index"] != index:
            raise ArtifactValidationError(f"full row {index} has invalid sequence_index")
        if set(request) != {"question", "answer", "max_tokens"}:
            raise ArtifactValidationError(f"requests row {index} has unexpected fields")
        if any(request[key] != full[key] for key in request):
            raise ArtifactValidationError(f"requests row {index} differs from full row")
    for key, path in (("full", full_path), ("requests", requests_path)):
        expected = files[key]["sha256"]
        actual = sha256_file(path)
        logger.info("[artifacts] validate_artifacts %s expected_sha256=%s actual_sha256=%s match=%s", key, expected, actual, actual == expected)
        if actual != expected:
            raise ArtifactValidationError(f"{key} SHA-256 mismatch")
    result = {"ok": True, "rows": len(full_rows), "run_id": manifest["run_id"]}
    logger.info("[artifacts] validate_artifacts result=%s", result)
    return result
