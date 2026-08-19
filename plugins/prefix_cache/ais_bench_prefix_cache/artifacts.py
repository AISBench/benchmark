from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import ArtifactValidationError


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
    return digest.hexdigest()


def _atomic_text(path: Path, text: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ArtifactValidationError(f"refusing to overwrite existing artifact: {path}")
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def write_json(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", overwrite)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], overwrite: bool) -> int:
    materialized = list(rows)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in materialized)
    _atomic_text(path, text, overwrite)
    return len(materialized)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"cannot read JSONL {path}: {exc}") from exc


def artifact_paths(output_dir: Path, run_id: str) -> ArtifactPaths:
    return ArtifactPaths(
        output_dir / f"{run_id}.full.jsonl",
        output_dir / f"{run_id}.requests.jsonl",
        output_dir / f"{run_id}.manifest.json",
        output_dir / f"{run_id}.analysis.json",
    )


def validate_artifacts(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"cannot read Manifest {manifest_path}: {exc}") from exc
    base = manifest_path.parent
    files = manifest.get("artifacts", {})
    full_path = base / files["full"]["name"]
    requests_path = base / files["requests"]["name"]
    full_rows = read_jsonl(full_path)
    request_rows = read_jsonl(requests_path)
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
        if sha256_file(path) != expected:
            raise ArtifactValidationError(f"{key} SHA-256 mismatch")
    return {"ok": True, "rows": len(full_rows), "run_id": manifest["run_id"]}
