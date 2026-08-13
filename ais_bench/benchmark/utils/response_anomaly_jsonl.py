"""Zstandard-compressed JSONL storage for response anomaly payloads."""

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


def _load_zstandard():
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError(
            "zstandard is required for compressed response anomaly payloads. "
            "Install the AISBench response_anomaly extra."
        ) from exc
    return zstandard


class ResponseAnomalyJsonlWriter:
    """Write payload records to bounded, atomically published JSONL.ZST shards."""

    def __init__(self, directory: Path, compression_level: int, rows_per_shard: int):
        self.directory = Path(directory)
        self.compression_level = compression_level
        self.rows_per_shard = rows_per_shard
        self.session_id = uuid.uuid4().hex[:8]
        self.shard_index = 0
        self.shard_rows = 0
        self.total_rows = 0
        self._raw_file = None
        self._stream = None
        self._inprogress_path: Optional[Path] = None
        self._shards = []
        manifest_path = self.directory / "payload_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._shards = list(manifest.get("shards", []))
            self.total_rows = int(manifest.get("total_rows", 0))

    def write(self, record: Dict[str, Any]) -> None:
        payload = record.get("response_anomaly_payload")
        if not isinstance(payload, dict):
            return
        if self._stream is None:
            self._open_shard()
        line = json.dumps(
            {
                "data_abbr": record.get("data_abbr"),
                "id": record.get("id"),
                "uuid": record.get("uuid"),
                "response_anomaly_payload": payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n"
        self._stream.write(line.encode("utf-8"))
        self.shard_rows += 1
        self.total_rows += 1
        if self.shard_rows >= self.rows_per_shard:
            self._close_shard()

    def close(self, payload_retention: Optional[str] = None) -> Dict[str, Any]:
        if self._stream is not None:
            self._close_shard()
        manifest = {
            "format": "jsonl",
            "compression": "zstd",
            "compression_level": self.compression_level,
            "total_rows": self.total_rows,
            "shards": self._shards,
        }
        if payload_retention is not None:
            manifest["payload_retention"] = payload_retention
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "payload_manifest.json"
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(str(tmp_path), str(path))
        return manifest

    def _open_shard(self) -> None:
        zstandard = _load_zstandard()
        self.directory.mkdir(parents=True, exist_ok=True)
        name = (
            f"part-p{os.getpid()}-{self.session_id}-"
            f"{self.shard_index:05d}.jsonl.zst"
        )
        self._inprogress_path = self.directory / f"{name}.inprogress"
        self._raw_file = self._inprogress_path.open("wb")
        compressor = zstandard.ZstdCompressor(level=self.compression_level)
        self._stream = compressor.stream_writer(self._raw_file, closefd=False)
        self.shard_rows = 0

    def _close_shard(self) -> None:
        self._stream.flush(_load_zstandard().FLUSH_FRAME)
        self._stream.close()
        self._raw_file.flush()
        os.fsync(self._raw_file.fileno())
        self._raw_file.close()
        final_path = self._inprogress_path.with_suffix("")
        os.replace(str(self._inprogress_path), str(final_path))
        self._shards.append(
            {
                "file": final_path.name,
                "rows": self.shard_rows,
                "size_bytes": final_path.stat().st_size,
                "sha256": f"sha256:{_sha256_file(final_path)}",
            }
        )
        self.shard_index += 1
        self._raw_file = None
        self._stream = None
        self._inprogress_path = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
