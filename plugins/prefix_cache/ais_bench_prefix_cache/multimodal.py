from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from base64 import b64decode, b64encode
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable

from . import __version__
from .artifacts import sha256_file, write_json, write_jsonl
from .errors import ArtifactValidationError, PrefixCacheError
from .generation import load_gsm8k, normalize_question


DEFAULT_REQUEST_COUNT = 1319
DEFAULT_TEXT_TOKENS = 30
DEFAULT_OUTPUT_TOKENS = 256
SINGLE_1080P = "single_1080p"
MULTI_720P_5 = "multi_720p_5"
SCENARIO_NAMES = (SINGLE_1080P, MULTI_720P_5)


@dataclass(frozen=True)
class MMMUImage:
    parquet_path: Path
    row_index: int
    sample_id: str
    image_column: str
    original_path: str
    data: bytes
    image_format: str
    mime_type: str
    width: int
    height: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def data_url(self) -> str:
        encoded = b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _decode(tokenizer: Any, token_ids: Iterable[int]) -> str:
    ids = list(token_ids)
    try:
        return tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(ids, skip_special_tokens=False)


def build_exact_token_texts(
    tokenizer: Any,
    questions: list[str],
    target_tokens: int = DEFAULT_TEXT_TOKENS,
) -> list[str]:
    """Build distinct GSM8K-derived texts that round-trip to exactly N tokens."""
    if target_tokens < 1:
        raise ArtifactValidationError("text_tokens must be a positive integer")
    texts: list[str] = []
    used: set[str] = set()
    for index, raw_question in enumerate(questions):
        question = normalize_question(raw_question)
        if not question:
            raise ArtifactValidationError(f"GSM8K question {index} is empty")
        repeated = question
        while len(_encode(tokenizer, repeated)) < target_tokens:
            repeated += "\n" + question
        token_ids = _encode(tokenizer, repeated)
        selected = None
        for start in range(0, len(token_ids) - target_tokens + 1):
            candidate = _decode(tokenizer, token_ids[start : start + target_tokens])
            if candidate in used:
                continue
            if len(_encode(tokenizer, candidate)) == target_tokens:
                selected = candidate
                break
        if selected is None:
            raise ArtifactValidationError(
                f"cannot construct a distinct round-trip-safe {target_tokens}-token text "
                f"from GSM8K question {index}"
            )
        used.add(selected)
        texts.append(selected)
    return texts


def _image_metadata(data: bytes) -> tuple[int, int, str, str]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ArtifactValidationError("Pillow is required to inspect MMMU images") from exc
    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
            image.verify()
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError(f"cannot decode an MMMU image: {exc}") from exc
    mime_type = Image.MIME.get(image_format)
    if not mime_type:
        raise ArtifactValidationError(f"unsupported MMMU image format: {image_format or 'unknown'}")
    return width, height, image_format, mime_type


def _parquet_sort_key(path: Path) -> tuple[int, str]:
    normalized = path.as_posix().lower()
    preferred = (
        "/agriculture/test-",
        "/literature/test-",
        "/sociology/test-",
        "/test-",
        "/validation-",
        "/dev-",
    )
    rank = next((index for index, marker in enumerate(preferred) if marker in normalized), 99)
    return rank, normalized


def select_mmmu_images(parquet_dir: Path | str) -> dict[str, MMMUImage]:
    """Select exact native-resolution images directly from MMMU Parquet binary fields."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ArtifactValidationError("pyarrow is required to read the MMMU Parquet dataset") from exc

    root = Path(parquet_dir).resolve()
    if not root.is_dir():
        raise ArtifactValidationError(f"MMMU Parquet directory does not exist: {root}")
    parquet_paths = sorted(root.rglob("*.parquet"), key=_parquet_sort_key)
    if not parquet_paths:
        raise ArtifactValidationError(f"no Parquet files found under MMMU directory: {root}")

    targets = {(1920, 1080): SINGLE_1080P, (1280, 720): MULTI_720P_5}
    selected: dict[str, MMMUImage] = {}
    for parquet_path in parquet_paths:
        try:
            parquet_file = pq.ParquetFile(parquet_path)
            image_columns = [name for name in parquet_file.schema_arrow.names if name.startswith("image_")]
            if not image_columns:
                continue
            columns = (["id"] if "id" in parquet_file.schema_arrow.names else []) + image_columns
            row_offset = 0
            for batch in parquet_file.iter_batches(batch_size=16, columns=columns):
                for batch_index, row in enumerate(batch.to_pylist()):
                    sample_id = str(row.get("id", ""))
                    for image_column in image_columns:
                        cell = row.get(image_column)
                        if not isinstance(cell, dict):
                            continue
                        raw = cell.get("bytes")
                        if not isinstance(raw, bytes) or not raw:
                            continue
                        try:
                            width, height, image_format, mime_type = _image_metadata(raw)
                        except ArtifactValidationError:
                            continue
                        scenario = targets.get((width, height))
                        if scenario is None or scenario in selected:
                            continue
                        selected[scenario] = MMMUImage(
                            parquet_path=parquet_path.resolve(),
                            row_index=row_offset + batch_index,
                            sample_id=sample_id,
                            image_column=image_column,
                            original_path=str(cell.get("path") or ""),
                            data=raw,
                            image_format=image_format,
                            mime_type=mime_type,
                            width=width,
                            height=height,
                        )
                    if len(selected) == len(targets):
                        return selected
                row_offset += batch.num_rows
        except (OSError, ValueError) as exc:
            raise ArtifactValidationError(f"cannot read MMMU Parquet file {parquet_path}: {exc}") from exc

    missing = [scenario for scenario in SCENARIO_NAMES if scenario not in selected]
    raise ArtifactValidationError(
        f"MMMU Parquet data does not contain exact native images for: {', '.join(missing)}"
    )


def _tokenizer_manifest(
    tokenizer: Any,
    tokenizer_path: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    return {
        "path": tokenizer_path,
        "trust_remote_code": trust_remote_code,
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "vocab_size": len(tokenizer),
    }


def _dataset_rows(
    texts: list[str],
    image_ref: str,
    image_count: int,
    text_tokens: int,
    output_tokens: int,
) -> list[dict[str, Any]]:
    rows = []
    for index, text in enumerate(texts):
        content = [
            {"type": "image_url", "image_url": {"base64_ref": image_ref}}
            for _ in range(image_count)
        ]
        content.append({"type": "text", "text": text})
        rows.append({
            "request_id": f"request-{index:08d}",
            "question": text,
            "answer": "none",
            "image_refs": [image_ref] * image_count,
            "prompt": [{"role": "user", "content": content}],
            "text_tokens": text_tokens,
            "max_out_len": output_tokens,
        })
    return rows


def prepare_multimodal_datasets(
    *,
    tokenizer_path: str,
    gsm8k_path: Path | str,
    mmmu_parquet_dir: Path | str,
    output_dir: Path | str,
    request_count: int = DEFAULT_REQUEST_COUNT,
    text_tokens: int = DEFAULT_TEXT_TOKENS,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    overwrite: bool = False,
    trust_remote_code: bool = False,
    tokenizer_loader: Callable[[str], Any] | None = None,
) -> Path:
    """Create both required multimodal JSONL datasets and an audit manifest."""
    for name, value in (
        ("request_count", request_count),
        ("text_tokens", text_tokens),
        ("output_tokens", output_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ArtifactValidationError(f"{name} must be a positive integer")

    corpus_path = Path(gsm8k_path).resolve()
    mmmu_root = Path(mmmu_parquet_dir).resolve()
    target_dir = Path(output_dir).resolve()
    if not corpus_path.is_file():
        raise ArtifactValidationError(f"GSM8K file does not exist: {corpus_path}")
    records = load_gsm8k(corpus_path)
    if len(records) < request_count:
        raise ArtifactValidationError(
            f"GSM8K has {len(records)} rows; {request_count} are required"
        )
    selected_records = records[:request_count]
    if len({record.question_sha256 for record in selected_records}) != request_count:
        raise ArtifactValidationError("selected GSM8K questions must be distinct")

    if tokenizer_loader is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ArtifactValidationError("transformers is required to load the tokenizer") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
        )
    else:
        tokenizer = tokenizer_loader(tokenizer_path)
    texts = build_exact_token_texts(
        tokenizer,
        [record.question for record in selected_records],
        text_tokens,
    )

    selected_images = select_mmmu_images(mmmu_root)

    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "multimodal_prefix_cache.manifest.json"
    dataset_paths = {
        SINGLE_1080P: target_dir / f"{SINGLE_1080P}_{request_count}.jsonl",
        MULTI_720P_5: target_dir / f"{MULTI_720P_5}_{request_count}.jsonl",
    }
    existing = [path for path in [manifest_path, *dataset_paths.values()] if path.exists()]
    if existing and not overwrite:
        raise ArtifactValidationError(
            f"output already exists: {existing[0]}; pass --overwrite to replace it"
        )

    rows_by_name = {
        SINGLE_1080P: _dataset_rows(
            texts, SINGLE_1080P, 1, text_tokens, output_tokens
        ),
        MULTI_720P_5: _dataset_rows(
            texts, MULTI_720P_5, 5, text_tokens, output_tokens
        ),
    }
    for name, path in dataset_paths.items():
        write_jsonl(path, rows_by_name[name], overwrite=overwrite)

    manifest = {
        "schema_version": "2.0",
        "plugin_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "path": str(corpus_path),
            "sha256": sha256_file(corpus_path),
            "rows_used": request_count,
            "field": "question",
        },
        "tokenizer": _tokenizer_manifest(tokenizer, tokenizer_path, trust_remote_code),
        "request_count": request_count,
        "text_tokens": text_tokens,
        "output_tokens": output_tokens,
        "content_order": ["image", "text"],
        "image_encoding": "base64",
        "mmmu_parquet_dir": str(mmmu_root),
        "datasets": {},
    }
    for name, path in dataset_paths.items():
        image = selected_images[name]
        image_count = 1 if name == SINGLE_1080P else 5
        manifest["datasets"][name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": request_count,
            "image_count_per_request": image_count,
            "same_image_within_request": True,
            "same_images_across_requests": True,
            "image": {
                "ref": name,
                "encoding": "base64",
                "data_url": image.data_url,
                "byte_length": len(image.data),
                "sha256": image.sha256,
                "format": image.image_format,
                "mime_type": image.mime_type,
                "width": image.width,
                "height": image.height,
                "parquet_path": str(image.parquet_path),
                "row_index": image.row_index,
                "sample_id": image.sample_id,
                "image_column": image.image_column,
                "original_path": image.original_path,
            },
        }
    write_json(manifest_path, manifest, overwrite=overwrite)
    validate_multimodal_manifest(manifest_path, tokenizer=tokenizer)
    return manifest_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"cannot read dataset {path}: {exc}") from exc
    return rows


def expand_prompt_image_refs(
    prompt: list[dict[str, Any]],
    media: dict[str, str],
) -> list[dict[str, Any]]:
    """Return a prompt whose compact image references are expanded to Base64 data URLs."""
    expanded = deepcopy(prompt)
    for message in expanded:
        content = message.get("content", message.get("prompt_mm"))
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                raise ArtifactValidationError("image_url must be an object")
            image_ref = image_url.pop("base64_ref", None)
            if image_ref is None:
                url = image_url.get("url")
                if isinstance(url, str) and url in media:
                    image_ref = url
            if not isinstance(image_ref, str) or image_ref not in media:
                raise ArtifactValidationError(f"unknown Base64 image reference: {image_ref!r}")
            data_url = media[image_ref]
            if not data_url.startswith("data:image/") or ";base64," not in data_url:
                raise ArtifactValidationError(f"invalid Base64 data URL for image reference {image_ref}")
            part["image_url"] = {"url": data_url}
    return expanded


def _decode_image_data_url(data_url: str, expected_mime: str) -> bytes:
    prefix = f"data:{expected_mime};base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise ArtifactValidationError(f"image data URL must start with {prefix}")
    try:
        return b64decode(data_url[len(prefix) :], validate=True)
    except (ValueError, TypeError) as exc:
        raise ArtifactValidationError("image data URL contains invalid Base64") from exc


def validate_multimodal_manifest(
    manifest_path: Path | str,
    *,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"cannot read multimodal manifest {path}: {exc}") from exc
    expected_count = int(manifest["request_count"])
    expected_text_tokens = int(manifest["text_tokens"])
    expected_output_tokens = int(manifest["output_tokens"])
    selected_summary: dict[str, dict[str, Any]] = {}
    if manifest.get("content_order") != ["image", "text"]:
        raise ArtifactValidationError("multimodal content order must be image then text")
    if manifest.get("image_encoding") != "base64":
        raise ArtifactValidationError("multimodal images must use Base64 encoding")
    for name in SCENARIO_NAMES:
        spec = manifest.get("datasets", {}).get(name)
        if not isinstance(spec, dict):
            raise ArtifactValidationError(f"manifest is missing dataset {name}")
        dataset_path = Path(spec["path"])
        if sha256_file(dataset_path) != spec["sha256"]:
            raise ArtifactValidationError(f"dataset checksum mismatch: {dataset_path}")
        rows = _read_jsonl(dataset_path)
        if len(rows) != expected_count or spec.get("rows") != expected_count:
            raise ArtifactValidationError(f"{name} must contain {expected_count} rows")
        expected_images = 1 if name == SINGLE_1080P else 5
        image_spec = spec.get("image", {})
        image_ref = image_spec.get("ref")
        expected_size = (1920, 1080) if name == SINGLE_1080P else (1280, 720)
        raw = _decode_image_data_url(image_spec.get("data_url"), image_spec.get("mime_type"))
        if len(raw) != image_spec.get("byte_length"):
            raise ArtifactValidationError(f"{name} decoded image byte length mismatch")
        if hashlib.sha256(raw).hexdigest() != image_spec.get("sha256"):
            raise ArtifactValidationError(f"{name} decoded image checksum mismatch")
        width, height, image_format, mime_type = _image_metadata(raw)
        if (width, height) != expected_size:
            raise ArtifactValidationError(
                f"{name} image must be exact native size {expected_size[0]}x{expected_size[1]}"
            )
        if (width, height) != (image_spec.get("width"), image_spec.get("height")):
            raise ArtifactValidationError(f"{name} image dimension metadata mismatch")
        if image_format != image_spec.get("format") or mime_type != image_spec.get("mime_type"):
            raise ArtifactValidationError(f"{name} image format metadata mismatch")
        parquet_path = Path(str(image_spec.get("parquet_path", "")))
        if not parquet_path.is_file():
            raise ArtifactValidationError(f"MMMU source Parquet file does not exist: {parquet_path}")
        selected_summary[name] = {
            "size": f"{width}x{height}",
            "sample_id": image_spec.get("sample_id"),
            "image_column": image_spec.get("image_column"),
            "parquet_path": str(parquet_path),
            "sha256": image_spec.get("sha256"),
        }
        questions = set()
        for index, row in enumerate(rows):
            if row.get("request_id") != f"request-{index:08d}":
                raise ArtifactValidationError(f"{name} request order mismatch at row {index}")
            image_refs = row.get("image_refs")
            if not isinstance(image_refs, list) or len(image_refs) != expected_images:
                raise ArtifactValidationError(f"{name} image count mismatch at row {index}")
            if len(set(image_refs)) != 1:
                raise ArtifactValidationError(f"{name} images must be identical at row {index}")
            if image_refs[0] != image_ref:
                raise ArtifactValidationError(f"{name} image changed at row {index}")
            prompt = row.get("prompt")
            if not isinstance(prompt, list) or len(prompt) != 1:
                raise ArtifactValidationError(f"{name} prompt structure mismatch at row {index}")
            content = prompt[0].get("content")
            if not isinstance(content, list) or len(content) != expected_images + 1:
                raise ArtifactValidationError(f"{name} prompt content mismatch at row {index}")
            expected_image_parts = [
                {"type": "image_url", "image_url": {"base64_ref": image_ref}}
                for _ in range(expected_images)
            ]
            if content[:-1] != expected_image_parts:
                raise ArtifactValidationError(f"{name} prompt image order mismatch at row {index}")
            if row.get("max_out_len") != expected_output_tokens:
                raise ArtifactValidationError(f"{name} output token mismatch at row {index}")
            if row.get("text_tokens") != expected_text_tokens:
                raise ArtifactValidationError(f"{name} text token metadata mismatch at row {index}")
            question = row.get("question")
            if not isinstance(question, str) or not question:
                raise ArtifactValidationError(f"{name} question is empty at row {index}")
            if content[-1] != {"type": "text", "text": question}:
                raise ArtifactValidationError(f"{name} prompt text mismatch at row {index}")
            questions.add(question)
            if tokenizer is not None and len(_encode(tokenizer, question)) != expected_text_tokens:
                raise ArtifactValidationError(f"{name} question token mismatch at row {index}")
        if len(questions) != expected_count:
            raise ArtifactValidationError(f"{name} questions must all be distinct")
    return {
        "ok": True,
        "request_count": expected_count,
        "text_tokens": expected_text_tokens,
        "output_tokens": expected_output_tokens,
        "datasets": list(SCENARIO_NAMES),
        "selected_images": selected_summary,
    }


def build_aisbench_config(
    *,
    dataset_path: Path | str,
    dataset_abbr: str,
    tokenizer_path: str,
    inference_url: str,
    model: str,
    work_dir: Path | str,
    image_ref: str,
    image_data_url: str,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    batch_size: int = 1,
) -> str:
    """Render a static AISBench Python config for one multimodal scenario."""
    values = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_abbr": dataset_abbr,
        "tokenizer_path": tokenizer_path,
        "inference_url": inference_url,
        "model": model,
        "work_dir": str(Path(work_dir).resolve()),
        "output_tokens": output_tokens,
        "batch_size": batch_size,
        "image_ref": image_ref,
        "image_data_url": image_data_url,
    }
    return f'''# Generated by ais-bench-prefix-cache-mm; do not edit.
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask
from ais_bench_prefix_cache.datasets import MultimodalPrefixCacheDataset
from ais_bench_prefix_cache.models import VLLMPrefixCacheChatAPI
from ais_bench_prefix_cache.openicl import Base64RefMMPromptTemplate

datasets = [dict(
    abbr={values['dataset_abbr']!r},
    type=MultimodalPrefixCacheDataset,
    path={values['dataset_path']!r},
    reader_cfg=dict(input_columns=['content', 'max_out_len'], output_column='answer'),
    infer_cfg=dict(
        prompt_template=dict(
            type=Base64RefMMPromptTemplate,
            media={{{values['image_ref']!r}: {values['image_data_url']!r}}},
            template=dict(round=[dict(role='HUMAN', prompt_mm={{
                'image': {{'type': 'image_url', 'image_url': {{'url': '{{image}}'}}}},
                'text': {{'type': 'text', 'text': '{{question}}'}},
            }})]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    ),
    eval_cfg=dict(evaluator=dict(type=AccEvaluator), pred_role='BOT'),
)]

models = [dict(
    type=VLLMPrefixCacheChatAPI,
    attr='service',
    abbr='prefix-cache-mm-vllm',
    path={values['tokenizer_path']!r},
    model={values['model']!r},
    inference_url={values['inference_url']!r},
    stream=True,
    max_out_len={values['output_tokens']!r},
    retry=2,
    batch_size={values['batch_size']!r},
    generation_kwargs=dict(temperature=0, ignore_eos=True),
)]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=1, task=dict(type=OpenICLApiInferTask)),
)
summarizer = dict(attr='accuracy', summary_groups=[])
work_dir = {values['work_dir']!r}
'''


def run_multimodal_benchmark(
    *,
    manifest_path: Path | str,
    scenarios: Iterable[str],
    inference_url: str,
    model: str,
    tokenizer_path: str,
    work_dir: Path | str,
    batch_size: int,
    extra_args: Iterable[str] = (),
) -> list[dict[str, Any]]:
    manifest_file = Path(manifest_path).resolve()
    validation = validate_multimodal_manifest(manifest_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    root = Path(work_dir).resolve()
    results = []
    selected = list(scenarios)
    for name in selected:
        if name not in SCENARIO_NAMES:
            raise ArtifactValidationError(f"unknown multimodal scenario: {name}")
        scenario_root = root / name
        image_spec = manifest["datasets"][name]["image"]
        config_dir = scenario_root / "generated_config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "multimodal_prefix_cache_perf.py"
        config_path.write_text(
            build_aisbench_config(
                dataset_path=manifest["datasets"][name]["path"],
                dataset_abbr=name,
                tokenizer_path=tokenizer_path,
                inference_url=inference_url,
                model=model,
                work_dir=scenario_root,
                image_ref=image_spec["ref"],
                image_data_url=image_spec["data_url"],
                output_tokens=validation["output_tokens"],
                batch_size=batch_size,
            ),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-m",
            "ais_bench.benchmark.cli.main",
            str(config_path),
            "--mode",
            "perf",
            "--num-warmups",
            "0",
            *map(str, extra_args),
        ]
        completed = subprocess.run(command, env=os.environ.copy(), check=False)
        if completed.returncode != 0:
            raise PrefixCacheError(
                f"AISBench multimodal scenario {name} failed with exit code {completed.returncode}"
            )
        results.append(
            report_performance(scenario_root, dataset_abbr=name)
            | {"scenario": name, "config": str(config_path)}
        )
    return results


def report_performance(work_dir: Path | str, dataset_abbr: str | None = None) -> dict[str, Any]:
    root = Path(work_dir).resolve()
    pattern = f"{dataset_abbr}.csv" if dataset_abbr else "*.csv"
    candidates = [path for path in root.rglob(pattern) if "performances" in path.parts]
    if not candidates:
        raise ArtifactValidationError(f"no AISBench performance CSV found under {root}")
    csv_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    metrics: dict[str, Any] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            name = row.get("Performance Parameters")
            if name in {"E2EL", "TTFT", "TPOT", "ITL", "InputTokens", "OutputTokens"}:
                metrics[name] = {
                    key: value
                    for key, value in row.items()
                    if key and key != "Performance Parameters"
                }
    missing = sorted({"TTFT", "TPOT", "ITL"} - set(metrics))
    if missing:
        raise ArtifactValidationError(
            f"AISBench performance CSV is missing acceptance metrics: {', '.join(missing)}; "
            "confirm stream=True and that requests produced multiple output chunks"
        )
    common_path = csv_path.with_suffix(".json")
    common = json.loads(common_path.read_text(encoding="utf-8")) if common_path.is_file() else {}
    return {
        "csv": str(csv_path),
        "json": str(common_path) if common_path.is_file() else None,
        "metrics": metrics,
        "common": common,
    }
