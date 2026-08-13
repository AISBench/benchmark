import json

import zstandard

from ais_bench.benchmark.utils.response_anomaly_jsonl import (
    ResponseAnomalyJsonlWriter,
    ResponseAnomalyStagingWriter,
    build_jsonl_zstd_manifest,
    iter_jsonl_zstd_records,
)


def _record(case_id):
    return {
        "data_abbr": "ds",
        "id": case_id,
        "uuid": f"u{case_id}",
        "response_anomaly_payload": {
            "tokens": [case_id],
            "topk_logprobs": [
                {
                    str(case_id): -0.123456789012345,
                    str(case_id + 1): -2.987654321098765,
                }
            ],
        },
    }


def _read_shard(path):
    with path.open("rb") as file:
        reader = zstandard.ZstdDecompressor().stream_reader(file)
        return [
            json.loads(line)
            for line in reader.read().decode("utf-8").splitlines()
        ]


def test_jsonl_zstd_writer_round_trips_and_shards(tmp_path):
    writer = ResponseAnomalyJsonlWriter(tmp_path, 3, 2)
    records = [_record(case_id) for case_id in range(3)]
    for record in records:
        writer.write(record)

    manifest = writer.close()

    shards = sorted(tmp_path.glob("*.jsonl.zst"))
    assert len(shards) == 2
    assert manifest["total_rows"] == 3
    assert [item["rows"] for item in manifest["shards"]] == [2, 1]
    assert manifest["shards"][0]["sha256"].startswith("sha256:")
    restored = [item for shard in shards for item in _read_shard(shard)]
    assert restored == records
    assert not list(tmp_path.glob("*.inprogress"))


def test_staging_writer_routes_datasets_and_streams_records(tmp_path):
    writer = ResponseAnomalyStagingWriter(
        {
            "work_dir": str(tmp_path),
            "model_abbr": "modelA",
            "compression_level": 3,
            "rows_per_shard": 2,
        }
    )
    first = _record(1)
    second = _record(2)
    second["data_abbr"] = "ds2"
    writer.write(first)
    writer.write(second)
    writer.close()

    root = tmp_path / "response_anomaly" / "modelA" / "payload_staging"
    assert [item["id"] for item in iter_jsonl_zstd_records(root / "ds")] == [1]
    assert [item["id"] for item in iter_jsonl_zstd_records(root / "ds2")] == [2]


def test_manifest_uses_streaming_row_counts_without_second_decode(tmp_path):
    writer = ResponseAnomalyJsonlWriter(tmp_path, 3, 10)
    writer.write(_record(1))
    writer.close(write_manifest=False)
    shard = next(tmp_path.glob("*.jsonl.zst"))

    manifest = build_jsonl_zstd_manifest(
        tmp_path, 3, "all", {shard.name: 1}
    )

    assert manifest["total_rows"] == 1
    assert manifest["shards"][0]["rows"] == 1
