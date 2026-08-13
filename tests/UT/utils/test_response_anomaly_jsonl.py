import json

import zstandard

from ais_bench.benchmark.utils.response_anomaly_jsonl import (
    ResponseAnomalyJsonlWriter,
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
