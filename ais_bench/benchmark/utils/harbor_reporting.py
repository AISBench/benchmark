"""Versioned Harbor report completion verification (no adapter dependency)."""

import hashlib
import json


def read_completed_report(path):
    """Read schema 1.0; prefer raw-byte hashes, retain canonical-v1 markers.

    Legacy canonical-v1 hashes use UTF-8 json.dumps(sort_keys=True,
    ensure_ascii=False, allow_nan=False), including its default separators.
    New publishers can use result_file_sha256 to hash the exact report bytes.
    """
    raw = path.read_bytes()
    report = json.loads(raw)
    marker = json.loads(path.with_name("reporting_complete.json").read_bytes())
    if report.get("schema_version") != "1.0":
        raise ValueError("Unsupported normalized report schema_version")
    if marker.get("schema_version", "1.0") != "1.0":
        raise ValueError("Unsupported completion marker schema_version")
    if "result_file_sha256" in marker:
        actual = hashlib.sha256(raw).hexdigest()
        expected = marker["result_file_sha256"]
    else:
        canonical = json.dumps(
            report, sort_keys=True, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        actual = hashlib.sha256(canonical).hexdigest()
        expected = marker["result_sha256"]
    if actual != expected:
        raise ValueError("Completion hash mismatch")
    accuracy = report["metrics"]["accuracy"]
    if accuracy is not None and (
        type(accuracy) not in (int, float) or not 0 <= accuracy <= 1
    ):
        raise ValueError("accuracy must be null or a number in [0, 1]")
    if report["integrity"]["status"] not in ("valid", "invalid"):
        raise ValueError("Invalid integrity status")
    if report["acceptance"]["status"] not in ("pass", "fail", "blocked"):
        raise ValueError("Invalid acceptance status")
    if report["integrity"]["status"] == "invalid" and accuracy is not None:
        raise ValueError("Invalid reports cannot publish accuracy")
    return report
