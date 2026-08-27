import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ais_bench_prefix_cache.artifacts import read_jsonl, sha256_file, validate_artifacts, write_json
from ais_bench_prefix_cache.errors import ArtifactValidationError


class AtomicWriteTest(unittest.TestCase):
    def test_refuses_to_overwrite_existing_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.json"
            write_json(path, {"a": 1}, overwrite=False)
            with self.assertRaisesRegex(ArtifactValidationError, "refusing to overwrite"):
                write_json(path, {"a": 2}, overwrite=False)

    def test_temp_file_is_cleaned_up_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.json"
            temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            with patch("os.replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    write_json(path, {"a": 1}, overwrite=True)
            self.assertFalse(temp.exists())


class ReadJsonlTest(unittest.TestCase):
    def test_unreadable_file_rejected(self):
        with self.assertRaisesRegex(ArtifactValidationError, "cannot read JSONL"):
            read_jsonl(Path("/nonexistent/x.jsonl"))

    def test_invalid_line_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.jsonl"
            path.write_text("not json\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactValidationError, "cannot read JSONL"):
                read_jsonl(path)


class ValidateArtifactsTest(unittest.TestCase):
    def _write_artifacts(self, folder, full_row, request_row, count=None, full_sha=None, request_sha=None) -> Path:
        """构造 manifest 与两个工件文件，返回 manifest 路径。"""
        root = Path(folder)
        result_dir = root / "result"
        result_dir.mkdir()
        full_path = result_dir / "x.full.jsonl"
        request_path = result_dir / "x.requests.jsonl"
        full_path.write_text(json.dumps(full_row) + "\n", encoding="utf-8")
        # requests 行要求严格为 {question, answer, max_tokens}，剥离 sequence_index。
        request_row = {key: value for key, value in request_row.items() if key != "sequence_index"}
        request_path.write_text(json.dumps(request_row) + "\n", encoding="utf-8")
        manifest = {
            "run_id": "pc-test",
            "requests": {"count": count if count is not None else 1},
            "artifacts": {
                "full": {"name": "x.full.jsonl", "sha256": full_sha or sha256_file(full_path)},
                "requests": {"name": "x.requests.jsonl", "sha256": request_sha or sha256_file(request_path)},
            },
        }
        manifest_path = result_dir / "x.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def _row(self, sequence_index=0, question="q", answer="a", max_tokens=2):
        return {"sequence_index": sequence_index, "question": question, "answer": answer, "max_tokens": max_tokens}

    def test_unreadable_manifest_rejected(self):
        with self.assertRaisesRegex(ArtifactValidationError, "cannot read Manifest"):
            validate_artifacts(Path("/nonexistent/x.manifest.json"))

    def test_row_count_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = self._write_artifacts(folder, self._row(), self._row(), count=5)
            with self.assertRaisesRegex(ArtifactValidationError, "row counts do not match"):
                validate_artifacts(manifest_path)

    def test_invalid_sequence_index_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = self._write_artifacts(folder, self._row(sequence_index=3), self._row())
            with self.assertRaisesRegex(ArtifactValidationError, "invalid sequence_index"):
                validate_artifacts(manifest_path)

    def test_unexpected_request_fields_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = self._write_artifacts(
                folder, self._row(), {"question": "q", "answer": "a", "max_tokens": 2, "bogus": 1}
            )
            with self.assertRaisesRegex(ArtifactValidationError, "unexpected fields"):
                validate_artifacts(manifest_path)

    def test_request_differing_from_full_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = self._write_artifacts(folder, self._row(question="q1"), self._row(question="q2"))
            with self.assertRaisesRegex(ArtifactValidationError, "differs from full row"):
                validate_artifacts(manifest_path)

    def test_sha256_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = self._write_artifacts(folder, self._row(), self._row(), full_sha="0" * 64)
            with self.assertRaisesRegex(ArtifactValidationError, "SHA-256 mismatch"):
                validate_artifacts(manifest_path)

    def test_valid_artifacts_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = self._write_artifacts(folder, self._row(), self._row())
            result = validate_artifacts(manifest_path)
            self.assertTrue(result["ok"])
            self.assertEqual(result["rows"], 1)
            self.assertEqual(result["run_id"], "pc-test")


if __name__ == "__main__":
    unittest.main()
