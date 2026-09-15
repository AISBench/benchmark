import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from ais_bench_prefix_cache.artifacts import read_jsonl
from ais_bench_prefix_cache.multimodal import (
    MULTI_720P_5,
    SINGLE_1080P,
    build_aisbench_config,
    build_exact_token_texts,
    expand_prompt_image_refs,
    prepare_multimodal_datasets,
    report_performance,
    validate_multimodal_manifest,
)


class FakeTokenizer:
    def __len__(self):
        return 256

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        return bytes(token_ids).decode("utf-8")


def _png_bytes(size: tuple[int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color=(12, 34, 56)).save(output, format="PNG")
    return output.getvalue()


def _write_mmmu_parquet(path: Path, sample_id: str, size: tuple[int, int]) -> None:
    image = {"bytes": _png_bytes(size), "path": f"{sample_id}_1.png"}
    table = pa.Table.from_pylist([{"id": sample_id, "image_1": image}])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


class MultimodalBuildTest(unittest.TestCase):
    def test_exact_token_texts_are_distinct_and_exact(self):
        tokenizer = FakeTokenizer()
        questions = [
            "alpha problem has enough ascii characters for truncation",
            "beta problem has enough ascii characters for truncation",
            "tiny",
        ]
        texts = build_exact_token_texts(tokenizer, questions, 30)
        self.assertEqual(len(set(texts)), 3)
        self.assertTrue(all(len(tokenizer.encode(text)) == 30 for text in texts))

    def test_prepare_builds_both_required_shapes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mmmu_dir = root / "MMMU"
            _write_mmmu_parquet(
                mmmu_dir / "Agriculture" / "test-00000-of-00001.parquet",
                "test_Agriculture_29",
                (1920, 1080),
            )
            _write_mmmu_parquet(
                mmmu_dir / "Literature" / "test-00000-of-00001.parquet",
                "test_Literature_48",
                (1280, 720),
            )
            questions = [
                "alpha arithmetic question with more than thirty bytes",
                "beta arithmetic question with more than thirty bytes",
                "gamma arithmetic question with more than thirty bytes",
            ]
            gsm = root / "test.jsonl"
            gsm.write_text(
                "".join(json.dumps({"question": question}) + "\n" for question in questions),
                encoding="utf-8",
            )
            manifest_path = prepare_multimodal_datasets(
                tokenizer_path="fake",
                gsm8k_path=gsm,
                mmmu_parquet_dir=mmmu_dir,
                output_dir=root / "out",
                request_count=3,
                tokenizer_loader=lambda _: FakeTokenizer(),
            )
            result = validate_multimodal_manifest(manifest_path, tokenizer=FakeTokenizer())
            self.assertTrue(result["ok"])
            self.assertEqual(result["output_tokens"], 256)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            single = read_jsonl(Path(manifest["datasets"][SINGLE_1080P]["path"]))
            multi = read_jsonl(Path(manifest["datasets"][MULTI_720P_5]["path"]))
            self.assertEqual(len(single), 3)
            self.assertTrue(all(len(row["image_refs"]) == 1 for row in single))
            self.assertTrue(all(len(row["image_refs"]) == 5 for row in multi))
            self.assertTrue(all(len(set(row["image_refs"])) == 1 for row in multi))
            self.assertTrue(all(row["max_out_len"] == 256 for row in single + multi))
            self.assertEqual(single[0]["question"], multi[0]["question"])
            self.assertEqual(manifest["schema_version"], "2.0")
            self.assertEqual(
                manifest["datasets"][SINGLE_1080P]["image"]["sample_id"],
                "test_Agriculture_29",
            )
            self.assertEqual(
                manifest["datasets"][MULTI_720P_5]["image"]["sample_id"],
                "test_Literature_48",
            )
            self.assertTrue(
                manifest["datasets"][SINGLE_1080P]["image"]["data_url"].startswith(
                    "data:image/png;base64,"
                )
            )
            prompt_text = json.dumps(single[0]["prompt"])
            self.assertNotIn("data:image", prompt_text)
            expanded = expand_prompt_image_refs(
                multi[0]["prompt"],
                {
                    MULTI_720P_5: manifest["datasets"][MULTI_720P_5]["image"]["data_url"]
                },
            )
            content = expanded[0]["content"]
            self.assertEqual([part["type"] for part in content], ["image_url"] * 5 + ["text"])
            self.assertTrue(all(part["image_url"]["url"].startswith("data:image/png;base64,") for part in content[:5]))

    def test_generated_config_is_image_first_streaming_chat(self):
        config = build_aisbench_config(
            dataset_path="dataset.jsonl",
            dataset_abbr=SINGLE_1080P,
            tokenizer_path="tokenizer",
            inference_url="http://127.0.0.1:8000/v1/chat/completions",
            model="vlm",
            work_dir="out",
            image_ref=SINGLE_1080P,
            image_data_url="data:image/png;base64,YQ==",
            batch_size=8,
        )
        self.assertLess(config.index("'image':"), config.index("'text':"))
        self.assertIn("stream=True", config)
        self.assertIn("max_out_len=256", config)
        self.assertIn("batch_size=8", config)
        self.assertIn("input_columns=['content', 'max_out_len']", config)
        self.assertIn("Base64RefMMPromptTemplate", config)
        self.assertEqual(config.count("data:image/png;base64,YQ=="), 1)
        compile(config, "generated.py", "exec")


class PerformanceReportTest(unittest.TestCase):
    def test_report_extracts_acceptance_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            perf = root / "run" / "performances" / "model"
            perf.mkdir(parents=True)
            csv_path = perf / f"{SINGLE_1080P}.csv"
            csv_path.write_text(
                "Performance Parameters,Stage,Average,Min,Max,Median,P75,P90,P99,N\n"
                "TTFT,total,10 ms,1 ms,20 ms,9 ms,11 ms,15 ms,19 ms,3\n"
                "TPOT,total,2 ms,1 ms,3 ms,2 ms,2 ms,3 ms,3 ms,3\n"
                "ITL,total,2 ms,1 ms,4 ms,2 ms,2 ms,3 ms,4 ms,765\n",
                encoding="utf-8",
            )
            csv_path.with_suffix(".json").write_text(
                json.dumps({"Request Throughput": {"total": "1 req/s"}}),
                encoding="utf-8",
            )
            result = report_performance(root, SINGLE_1080P)
            self.assertEqual(set(result["metrics"]), {"TTFT", "TPOT", "ITL"})
            self.assertEqual(result["metrics"]["TTFT"]["Average"], "10 ms")
            self.assertEqual(result["common"]["Request Throughput"]["total"], "1 req/s")


if __name__ == "__main__":
    unittest.main()
