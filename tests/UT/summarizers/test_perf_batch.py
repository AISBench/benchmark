import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from ais_bench.benchmark.summarizers import perf_batch
from ais_bench.benchmark.summarizers.perf_batch import (
    CSV_HEADERS,
    REPORT_HTML_NAME,
    REPORT_MARKER,
    SUMMARY_CSV_NAME,
    _escape_csv_text,
    _format_value,
    _parse_number,
    _saturation_note,
    aggregate_batch_perf,
    render_batch_perf_report,
)

SYNTHETIC_TYPE = "ais_bench.benchmark.datasets.synthetic.SyntheticDataset"


def _make_combo(index, batch_size, model_abbr, dataset_abbr="synthetic",
                stream=True, ttft_valid=True, request_size=1024,
                prefix_len=0, num_prompts=64):
    return {
        "index": index,
        "model_abbr": model_abbr,
        "dataset_abbr": dataset_abbr,
        "batch_size": batch_size,
        "num_prompts": num_prompts,
        "request_count": 128,
        "model_config": {
            "abbr": model_abbr,
            "stream": stream,
            "ttft_valid": ttft_valid,
            "use_timestamp": False,
            "max_out_len": 512,
        },
        "dataset_config": {
            "Type": SYNTHETIC_TYPE,
            "RequestSize": request_size,
            "PrefixLen": prefix_len,
        },
    }


def _make_manifest(combos):
    return {
        "schema_version": 1,
        "ais_bench_version": "test",
        "created_at": "2026-01-01T00:00:00",
        "cli_args": {"batch_size": [c["batch_size"] for c in combos]},
        "combos": combos,
    }


def _write_perf(work_dir, model_abbr, dataset_abbr, throughput=1000.0,
                total=10, success=10, duration_ms=2000.0,
                ttft_average=12.5, tpot_average=3.5, itl_p99=4.2):
    perf_dir = Path(work_dir) / "performances" / model_abbr
    perf_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "Output Token Throughput": {"total": f"{throughput} token/s"},
        "Total Requests": {"total": str(total)},
        "Success Requests": {"total": str(success)},
        # 真实 synthetic.json 必含该指标，汇总区原样取值，缺失时留空
        "Failed Requests": {"total": str(total - success)},
        "Benchmark Duration": {"total": f"{duration_ms} ms"},
    }
    with open(perf_dir / f"{dataset_abbr}.json", "w", encoding="utf-8") as file:
        json.dump(common, file)
    rows = [
        ["Performance Parameters", "Stage", "Average", "Min", "Max",
         "Median", "P75", "P90", "P99"],
        ["TTFT", "total", str(ttft_average), "1", "30", "10", "15", "20", "25"],
        ["TPOT", "total", str(tpot_average), "1", "5", "3", "3.4", "4", "4.5"],
        ["ITL", "total", "3.0", "1", "5", "3", "3.2", "3.8", str(itl_p99)],
    ]
    with open(perf_dir / f"{dataset_abbr}.csv", "w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(rows)


def _write_manifest(work_dir, combos):
    manifest = _make_manifest(combos)
    with open(Path(work_dir) / "combos_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file)
    return manifest


def _read_sections(work_dir):
    """按空行切分汇总 CSV：返回 (指标区, 汇总区)；汇总区自带独立表头。"""
    with open(Path(work_dir) / SUMMARY_CSV_NAME, "r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    header, body = rows[0], rows[1:]
    if [] not in body:
        return [dict(zip(header, row)) for row in body], []
    split_at = body.index([])
    run_header = body[split_at + 1]
    return ([dict(zip(header, row)) for row in body[:split_at]],
            [dict(zip(run_header, row)) for row in body[split_at + 2:]])


def _read_summary(work_dir):
    return _read_sections(work_dir)[0]


class TestAggregateBatchPerf(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.work_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_skip_without_manifest(self):
        """无 manifest 时零开销跳过，不落盘任何产物"""
        artifacts = aggregate_batch_perf(self.work_dir)
        self.assertIsNone(artifacts.csv_path)
        self.assertIsNone(artifacts.html_path)
        self.assertFalse((self.work_dir / SUMMARY_CSV_NAME).exists())

    def test_skip_with_single_combo(self):
        """单组合不聚合（单值路径零开销）"""
        _write_manifest(self.work_dir, [_make_combo(0, 4, "m-bs4")])
        artifacts = aggregate_batch_perf(self.work_dir)
        self.assertIsNone(artifacts.csv_path)
        self.assertFalse((self.work_dir / SUMMARY_CSV_NAME).exists())

    def test_missing_combo_marked_and_others_kept(self):
        """个别组合结果缺失时标记 fail，不阻塞其余行"""
        _write_manifest(
            self.work_dir,
            [_make_combo(0, 4, "m-bs4"), _make_combo(1, 16, "m-bs16")],
        )
        _write_perf(self.work_dir, "m-bs4", "synthetic")

        artifacts = aggregate_batch_perf(self.work_dir)
        self.assertEqual(artifacts.csv_path, self.work_dir / SUMMARY_CSV_NAME)
        rows = _read_summary(self.work_dir)
        # 正常组合展开为 3 个指标行（身份列逐行填充），缺失组合只保留一行标识
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            [row["Batch Size"] for row in rows], ["4", "4", "4", "16"]
        )
        self.assertEqual(rows[0]["Task Status"], "success")
        self.assertEqual(rows[-1]["Task Status"], "fail")
        self.assertEqual(rows[-1]["Performance Parameters"], "")

    def test_csv_values_and_header(self):
        """CSV 与单任务 synthetic.csv 对齐：每个指标一行，身份列逐行填充"""
        _write_perf(self.work_dir, "m-bs4", "synthetic")
        # 单组合会跳过聚合，这里直接走写表路径验证列口径
        perf_batch._write_summary_csv(
            self.work_dir / SUMMARY_CSV_NAME,
            [perf_batch._build_row(self.work_dir, _make_combo(0, 4, "m-bs4"))],
            self.work_dir,
        )
        with open(self.work_dir / SUMMARY_CSV_NAME, "r", encoding="utf-8") as file:
            header = next(csv.reader(file))
        self.assertEqual(header, CSV_HEADERS)

        rows = _read_summary(self.work_dir)
        self.assertEqual(
            [row["Performance Parameters"] for row in rows],
            ["TTFT", "TPOT", "ITL"],
        )
        self.assertEqual(rows[0]["Model Abbr"], "m-bs4")
        self.assertEqual(rows[0]["Batch Size"], "4")
        self.assertEqual([row["Model Abbr"] for row in rows[1:]],
                         ["m-bs4", "m-bs4"])
        self.assertEqual(rows[0]["Average"], "12.5")
        self.assertEqual(rows[0]["P90"], "20")
        self.assertEqual(rows[0]["P99"], "25")
        self.assertEqual(rows[1]["Average"], "3.5")
        self.assertEqual(rows[2]["P99"], "4.2")

    def test_non_stream_blanks_ttft_tpot_itl(self):
        """非流式（ttft_valid=false）时延迟指标只保留指标名，统计值置空"""
        combo = _make_combo(0, 4, "m-bs4", stream=False, ttft_valid=False)
        _write_manifest(self.work_dir, [combo, _make_combo(1, 16, "m-bs16")])
        _write_perf(self.work_dir, "m-bs4", "synthetic")

        aggregate_batch_perf(self.work_dir)
        rows = _read_summary(self.work_dir)
        for row in rows[:3]:
            for key in ("Average", "Min", "Max", "Median", "P75", "P90", "P99"):
                self.assertEqual(row[key], "", f"{key} should be empty")

    def test_csv_formula_injection_escaped(self):
        """文本列以 = + - @ 开头时加前缀引号，避免 CSV 公式注入"""
        combo = _make_combo(0, 4, "=cmd|'/c calc'!A1")
        _write_manifest(self.work_dir, [combo, _make_combo(1, 16, "m-bs16")])
        _write_perf(self.work_dir, "=cmd|'/c calc'!A1", "synthetic")

        aggregate_batch_perf(self.work_dir)
        rows = _read_summary(self.work_dir)
        self.assertEqual(rows[0]["Model Abbr"], "'=cmd|'/c calc'!A1")

    def test_rows_sorted_by_batch_size(self):
        """行序按 batch_size 升序，便于直接读出并发-吞吐曲线"""
        _write_manifest(
            self.work_dir,
            [
                _make_combo(0, 32, "m-bs32"),
                _make_combo(1, 4, "m-bs4"),
                _make_combo(2, 16, "m-bs16"),
            ],
        )
        for batch_size, abbr in ((32, "m-bs32"), (4, "m-bs4"), (16, "m-bs16")):
            _write_perf(self.work_dir, abbr, "synthetic", throughput=batch_size * 10)
        aggregate_batch_perf(self.work_dir)
        rows = _read_summary(self.work_dir)
        # 身份列逐行填充，故按首次出现顺序取组合档位
        batch_sizes = []
        for row in rows:
            if not batch_sizes or batch_sizes[-1] != row["Batch Size"]:
                batch_sizes.append(row["Batch Size"])
        self.assertEqual(batch_sizes, ["4", "16", "32"])

    def test_run_summary_appended_after_blank_line(self):
        """synthetic.json 汇总区：空行 + 独立表头 + 每组合一行，身份列不变"""
        _write_manifest(
            self.work_dir,
            [_make_combo(0, 4, "m-bs4"), _make_combo(1, 16, "m-bs16")],
        )
        _write_perf(self.work_dir, "m-bs4", "synthetic")

        aggregate_batch_perf(self.work_dir)
        metric_rows, run_rows = _read_sections(self.work_dir)
        # 指标区仍是 CSV 的 3 行 + 缺失组合的 1 行标识
        self.assertEqual(len(metric_rows), 4)
        self.assertEqual(len(run_rows), 2)
        self.assertEqual(
            list(run_rows[0])[:5],
            ["Model Abbr", "Dataset Abbr", "Batch Size", "Num Prompts", "Task Status"],
        )
        self.assertEqual(run_rows[0]["Model Abbr"], "m-bs4")
        self.assertEqual(run_rows[0]["Batch Size"], "4")
        self.assertEqual(run_rows[0]["Task Status"], "success")
        # 表头只写指标名，取值原样保留单位
        self.assertEqual(run_rows[0]["Benchmark Duration"], "2000.0 ms")
        self.assertEqual(run_rows[0]["Total Requests"], "10")
        self.assertEqual(run_rows[0]["Failed Requests"], "0")
        self.assertEqual(run_rows[0]["Success Requests"], "10")
        self.assertEqual(run_rows[0]["Output Token Throughput"], "1000.0 token/s")
        # 结果缺失的组合只保留身份列，指标列留空
        self.assertEqual(run_rows[1]["Batch Size"], "16")
        self.assertEqual(run_rows[1]["Task Status"], "fail")
        self.assertEqual(run_rows[1]["Benchmark Duration"], "")


class TestRenderBatchPerfReport(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.work_dir = Path(self._tmp.name)
        self.combos = [
            _make_combo(0, 4, "m-bs4"),
            _make_combo(1, 16, "m-bs16"),
        ]
        _write_perf(self.work_dir, "m-bs4", "synthetic", throughput=1000.0)
        _write_perf(self.work_dir, "m-bs16", "synthetic", throughput=1200.0)
        self.rows = [
            perf_batch._build_row(self.work_dir, combo) for combo in self.combos
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def test_marker_line_and_escape(self):
        """HTML 首两行为 DOCTYPE 与渲染标记；模板变量被转义"""
        if perf_batch._load_template() is None:
            self.skipTest("jinja2 or report template is unavailable")
        warnings = ["<script>alert(1)</script>"]
        path = render_batch_perf_report(
            self.rows, _make_manifest(self.combos), warnings, self.work_dir
        )
        self.assertIsNotNone(path)
        self.assertEqual(path.name, REPORT_HTML_NAME)
        with open(path, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()
        self.assertEqual(lines[0].strip(), "<!DOCTYPE html>")
        self.assertEqual(lines[1].strip(), REPORT_MARKER)
        content = "\n".join(lines)
        self.assertIn("&lt;script&gt;", content)
        self.assertNotIn("<script>alert(1)</script>", content)
        self.assertIn("AISBench 性能测试报告", content)

    def test_render_failure_degrades_to_none(self):
        """渲染异常只告警，返回 None，不影响已落盘的 CSV"""
        _write_manifest(self.work_dir, self.combos)
        with mock.patch.object(
            perf_batch, "_build_charts", side_effect=RuntimeError("boom")
        ):
            artifacts = aggregate_batch_perf(self.work_dir)
        self.assertIsNotNone(artifacts.csv_path)
        self.assertIsNone(artifacts.html_path)
        self.assertTrue((self.work_dir / SUMMARY_CSV_NAME).exists())
        self.assertFalse((self.work_dir / REPORT_HTML_NAME).exists())

    def test_missing_combo_not_in_details(self):
        """fail 组合只进汇总表，不进入逐组合详情与曲线"""
        _write_manifest(self.work_dir, self.combos)
        missing = _make_combo(2, 64, "m-bs64")
        row = perf_batch._build_row(self.work_dir, missing)
        self.assertEqual(row["status"], "fail")
        groups = perf_batch._sorted_groups(self.rows + [row])
        self.assertTrue(all(
            all(item.get("status") == "success" for item in group_rows)
            for _, group_rows in groups
        ))


class TestPerfBatchHelpers(unittest.TestCase):
    def test_parse_number(self):
        self.assertEqual(_parse_number("123.4 ms"), 123.4)
        self.assertEqual(_parse_number(12), 12.0)
        self.assertEqual(_parse_number("1.2e+03"), 1200.0)
        self.assertIsNone(_parse_number(""))
        self.assertIsNone(_parse_number("abc"))
        self.assertIsNone(_parse_number(None))

    def test_format_value(self):
        self.assertEqual(_format_value(None), "")
        self.assertEqual(_format_value(1.0), "1")
        self.assertEqual(_format_value(1.25), "1.25")
        self.assertEqual(_format_value(True), "true")
        self.assertEqual(_format_value(3), "3")

    def test_escape_csv_text(self):
        self.assertEqual(_escape_csv_text("=1+1"), "'=1+1")
        self.assertEqual(_escape_csv_text("+1"), "'+1")
        self.assertEqual(_escape_csv_text("-1"), "'-1")
        self.assertEqual(_escape_csv_text("@x"), "'@x")
        self.assertEqual(_escape_csv_text("ok"), "ok")
        self.assertEqual(_escape_csv_text(None), "")

    def test_saturation_note(self):
        def row(batch_size, throughput, success_rate=100):
            return {
                "status": "success",
                "batch_size": batch_size,
                "throughput_tok_s": throughput,
                "success_rate": success_rate,
            }

        # 返回 (中文, English)，断言中文文案
        self.assertIn("不足 2 个", _saturation_note([row(4, 1000)])[0])
        self.assertIn("未饱和", _saturation_note([row(4, 1000), row(16, 1200)])[0])
        self.assertIn("接近饱和", _saturation_note([row(4, 1000), row(16, 1020)])[0])
        self.assertIn("已饱和", _saturation_note([row(4, 1000), row(16, 900)])[0])
        self.assertIn("已过载", _saturation_note([row(4, 1000), row(16, 1200, 90)])[0])


if __name__ == "__main__":
    unittest.main()
