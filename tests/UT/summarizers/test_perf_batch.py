import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from ais_bench.benchmark.summarizers import perf_batch
from ais_bench.benchmark.utils.perf_details import derive_request_metrics
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
                ttft_average=12.5, tpot_average=3.5, itl_p99=4.2,
                stage="total"):
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
        ["TTFT", stage, str(ttft_average), "1", "30", "10", "15", "20", "25"],
        ["TPOT", stage, str(tpot_average), "1", "5", "3", "3.4", "4", "4.5"],
        ["ITL", stage, "3.0", "1", "5", "3", "3.2", "3.8", str(itl_p99)],
    ]
    with open(perf_dir / f"{dataset_abbr}.csv", "w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(rows)


def _write_manifest(work_dir, combos, warnings=None):
    manifest = _make_manifest(combos)
    if warnings is not None:
        manifest["warnings"] = list(warnings)
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

    def test_stable_stage_metrics_used(self):
        """perf/stable_stage 的 stage 是 stable，时延指标需回退取值而非留空"""
        combo = _make_combo(0, 4, "m-bs4")
        _write_perf(self.work_dir, "m-bs4", "synthetic", stage="stable",
                    ttft_average=20.0, tpot_average=4.0)

        row = perf_batch._build_row(self.work_dir, combo)

        self.assertEqual(row["ttft_avg"], 20.0)
        self.assertEqual(row["ttft_p99"], 25.0)
        self.assertEqual(row["tpot_avg"], 4.0)

    def test_csv_formula_injection_escaped(self):
        """文本列以 = + - @ 开头时加前缀引号，避免 CSV 公式注入

        abbr 需同时通过路径安全校验（不含分隔符），否则该行会被整体跳过。
        """
        combo = _make_combo(0, 4, "=1+1")
        _write_manifest(self.work_dir, [combo, _make_combo(1, 16, "m-bs16")])
        _write_perf(self.work_dir, "=1+1", "synthetic")

        aggregate_batch_perf(self.work_dir)
        rows = _read_summary(self.work_dir)
        self.assertEqual(rows[0]["Model Abbr"], "'=1+1")

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

    def test_plotly_inlined_by_default(self):
        """默认内联 plotly.min.js，离线机器打开报告也能出图"""
        if perf_batch._load_template() is None:
            self.skipTest("jinja2 or report template is unavailable")
        with mock.patch.object(
            perf_batch, "_plotly_inline_js",
            return_value="window.__INLINE_PLOTLY__ = 1;",
        ):
            path = render_batch_perf_report(
                self.rows, _make_manifest(self.combos), [], self.work_dir
            )
        content = path.read_text(encoding="utf-8")
        self.assertIn("window.__INLINE_PLOTLY__ = 1;", content)
        self.assertNotIn("cdn.plot.ly", content)

    def test_plotly_cdn_fallback_noted(self):
        """内联不可用时回退 CDN，并在说明区提示离线限制"""
        if perf_batch._load_template() is None:
            self.skipTest("jinja2 or report template is unavailable")
        with mock.patch.object(perf_batch, "_plotly_inline_js", return_value=None):
            path = render_batch_perf_report(
                self.rows, _make_manifest(self.combos), [], self.work_dir
            )
        content = path.read_text(encoding="utf-8")
        self.assertIn(
            'src="https://cdn.plot.ly/plotly-2.27.0.min.js"', content)
        self.assertIn("未内联 plotly", content)

    def test_manifest_warnings_in_report(self):
        """manifest 落盘的展开期告警需透传到报告，不依赖调用方再次传入"""
        if perf_batch._load_template() is None:
            self.skipTest("jinja2 or report template is unavailable")
        _write_manifest(
            self.work_dir, self.combos,
            warnings=["`--batch-size`=16 is greater than the effective request count"],
        )

        artifacts = aggregate_batch_perf(self.work_dir)

        self.assertIsNotNone(artifacts.html_path)
        content = artifacts.html_path.read_text(encoding="utf-8")
        self.assertIn("greater than the effective request count", content)

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

    def test_unexpected_render_error_keeps_traceback(self):
        """未预期异常仍降级，但要带堆栈，避免真实 bug 被静默吞掉"""
        _write_manifest(self.work_dir, self.combos)
        with mock.patch.object(
            perf_batch, "_build_charts", side_effect=RuntimeError("boom")
        ), mock.patch.object(perf_batch.logger, "warning") as warn:
            artifacts = aggregate_batch_perf(self.work_dir)

        self.assertIsNone(artifacts.html_path)
        self.assertTrue(any(
            call.kwargs.get("exc_info") for call in warn.call_args_list
        ))

    def test_template_error_is_expected_failure(self):
        """模板语法/运行时错误属于可预期失败，告警不带堆栈"""
        try:
            from jinja2 import TemplateError
        except ImportError:
            self.skipTest("jinja2 is unavailable")
        _write_manifest(self.work_dir, self.combos)
        template = mock.MagicMock()
        template.render.side_effect = TemplateError("bad template")
        with mock.patch.object(
            perf_batch, "_load_template", return_value=template
        ), mock.patch.object(perf_batch.logger, "warning") as warn:
            artifacts = aggregate_batch_perf(self.work_dir)

        self.assertIsNone(artifacts.html_path)
        self.assertFalse(any(
            call.kwargs.get("exc_info") for call in warn.call_args_list
        ))
        self.assertTrue(warn.called)

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


class TestAbbrPathSafety(unittest.TestCase):
    def test_unsafe_abbr_skipped(self):
        """abbr 含路径分隔符/.. 时跳过该组合，避免读写越出工作目录"""
        for abbr in ("../evil", "a/b", "..", ""):
            combo = _make_combo(1, 8, abbr)
            self.assertIsNone(perf_batch._build_row(Path("."), combo), abbr)

    def test_safe_abbr_kept(self):
        combo = _make_combo(1, 8, "qwen-bs8")
        row = perf_batch._build_row(Path("."), combo)
        self.assertIsNotNone(row)
        self.assertEqual(row["model_abbr"], "qwen-bs8")


class TestDerivedMetrics(unittest.TestCase):
    """公共函数返回秒，调用方自行换算：保证抽取前后报告数值与单位不变。"""

    def test_shared_metrics_are_seconds(self):
        derived = derive_request_metrics([0.0, 0.5, 1.5, 4.0], 4)
        self.assertEqual(derived["latency"], 4.0)
        self.assertEqual(derived["ttft"], 0.5)
        self.assertAlmostEqual(derived["tpot"], (4.0 - 0.5) / 3)
        self.assertEqual(list(derived["itl"]), [1.0, 2.5])
        # 输出 token 不足 2 时不计算 TPOT；点数不足 3 时 ITL 为空
        self.assertIsNone(derive_request_metrics([0.0, 1.0], None)["tpot"])
        self.assertIsNone(derive_request_metrics([0.0, 1.0], 1)["tpot"])
        self.assertEqual(derive_request_metrics([0.0, 1.0], 2)["itl"].size, 0)

    def test_detail_rows_keep_ms_units(self):
        """perf_batch 逐请求曲线仍以毫秒输出，数值与抽取前公式一致"""
        with TemporaryDirectory() as tmp:
            perf_dir = Path(tmp)
            records = [
                {"db_name": "d", "success": True,
                 "time_points": [0.0, 0.5, 1.5, 4.0], "output_tokens": 4},
                {"db_name": "d", "success": True,
                 "time_points": [0.0, 1.0], "output_tokens": None},
            ]
            (perf_dir / "synthetic_details.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )
            rows = perf_batch._load_detail_rows(perf_dir, "synthetic",
                                                ttft_valid=True)

        self.assertEqual(rows[0]["latency"], 4000.0)
        self.assertEqual(rows[0]["ttft"], 500.0)
        self.assertAlmostEqual(rows[0]["tpot"], 3500.0 / 3)
        self.assertAlmostEqual(rows[0]["itl"], 1750.0)
        self.assertEqual(rows[1]["latency"], 1000.0)
        self.assertEqual(rows[1]["ttft"], 1000.0)
        self.assertIsNone(rows[1]["tpot"])
        self.assertIsNone(rows[1]["itl"])

    def test_detail_rows_skip_ttft_when_invalid(self):
        """非流式（ttft_valid=false）不推导 TTFT/TPOT/ITL，避免两时间点退化"""
        with TemporaryDirectory() as tmp:
            perf_dir = Path(tmp)
            (perf_dir / "synthetic_details.jsonl").write_text(
                json.dumps({"db_name": "d", "success": True,
                            "time_points": [0.0, 4.0], "output_tokens": 8}),
                encoding="utf-8",
            )
            rows = perf_batch._load_detail_rows(perf_dir, "synthetic",
                                                ttft_valid=False)

        self.assertEqual(rows[0]["latency"], 4000.0)
        self.assertIsNone(rows[0]["ttft"])
        self.assertIsNone(rows[0]["tpot"])
        self.assertIsNone(rows[0]["itl"])
        self.assertEqual(rows[0]["output_len"], 8)

    def test_request_charts_skip_ttft_when_invalid(self):
        """非流式组合的请求明细不渲染 TTFT/TPOT/ITL 图"""
        detail_rows = [{"success": True, "latency": 4000.0, "ttft": 4000.0,
                        "tpot": 0.0, "itl": None, "input_len": 10,
                        "output_len": 8}]
        charts = perf_batch._build_request_charts(detail_rows, ttft_valid=False)
        if not charts:
            self.skipTest("plotly is unavailable")
        titles = [chart["title_zh"] for chart in charts]
        self.assertNotIn("首Token/每Token/Token间时延", titles)
        self.assertIn("时延", titles)

        valid_charts = perf_batch._build_request_charts(detail_rows,
                                                        ttft_valid=True)
        self.assertIn("首Token/每Token/Token间时延",
                      [chart["title_zh"] for chart in valid_charts])


class TestComparableGroupConclusions(unittest.TestCase):
    """最佳配置/饱和判定/时延增长需按可比分组给出，组间不混算。"""

    @staticmethod
    def _row(batch_size, *, request_size=1024, throughput=1000.0,
             ttft=10.0, e2el=1.0, success_rate=100):
        return {
            "status": "success",
            "batch_size": batch_size,
            "model_abbr": "m",
            "dataset_abbr": "synthetic",
            "stream": True,
            "ttft_valid": True,
            "use_timestamp": False,
            "request_size": request_size,
            "prefix_len": 0,
            "throughput_tok_s": throughput,
            "ttft_avg": ttft,
            "e2el_avg_s": e2el,
            "request_throughput": 10.0,
            "success_rate": success_rate,
            "success_rate_text": f"{success_rate}%",
        }

    def _multi_group_rows(self):
        return [
            self._row(4, request_size=1024, throughput=900.0, ttft=12.0, e2el=1.0),
            self._row(16, request_size=1024, throughput=1200.0, ttft=20.0, e2el=1.2),
            self._row(4, request_size=8192, throughput=100.0, ttft=30.0, e2el=5.0),
            self._row(16, request_size=8192, throughput=90.0, ttft=40.0, e2el=6.0),
        ]

    def test_best_configs_per_group(self):
        """最高吞吐/最低 TTFT 只在组内比较，不被其它组数据带偏"""
        best = perf_batch._best_configs(self._multi_group_rows())

        self.assertEqual(len(best), 2)
        by_label = {group["label"]: group for group in best}
        small = by_label["m/synthetic (size=1024, prefix=0)"]
        large = by_label["m/synthetic (size=8192, prefix=0)"]
        self.assertEqual(small["throughput"]["batch_size"], 16)
        self.assertEqual(large["throughput"]["batch_size"], 4)
        self.assertEqual(small["ttft"]["batch_size"], 4)
        self.assertEqual(large["ttft"]["batch_size"], 4)

    def test_saturation_per_group(self):
        """饱和判定按组给出，多组时标注所属组"""
        notes = perf_batch._saturation_notes(self._multi_group_rows())

        self.assertEqual(len(notes), 2)
        self.assertIn("未饱和", notes[0][0])
        self.assertIn("已饱和", notes[1][0])
        self.assertIn("size=1024", notes[0][0])
        self.assertIn("size=8192", notes[1][0])

    def test_latency_growth_per_group(self):
        """时延增长说明按组计算，不出现跨组首末档对比"""
        rows = self._multi_group_rows()
        notes = perf_batch._recommendations(rows, perf_batch._best_configs(rows))
        growth = [zh for zh, _ in notes if "端到端平均时延" in zh]

        self.assertEqual(len(growth), 2)
        self.assertIn("size=1024", growth[0])
        self.assertIn("size=8192", growth[1])
        # 组内分别为 1.0→1.2 与 5.0→6.0；跨组会得到 6.0×
        self.assertTrue(all("1.2×" in note for note in growth), growth)

    def test_single_group_output_unchanged(self):
        """单组时标签为空、文案与分组前一致"""
        rows = [
            self._row(4, throughput=1000.0),
            self._row(16, throughput=1200.0),
        ]
        best = perf_batch._best_configs(rows)
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0]["label"], "")
        self.assertEqual(perf_batch._saturation_notes(rows),
                         [perf_batch._saturation_note(rows)])
        notes = perf_batch._recommendations(rows, best)
        self.assertTrue(all(not zh.startswith("[") for zh, _ in notes), notes)


class TestChartLabelEscaping(unittest.TestCase):
    """trace name 会内联进 plotly 的 <script>，用户可控 abbr 不能提前闭合脚本。"""

    PAYLOAD = "</script><script>alert(1)</script>"

    def _rows(self):
        rows = []
        for request_size in (1024, 8192):
            rows.append({
                "status": "success",
                "batch_size": 4,
                "model_abbr": self.PAYLOAD,
                "dataset_abbr": "synthetic",
                "stream": True,
                "ttft_valid": True,
                "use_timestamp": False,
                "request_size": request_size,
                "prefix_len": 0,
                "e2el_avg_s": 1.0,
                "e2el_p99_s": 1.5,
                "ttft_avg": 10.0,
                "ttft_p99": 12.0,
                "tpot_avg": 1.0,
                "tpot_p99": 1.2,
                "request_throughput": 10.0,
                "throughput_tok_s": 100.0,
                "total_throughput_tok_s": 120.0,
                "success_rate": 100,
            })
        return rows

    def test_user_abbr_cannot_close_script_tag(self):
        charts = perf_batch._build_charts(self._rows())
        if not (charts.get("latency") or charts.get("throughput")):
            self.skipTest("plotly is unavailable")

        chart_count = sum(len(items) for items in charts.values())
        html_text = "".join(
            chart["html"] for items in charts.values() for chart in items
        )
        self.assertNotIn("<script>alert(1)</script>", html_text)
        self.assertNotIn("</script><script>", html_text)
        # plotly 会把 JSON 字符串中的 "/" 转义为 \u002f，还原后再比对实体化结果
        escaped = perf_batch._escape_chart_text(self.PAYLOAD)
        self.assertIn(escaped, html_text.replace("\\u002f", "/"))
        # 只剩 plotly 自己为每张图输出的闭合标签
        self.assertEqual(html_text.count("</script>"), chart_count)


class TestDetailSampling(unittest.TestCase):
    def test_detail_rows_sampled(self):
        """逐请求点数超限时等距采样，保留首末点"""
        rows = list(range(perf_batch.MAX_DETAIL_POINTS * 2))
        sampled = perf_batch._limit_detail_rows(rows)

        self.assertLessEqual(len(sampled), perf_batch.MAX_DETAIL_POINTS + 1)
        self.assertEqual(sampled[0], rows[0])
        self.assertEqual(sampled[-1], rows[-1])

    def test_detail_rows_kept_when_small(self):
        rows = list(range(10))
        self.assertEqual(perf_batch._limit_detail_rows(rows), rows)


class TestExtraMetricColumns(unittest.TestCase):
    def test_extra_columns_warned(self):
        """calculator 自定义统计项超出固定列宽时需告警，避免静默丢弃"""
        row = {"model_abbr": "m", "dataset_abbr": "synthetic", "status": "success"}
        header = ["Metric"] + ["Total"] * len(perf_batch.PERF_METRIC_HEADERS) + ["Extra"]
        metric_row = ["TTFT"] + ["1"] * (len(header) - 1)
        with TemporaryDirectory() as tmp, \
                mock.patch.object(perf_batch, "_read_metric_rows",
                                  return_value=(header, [metric_row])), \
                mock.patch.object(perf_batch.logger, "warning") as warn:
            expanded = perf_batch._expand_summary_rows([row], Path(tmp))

        self.assertTrue(warn.called)
        self.assertIn("Extra", str(warn.call_args))
        for values in expanded:
            self.assertEqual(len(values),
                             len(perf_batch.IDENTITY_COLUMNS)
                             + len(perf_batch.PERF_METRIC_HEADERS))


class TestTemplatePackaging(unittest.TestCase):
    def test_template_declared_in_manifest(self):
        """报告模板需随包分发：位于包内且登记到 MANIFEST.in"""
        template = (Path(perf_batch.__file__).resolve().parent
                    / "templates" / perf_batch.TEMPLATE_NAME)
        self.assertTrue(template.exists(), f"template missing: {template}")
        manifest = (Path(__file__).resolve().parents[3] / "MANIFEST.in").read_text(
            encoding="utf-8")
        self.assertIn("summarizers/templates *.j2", manifest)


if __name__ == "__main__":
    unittest.main()
