import http.client
import io
import json
import os
import os.path as osp
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mmengine.config import ConfigDict

from ais_bench.benchmark.tasks.custom_tasks.agentx_result import HARNESS_SUMMARY_JSON
from ais_bench.benchmark.tasks.custom_tasks.agentx_task import (
    AGENTX_SCENARIO,
    DATASETS,
    IMAGE_HARNESS_PYTHON,
    PROCESS_TIMEOUT_ENV,
    AgentXTask,
    _apply_extra_args,
    _extra_arg_secrets,
    _http_json,
    _process_timeout,
    _sha256_and_line_count,
    build_agentx_command,
    probe_harness_environment,
    redact_command,
    resolve_agentx_bin,
    resolve_cooldown,
    resolve_harness_python,
    resolve_smoke_overrides,
    resolve_variant,
    validate_agentx_args,
)


class AgentXTaskTestBase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        self.work_dir = self.root / "work"
        self.work_dir.mkdir()

    def model_cfg(self, **overrides):
        model = {
            "abbr": "qwen",
            "model": "qwen3.8",
            "url": "http://127.0.0.1:8000",
            "path": str(self.model_dir),
        }
        model.update(overrides)
        return model

    def dataset_args(self, **overrides):
        args = {"concurrency": 8, "variant": "full"}
        args.update(overrides)
        return args

    def task(self, nested=False, **dataset_overrides):
        dataset = {"abbr": "weka", "args": self.dataset_args(**dataset_overrides)}
        datasets = [[dataset]] if nested else [dataset]
        return AgentXTask(ConfigDict({
            "work_dir": str(self.work_dir),
            "models": [self.model_cfg()],
            "datasets": datasets,
            "cli_args": {"debug": False},
        }))

    def harness_metadata(self):
        return {
            "distribution": "agentx-harness",
            "version": "1.0.13",
            "agentx_version": "1.0.13",
            "python_version": "3.11.9",
            "python": "python",
            "entry": ["agentx"],
            "platform": "test",
        }


class TestAgentXValidation(AgentXTaskTestBase):
    def test_variant_must_be_explicit(self):
        self.assertEqual(resolve_variant("full"), "full")
        self.assertEqual(resolve_variant("256k"), "256k")
        for value in (None, "", "  ", "other"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_variant(value)
        with self.assertRaisesRegex(ValueError, "explicit dataset variant"):
            validate_agentx_args(self.model_cfg(), {"concurrency": 8})
        with self.assertRaisesRegex(ValueError, "Unknown AgentX dataset variant"):
            validate_agentx_args(self.model_cfg(), self.dataset_args(variant="other"))

    def test_concurrency_levels(self):
        self.assertEqual(validate_agentx_args(self.model_cfg(), self.dataset_args(concurrency=8)), [8])
        self.assertEqual(validate_agentx_args(self.model_cfg(), self.dataset_args(concurrency=[8])), [8])
        self.assertEqual(
            validate_agentx_args(
                self.model_cfg(), self.dataset_args(concurrency=[1, 8, 32])
            ),
            [1, 8, 32],
        )

    def test_invalid_concurrency_rejected_before_execution(self):
        for value in (0, -1, 1.5, True, "8", [], [8, 8], [1, 0]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_agentx_args(self.model_cfg(), self.dataset_args(concurrency=value))

    def test_cooldown_default_and_validation(self):
        for value in (None, 0, 5, 2.5):
            with self.subTest(value=value):
                validate_agentx_args(self.model_cfg(), self.dataset_args(cooldown=value))
        self.assertEqual(resolve_cooldown(None), 5.0)
        self.assertEqual(resolve_cooldown(2), 2.0)
        for value in (-1, True, "5", float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "cooldown must be"):
                    validate_agentx_args(self.model_cfg(), self.dataset_args(cooldown=value))

    def test_required_model_url_and_tokenizer(self):
        with self.assertRaisesRegex(ValueError, "non-empty model name"):
            validate_agentx_args(self.model_cfg(model=""), self.dataset_args())
        with self.assertRaisesRegex(ValueError, "base URL"):
            validate_agentx_args(self.model_cfg(url=None), self.dataset_args())
        with self.assertRaisesRegex(ValueError, "base URL without"):
            validate_agentx_args(
                self.model_cfg(url="http://127.0.0.1:8000/v1/chat/completions"),
                self.dataset_args(),
            )
        for endpoint in ("", "v1/chat/completions", "/v1/chat?x=1", "/v1/chat#frag", "/v1 chat"):
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(ValueError, "endpoint must be"):
                    validate_agentx_args(
                        self.model_cfg(endpoint=endpoint), self.dataset_args()
                    )
        validate_agentx_args(
            self.model_cfg(endpoint="/custom/v2/chat"), self.dataset_args()
        )
        with self.assertRaisesRegex(ValueError, "base URL without /custom/v2/chat"):
            validate_agentx_args(
                self.model_cfg(
                    url="http://127.0.0.1:8000/custom/v2/chat",
                    endpoint="/custom/v2/chat",
                ),
                self.dataset_args(),
            )
        with self.assertRaisesRegex(ValueError, r"models\[0\].path"):
            validate_agentx_args(self.model_cfg(path=None), self.dataset_args())
        with self.assertRaises(FileNotFoundError):
            validate_agentx_args(
                self.model_cfg(path=str(self.root / "missing")), self.dataset_args()
            )

    def test_smoke_duration_and_trace_limit_policy(self):
        args = self.dataset_args(run_mode="smoke", duration=2.5, trace_limit=3)
        self.assertEqual(validate_agentx_args(self.model_cfg(), args), [8])
        validate_agentx_args(
            self.model_cfg(), self.dataset_args(run_mode="benchmark", duration=1800)
        )
        for key, value, message in (
            ("duration", 2, ">= 900"),
            ("trace_limit", 3, "only valid in smoke"),
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, message):
                    validate_agentx_args(self.model_cfg(), self.dataset_args(**{key: value}))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            validate_agentx_args(
                self.model_cfg(), self.dataset_args(run_mode="smoke", trace_limit=1.5)
            )

    def test_extra_args_only_add_parameters_not_generated_by_aisbench(self):
        extra_args = (
            "--log-level INFO --verbose --benchmark-duration 901 "
            "--random-seed 7 --concurrency 2 --scenario custom_trace --model Qwen"
        )
        validate_agentx_args(self.model_cfg(), self.dataset_args(extra_args=extra_args))
        ignored = []
        command = build_agentx_command(
            ["agentx"],
            url=self.model_cfg()["url"],
            model="qwen3.8",
            concurrency=8,
            dataset_alias=DATASETS["full"]["alias"],
            artifact_dir="/out/run/artifacts",
            extra_args=extra_args,
            ignored_out=ignored,
        )
        # 同名参数以 AISBench 生成为准，extra args 中的冲突项被忽略并原样返回；
        # --scenario 不是 AISBench 生成的参数，完全由 extra args 决定。
        self.assertEqual(command.count("--model"), 1)
        self.assertEqual(command[command.index("--model") + 1], "qwen3.8")
        self.assertEqual(command.count("--concurrency"), 1)
        self.assertEqual(command[command.index("--concurrency") + 1], "8")
        self.assertEqual(command.count("--scenario"), 1)
        self.assertEqual(command[command.index("--scenario") + 1], "custom_trace")
        self.assertNotIn("Qwen", command)
        self.assertEqual(ignored, ["--concurrency", "2", "--model", "Qwen"])
        # AISBench 未生成的参数照常透传，追加在末尾
        self.assertEqual(command[command.index("--url") + 1], self.model_cfg()["url"])
        self.assertEqual(command[-9:], ["--log-level", "INFO", "--verbose", "--benchmark-duration",
                                        "901", "--random-seed", "7", "--scenario", "custom_trace"])
        braced = build_agentx_command(
            ["agentx"],
            url=self.model_cfg()["url"],
            model="qwen3.8",
            concurrency=8,
            dataset_alias=DATASETS["full"]["alias"],
            artifact_dir="/out/run/artifacts",
            extra_args="{ --concurrency 1 --model Qwen }",
        )
        self.assertEqual(braced.count("--concurrency"), 1)
        self.assertEqual(braced[braced.index("--concurrency") + 1], "8")
        self.assertNotIn("Qwen", braced)
        for item in ("", "  ", [], ["--log-level=INFO"]):
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    validate_agentx_args(self.model_cfg(), self.dataset_args(extra_args=item))

    def test_extra_args_cannot_switch_dataset_source(self):
        """语料来源选项互斥：AISBench 生成一个后，extra args 里的另一个也不生效。"""
        other_dir = str(self.root / "other-corpus")
        ignored = []
        command = build_agentx_command(
            ["agentx"],
            url=self.model_cfg()["url"],
            model="qwen3.8",
            concurrency=8,
            dataset_alias=DATASETS["full"]["alias"],
            artifact_dir="/out/run/artifacts",
            extra_args=f"--hf-weka-dataset {other_dir} --random-seed 7",
            ignored_out=ignored,
        )
        self.assertEqual(command.count("--public-dataset"), 1)
        self.assertNotIn("--hf-weka-dataset", command)
        self.assertEqual(ignored, ["--hf-weka-dataset", other_dir])
        self.assertEqual(command[-2:], ["--random-seed", "7"])

        ignored = []
        command = build_agentx_command(
            ["agentx"],
            url=self.model_cfg()["url"],
            model="qwen3.8",
            concurrency=8,
            dataset_dir=str(self.model_dir),
            artifact_dir="/out/run/artifacts",
            extra_args="--public-dataset other-alias",
            ignored_out=ignored,
        )
        self.assertEqual(command.count("--hf-weka-dataset"), 1)
        self.assertNotIn("--public-dataset", command)
        self.assertEqual(ignored, ["--public-dataset", "other-alias"])

    def test_raw_mode_skips_model_url_and_variant_checks(self):
        args = {
            "concurrency": 8, "raw_mode": True,
            "extra_args": "{ --url http://127.0.0.1:8000 --model qwen3.8 }",
        }
        self.assertEqual(validate_agentx_args({}, args), [8])
        with self.assertRaises(ValueError):
            validate_agentx_args({}, dict(args, concurrency=0))

    def test_raw_command_relies_on_extra_args(self):
        extra = "{ --url http://127.0.0.1:8000 --model Qwen --public-dataset alias }"
        command = build_agentx_command(
            ["agentx"], concurrency=8, artifact_dir="/out/raw",
            extra_args=extra, run_mode="smoke", raw=True,
        )
        # raw 模式基础命令只保留并发/产物目录，其余全部来自 extra args
        self.assertEqual(command[:4], ["agentx", "profile", "--concurrency", "8"])
        self.assertEqual(command[4:6], ["--output-artifact-dir", "/out/raw"])
        for option in ("--url", "--model", "--public-dataset"):
            with self.subTest(option=option):
                self.assertEqual(command.count(option), 1)
        # 非 raw 模式仍要求语料来源，避免静默生成无 --public-dataset/--hf-weka-dataset 的命令
        with self.assertRaises(ValueError):
            build_agentx_command(["agentx"], concurrency=1, artifact_dir="/out")

    def test_probe_engine_is_skipped_without_url(self):
        task = AgentXTask(ConfigDict({
            "work_dir": str(self.work_dir),
            "models": [{"abbr": "agentx-raw"}],
            "datasets": [{"abbr": "weka", "args": {"concurrency": 8, "raw_mode": True}}],
            "cli_args": {"debug": False},
        }))
        engine, errors = task._probe_engine({})
        self.assertEqual(errors, [])
        self.assertIsNone(engine["base_url"])
        self.assertIn("skipped", engine["probe"])

    def test_environment_validation(self):
        with self.assertRaisesRegex(ValueError, "KEY=VALUE|env variable"):
            validate_agentx_args(self.model_cfg(), self.dataset_args(env={"BAD-KEY": "x"}))
        with self.assertRaisesRegex(ValueError, "must not override"):
            validate_agentx_args(self.model_cfg(), self.dataset_args(env={"AIPERF_SCENARIO": "x"}))


class TestAgentXCommand(AgentXTaskTestBase):
    def test_public_benchmark_command_golden(self):
        command = build_agentx_command(
            ["agentx"],
            url=self.model_cfg()["url"] + "/",
            model="qwen3.8",
            concurrency=8,
            dataset_alias=DATASETS["full"]["alias"],
            artifact_dir="/out/run/artifacts",
            tokenizer_path=str(self.model_dir),
            tokenizer_trust_remote_code=True,
            api_key="secret",
            extra_args="--log-level=INFO",
        )
        self.assertEqual(command[:2], ["agentx", "profile"])
        # --scenario 属于 agentx 入口参数，AISBench 不注入
        self.assertNotIn("--scenario", command)
        self.assertEqual(command[command.index("--url") + 1], "http://127.0.0.1:8000")
        self.assertEqual(command[command.index("--endpoint") + 1], "/v1/chat/completions")
        self.assertEqual(command[command.index("--concurrency") + 1], "8")
        self.assertEqual(command[command.index("--tokenizer") + 1], str(self.model_dir))
        self.assertIn("--tokenizer-trust-remote-code", command)
        self.assertIn("--public-dataset", command)
        self.assertNotIn("--hf-weka-dataset", command)
        self.assertNotIn("--input-file", command)
        for forbidden in ("--benchmark-duration", "--num-dataset-entries", "--seed", "--random-seed"):
            self.assertNotIn(forbidden, command)

    def test_custom_endpoint_is_forwarded(self):
        command = build_agentx_command(
            ["agentx"],
            url=self.model_cfg()["url"],
            model="qwen3.8",
            concurrency=8,
            dataset_alias=DATASETS["full"]["alias"],
            artifact_dir="/out/custom",
            endpoint="/custom/v2/chat",
        )
        self.assertEqual(command[command.index("--endpoint") + 1], "/custom/v2/chat")
        self.assertEqual(command[command.index("--url") + 1], "http://127.0.0.1:8000")

    def test_local_and_smoke_command_golden(self):
        dataset_dir = self.root / "dataset"
        dataset_dir.mkdir()
        command = build_agentx_command(
            ["python", "-m", "agentx_cli"],
            url=self.model_cfg()["url"],
            model="qwen3.8",
            concurrency=1,
            dataset_alias=None,
            dataset_dir=str(dataset_dir),
            artifact_dir="/out/smoke",
            run_mode="smoke",
            duration=2,
            trace_limit=4,
        )
        self.assertEqual(command[command.index("--hf-weka-dataset") + 1], str(dataset_dir))
        self.assertIn("--unsafe-override", command)
        self.assertNotIn("--public-dataset", command)
        self.assertEqual(command[command.index("--benchmark-duration") + 1], "2")
        self.assertEqual(command[command.index("--num-dataset-entries") + 1], "4")

    def test_extra_args_dedup_of_flag_options_keeps_next_token(self):
        # --unsafe-override / --tokenizer-trust-remote-code 都不带取值，
        # 与基础命令同名时只丢弃旗标本身，不能连带删掉后面的选项。
        command = build_agentx_command(
            ["agentx"],
            url=self.model_cfg()["url"],
            model="qwen3.8",
            concurrency=8,
            dataset_alias=None,
            dataset_dir=str(self.root / "dataset"),
            artifact_dir="/out/local",
            tokenizer_path=str(self.model_dir),
            tokenizer_trust_remote_code=True,
            api_key="k",
            extra_args=(
                "--unsafe-override --verbose --tokenizer-trust-remote-code --log-level INFO"
            ),
        )
        self.assertEqual(command.count("--unsafe-override"), 1)
        self.assertEqual(command.count("--tokenizer-trust-remote-code"), 1)
        self.assertEqual(command.count("--hf-weka-dataset"), 1)
        self.assertEqual(command[command.index("--tokenizer") + 1], str(self.model_dir))
        self.assertEqual(command[command.index("--api-key") + 1], "k")
        self.assertEqual(command[-3:], ["--verbose", "--log-level", "INFO"])

    def test_extra_args_dedup_keeps_aisbench_value(self):
        # 去重不依赖名单：任何无值开关都按“下一个 token 不是选项”判定，
        # 被忽略的片段完整返回给调用方记录。
        self.assertEqual(
            _apply_extra_args(["--custom-flag", "--model", "m"], ["--custom-flag"]),
            (["--custom-flag", "--model", "m"], ["--custom-flag"]),
        )
        self.assertEqual(
            _apply_extra_args(["--unsafe-override"], ["--unsafe-override", "--verbose"]),
            (["--unsafe-override", "--verbose"], ["--unsafe-override"]),
        )
        self.assertEqual(
            _apply_extra_args(["--model", "m"], ["--verbose"]),
            (["--model", "m", "--verbose"], []),
        )

    def test_benchmark_duration_override_is_forwarded(self):
        command = build_agentx_command(
            ["agentx"],
            url=self.model_cfg()["url"],
            model="qwen3.8",
            concurrency=8,
            dataset_alias=DATASETS["full"]["alias"],
            artifact_dir="/out/benchmark",
            run_mode="benchmark",
            duration=1800.0,
        )
        self.assertEqual(command[command.index("--benchmark-duration") + 1], "1800.0")
        self.assertNotIn("--unsafe-override", command)
        self.assertNotIn("--num-dataset-entries", command)

    def test_command_redaction(self):
        command = ["agentx", "--api-key", "secret", "--api-key=other", "--model", "m"]
        redacted = redact_command(command)
        self.assertNotIn("secret", redacted)
        self.assertNotIn("other", redacted)
        self.assertIn("--api-key=<redacted>", redacted)
        self.assertEqual(redacted[-1], "m")

    def test_command_redaction_does_not_forge_tail_value(self):
        # --api-key 位于末尾时没有取值，不能凭空补一个 <redacted>。
        self.assertEqual(redact_command(["agentx", "--api-key"]), ["agentx", "--api-key"])

    def test_tokenizer_path_is_expanded(self):
        command = build_agentx_command(
            ["agentx"], url="http://127.0.0.1:8000", model="m", concurrency=1,
            dataset_alias=DATASETS["full"]["alias"], artifact_dir="/out/benchmark",
            tokenizer_path="~/tokenizer",
        )
        self.assertEqual(
            command[command.index("--tokenizer") + 1],
            osp.abspath(osp.expanduser("~/tokenizer")),
        )

    def test_command_makes_relative_dataset_and_tokenizer_absolute(self):
        """Harness 的 cwd 是 work_dir，相对路径必须在生成命令时转成绝对路径。"""
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.root)
        expected_corpus = osp.join(os.getcwd(), "corpus")
        expected_tokenizer = osp.join(os.getcwd(), "model")
        command = build_agentx_command(
            ["agentx"], url="http://127.0.0.1:8000", model="m", concurrency=1,
            dataset_dir="corpus", artifact_dir="/out/benchmark", tokenizer_path="model",
        )
        self.assertEqual(
            command[command.index("--hf-weka-dataset") + 1], expected_corpus
        )
        self.assertEqual(command[command.index("--tokenizer") + 1], expected_tokenizer)

    def test_effective_artifact_dir_resolves_relative_against_harness_cwd(self):
        self.assertEqual(
            AgentXTask._effective_artifact_dir(
                {"--output-artifact-dir": "relative_results"}, str(self.work_dir)
            ),
            osp.join(str(self.work_dir), "relative_results"),
        )

    def test_drain_stdout_reads_partial_output_without_blocking(self):
        """管道里只有无换行的半行数据时，排空也不能阻塞在 readline 上。"""
        read_fd, write_fd = os.pipe()
        stream = os.fdopen(read_fd, "r", encoding="utf-8")
        try:
            os.write(write_fd, b"partial-line-without-newline")
            log = io.StringIO()
            start = time.monotonic()
            self.task()._drain_stdout(SimpleNamespace(stdout=stream), log, [])
            self.assertLess(time.monotonic() - start, 2.0)
            self.assertIn("partial-line-without-newline", log.getvalue())
        finally:
            stream.close()
            os.close(write_fd)

    def test_process_timeout_env(self):
        self.assertIsNone(_process_timeout({}))
        self.assertEqual(_process_timeout({PROCESS_TIMEOUT_ENV: "1800"}), 1800.0)
        for bad in ("0", "-5", "abc", "nan"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _process_timeout({PROCESS_TIMEOUT_ENV: bad})

    def test_extra_arg_secrets_collect_known_sensitive_values(self):
        self.assertEqual(
            _extra_arg_secrets("--api-key k1 --api-key=k2 --log-level INFO"),
            ["k1", "k2"],
        )
        self.assertEqual(_extra_arg_secrets(None), [])
        self.assertEqual(_extra_arg_secrets("--log-level INFO"), [])

    def test_http_json_degrades_http_exceptions_to_error(self):
        # IncompleteRead 等 HTTPException 不是 URLError/OSError/ValueError，
        # 必须同样降级为探测错误，不能逃逸吞掉整档结果。
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=http.client.IncompleteRead(b"partial"),
        ):
            payload, error = _http_json("http://127.0.0.1:8000/version")
        self.assertIsNone(payload)
        self.assertIn("IncompleteRead", error)


class TestAgentXDatasetAndEnvironment(AgentXTaskTestBase):
    def test_local_hash_match_and_mismatch_policy(self):
        dataset_dir = self.root / "dataset"
        dataset_dir.mkdir()
        trace = dataset_dir / "traces.jsonl"
        trace.write_bytes(b'{"session": 1}\n')
        actual = _sha256_and_line_count(str(trace))[0]
        task = self.task(dataset_path=str(dataset_dir))
        with mock.patch.dict(DATASETS["full"], {"sha256": actual}):
            dataset = task._resolve_dataset("benchmark")
        self.assertEqual(dataset["actual_sha256"], actual)
        self.assertEqual(dataset["verification"], ["local_content_hash_verified"])
        self.assertFalse(dataset["sha256_mismatch"])

        with self.assertRaisesRegex(ValueError, "AGENTX_DATASET_HASH_MISMATCH"):
            task._resolve_dataset("benchmark")
        smoke = self.task(dataset_path=str(dataset_dir), run_mode="smoke")
        dataset = smoke._resolve_dataset("smoke")
        self.assertTrue(dataset["sha256_mismatch"])
        self.assertEqual(dataset["actual_sha256"], actual)
        self.assertEqual(dataset["expected_sha256"], DATASETS["full"]["sha256"])

    def test_raw_dataset_provenance_from_public_dataset_flag(self):
        """raw 模式溯源直接取命令里的 --public-dataset，已知别名映射到内置 registry。"""
        alias = DATASETS["full"]["alias"]
        task = self.task(
            raw_mode=True,
            extra_args=f"{{ --public-dataset {alias} --num-dataset-entries 12 }}",
        )
        dataset = task._resolve_dataset("smoke")
        self.assertEqual(dataset["source"], "public")
        self.assertTrue(dataset["registered"])
        self.assertEqual(dataset["variant"], "full")
        self.assertEqual(dataset["registry_alias"], alias)
        self.assertEqual(dataset["repo"], DATASETS["full"]["hf_repo"])
        self.assertEqual(dataset["revision"], DATASETS["full"]["hf_revision"])
        self.assertEqual(dataset["expected_sha256"], DATASETS["full"]["sha256"])
        self.assertEqual(dataset["num_dataset_entries"], 12)
        self.assertEqual(dataset["provenance_origin"], "--public-dataset")

    def test_raw_unregistered_public_dataset_alias_is_not_official(self):
        """未注册别名只记录来源，不能伪装成官方语料的 registry 记录。"""
        task = self.task(raw_mode=True, extra_args="{ --public-dataset custom-trace }")
        dataset = task._resolve_dataset("benchmark")
        self.assertEqual(dataset["source"], "public")
        self.assertFalse(dataset["registered"])
        self.assertIsNone(dataset["registry_alias"])
        self.assertEqual(dataset["variant"], "raw")
        self.assertEqual(dataset["provenance_origin"], "--public-dataset")

    def test_raw_dataset_provenance_from_local_dataset_flag(self):
        dataset_dir = self.root / "dataset"
        dataset_dir.mkdir()
        task = self.task(raw_mode=True, extra_args=f"{{ --hf-weka-dataset {dataset_dir} }}")
        dataset = task._resolve_dataset("smoke")
        self.assertEqual(dataset["source"], "local")
        self.assertEqual(dataset["path"], osp.realpath(str(dataset_dir)))
        self.assertEqual(dataset["verification"], ["not_verified"])
        self.assertEqual(dataset["provenance_origin"], "--hf-weka-dataset")

    def test_raw_dataset_without_flags_stays_raw(self):
        dataset = self.task(raw_mode=True)._resolve_dataset("smoke")
        self.assertEqual(dataset["source"], "raw")
        self.assertEqual(dataset["variant"], "raw")

    def test_probe_accepts_supported_environment(self):
        probe = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "python_version": "3.11.9",
                "agentx_version": "1.0.13",
                "upstream_aiperf": False,
            }),
            stderr="",
        )
        help_result = SimpleNamespace(returncode=0, stdout="help", stderr="")
        with mock.patch("subprocess.run", side_effect=[probe, help_result]) as run:
            metadata = probe_harness_environment("python", {}, agentx_bin="agentx")
        self.assertEqual(metadata["version"], "1.0.13")
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            # 子进程一律按 UTF-8 解码，非 UTF-8 locale 下上游输出不会中断整档测评。
            self.assertEqual(call.kwargs.get("encoding"), "utf-8")
            self.assertEqual(call.kwargs.get("errors"), "replace")

    def test_probe_rejects_python_version_harness_version_and_aiperf(self):
        cases = (
            ({"python_version": "3.10.14", "agentx_version": "1.0.13", "upstream_aiperf": False},
             "Python 3.11~3.13"),
            ({"python_version": "3.11.9", "agentx_version": "1.0.12", "upstream_aiperf": False},
             "AGENTX_VERSION_UNSUPPORTED"),
            ({"python_version": "3.11.9", "agentx_version": "1.0.13", "upstream_aiperf": True},
             "upstream PyPI aiperf"),
        )
        for metadata, message in cases:
            with self.subTest(message=message):
                completed = SimpleNamespace(
                    returncode=0, stdout=json.dumps(metadata), stderr=""
                )
                with mock.patch("subprocess.run", return_value=completed):
                    with self.assertRaisesRegex(RuntimeError, message):
                        probe_harness_environment("python", {}, agentx_bin="agentx")

    def test_explicit_python_and_environment_variable_resolution(self):
        environ = {"AISBENCH_AGENTX_PYTHON": "/env/agentx/bin/python"}
        self.assertEqual(
            resolve_harness_python({"agentx_python": "~/agentx/bin/python"}, environ),
            osp.expanduser("~/agentx/bin/python"),
        )
        self.assertEqual(resolve_harness_python({}, environ), "/env/agentx/bin/python")

    def test_image_and_current_interpreter_fallbacks(self):
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.osp.isfile", return_value=True
        ):
            self.assertEqual(resolve_harness_python({}, {}), IMAGE_HARNESS_PYTHON)

        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.osp.isfile", return_value=False
        ):
            self.assertEqual(resolve_harness_python({}, {}), sys.executable)

    def test_agentx_entry_stays_with_selected_python(self):
        bin_dir = self.root / "venv" / "bin"
        bin_dir.mkdir(parents=True)
        python_exe = bin_dir / "python"
        python_exe.touch()
        entry = bin_dir / "agentx"
        entry.touch()
        self.assertEqual(
            resolve_agentx_bin(None, str(python_exe)), [str(entry.resolve())]
        )
        entry.unlink()
        self.assertEqual(
            resolve_agentx_bin(None, str(python_exe)),
            [str(python_exe), "-m", "agentx_cli"],
        )
        self.assertEqual(
            resolve_agentx_bin("/custom/agentx", str(python_exe)), ["/custom/agentx"]
        )
        # 校验阶段按 expanduser 判定存在，执行阶段必须同样展开。
        self.assertEqual(
            resolve_agentx_bin("~/agentx", str(python_exe)), [osp.expanduser("~/agentx")]
        )


    def test_test_params_scenario_comes_from_final_command(self):
        """test_params 以最终 argv 的 --scenario 为准，不硬编码固定值。"""
        plan = {
            "argv": ["agentx", "profile", "--scenario", "custom_trace", "--concurrency", "8"],
            "concurrency": 8, "trace_limit": None,
        }
        params = self.task(raw_mode=True)._test_params(plan, {})
        self.assertEqual(params["scenario"], "custom_trace")

    def test_test_params_falls_back_to_harness_default_scenario(self):
        plan = {
            "argv": ["agentx", "profile", "--concurrency", "8"],
            "concurrency": 8, "trace_limit": None,
        }
        params = self.task()._test_params(plan, {})
        # AISBench 不注入 --scenario，未显式传入时记录 agentx 入口的默认口径
        self.assertEqual(params["scenario"], AGENTX_SCENARIO)


class TestAgentXTaskLifecycle(AgentXTaskTestBase):
    def test_direct_and_legacy_nested_dataset_shapes(self):
        direct = self.task(nested=False)
        legacy = self.task(nested=True)
        self.assertEqual(direct.dataset_cfg["abbr"], "weka")
        self.assertEqual(legacy.dataset_cfg["abbr"], "weka")

    def test_dry_run_writes_redacted_plan_without_popen(self):
        task = self.task(api_key="secret")
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value="python",
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=self.harness_metadata(),
        ), mock.patch("subprocess.Popen") as popen, mock.patch.object(
            task, "_execute_plan"
        ) as execute:
            plan = task.dry_run()
        popen.assert_not_called()
        execute.assert_not_called()
        run_id = "agentx_full_batchsize_8"
        plan_path = self.work_dir / "performances" / run_id / "execution_plan.redacted.json"
        self.assertTrue(plan_path.is_file())
        self.assertFalse((self.work_dir / "configs" / run_id).exists())
        content = plan_path.read_text(encoding="utf-8")
        self.assertNotIn("secret", content)
        self.assertIn("<redacted>", content)
        self.assertEqual(plan["runs"][0]["run_id"], "agentx_full_batchsize_8")

    def test_reuse_overwrites_without_deleting_run_dirs(self):
        """--reuse 只清理上一轮结果文件，不删除目录和其他产物"""
        run_id = "agentx_full_batchsize_8"
        artifact_dir = self.work_dir / "performances" / run_id / "agentx_raw"
        artifact_dir.mkdir(parents=True)
        keep = artifact_dir / "user_note.txt"
        keep.write_text("keep", encoding="utf-8")
        stale_summary = artifact_dir / "profile_export_aiperf.json"
        stale_summary.write_text("{}", encoding="utf-8")
        logs_dir = self.work_dir / "logs" / run_id
        logs_dir.mkdir(parents=True)
        summary_log = logs_dir / "agentx_summary.log"
        summary_log.write_text("stale", encoding="utf-8")
        manifest_path = self.work_dir / "performances" / run_id / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")

        task = AgentXTask(ConfigDict({
            "work_dir": str(self.work_dir),
            "models": [self.model_cfg()],
            "datasets": [{"abbr": "weka", "args": self.dataset_args()}],
            "cli_args": {"debug": False, "reuse": "latest"},
        }))
        completed = {
            "run": {
                "status": "completed", "parse_status": "success",
                "benchmark_valid": True, "comparison_scope": "official_agentx",
            },
            "artifacts": {"harness_log": "agentx_stdout.log"},
        }
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value="python",
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=self.harness_metadata(),
        ), mock.patch.object(task, "_execute_plan", return_value=completed):
            task.run()

        self.assertTrue(artifact_dir.is_dir())
        self.assertTrue(logs_dir.is_dir())
        self.assertTrue(keep.exists())
        self.assertTrue(summary_log.exists())
        self.assertFalse(stale_summary.exists())
        self.assertFalse(manifest_path.exists())

    def test_prepare_ignores_conflicting_output_artifact_dir_override(self):
        """extra args 与 AISBench 同名时以 AISBench 为准，产物目录仍按档位生成。"""
        custom = self.root / "custom_artifacts"
        task = self.task(
            raw_mode=True,
            extra_args=f"{{ --random-seed 7 --output-artifact-dir {custom} }}",
        )
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value="python",
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=self.harness_metadata(),
        ):
            plan = task._prepare()[0]
        expected = osp.abspath(osp.join(
            self.work_dir, "performances", "agentx_raw_batchsize_8", "agentx_raw"
        ))
        self.assertEqual(plan["run_id"], "agentx_raw_batchsize_8")
        self.assertEqual(plan["artifact_dir"], expected)
        self.assertEqual(plan["summary_path"], str(Path(expected) / "profile_export_aiperf.json"))
        self.assertEqual(plan["ignored_extra_args"], ["--output-artifact-dir", str(custom)])
        self.assertTrue(Path(expected).is_dir())
        self.assertFalse(custom.exists())
        plan_text = Path(plan["plan_path"]).read_text(encoding="utf-8")
        self.assertIn("ignored_extra_args", plan_text)
        self.assertIn("--output-artifact-dir", plan_text)

    def _reuse_raw_task(self):
        return AgentXTask(ConfigDict({
            "work_dir": str(self.work_dir),
            "models": [self.model_cfg()],
            "datasets": [{"abbr": "weka", "args": self.dataset_args(
                raw_mode=True,
                extra_args="{ --random-seed 7 }",
            )}],
            "cli_args": {"debug": False, "reuse": "latest"},
        }))

    def test_reuse_keeps_files_in_aisbench_artifact_dir(self):
        """产物目录只清理本轮结果文件，保留其他文件"""
        artifact_dir = (
            self.work_dir / "performances" / "agentx_raw_batchsize_8" / "agentx_raw"
        )
        artifact_dir.mkdir(parents=True)
        keep = artifact_dir / "stale.txt"
        keep.write_text("stale", encoding="utf-8")
        old_summary = artifact_dir / "profile_export_aiperf.json"
        old_summary.write_text("{}", encoding="utf-8")
        task = self._reuse_raw_task()
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value="python",
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=self.harness_metadata(),
        ):
            plan = task._prepare()[0]
        self.assertEqual(plan["artifact_dir"], osp.abspath(str(artifact_dir)))
        self.assertTrue(artifact_dir.is_dir())
        self.assertTrue(keep.exists())
        self.assertFalse(old_summary.exists())

    def test_run_rejects_existing_artifacts_without_reuse(self):
        run_id = "agentx_full_batchsize_8"
        artifact_dir = self.work_dir / "performances" / run_id / "agentx_raw"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "summary_export.json").write_text("{}", encoding="utf-8")
        task = self.task()
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value="python",
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=self.harness_metadata(),
        ):
            with self.assertRaisesRegex(RuntimeError, "AGENTX_OUTPUT_WRITE_FAILED"):
                task.run()

    def test_cancelled_run_does_not_record_process_failure(self):
        """取消时退出码来自我方 kill，不应同时记 AGENTX_PROCESS_FAILED。"""
        task = self.task()
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value="python",
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=self.harness_metadata(),
        ):
            plan = task._prepare()[0]
        with mock.patch.object(task, "_execute", return_value=(-9, True, None, None)), \
                mock.patch.object(
                    task, "_read_summary", return_value=({}, "missing", "no summary")), \
                mock.patch.object(task, "_probe_engine", return_value=(None, [])):
            result = task._execute_plan(plan)

        codes = [item["code"] for item in result["errors"]]
        self.assertNotIn("AGENTX_PROCESS_FAILED", codes)
        self.assertIn("AGENTX_CANCELLED", codes)

    def test_smoke_defaults_are_applied_when_unset(self):
        self.assertEqual(resolve_smoke_overrides("smoke", None, None), (60.0, 4))
        self.assertEqual(resolve_smoke_overrides("smoke", 2.5, 3), (2.5, 3))
        self.assertEqual(resolve_smoke_overrides("benchmark", None, None), (None, None))

        def dry_run_plan(**dataset_overrides):
            task = self.task(**dataset_overrides)
            with mock.patch(
                "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
                return_value="python",
            ), mock.patch(
                "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
                return_value=self.harness_metadata(),
            ):
                return task.dry_run()["runs"][0]

        smoke = dry_run_plan(run_mode="smoke")
        self.assertEqual(smoke["duration"], 60.0)
        self.assertEqual(smoke["trace_limit"], 4)
        self.assertEqual(
            smoke["argv"][smoke["argv"].index("--benchmark-duration") + 1], "60.0"
        )
        self.assertEqual(
            smoke["argv"][smoke["argv"].index("--num-dataset-entries") + 1], "4"
        )

        benchmark = dry_run_plan(run_mode="benchmark")
        self.assertIsNone(benchmark["duration"])
        self.assertIsNone(benchmark["trace_limit"])
        self.assertNotIn("--benchmark-duration", benchmark["argv"])
        self.assertNotIn("--num-dataset-entries", benchmark["argv"])

    def test_multiple_concurrency_levels_run_serially_with_fresh_dirs(self):
        task = self.task(concurrency=[1, 8, 32], cooldown=7)
        calls = []

        def fake_execute(plan):
            calls.append(plan)
            return {
                "run": {"status": "completed", "parse_status": "success",
                        "benchmark_valid": True, "comparison_scope": "official_agentx"},
                "artifacts": {"manifest": plan["manifest_path"],
                              "harness_log": plan["log_path"]},
            }

        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value="python",
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=self.harness_metadata(),
        ), mock.patch.object(task, "_execute_plan", side_effect=fake_execute), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.time.sleep"
        ) as sleep:
            result = task.run()

        self.assertEqual([plan["concurrency"] for plan in calls], [1, 8, 32])
        self.assertEqual(len({plan["run_id"] for plan in calls}), 3)
        self.assertEqual(len({plan["artifact_dir"] for plan in calls}), 3)
        for plan in calls:
            argv = plan["argv"]
            self.assertEqual(argv[argv.index("--concurrency") + 1], str(plan["concurrency"]))
            self.assertEqual(Path(plan["plan_path"]).is_file(), True)
            self.assertNotIn("run_config_path", plan)
        self.assertEqual(
            [plan["run_id"] for plan in calls],
            ["agentx_full_batchsize_1", "agentx_full_batchsize_8", "agentx_full_batchsize_32"],
        )
        self.assertEqual(sleep.call_args_list, [mock.call(7.0), mock.call(7.0)])
        self.assertEqual([item["run"]["status"] for item in result["runs"]], ["completed"] * 3)

    def test_start_failure_writes_structured_redacted_result(self):
        task = self.task(api_key="secret")
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value="python",
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=self.harness_metadata(),
        ), mock.patch.object(
            task, "_probe_engine", return_value=({"type": None}, [])
        ), mock.patch.object(task, "_execute", return_value=(None, False, "spawn secret failed", None)):
            with self.assertRaisesRegex(RuntimeError, "start_failed/missing"):
                task.run()
        manifest_path = (
            self.work_dir / "performances" / "agentx_full_batchsize_8" / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_id"], "agentx_full_batchsize_8")
        self.assertEqual(manifest["result"]["status"], "start_failed")
        self.assertEqual(manifest["result"]["parse_status"], "missing")
        self.assertFalse(manifest["result"]["benchmark_valid"])
        self.assertEqual(manifest["result"]["errors"][0]["code"], "AGENTX_PROCESS_START_FAILED")
        self.assertIsNone(manifest["result"]["summary_json"])
        self.assertEqual(manifest["engine"], {"type": None})
        self.assertNotIn("secret", manifest_path.read_text(encoding="utf-8"))

    def test_invalid_process_timeout_env_aborts_as_config_error(self):
        """非法超时属于配置错误，必须在启动 Harness 前按 CONFIG_INVALID 报出。"""
        task = self.task(env={PROCESS_TIMEOUT_ENV: "abc"})
        with mock.patch.object(task, "_execute") as execute:
            with self.assertRaisesRegex(RuntimeError, "AGENTX_CONFIG_INVALID"):
                task.run()
        execute.assert_not_called()
        self.assertFalse(
            (self.work_dir / "performances" / "agentx_full_batchsize_8" / "manifest.json").exists()
        )

    def test_summary_missing_corrupt_schema_and_ambiguity(self):
        task = self.task()
        artifact = self.root / "artifacts"
        artifact.mkdir()
        expected = artifact / HARNESS_SUMMARY_JSON
        raw, status, error = task._read_summary(str(artifact), str(expected))
        self.assertEqual((raw, status), ({}, "missing"))
        self.assertIn("missing", error)

        expected.write_text("{bad", encoding="utf-8")
        _, status, _ = task._read_summary(str(artifact), str(expected))
        self.assertEqual(status, "invalid_json")

        expected.write_text(json.dumps({"schema_version": "1.3"}), encoding="utf-8")
        _, status, _ = task._read_summary(str(artifact), str(expected))
        self.assertEqual(status, "unsupported_schema")

        expected.write_text(json.dumps({"schema_version": "1.4"}), encoding="utf-8")
        nested = artifact / "nested"
        nested.mkdir()
        (nested / HARNESS_SUMMARY_JSON).write_text("{}", encoding="utf-8")
        _, status, error = task._read_summary(str(artifact), str(expected))
        self.assertEqual(status, "invalid_json")
        self.assertIn("Multiple", error)


if __name__ == "__main__":
    unittest.main()
