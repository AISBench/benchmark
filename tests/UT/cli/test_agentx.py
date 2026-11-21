import copy
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mmengine.config import Config, ConfigDict

from ais_bench.benchmark.cli.argument_parser import ArgumentParser
from ais_bench.benchmark.cli.config_manager import ConfigManager
from ais_bench.benchmark.cli.task_manager import TaskManager
from ais_bench.benchmark.cli.workers import (
    WORK_FLOW,
    AgentXRun,
    AgentXViz,
    WorkFlowExecutor,
)
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError


class AgentXCLITestBase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()

    def parse(self, *extra, include_required=True, mode="agentx"):
        argv = ["ais_bench", "--mode", mode]
        if include_required:
            argv += [
                "--url", "http://127.0.0.1:8000",
                "--model-name", "qwen3.8",
                "--path", str(self.model_dir),
                "--agentx-dataset-variant", "full",
            ]
        argv += list(extra)
        with mock.patch.object(sys, "argv", argv), mock.patch(
            "ais_bench.benchmark.cli.argument_parser.get_current_time_str",
            return_value="20261106_000000",
        ):
            return ArgumentParser().parse_args()


class TestAgentXArgumentParser(AgentXCLITestBase):
    def test_agentx_options_and_batch_size(self):
        args = self.parse(
            "--batch-size", "8",
            "--agentx-run-mode", "smoke",
            "--agentx-duration", "2.5",
            "--agentx-trace-limit", "4",
            "--endpoint", "/custom/v2/chat",
            "--cooldown", "2.5",
            "--agentx-dataset-variant", "256k",
            "--agentx-env", "PUBLIC=value",
            "--agentx-extra-arg", "--log-level INFO --concurrency 2",
        )
        self.assertEqual(args.endpoint, "/custom/v2/chat")
        self.assertEqual(args.cooldown, 2.5)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.agentx_run_mode, "smoke")
        self.assertEqual(args.agentx_duration, 2.5)
        self.assertEqual(args.agentx_trace_limit, 4)
        self.assertEqual(args.agentx_dataset_variant, "256k")
        self.assertEqual(args.agentx_env, ["PUBLIC=value"])
        self.assertEqual(args.agentx_extra_arg, "--log-level INFO --concurrency 2")

    def test_agentx_defaults_do_not_override_config(self):
        args = self.parse()
        self.assertIsNone(args.endpoint)
        # --cooldown 缺省 None，AgentX 侧由 resolve_cooldown 回落 5 秒
        self.assertIsNone(args.cooldown)
        self.assertIsNone(args.agentx_run_mode)
        self.assertIsNone(args.agentx_duration)
        self.assertIsNone(args.agentx_trace_limit)
        self.assertIsNone(args.agentx_dataset_path)
        self.assertEqual(args.agentx_env, [])
        self.assertIsNone(args.agentx_extra_arg)


class TestAgentXConfigManager(AgentXCLITestBase):
    def test_minimal_cli_builds_model_and_dataset(self):
        args = self.parse("--batch-size", "8", "--api-key", "secret")
        cfg = ConfigManager(args)._get_config_from_arg()
        self.assertEqual(len(cfg["models"]), 1)
        self.assertEqual(len(cfg["datasets"]), 1)
        model = cfg["models"][0]
        dataset_args = cfg["datasets"][0]["args"]
        self.assertEqual(model["model"], "qwen3.8")
        self.assertEqual(model["url"], "http://127.0.0.1:8000")
        self.assertNotIn("endpoint", model)
        self.assertNotIn("endpoint", dataset_args)
        self.assertEqual(model["path"], str(self.model_dir))
        self.assertEqual(model["api_key"], "secret")
        self.assertEqual(dataset_args["concurrency"], 8)
        self.assertEqual(dataset_args["api_key"], "secret")

    def test_raw_mode_builds_config_from_extra_args_only(self):
        """agentx_raw 只给 --agentx-extra-arg 时，基础参数全部交给 Harness 命令。"""
        extra = "{ --url http://127.0.0.1:8000 --model qwen3.8 --concurrency 8 }"
        args = self.parse(
            "--agentx-extra-arg", extra, include_required=False, mode="agentx_raw"
        )
        cfg = ConfigManager(args)._get_config_from_arg()
        model = cfg["models"][0]
        dataset_args = cfg["datasets"][0]["args"]
        self.assertEqual(model["abbr"], "agentx-raw")
        for key in ("model", "url", "path", "endpoint"):
            self.assertNotIn(key, model)
        self.assertTrue(dataset_args["raw_mode"])
        self.assertEqual(dataset_args["concurrency"], 8)
        self.assertEqual(dataset_args["extra_args"], extra)

    def test_agentx_mode_keeps_base_args_required_with_extra_args(self):
        """--mode agentx 不会因为 extra args 就降级为 raw，缺基础参数仍然报错。"""
        extra = "{ --url http://127.0.0.1:8000 --model qwen3.8 --concurrency 8 }"
        args = self.parse("--agentx-extra-arg", extra, include_required=False)
        with self.assertRaisesRegex(AISBenchConfigError, "--url"):
            ConfigManager(args)._get_config_from_arg()

    def test_raw_mode_requires_extra_args(self):
        args = self.parse(include_required=False, mode="agentx_raw")
        with self.assertRaisesRegex(AISBenchConfigError, "requires --agentx-extra-arg"):
            ConfigManager(args)._get_config_from_arg()

    def test_raw_mode_rejects_non_subprocess_args(self):
        """agentx_raw 下命令完全由 extra args 描述，其余参数必须直接报错。"""
        extra = "{ --url http://127.0.0.1:8000 --model qwen3.8 }"
        for flag, value in (
            ("--batch-size", "8"),
            ("--url", "http://127.0.0.1:8000"),
            ("--agentx-duration", "60"),
            ("--agentx-run-mode", "smoke"),
            ("--api-key", "secret"),
        ):
            with self.subTest(flag=flag):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(
                    SystemExit
                ) as caught:
                    self.parse(
                        "--agentx-extra-arg", extra, flag, value,
                        include_required=False, mode="agentx_raw",
                    )
                self.assertEqual(caught.exception.code, 2)
                self.assertIn(flag, stderr.getvalue())

    def test_raw_mode_accepts_lifecycle_args(self):
        extra = "{ --url http://127.0.0.1:8000 --concurrency 8 }"
        args = self.parse(
            "--agentx-extra-arg", extra,
            "--work-dir", str(self.root / "work"),
            "--agentx-env", "A=B",
            include_required=False, mode="agentx_raw",
        )
        self.assertEqual(args.mode, "agentx_raw")
        self.assertEqual(args.agentx_env, ["A=B"])

    def test_custom_endpoint_is_mapped_to_model(self):
        args = self.parse("--endpoint", "/custom/v2/chat")
        cfg = ConfigManager(args)._get_config_from_arg()
        self.assertEqual(cfg["models"][0]["endpoint"], "/custom/v2/chat")
        self.assertNotIn("endpoint", cfg["datasets"][0]["args"])

    def test_cooldown_is_mapped_to_dataset_args(self):
        args = self.parse("--cooldown", "1.5")
        cfg = ConfigManager(args)._get_config_from_arg()
        self.assertEqual(cfg["datasets"][0]["args"]["cooldown"], 1.5)

    def test_invalid_cooldown_is_rejected_before_execution(self):
        with self.assertRaisesRegex(AISBenchConfigError, "non-negative finite"):
            self.parse("--cooldown", "-1")

    def test_invalid_endpoint_is_rejected_before_execution(self):
        for endpoint in ("v1/chat/completions", "/v1/chat?x=1", "/v1 chat"):
            with self.subTest(endpoint=endpoint):
                args = self.parse("--endpoint", endpoint)
                with self.assertRaisesRegex(AISBenchConfigError, "endpoint must be"):
                    ConfigManager(args)._get_config_from_arg()

    def test_smoke_cli_fields_are_mapped(self):
        args = self.parse(
            "--agentx-run-mode", "smoke",
            "--agentx-duration", "2",
            "--agentx-trace-limit", "3",
        )
        cfg = ConfigManager(args)._get_config_from_arg()
        dataset_args = cfg["datasets"][0]["args"]
        self.assertEqual(dataset_args["run_mode"], "smoke")
        self.assertEqual(dataset_args["duration"], 2)
        self.assertEqual(dataset_args["trace_limit"], 3)

    def test_missing_required_args_fail_early(self):
        for extra in (
            ["--model-name", "qwen3.8", "--path", str(self.model_dir)],
            ["--url", "http://127.0.0.1:8000", "--path", str(self.model_dir)],
            ["--url", "http://127.0.0.1:8000", "--model-name", "qwen3.8"],
        ):
            with self.subTest(extra=extra):
                args = self.parse(*extra, include_required=False)
                with self.assertRaises(AISBenchConfigError):
                    ConfigManager(args)._get_config_from_arg()

    def test_missing_variant_fails_early(self):
        args = self.parse(
            "--url", "http://127.0.0.1:8000",
            "--model-name", "qwen3.8",
            "--path", str(self.model_dir),
            include_required=False,
        )
        with self.assertRaisesRegex(AISBenchConfigError, "agentx-dataset-variant"):
            ConfigManager(args)._get_config_from_arg()

    def test_reuse_still_requires_startup_args(self):
        """无配置文件时 --reuse 只复用产物目录，仍要求 url/model-name/path/dataset-variant"""
        for reuse in ("latest", "20261106_000000"):
            with self.subTest(reuse=reuse):
                args = self.parse("--reuse", reuse, include_required=False)
                with self.assertRaisesRegex(AISBenchConfigError, "requires"):
                    ConfigManager(args)._get_config_from_arg()

    def test_reuse_with_dry_run_is_rejected(self):
        """reuse 要求重跑覆盖、dry-run 要求零写入，组合必须在配置期拒绝。"""
        args = self.parse("--reuse", "latest", "--dry-run")
        with self.assertRaisesRegex(AISBenchConfigError, "--reuse cannot be combined"):
            ConfigManager(args)._get_config_from_arg()

    def test_multiple_concurrency_levels_are_mapped(self):
        args = self.parse("--batch-size", "1", "8", "32")
        cfg = ConfigManager(args)._get_config_from_arg()
        self.assertEqual(cfg["datasets"][0]["args"]["concurrency"], [1, 8, 32])

    def test_duplicate_concurrency_levels_are_rejected(self):
        args = self.parse("--batch-size", "8", "8")
        with self.assertRaisesRegex(AISBenchConfigError, "duplicated"):
            ConfigManager(args)._get_config_from_arg()

    def test_full_chat_completions_url_is_rejected_before_execution(self):
        args = self.parse(
            "--url", "http://127.0.0.1:8000/v1/chat/completions",
            "--model-name", "qwen3.8",
            "--path", str(self.model_dir),
            "--agentx-dataset-variant", "full",
            include_required=False,
        )
        with self.assertRaisesRegex(AISBenchConfigError, "base URL without"):
            ConfigManager(args)._get_config_from_arg()

    def test_explicit_cli_overrides_config_and_unset_values_preserve_it(self):
        config = ConfigDict({
            "models": [{
                "abbr": "old", "model": "old-model", "path": str(self.model_dir),
                "url": "http://old:8000", "endpoint": "/config/chat",
                "trust_remote_code": False,
            }],
            "datasets": [{"abbr": "old-data", "args": {
                "concurrency": 2, "cooldown": 3.0, "variant": "full",
            }}],
        })
        args = self.parse(
            "--batch-size", "8", "--model-name", "qwen3.8next",
            "--model-abbr", "new", "--dataset-abbr", "new-data",
            "--trust-remote-code", "--endpoint", "/cli/chat",
            "--cooldown", "1.5",
        )
        ConfigManager(args)._apply_agentx_cli_overrides(config)
        self.assertEqual(config["models"][0]["model"], "qwen3.8next")
        self.assertEqual(config["models"][0]["abbr"], "new")
        self.assertTrue(config["models"][0]["trust_remote_code"])
        self.assertEqual(config["datasets"][0]["abbr"], "new-data")
        self.assertEqual(config["datasets"][0]["args"]["concurrency"], 8)
        self.assertEqual(config["datasets"][0]["args"]["cooldown"], 1.5)
        self.assertEqual(config["datasets"][0]["args"]["variant"], "full")
        self.assertEqual(config["models"][0]["endpoint"], "/cli/chat")
        self.assertNotIn("endpoint", config["datasets"][0]["args"])

        preserved = ConfigDict({
            "models": [dict(self._valid_model(), endpoint="/config/chat")],
            "datasets": [{"abbr": "data", "args": {"concurrency": 3, "cooldown": 4}}],
        })
        args = self.parse()
        ConfigManager(args)._apply_agentx_cli_overrides(preserved)
        self.assertEqual(preserved["datasets"][0]["args"]["concurrency"], 3)
        # --cooldown 只认 CLI，配置文件里的同名字段不参与；缺省写 None 后回落 5 秒
        self.assertIsNone(preserved["datasets"][0]["args"]["cooldown"])
        self.assertEqual(preserved["models"][0]["endpoint"], "/config/chat")
        self.assertEqual(preserved["models"][0]["model"], "qwen3.8")

    def test_agentx_env_merges_with_config_env_by_key(self):
        """--agentx-env 按键合并配置 env：CLI 同名键覆盖，其余保留"""
        config = ConfigDict({
            "models": [self._valid_model()],
            "datasets": [{"abbr": "data", "args": {
                "concurrency": 2, "variant": "full",
                "env": {"API_TOKEN": "cfg-token", "PUBLIC": "cfg-public"},
            }}],
        })
        args = self.parse(
            "--agentx-env", "PUBLIC=cli-public",
            "--agentx-env", "EXTRA=cli-extra",
        )
        ConfigManager(args)._apply_agentx_cli_overrides(config)
        env = config["datasets"][0]["args"]["env"]
        self.assertEqual(env["API_TOKEN"], "cfg-token")
        self.assertEqual(env["PUBLIC"], "cli-public")
        self.assertEqual(env["EXTRA"], "cli-extra")

    def test_config_file_uses_agentx_override_without_model_type(self):
        config_path = self.root / "agentx_config.py"
        config_path.write_text(
            "models = [dict(abbr='cfg', model='qwen3.8', "
            f"path={str(self.model_dir)!r}, "
            "url='http://127.0.0.1:8000')]\n"
            "datasets = [dict(abbr='weka', args=dict(concurrency=2, variant='full'))]\n",
            encoding="utf-8",
        )
        args = self.parse("--model-name", "qwen3.8next", str(config_path))
        cfg = ConfigManager(args)._get_config_from_arg()
        self.assertEqual(cfg["models"][0]["model"], "qwen3.8next")
        self.assertEqual(cfg["datasets"][0]["args"]["concurrency"], 2)
        self.assertNotIn("type", cfg["models"][0])

    def _valid_model(self):
        return {
            "abbr": "qwen",
            "model": "qwen3.8",
            "path": str(self.model_dir),
            "url": "http://127.0.0.1:8000",
        }

    def test_config_file_validation_errors(self):
        """配置文件入口与纯 CLI 共用同一套启动前校验。"""
        cases = {
            "multiple_models": (
                "models = [dict(abbr='a', model='m', url='http://127.0.0.1:8000', path=%r), "
                "dict(abbr='b', model='m', url='http://127.0.0.1:8000', path=%r)]\n"
                "datasets = [dict(abbr='d', args=dict(concurrency=1, variant='full'))]\n"
                % (str(self.model_dir), str(self.model_dir)),
                "exactly one model",
            ),
            "missing_variant": (
                "models = [dict(abbr='a', model='m', url='http://127.0.0.1:8000', path=%r)]\n"
                "datasets = [dict(abbr='d', args=dict(concurrency=1))]\n" % str(self.model_dir),
                "explicit dataset variant",
            ),
            "missing_path": (
                "models = [dict(abbr='a', model='m', url='http://127.0.0.1:8000', path=%r)]\n"
                "datasets = [dict(abbr='d', args=dict(concurrency=1, variant='full'))]\n"
                % str(self.root / "missing"),
                "does not exist",
            ),
        }
        for name, (body, pattern) in cases.items():
            with self.subTest(case=name):
                config_path = self.root / f"{name}.py"
                config_path.write_text(body, encoding="utf-8")
                args = self.parse(str(config_path), include_required=False)
                with self.assertRaisesRegex(AISBenchConfigError, pattern):
                    ConfigManager(args)._get_config_from_arg()

    def test_config_file_keeps_work_dir_summarizer_and_merges_env(self):
        """配置文件模式下 work_dir/summarizer 原样保留，env 按键合并，且不写 CLI 来源标记。"""
        config_path = self.root / "agentx_config_env.py"
        config_path.write_text(
            "work_dir = %r\n"
            "models = [dict(abbr='cfg', model='qwen3.8', url='http://127.0.0.1:8000', path=%r)]\n"
            "datasets = [dict(abbr='weka', args=dict(concurrency=2, variant='full', "
            "env=dict(API_TOKEN='cfg-token', PUBLIC='cfg-public')))]\n"
            "summarizer = dict(attr='agentx')\n"
            % (str(self.root / "work"), str(self.model_dir)),
            encoding="utf-8",
        )
        args = self.parse(
            str(config_path),
            "--agentx-env", "PUBLIC=cli-public",
            include_required=False,
        )
        cfg = ConfigManager(args)._get_config_from_arg()
        self.assertEqual(cfg["work_dir"], str(self.root / "work"))
        self.assertEqual(cfg["summarizer"]["attr"], "agentx")
        self.assertEqual(
            cfg["datasets"][0]["args"]["env"],
            {"API_TOKEN": "cfg-token", "PUBLIC": "cli-public"},
        )

    def test_example_config_template_stays_parseable(self):
        """示例模板是用户复制起点：只校验结构与字段名，不要求本机路径存在。"""
        template = (
            Path(__file__).resolve().parents[3]
            / "ais_bench" / "configs" / "agentx_example" / "agentx_weka_trace.py"
        )
        cfg = Config.fromfile(str(template), format_python_code=False)
        self.assertEqual(len(cfg["models"]), 1)
        self.assertEqual(len(cfg["datasets"]), 1)
        self.assertLessEqual(
            set(cfg["models"][0]),
            {"abbr", "model", "url", "endpoint", "path", "trust_remote_code", "api_key"},
        )
        dataset_args = cfg["datasets"][0]["args"]
        self.assertLessEqual(
            set(dataset_args),
            {"concurrency", "run_mode", "duration", "trace_limit", "variant",
             "dataset_path", "agentx_python", "agentx_bin", "extra_args", "env"},
        )
        self.assertIn(dataset_args["variant"], {"full", "256k"})
        self.assertIn(dataset_args["run_mode"], {"benchmark", "smoke"})
        self.assertEqual(cfg["summarizer"]["attr"], "agentx")

    def test_snapshot_is_redacted_and_memory_secrets_are_restored(self):
        args = self.parse(
            "--api-key", "secret",
            "--agentx-env", "API_TOKEN=env-token-value",
            "--agentx-env", "PUBLIC=visible",
            # 与真实用法一致带花括号：裸的 "--api-key=..." 会被 argparse 当成 AISBench 自己的
            # --api-key 选项，导致 --agentx-extra-arg 缺少参数。
            "--agentx-extra-arg",
            "{ --api-key=extra-secret --header 'Authorization: Bearer header-secret' "
            "--max-context-length 204800 }",
        )
        manager = ConfigManager(args)
        manager.cfg_time_str = "20261106_000000"
        manager.cfg = Config({
            "work_dir": str(self.root / "work"),
            "models": [dict(self._valid_model(), api_key="secret")],
            "datasets": [{
                "abbr": "data",
                "args": {
                    "concurrency": 1,
                    "api_key": "secret",
                    "env": {"API_TOKEN": "env-token-value", "PUBLIC": "visible"},
                    "extra_args": (
                        "--api-key=extra-secret "
                        "--header 'Authorization: Bearer header-secret'"
                    ),
                },
            }],
            "cli_args": vars(args),
        })
        (self.root / "work" / "configs").mkdir(parents=True)
        manager._dump_and_reload_config()
        snapshots = list((self.root / "work" / "configs").glob("*.py"))
        self.assertEqual(len(snapshots), 1)
        content = snapshots[0].read_text(encoding="utf-8")
        self.assertNotIn("secret", content)
        self.assertNotIn("env-token-value", content)
        self.assertIn("<redacted>", content)
        self.assertIn("visible", content)
        self.assertIn("--max-context-length 204800", content)
        self.assertEqual(manager.cfg["models"][0]["api_key"], "secret")
        self.assertEqual(manager.cfg["datasets"][0]["args"]["api_key"], "secret")
        self.assertEqual(manager.cfg["datasets"][0]["args"]["env"]["API_TOKEN"], "env-token-value")
        self.assertEqual(manager.cfg["datasets"][0]["args"]["env"]["PUBLIC"], "visible")
        self.assertEqual(
            manager.cfg["cli_args"]["agentx_extra_arg"],
            "{ --api-key=extra-secret --header 'Authorization: Bearer header-secret' "
            "--max-context-length 204800 }",
        )
        self.assertEqual(
            manager.cfg["datasets"][0]["args"]["extra_args"],
            "--api-key=extra-secret --header 'Authorization: Bearer header-secret'",
        )


class TestAgentXLifecycle(unittest.TestCase):
    def test_workflow_registration(self):
        self.assertEqual(WORK_FLOW["agentx"], [AgentXRun, AgentXViz])
        self.assertEqual(WORK_FLOW["agentx_raw"], [AgentXRun, AgentXViz])
        self.assertNotIn("agentx_viz", WORK_FLOW)

    def test_worker_directly_executes_task_without_partitioner(self):
        task = mock.MagicMock()
        with mock.patch("ais_bench.benchmark.cli.workers.TASKS.build", return_value=task) as build:
            AgentXRun(SimpleNamespace(dry_run=False)).do_work(ConfigDict({"models": [], "datasets": []}))
        build.assert_called_once()
        task.run.assert_called_once_with()
        task.dry_run.assert_not_called()

        task.reset_mock()
        with mock.patch("ais_bench.benchmark.cli.workers.TASKS.build", return_value=task):
            AgentXRun(SimpleNamespace(dry_run=True)).do_work(ConfigDict({"models": [], "datasets": []}))
        task.dry_run.assert_called_once_with()
        task.run.assert_not_called()

    def test_reuse_keeps_full_workflow_and_dry_run_skips_viz(self):
        for reuse, dry_run, expected in (
            ("latest", False, [AgentXRun, AgentXViz]),
            (None, True, [AgentXRun]),
        ):
            with self.subTest(reuse=reuse, dry_run=dry_run):
                manager = TaskManager.__new__(TaskManager)
                manager.logger = mock.MagicMock()
                manager.args = SimpleNamespace(
                    search=False, mode="agentx", reuse=reuse, dry_run=dry_run
                )
                manager.config_manager = mock.MagicMock()
                manager.config_manager.load_config.return_value = ConfigDict({"work_dir": "unused"})
                with mock.patch("ais_bench.benchmark.cli.workers.WorkFlowExecutor") as executor:
                    manager.run()
                workflow = manager.config_manager.load_config.call_args.args[0]
                self.assertEqual([type(worker) for worker in workflow], expected)
                executor.return_value.execute.assert_called_once_with()

    def test_run_failure_is_raised_after_viz(self):
        events = []

        # 必须继承真实的 AgentXRun：WorkFlowExecutor 通过 isinstance 判断
        # 是否延后抛错，普通伪类会让异常立即抛出、AgentXViz 不会执行。
        class FailingAgentXRun(AgentXRun):
            def do_work(self, cfg):
                events.append("run")
                raise RuntimeError("harness failed")

        class RecordingAgentXViz(AgentXViz):
            def do_work(self, cfg):
                events.append("viz")

        executor = WorkFlowExecutor(
            ConfigDict({"work_dir": "unused"}),
            [FailingAgentXRun(SimpleNamespace()), RecordingAgentXViz(SimpleNamespace())],
        )
        with self.assertRaisesRegex(RuntimeError, "harness failed"):
            executor.execute()
        self.assertEqual(events, ["run", "viz"])


if __name__ == "__main__":
    unittest.main()
