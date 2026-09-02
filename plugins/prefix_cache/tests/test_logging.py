import ast
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).parents[1]
AISBENCH_INTEGRATION_FILES = (
    PLUGIN_ROOT / "ais_bench_prefix_cache" / "datasets" / "prefix_cache_dataset.py",
    PLUGIN_ROOT / "ais_bench_prefix_cache" / "models" / "vllm_prefix_cache_api.py",
    PLUGIN_ROOT
    / "ais_bench_prefix_cache"
    / "openicl"
    / "icl_inferencer"
    / "prefix_cache_gen_inferencer.py",
)


class AISBenchIntegrationLoggingTest(unittest.TestCase):
    def test_integration_modules_use_aislogger_and_debug_only(self):
        for path in AISBENCH_INTEGRATION_FILES:
            with self.subTest(path=path):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                aislogger_imports = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    and node.module == "ais_bench.benchmark.utils.logging.logger"
                    and any(alias.name == "AISLogger" for alias in node.names)
                ]
                self.assertEqual(len(aislogger_imports), 1)

                logger_assignments = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "logger"
                        for target in node.targets
                    )
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "AISLogger"
                ]
                self.assertEqual(len(logger_assignments), 1)

                calls = [
                    node.func.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "logger"
                ]
                self.assertTrue(calls)
                self.assertEqual(set(calls), {"debug"})

    def test_example_runner_falls_back_to_outputs_default(self):
        path = PLUGIN_ROOT / "config_examples" / "prefix_cache_perf.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        defaults = [
            node.args[1].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "AISBENCH_PREFIX_CACHE_WORK_DIR"
            and isinstance(node.args[1], ast.Constant)
        ]
        self.assertEqual(defaults, ["outputs/default"])

        runtime_path = PLUGIN_ROOT / "ais_bench_prefix_cache" / "runtime.py"
        runtime_source = runtime_path.read_text(encoding="utf-8")
        self.assertIn('namespace.get("work_dir", "outputs/default")', runtime_source)
        self.assertNotIn('namespace.get("work_dir", "outputs/prefix_cache")', runtime_source)


if __name__ == "__main__":
    unittest.main()
