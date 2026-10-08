import ast
import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path


class ModelImportTest(unittest.TestCase):
    def test_model_extends_vllm_through_public_aisbench_facade(self):
        model_path = (
            Path(__file__).resolve().parents[1]
            / "ais_bench_prefix_cache"
            / "models"
            / "vllm_prefix_cache_api.py"
        )
        tree = ast.parse(model_path.read_text(encoding="utf-8"))
        imports = {
            (node.module, alias.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }

        self.assertIn(("ais_bench.benchmark.models", "VLLMCustomAPI"), imports)
        self.assertNotIn(
            ("ais_bench.benchmark.models.api_models.vllm_custom_api", "VLLMCustomAPI"),
            imports,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "full AISBench model dependencies are not installed",
    )
    def test_mmengine_lazy_model_import_in_fresh_process(self):
        """The generated AISBench config must resolve the plugin model cleanly."""
        repository_root = Path(__file__).resolve().parents[3]
        plugin_root = repository_root / "plugins" / "prefix_cache"
        environment = os.environ.copy()
        python_path = [str(plugin_root), str(repository_root)]
        if environment.get("PYTHONPATH"):
            python_path.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_path)

        script = r'''
import tempfile
from pathlib import Path

from mmengine.config import Config
from ais_bench.benchmark.cli.utils import recur_convert_config_type

with tempfile.TemporaryDirectory() as directory:
    config_path = Path(directory) / "config.py"
    config_path.write_text(
        "from ais_bench_prefix_cache.models.vllm_prefix_cache_api "
        "import VLLMPrefixCacheAPI as _model_type\n"
        "models = [dict(type=_model_type)]\n",
        encoding="utf-8",
    )
    config = Config.fromfile(str(config_path), format_python_code=False)
    recur_convert_config_type(config)
    assert config["models"][0]["type"].endswith(".VLLMPrefixCacheAPI")
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
