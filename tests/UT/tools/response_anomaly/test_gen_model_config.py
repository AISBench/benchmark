import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ais_bench.tools.response_anomaly.gen_model_config import (
    generate_model_config,
)


class TestGenerateModelConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

        self.msprobe_dir = self.root / "msprobe"
        (self.msprobe_dir / "tools").mkdir(parents=True)
        (self.msprobe_dir / "tools" / "gen_model_config.py").write_text(
            "# fake", encoding="utf-8"
        )
        (self.msprobe_dir / "configs").mkdir(parents=True)
        (self.msprobe_dir / "configs" / "config.yaml").write_text(
            "window_size: 128", encoding="utf-8"
        )

        self.output_dir = self.root / "msprobe_configs"
        (self.output_dir / "configs").mkdir(parents=True)
        (self.output_dir / "configs" / "mtype_config.json").write_text(
            json.dumps({"old-model": {"eos": 1}}), encoding="utf-8"
        )

    def _patch_msprobe_dir(self):
        return mock.patch(
            "ais_bench.tools.response_anomaly.gen_model_config."
            "_msprobe_response_anomaly_dir",
            return_value=self.msprobe_dir,
        )

    def test_generate_model_config_merges_and_copies_defaults(self):
        def fake_run(command, cwd, **kwargs):
            output_root = Path(cwd).parent
            (output_root / "configs").mkdir(parents=True, exist_ok=True)
            (output_root / "configs" / "mtype_config.json").write_text(
                json.dumps({"new-model": {"eos": 2}}), encoding="utf-8"
            )
            (output_root / "token2category").mkdir(parents=True, exist_ok=True)
            (output_root / "token2category" / "new-model_10.json").write_text(
                json.dumps({"0": "other"}), encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0)

        with self._patch_msprobe_dir(), mock.patch(
            "ais_bench.tools.response_anomaly.gen_model_config.subprocess.run",
            side_effect=fake_run,
        ):
            generated = generate_model_config(
                model_path="/models/new",
                model_name="New-Model",
                output_dir=str(self.output_dir),
            )

        mtype = json.loads(
            (self.output_dir / "configs" / "mtype_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(mtype), {"old-model", "new-model"})
        self.assertTrue(
            (self.output_dir / "configs" / "config.yaml").exists()
        )
        self.assertTrue(
            (self.output_dir / "token2category" / "new-model_10.json").exists()
        )
        self.assertEqual(
            generated["msprobe_config_path"],
            str(self.output_dir / "configs" / "config.yaml"),
        )
        self.assertEqual(generated["model_name"], "new-model")
        self.assertFalse((self.output_dir / "_gen_tmp_new-model").exists())

    def test_generate_model_config_failure_raises(self):
        failed = subprocess.CompletedProcess(
            ["python", "gen_model_config.py"], 1, stdout="", stderr="boom"
        )
        with self._patch_msprobe_dir(), mock.patch(
            "ais_bench.tools.response_anomaly.gen_model_config.subprocess.run",
            return_value=failed,
        ):
            with self.assertRaises(RuntimeError) as cm:
                generate_model_config(
                    model_path="/models/new",
                    model_name="New-Model",
                    output_dir=str(self.output_dir),
                )
        self.assertIn("boom", str(cm.exception))
        self.assertTrue((self.output_dir / "_gen_tmp_new-model").exists())
        # Old mtype_config.json must not have been clobbered.
        mtype = json.loads(
            (self.output_dir / "configs" / "mtype_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(mtype), {"old-model"})

    def test_generate_model_config_keeps_tools_dir_when_run_raises(self):
        with self._patch_msprobe_dir(), mock.patch(
            "ais_bench.tools.response_anomaly.gen_model_config.subprocess.run",
            side_effect=OSError("cannot run"),
        ):
            with self.assertRaises(RuntimeError) as cm:
                generate_model_config(
                    model_path="/models/new",
                    model_name="New-Model",
                    output_dir=str(self.output_dir),
                )

        self.assertIn("cannot run", str(cm.exception))
        self.assertTrue((self.output_dir / "_gen_tmp_new-model").exists())
        mtype = json.loads(
            (self.output_dir / "configs" / "mtype_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(mtype), {"old-model"})


if __name__ == "__main__":
    unittest.main()
