import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ais_bench.benchmark.tasks.custom_tasks.harbor_agent_task import (
    HarborAgentTask,
)
from ais_bench.benchmark.utils.config import ConfigDict


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class _FakeAgentName:
    """Deterministic substitute for harbor's AgentName enum .values()."""
    _values = ["oracle", "claude-code", "terminus-2"]

    @classmethod
    def values(cls):
        return list(cls._values)


class _FakeTaskModel:
    """Deterministic substitute for harbor Task.is_valid_dir()."""
    is_valid_dir_result = True

    @classmethod
    def is_valid_dir(cls, *args, **kwargs):
        return cls.is_valid_dir_result


_HARBOR_MODULES = {
    "harbor": mock.MagicMock(),
    "harbor.models": mock.MagicMock(),
    "harbor.models.agent": mock.MagicMock(),
    "harbor.models.agent.name": mock.MagicMock(),
    "harbor.models.job": mock.MagicMock(),
    "harbor.models.job.config": mock.MagicMock(),
    "harbor.models.task": mock.MagicMock(),
    "harbor.models.task.task": mock.MagicMock(),
    "harbor.models.trial": mock.MagicMock(),
    "harbor.models.trial.config": mock.MagicMock(),
    "harbor.models.environment_type": mock.MagicMock(),
}
# deterministic fakes for the parts whose return values our tests must control
_HARBOR_MODULES["harbor.models.agent.name"].AgentName = _FakeAgentName
_HARBOR_MODULES["harbor.models.task.task"].Task = _FakeTaskModel


def _make_cfg(temp_dir, model=None, dataset=None):
    model = model or {
        "abbr": "m",
        "agent_name": "oracle",
        "model_names": ["openai/qwen3"],
    }
    dataset = dataset or {"abbr": "d", "args": {}}
    return ConfigDict(
        {
            "work_dir": temp_dir,
            "models": [model],
            "datasets": [[dataset]],
            "cli_args": {"debug": False},
        }
    )


class TestHarborAgentTaskJobMetrics(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.job_dir = Path(self.temp_dir) / "details"
        self.job_dir.mkdir(parents=True)
        self.task = HarborAgentTask(_make_cfg(self.temp_dir))
        self.task.logger = mock.MagicMock()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _write_result(self, trials):
        _write_json(self.job_dir / "result.json", {"trial_results": trials})

    def test_counts(self):
        self._write_result(
            [
                {"verifier_result": {"rewards": {"reward": 1.0}}},
                {"verifier_result": {"rewards": {"reward": 0.0}}},
                {"exception_info": {"type": "T"}},
                {"verifier_result": {"rewards": {"reward": 2.0}}},
            ]
        )
        m = self.task._job_metrics(self.job_dir)
        self.assertEqual(m, {"correct": 2, "wrong": 1, "exception": 1, "avg_score": 1.0})

    def test_no_verifier_result_skipped(self):
        self._write_result([{"trial_name": "a"}])
        m = self.task._job_metrics(self.job_dir)
        self.assertEqual(m["correct"], 0)
        self.assertEqual(m["wrong"], 0)
        self.assertEqual(m["avg_score"], None)

    def test_reward_none_skipped(self):
        self._write_result([{"verifier_result": {"rewards": {"reward": None}}}])
        m = self.task._job_metrics(self.job_dir)
        self.assertEqual(m["avg_score"], None)

    def test_reward_non_numeric_skipped(self):
        self._write_result([{"verifier_result": {"rewards": {"reward": "abc"}}}])
        m = self.task._job_metrics(self.job_dir)
        self.assertEqual(m["avg_score"], None)

    def test_exception_trial_excluded_from_score(self):
        self._write_result(
            [{"exception_info": {"type": "T"}}, {"verifier_result": {"rewards": {"reward": 1.0}}}]
        )
        m = self.task._job_metrics(self.job_dir)
        self.assertEqual(m["exception"], 1)
        self.assertEqual(m["correct"], 1)
        self.assertEqual(m["avg_score"], 1.0)

    def test_missing_file(self):
        empty = self.job_dir  # no result.json
        self.assertIsNone(self.task._job_metrics(empty))

    def test_invalid_json(self):
        (self.job_dir / "result.json").write_text("{bad", encoding="utf-8")
        self.assertIsNone(self.task._job_metrics(self.job_dir))

    def test_non_dict_result(self):
        _write_json(self.job_dir / "result.json", [1, 2, 3])
        self.assertIsNone(self.task._job_metrics(self.job_dir))

    def test_non_dict_trial_skipped(self):
        self._write_result(['string', {"exception_info": {"type": "T"}}])
        m = self.task._job_metrics(self.job_dir)
        self.assertEqual(m["exception"], 1)


class TestHarborAgentTaskRefreshMetrics(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.job_dir = Path(self.temp_dir) / "details"
        self.job_dir.mkdir(parents=True)
        self.task = HarborAgentTask(_make_cfg(self.temp_dir))
        self.task.logger = mock.MagicMock()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_no_state_manager_returns(self):
        self.task.task_state_manager = None
        self.task._progress_job_dir = self.job_dir
        self.task._refresh_progress_metrics()

    def test_no_job_dir_returns(self):
        self.task.task_state_manager = mock.MagicMock()
        self.task._progress_job_dir = None
        self.task.job = mock.MagicMock()
        self.task.job.job_dir = None
        self.task._refresh_progress_metrics()

    def test_within_interval_returns(self):
        import time
        self.task.task_state_manager = mock.MagicMock()
        self.task._progress_job_dir = self.job_dir
        self.task._last_metrics_ts = time.time()  # just updated -> below interval
        _write_json(self.job_dir / "result.json", {"trial_results": []})
        self.task._refresh_progress_metrics()
        self.task.task_state_manager.update_task_state.assert_not_called()

    def test_updates_state(self):
        import time
        _write_json(
            self.job_dir / "result.json",
            {"trial_results": [{"verifier_result": {"rewards": {"reward": 1.0}}}]},
        )
        mgr = mock.MagicMock()
        self.task.task_state_manager = mgr
        self.task.task_state_manager = mgr
        self.task._progress_job_dir = self.job_dir
        self.task._last_metrics_ts = 0.0
        self.task._refresh_progress_metrics()
        mgr.update_task_state.assert_called_once()
        args = mgr.update_task_state.call_args[0][0]
        self.assertEqual(args["other_kwargs"]["correct"], 1)


class TestHarborAgentTaskBuildAgents(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # reset deterministic fakes per test
        _FakeAgentName._values = ["oracle", "claude-code", "terminus-2"]

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @mock.patch.dict("sys.modules", _HARBOR_MODULES)
    def test_unknown_agent_raises(self):
        model = {"abbr": "m", "agent_name": "bogus-agent", "model_names": ["x"]}
        task = HarborAgentTask(_make_cfg(self.temp_dir, model=model))
        task.logger = mock.MagicMock()
        with self.assertRaises(ValueError):
            task._build_agents()

    @mock.patch.dict("sys.modules", _HARBOR_MODULES)
    def test_custom_import_path_passes(self):
        # an agent name without a colon must be a known AgentName; a custom
        # agent is expressed as an import path (contains ':') plus import_path
        model = {
            "abbr": "m",
            "agent_name": "my.module:Agent",
            "agent_import_path": "my.module:Agent",
            "model_names": ["x"],
        }
        task = HarborAgentTask(_make_cfg(self.temp_dir, model=model))
        task.logger = mock.MagicMock()
        agents = task._build_agents()
        self.assertIsInstance(agents, list)
        self.assertEqual(len(agents), 1)

    @mock.patch.dict("sys.modules", _HARBOR_MODULES)
    def test_oracle_default_when_no_agent(self):
        model = {"abbr": "m", "model_names": ["x"]}
        task = HarborAgentTask(_make_cfg(self.temp_dir, model=model))
        task.logger = mock.MagicMock()
        agents = task._build_agents()
        self.assertEqual(len(agents), 1)


class TestHarborAgentTaskDatasetSource(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        _FakeTaskModel.is_valid_dir_result = True

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _task(self, args):
        dataset = {"abbr": "d", "args": args}
        return HarborAgentTask(_make_cfg(self.temp_dir, dataset=dataset))

    def _real_path(self):
        p = Path(self.temp_dir) / "taskdir"
        p.mkdir(exist_ok=True)
        return str(p)

    @mock.patch.dict("sys.modules", _HARBOR_MODULES)
    def test_path_single_task(self):
        _FakeTaskModel.is_valid_dir_result = True
        config = mock.MagicMock()
        config.verifier.disable = False
        self._task({"path": self._real_path()})._apply_dataset_source(
            config, {"path": self._real_path()}
        )
        # single task -> datasets empty, tasks set
        self.assertEqual(config.datasets, [])
        self.assertNotEqual(config.tasks, [])

    @mock.patch.dict("sys.modules", _HARBOR_MODULES)
    def test_path_dataset_dir(self):
        _FakeTaskModel.is_valid_dir_result = False
        config = mock.MagicMock()
        config.verifier.disable = False
        self._task({"path": self._real_path()})._apply_dataset_source(
            config, {"path": self._real_path()}
        )
        self.assertNotEqual(config.datasets, [])
        self.assertEqual(config.tasks, [])

    @mock.patch.dict("sys.modules", _HARBOR_MODULES)
    def test_registry_name_version(self):
        config = mock.MagicMock()
        config.verifier.disable = False
        self._task({"dataset_name_version": "my_dataset@v1"})._apply_dataset_source(
            config, {"dataset_name_version": "my_dataset@v1"}
        )
        self.assertEqual(config.tasks, [])

    @mock.patch.dict("sys.modules", _HARBOR_MODULES)
    def test_package_reference(self):
        config = mock.MagicMock()
        config.verifier.disable = False
        self._task({"dataset_name_version": "org/name@ref"})._apply_dataset_source(
            config, {"dataset_name_version": "org/name@ref"}
        )
        self.assertEqual(config.tasks, [])

    @mock.patch.dict("sys.modules", _HARBOR_MODULES)
    def test_missing_source_raises(self):
        config = mock.MagicMock()
        config.verifier.disable = False
        with self.assertRaises(ValueError):
            self._task({})._apply_dataset_source(config, {})


class TestHarborAgentTaskRunResume(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.task = HarborAgentTask(_make_cfg(self.temp_dir))
        self.task.logger = mock.MagicMock()
        self.task.out_detail_dir = Path(self.temp_dir) / "out"

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @mock.patch.object(HarborAgentTask, "_resume_job")
    def test_resume_when_config_exists(self, mock_resume):
        mock_resume.return_value = ("job", "result")
        details = self.task.out_detail_dir / "details"
        details.mkdir(parents=True)
        (details / "config.json").write_text("{}", encoding="utf-8")
        job, result = self.task._run_harbor_job()
        mock_resume.assert_called_once_with(details)
        self.assertEqual(job, "job")

    @mock.patch.object(HarborAgentTask, "_build_job_config")
    @mock.patch.object(HarborAgentTask, "_get_task_count", return_value=0)
    @mock.patch.object(HarborAgentTask, "_run_with_tqdm")
    def test_build_when_no_config(self, mock_run, mock_count, mock_build):
        mock_config = mock.MagicMock()
        mock_config.n_attempts = 1
        mock_build.return_value = mock_config
        mock_run.return_value = ("job", "result")
        job, result = self.task._run_harbor_job()
        self.assertEqual(job, "job")
        mock_build.assert_called_once()


if __name__ == "__main__":
    unittest.main()