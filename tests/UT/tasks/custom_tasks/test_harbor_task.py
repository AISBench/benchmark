import unittest
import os
import tempfile
import json
import shutil
from unittest import mock
from pathlib import Path
import sys

sys.modules['harbor'] = mock.MagicMock()
sys.modules['harbor.cli'] = mock.MagicMock()
sys.modules['harbor.cli.utils'] = mock.MagicMock()
sys.modules['harbor.job'] = mock.MagicMock()
sys.modules['harbor.models'] = mock.MagicMock()
sys.modules['harbor.models.job'] = mock.MagicMock()
sys.modules['harbor.models.job.config'] = mock.MagicMock()
sys.modules['harbor.models.agent'] = mock.MagicMock()
sys.modules['harbor.models.agent.name'] = mock.MagicMock()
sys.modules['harbor.models.environment_type'] = mock.MagicMock()

from ais_bench.benchmark.tasks.custom_tasks.harbor_task import HarborTask, DEFAULT_FAKE_API_KEY
from ais_bench.benchmark.utils.config import ConfigDict
from ais_bench.benchmark.tasks.base import TaskStateManager


class TestHarborTask(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.task_state_manager = mock.MagicMock(spec=TaskStateManager)

        self.model_cfg = {
            "type": "openai",
            "abbr": "test_model",
            "agent_name": "test_agent",
            "model_names": ["hosted_vllm/qwen3"],
            "agent_kwargs": {"api_base": "http://0.0.0.0:8080/v1"},
            "agent_env": {"TEST_VAR": "test_value"}
        }

        self.dataset_cfg = {
            "abbr": "test_dataset",
            "args": {
                "n_attempts": 2,
                "timeout_multiplier": 1.5,
                "n_concurrent_trials": 3,
                "path": "/path/to/dataset/",
                "retry_include_exceptions": ["AgentTimeoutError"],
                "retry_exclude_exceptions": ["VerifierError"],
                "environment_type": "docker",
                "verifier_env": ["VAR1=value1", "VAR2=value2"]
            }
        }

        self.cfg = ConfigDict({
            "work_dir": self.temp_dir,
            "models": [self.model_cfg],
            "datasets": [[self.dataset_cfg]],
            "cli_args": {"debug": False}
        })

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        task = HarborTask(self.cfg)
        self.assertIsNone(task.captured_metrics)
        self.assertEqual(task.name_prefix, "HarborTask")
        self.assertEqual(task.output_subdir, "results")

    def test_get_command(self):
        task = HarborTask(self.cfg)
        command = task.get_command("test_config.py", "test_template")
        self.assertIn("test_config.py", command)
        self.assertIn("harbor_task.py", command)

    def test_set_api_key_with_api_key(self):
        original_env = os.environ.get("OPENAI_API_KEY")
        cfg = ConfigDict({
            "work_dir": self.temp_dir,
            "models": [{"abbr": "test_model", "api_key": "test_key"}],
            "datasets": [[{"abbr": "test_dataset", "args": {}}]],
            "cli_args": {"debug": False}
        })
        task = HarborTask(cfg)
        task._set_api_key()
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), "test_key")
        if original_env:
            os.environ["OPENAI_API_KEY"] = original_env
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_set_api_key_without_api_key(self):
        original_env = os.environ.get("OPENAI_API_KEY")
        cfg = ConfigDict({
            "work_dir": self.temp_dir,
            "models": [{"abbr": "test_model"}],
            "datasets": [[{"abbr": "test_dataset", "args": {}}]],
            "cli_args": {"debug": False}
        })
        task = HarborTask(cfg)
        task._set_api_key()
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), DEFAULT_FAKE_API_KEY)
        if original_env:
            os.environ["OPENAI_API_KEY"] = original_env
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_prepare_out_dir(self):
        task = HarborTask(self.cfg)
        task._prepare_out_dir()
        expected_dir = os.path.join(self.temp_dir, "results", "test_model", "test_dataset")
        self.assertTrue(os.path.exists(expected_dir))

    def test_dump_eval_results_with_none_result(self):
        task = HarborTask(self.cfg)
        task.dataset_cfgs = [{"abbr": "test_dataset"}]
        task.out_dir = os.path.join(self.temp_dir, "results")
        task.logger = mock.MagicMock()
        task._dump_eval_results(mock.MagicMock(), None)
        task.logger.error.assert_called_once()

    def test_dump_eval_results_normal(self):
        task = HarborTask(self.cfg)
        task.dataset_cfgs = [{"abbr": "test_dataset"}]
        task.out_dir = os.path.join(self.temp_dir, "results")
        task.logger = mock.MagicMock()
        os.makedirs(task.out_dir, exist_ok=True)

        mock_job = mock.MagicMock()
        mock_job.config = mock.MagicMock()

        mock_trial_1 = mock.MagicMock()
        mock_trial_1.exception_info = None
        mock_trial_1.verifier_result = mock.MagicMock()
        mock_trial_1.verifier_result.rewards = {"accuracy": 1.0}

        mock_trial_2 = mock.MagicMock()
        mock_trial_2.exception_info = mock.MagicMock()
        mock_trial_2.exception_info.exception_type = "AgentTimeoutError"

        mock_job_result = mock.MagicMock()
        mock_job_result.trial_results = [mock_trial_1, mock_trial_2]
        mock_job_result.n_total_trials = 2
        mock_job_result.stats = mock.MagicMock()
        mock_job_result.stats.evals = {
            "test_eval": mock.MagicMock(pass_at_k={"pass@1": 0.5})
        }

        with mock.patch.object(task, '_get_task_count', return_value=2):
            task._dump_eval_results(mock_job, mock_job_result)

        out_json_path = os.path.join(task.out_dir, "test_dataset.json")
        self.assertTrue(os.path.exists(out_json_path))
        with open(out_json_path, 'r') as f:
            results = json.load(f)
        self.assertEqual(results["n_errors"], 1)
        self.assertIn("pass@1", results["pass_at_k"])

    def test_dump_eval_results_all_errors(self):
        task = HarborTask(self.cfg)
        task.dataset_cfgs = [{"abbr": "test_dataset"}]
        task.out_dir = os.path.join(self.temp_dir, "results")
        task.logger = mock.MagicMock()
        os.makedirs(task.out_dir, exist_ok=True)

        mock_job = mock.MagicMock()
        mock_job.config = mock.MagicMock()

        mock_trial = mock.MagicMock()
        mock_trial.exception_info = mock.MagicMock()
        mock_trial.exception_info.exception_type = "AgentTimeoutError"

        mock_job_result = mock.MagicMock()
        mock_job_result.trial_results = [mock_trial]
        mock_job_result.n_total_trials = 1
        mock_job_result.stats = None

        with mock.patch.object(task, '_get_task_count', return_value=1):
            task._dump_eval_results(mock_job, mock_job_result)

        out_json_path = os.path.join(task.out_dir, "test_dataset.json")
        with open(out_json_path, 'r') as f:
            results = json.load(f)
        self.assertEqual(results["avg_score"], 0.0)
        self.assertEqual(results["n_errors"], 1)

    def test_dump_eval_results_reward_distribution(self):
        task = HarborTask(self.cfg)
        task.dataset_cfgs = [{"abbr": "test_dataset"}]
        task.out_dir = os.path.join(self.temp_dir, "results")
        task.logger = mock.MagicMock()
        os.makedirs(task.out_dir, exist_ok=True)

        mock_job = mock.MagicMock()
        mock_job.config = mock.MagicMock()

        mock_trials = []
        for score in [1.0, 1.0, 0.0]:
            mock_trial = mock.MagicMock()
            mock_trial.exception_info = None
            mock_trial.verifier_result = mock.MagicMock()
            mock_trial.verifier_result.rewards = {"score": score}
            mock_trials.append(mock_trial)

        mock_job_result = mock.MagicMock()
        mock_job_result.trial_results = mock_trials
        mock_job_result.n_total_trials = 3
        mock_job_result.stats = None

        with mock.patch.object(task, '_get_task_count', return_value=3):
            task._dump_eval_results(mock_job, mock_job_result)

        out_json_path = os.path.join(task.out_dir, "test_dataset.json")
        with open(out_json_path, 'r') as f:
            results = json.load(f)
        self.assertEqual(len(results["reward_distribution"]), 2)

    def test_dump_eval_results_exception_distribution(self):
        task = HarborTask(self.cfg)
        task.dataset_cfgs = [{"abbr": "test_dataset"}]
        task.out_dir = os.path.join(self.temp_dir, "results")
        task.logger = mock.MagicMock()
        os.makedirs(task.out_dir, exist_ok=True)

        mock_job = mock.MagicMock()
        mock_job.config = mock.MagicMock()

        mock_trials = []
        for exc_type in ["AgentTimeoutError", "AgentTimeoutError", "VerifierError"]:
            mock_trial = mock.MagicMock()
            mock_trial.exception_info = mock.MagicMock()
            mock_trial.exception_info.exception_type = exc_type
            mock_trials.append(mock_trial)

        mock_job_result = mock.MagicMock()
        mock_job_result.trial_results = mock_trials
        mock_job_result.n_total_trials = 3
        mock_job_result.stats = None

        with mock.patch.object(task, '_get_task_count', return_value=3):
            task._dump_eval_results(mock_job, mock_job_result)

        out_json_path = os.path.join(task.out_dir, "test_dataset.json")
        with open(out_json_path, 'r') as f:
            results = json.load(f)
        self.assertEqual(results["n_errors"], 3)
        self.assertEqual(len(results["exception_distribution"]), 2)

    def test_dump_eval_results_no_verifier_result(self):
        task = HarborTask(self.cfg)
        task.dataset_cfgs = [{"abbr": "test_dataset"}]
        task.out_dir = os.path.join(self.temp_dir, "results")
        task.logger = mock.MagicMock()
        os.makedirs(task.out_dir, exist_ok=True)

        mock_job = mock.MagicMock()
        mock_job.config = mock.MagicMock()

        mock_trial_1 = mock.MagicMock()
        mock_trial_1.exception_info = None
        mock_trial_1.verifier_result = None

        mock_trial_2 = mock.MagicMock()
        mock_trial_2.exception_info = None
        mock_trial_2.verifier_result = mock.MagicMock()
        mock_trial_2.verifier_result.rewards = {"accuracy": 1.0}

        mock_job_result = mock.MagicMock()
        mock_job_result.trial_results = [mock_trial_1, mock_trial_2]
        mock_job_result.n_total_trials = 2
        mock_job_result.stats = None

        with mock.patch.object(task, '_get_task_count', return_value=2):
            task._dump_eval_results(mock_job, mock_job_result)

        out_json_path = os.path.join(task.out_dir, "test_dataset.json")
        with open(out_json_path, 'r') as f:
            results = json.load(f)
        self.assertEqual(results["avg_score"], 0.5)

    def test_dump_eval_results_multiple_evals(self):
        task = HarborTask(self.cfg)
        task.dataset_cfgs = [{"abbr": "test_dataset"}]
        task.out_dir = os.path.join(self.temp_dir, "results")
        task.logger = mock.MagicMock()
        os.makedirs(task.out_dir, exist_ok=True)

        mock_job = mock.MagicMock()
        mock_job.config = mock.MagicMock()

        mock_trial = mock.MagicMock()
        mock_trial.exception_info = None
        mock_trial.verifier_result = mock.MagicMock()
        mock_trial.verifier_result.rewards = {"accuracy": 1.0}

        mock_job_result = mock.MagicMock()
        mock_job_result.trial_results = [mock_trial]
        mock_job_result.n_total_trials = 1
        mock_job_result.stats = mock.MagicMock()
        mock_job_result.stats.evals = {
            "eval1": mock.MagicMock(pass_at_k=None),
            "eval2": mock.MagicMock(pass_at_k={"pass@1": 0.8})
        }

        with mock.patch.object(task, '_get_task_count', return_value=1):
            task._dump_eval_results(mock_job, mock_job_result)

        out_json_path = os.path.join(task.out_dir, "test_dataset.json")
        with open(out_json_path, 'r') as f:
            results = json.load(f)
        self.assertEqual(results["pass_at_k"], {"pass@1": 0.8})

    def test_run_method_integration(self):
        task = HarborTask(self.cfg)
        task.logger = mock.MagicMock()
        task.task_state_manager = self.task_state_manager

        with mock.patch.object(task, '_set_api_key') as mock_set_key, \
             mock.patch.object(task, '_prepare_out_dir') as mock_prepare, \
             mock.patch.object(task, '_run_harbor_job') as mock_run_job, \
             mock.patch.object(task, '_dump_eval_results') as mock_dump:
            mock_run_job.return_value = (mock.MagicMock(), mock.MagicMock())
            task.run(self.task_state_manager)
            mock_set_key.assert_called_once()
            mock_prepare.assert_called_once()
            mock_dump.assert_called_once()

    def test_parse_args(self):
        from ais_bench.benchmark.tasks.custom_tasks.harbor_task import parse_args
        original_argv = sys.argv
        try:
            sys.argv = ["harbor_task.py", "test_config.py"]
            args = parse_args()
            self.assertEqual(args.config, "test_config.py")
        finally:
            sys.argv = original_argv


class TestHarborTaskRunHarborJob(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_cfg_base = {
            "type": "openai",
            "abbr": "test_model",
            "agent_name": "test_agent",
            "model_names": ["hosted_vllm/qwen3"],
            "agent_kwargs": {"api_base": "http://0.0.0.0:8080/v1"},
            "agent_env": {"TEST_VAR": "test_value"}
        }
        self.dataset_cfg_base = {
            "abbr": "test_dataset",
            "args": {
                "n_attempts": 2,
                "timeout_multiplier": 1.5,
                "n_concurrent_trials": 3,
                "path": "/path/to/dataset/"
            }
        }
        self.cfg_base = ConfigDict({
            "work_dir": self.temp_dir,
            "models": [self.model_cfg_base],
            "datasets": [[self.dataset_cfg_base]],
            "cli_args": {"debug": False}
        })

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @mock.patch.dict(sys.modules, {
        'harbor': mock.MagicMock(),
        'harbor.cli': mock.MagicMock(),
        'harbor.cli.utils': mock.MagicMock(),
        'harbor.job': mock.MagicMock(),
        'harbor.models': mock.MagicMock(),
        'harbor.models.job': mock.MagicMock(),
        'harbor.models.job.config': mock.MagicMock(),
        'harbor.models.agent': mock.MagicMock(),
        'harbor.models.agent.name': mock.MagicMock(),
        'harbor.models.environment_type': mock.MagicMock(),
    })
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.harbor_task.AISLogger')
    def test_run_harbor_job_basic(self, mock_ais_logger):
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        task = HarborTask(self.cfg_base)
        task.logger = mock_logger
        task.out_detail_dir = os.path.join(self.temp_dir, "results", "test_model", "test_dataset")

        from harbor.cli.utils import run_async
        from harbor.job import Job

        mock_job = mock.MagicMock()
        mock_job_result = mock.MagicMock()
        run_async.return_value = (mock_job, mock_job_result)

        with mock.patch.object(task, '_get_task_count', return_value=5):
            with mock.patch.object(task, '_run_with_tqdm') as mock_run_tqdm:
                mock_run_tqdm.return_value = (mock_job, mock_job_result)
                job, result = task._run_harbor_job()
                mock_run_tqdm.assert_called_once()

    @mock.patch.dict(sys.modules, {
        'harbor': mock.MagicMock(),
        'harbor.cli': mock.MagicMock(),
        'harbor.cli.utils': mock.MagicMock(),
        'harbor.job': mock.MagicMock(),
        'harbor.models': mock.MagicMock(),
        'harbor.models.job': mock.MagicMock(),
        'harbor.models.job.config': mock.MagicMock(),
        'harbor.models.agent': mock.MagicMock(),
        'harbor.models.agent.name': mock.MagicMock(),
        'harbor.models.environment_type': mock.MagicMock(),
    })
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.harbor_task.AISLogger')
    def test_run_harbor_job_with_all_options(self, mock_ais_logger):
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        cfg = ConfigDict({
            "work_dir": self.temp_dir,
            "models": [self.model_cfg_base],
            "datasets": [[{
                "abbr": "test_dataset",
                "args": {
                    "n_attempts": 1,
                    "timeout_multiplier": 1.0,
                    "agent_timeout_multiplier": 2.0,
                    "verifier_timeout_multiplier": 1.5,
                    "agent_setup_timeout_multiplier": 1.2,
                    "environment_build_timeout_multiplier": 1.1,
                    "debug": True,
                    "n_concurrent_trials": 5,
                    "quiet": True,
                    "max_retries": 3,
                    "retry_include_exceptions": ["AgentTimeoutError"],
                    "retry_exclude_exceptions": ["VerifierError"],
                    "environment_type": "docker",
                    "environment_force_build": True,
                    "environment_delete": True,
                    "path": "/path/to/dataset/",
                    "disable_verification": True,
                    "verifier_env": ["VAR1=value1", "VAR2=value2"]
                }
            }]],
            "cli_args": {"debug": False}
        })

        task = HarborTask(cfg)
        task.logger = mock_logger
        task.out_detail_dir = os.path.join(self.temp_dir, "results", "test_model", "test_dataset")

        from harbor.cli.utils import run_async
        from harbor.job import Job

        mock_job = mock.MagicMock()
        mock_job_result = mock.MagicMock()
        run_async.return_value = (mock_job, mock_job_result)

        with mock.patch.object(task, '_get_task_count', return_value=5):
            with mock.patch.object(task, '_run_with_tqdm') as mock_run_tqdm:
                mock_run_tqdm.return_value = (mock_job, mock_job_result)
                job, result = task._run_harbor_job()
                mock_run_tqdm.assert_called_once()

    @mock.patch.dict(sys.modules, {
        'harbor': mock.MagicMock(),
        'harbor.cli': mock.MagicMock(),
        'harbor.cli.utils': mock.MagicMock(),
        'harbor.job': mock.MagicMock(),
        'harbor.models': mock.MagicMock(),
        'harbor.models.job': mock.MagicMock(),
        'harbor.models.job.config': mock.MagicMock(),
        'harbor.models.agent': mock.MagicMock(),
        'harbor.models.agent.name': mock.MagicMock(),
        'harbor.models.environment_type': mock.MagicMock(),
    })
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.harbor_task.AISLogger')
    def test_run_harbor_job_with_name_version(self, mock_ais_logger):
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        cfg = ConfigDict({
            "work_dir": self.temp_dir,
            "models": [self.model_cfg_base],
            "datasets": [[{
                "abbr": "test_dataset",
                "args": {"dataset_name_version": "my_dataset@v1.0"}
            }]],
            "cli_args": {"debug": False}
        })

        task = HarborTask(cfg)
        task.logger = mock_logger
        task.out_detail_dir = os.path.join(self.temp_dir, "results", "test_model", "test_dataset")

        from harbor.cli.utils import run_async
        from harbor.job import Job

        mock_job = mock.MagicMock()
        mock_job_result = mock.MagicMock()
        run_async.return_value = (mock_job, mock_job_result)

        with mock.patch.object(task, '_get_task_count', return_value=5):
            with mock.patch.object(task, '_run_with_tqdm') as mock_run_tqdm:
                mock_run_tqdm.return_value = (mock_job, mock_job_result)
                job, result = task._run_harbor_job()
                mock_run_tqdm.assert_called_once()

    @mock.patch.dict(sys.modules, {
        'harbor': mock.MagicMock(),
        'harbor.cli': mock.MagicMock(),
        'harbor.cli.utils': mock.MagicMock(),
        'harbor.job': mock.MagicMock(),
        'harbor.models': mock.MagicMock(),
        'harbor.models.job': mock.MagicMock(),
        'harbor.models.job.config': mock.MagicMock(),
        'harbor.models.agent': mock.MagicMock(),
        'harbor.models.agent.name': mock.MagicMock(),
        'harbor.models.environment_type': mock.MagicMock(),
    })
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.harbor_task.AISLogger')
    def test_run_harbor_job_without_model_names(self, mock_ais_logger):
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        cfg = ConfigDict({
            "work_dir": self.temp_dir,
            "models": [{"abbr": "test_model", "agent_name": "oracle"}],
            "datasets": [[self.dataset_cfg_base]],
            "cli_args": {"debug": False}
        })

        task = HarborTask(cfg)
        task.logger = mock_logger
        task.out_detail_dir = os.path.join(self.temp_dir, "results", "test_model", "test_dataset")

        from harbor.cli.utils import run_async
        from harbor.job import Job

        mock_job = mock.MagicMock()
        mock_job_result = mock.MagicMock()
        run_async.return_value = (mock_job, mock_job_result)

        with mock.patch.object(task, '_get_task_count', return_value=5):
            with mock.patch.object(task, '_run_with_tqdm') as mock_run_tqdm:
                mock_run_tqdm.return_value = (mock_job, mock_job_result)
                job, result = task._run_harbor_job()
                mock_run_tqdm.assert_called_once()

    @mock.patch.dict(sys.modules, {
        'harbor': mock.MagicMock(),
        'harbor.cli': mock.MagicMock(),
        'harbor.cli.utils': mock.MagicMock(),
        'harbor.job': mock.MagicMock(),
        'harbor.models': mock.MagicMock(),
        'harbor.models.job': mock.MagicMock(),
        'harbor.models.job.config': mock.MagicMock(),
        'harbor.models.agent': mock.MagicMock(),
        'harbor.models.agent.name': mock.MagicMock(),
        'harbor.models.environment_type': mock.MagicMock(),
    })
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.harbor_task.AISLogger')
    def test_run_harbor_job_resume_existing(self, mock_ais_logger):
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        task = HarborTask(self.cfg_base)
        task.logger = mock_logger

        existing_job_dir = os.path.join(self.temp_dir, "results", "test_model", "test_dataset", "details")
        os.makedirs(existing_job_dir, exist_ok=True)
        config_path = os.path.join(existing_job_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump({"job_name": "details"}, f)

        task.out_detail_dir = os.path.join(self.temp_dir, "results", "test_model", "test_dataset")

        from harbor.cli.utils import run_async
        from harbor.job import Job

        mock_job = mock.MagicMock()
        mock_job_result = mock.MagicMock()
        run_async.return_value = (mock_job, mock_job_result)

        with mock.patch.object(task, '_resume_job') as mock_resume:
            mock_resume.return_value = (mock_job, mock_job_result)
            job, result = task._run_harbor_job()
            mock_resume.assert_called_once()


class TestHarborTaskGetTaskCount(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_cfg = {"abbr": "test_model"}
        self.dataset_cfg = {"abbr": "test_dataset", "args": {"path": "/path/to/dataset/"}}
        self.cfg = ConfigDict({
            "work_dir": self.temp_dir,
            "models": [self.model_cfg],
            "datasets": [[self.dataset_cfg]],
            "cli_args": {"debug": False}
        })

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @mock.patch.dict(sys.modules, {
        'harbor': mock.MagicMock(),
        'harbor.cli': mock.MagicMock(),
        'harbor.cli.utils': mock.MagicMock(),
        'harbor.job': mock.MagicMock(),
        'harbor.models': mock.MagicMock(),
        'harbor.models.job': mock.MagicMock(),
        'harbor.models.job.config': mock.MagicMock(),
        'harbor.models.agent': mock.MagicMock(),
        'harbor.models.agent.name': mock.MagicMock(),
        'harbor.models.environment_type': mock.MagicMock(),
    })
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.harbor_task.AISLogger')
    def test_get_task_count(self, mock_ais_logger):
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        task = HarborTask(self.cfg)
        task.logger = mock_logger

        from harbor.cli.utils import run_async
        run_async.return_value = 2

        mock_config = mock.MagicMock()
        count = task._get_task_count(mock_config)
        self.assertEqual(count, 2)


class TestHarborTaskRunWithProgress(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_cfg = {"abbr": "test_model"}
        self.dataset_cfg = {"abbr": "test_dataset", "args": {"path": "/path/to/dataset/"}}
        self.cfg = ConfigDict({
            "work_dir": self.temp_dir,
            "models": [self.model_cfg],
            "datasets": [[self.dataset_cfg]],
            "cli_args": {"debug": False}
        })
        self.task_state_manager = mock.MagicMock(spec=TaskStateManager)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @mock.patch.dict(sys.modules, {
        'harbor': mock.MagicMock(),
        'harbor.cli': mock.MagicMock(),
        'harbor.cli.utils': mock.MagicMock(),
        'harbor.job': mock.MagicMock(),
        'harbor.models': mock.MagicMock(),
        'harbor.models.job': mock.MagicMock(),
        'harbor.models.job.config': mock.MagicMock(),
        'harbor.models.agent': mock.MagicMock(),
        'harbor.models.agent.name': mock.MagicMock(),
        'harbor.models.environment_type': mock.MagicMock(),
    })
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.harbor_task.AISLogger')
    def test_run_with_tqdm(self, mock_ais_logger):
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        task = HarborTask(self.cfg)
        task.logger = mock_logger
        task.task_state_manager = self.task_state_manager

        from harbor.cli.utils import run_async
        from harbor.job import Job

        mock_job = mock.MagicMock()
        mock_job.job_dir = Path(self.temp_dir)
        mock_job_result = mock.MagicMock()
        run_async.return_value = (mock_job, mock_job_result)

        mock_config = mock.MagicMock()
        job, result = task._run_with_tqdm(mock_config, 10)
        self.assertEqual(job, mock_job)
        self.assertEqual(result, mock_job_result)

    @mock.patch.dict(sys.modules, {
        'harbor': mock.MagicMock(),
        'harbor.cli': mock.MagicMock(),
        'harbor.cli.utils': mock.MagicMock(),
        'harbor.job': mock.MagicMock(),
        'harbor.models': mock.MagicMock(),
        'harbor.models.job': mock.MagicMock(),
        'harbor.models.job.config': mock.MagicMock(),
        'harbor.models.agent': mock.MagicMock(),
        'harbor.models.agent.name': mock.MagicMock(),
        'harbor.models.environment_type': mock.MagicMock(),
    })
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.harbor_task.AISLogger')
    def test_run_with_tqdm_no_state_manager(self, mock_ais_logger):
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        task = HarborTask(self.cfg)
        task.logger = mock_logger
        task.task_state_manager = None

        from harbor.cli.utils import run_async
        from harbor.job import Job

        mock_job = mock.MagicMock()
        mock_job.job_dir = None
        mock_job_result = mock.MagicMock()
        run_async.return_value = (mock_job, mock_job_result)

        mock_config = mock.MagicMock()
        job, result = task._run_with_tqdm(mock_config, 10)
        self.assertEqual(job, mock_job)


if __name__ == '__main__':
    unittest.main()