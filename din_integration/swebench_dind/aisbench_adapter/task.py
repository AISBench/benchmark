"""AISBench ``BaseTask`` subclass that wraps a SWE-bench DinD trial.

Loaded by aisbench when ``task.type="SwebenchDindTask"`` is set in a
config. The class:

- Inherits ``BaseTask`` from ais_bench so it auto-registers with the
  ``TASKS`` registry (when ``@TASKS.register_module()`` is applied).
- Implements ``get_command(cfg_path)`` which returns the shell command
  aisbench's ``LocalRunner`` should spawn as a subprocess.
- The subprocess entrypoint is ``swebench_dind.aisbench_adapter.runner``,
  which reads the same ``cfg_path``, parses ``model_cfg`` and
  ``dataset_cfg``, runs the harbor trial via the in-container
  ``swebench-dind launch`` CLI, then writes the AISBench-format
  ``<work_dir>/results/<model>/<dataset>.json``.

NOTE: This module assumes aisbench is installed (so we can import
``BaseTask``). If aisbench is not on PYTHONPATH, the ``@register``
decorator is a no-op so the module still imports.
"""
from __future__ import annotations

from pathlib import Path

try:
    from ais_bench.benchmark.tasks.base import BaseTask
    from ais_bench.benchmark.registry import TASKS
    _AISBENCH_AVAILABLE = True
except ImportError:
    _AISBENCH_AVAILABLE = False

    class BaseTask:  # type: ignore[no-redef]
        """Fallback stub so the module is importable without aisbench."""
        name_prefix = "swebench_dind"
        log_subdir = "logs/eval"
        output_subdir = "results"

        def __init__(self, cfg):
            self.cfg = cfg

        def get_command(self, cfg_path, template=None):
            raise NotImplementedError

        def run(self, task_state_manager):
            raise NotImplementedError

    class _StubRegistry:
        @staticmethod
        def register_module():
            def decorator(cls):
                return cls
            return decorator

    TASKS = _StubRegistry()


@TASKS.register_module()
class SwebenchDindTask(BaseTask):
    """BaseTask wrapper that runs a single harbor trial."""

    name_prefix = "swebench_dind"
    log_subdir = "logs/eval"
    output_subdir = "results"

    def get_command(self, cfg_path: str, template=None) -> str:
        """Return the subprocess command for LocalRunner to spawn."""
        # The actual work happens in the subprocess; ``runner.py`` reads
        # ``cfg_path`` and calls ``launcher.launch_trial`` programmatically.
        return (
            f"python -m swebench_dind.aisbench_adapter.runner "
            f"--config {cfg_path} "
            f"--work-dir {self.cfg.get('work_dir', './outputs/')}"
        )

    def run(self, task_state_manager):
        """In subprocess: invoke ``runner.main()`` and report progress."""
        from .runner import main as runner_main
        runner_main(self.cfg, task_state_manager)


# === Helper for AISBench dataset registration ===
try:
    from ais_bench.benchmark.datasets.base import BaseDataset
    from ais_bench.benchmark.registry import DATASETS
    _DATASET_AVAILABLE = True
except ImportError:
    _DATASET_AVAILABLE = False

    class BaseDataset:  # type: ignore[no-redef]
        def __init__(self, path=None):
            self.path = path

    DATASETS = _StubRegistry()


@DATASETS.register_module()
class SwebenchDindDataset(BaseDataset):
    """Minimal dataset that points at our local task directory.

    AISBench expects ``SwebenchDindDataset`` to expose a list of task
    records (one per (case, agent)). For the simple 3×3 use case we
    auto-derive 3 records from the path naming convention.
    """

    def __init__(self, path: str = "", **kwargs):
        super().__init__(path=path)
        self._records = self._parse_records(Path(path))

    def _parse_records(self, path: Path) -> list[dict]:
        """One record per subdirectory of ``path`` matching our convention.

        Path format expected: ``.../django__django-{case}-{agent}/task.toml``
        """
        records: list[dict] = []
        if not path.exists():
            return records
        for sub in sorted(path.parent.iterdir()):
            if not sub.is_dir() or not sub.name.startswith("django__django-"):
                continue
            # django__django-11099-aider → case=11099, agent=aider
            try:
                _, _, case_agent = sub.name.partition("django__django-")
                case, _, agent = case_agent.partition("-")
            except ValueError:
                continue
            toml = sub / "task.toml"
            if not toml.exists():
                continue
            records.append({
                "case": case,
                "agent": agent,
                "task_path": str(sub),
                "task_toml": str(toml),
            })
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)