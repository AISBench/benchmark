import subprocess
import unittest
from unittest.mock import MagicMock, call, patch

from ais_bench.benchmark.tasks.swebench import utils


class TestSWEBenchContainerCleanup(unittest.TestCase):
    def test_list_swebench_container_ids_queries_session_label(self):
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout="one\ntwo\n")

        with patch.object(utils.subprocess, "run", return_value=result) as mock_run:
            container_ids = utils.list_swebench_container_ids("session-1")

        self.assertEqual(container_ids, {"one", "two"})
        self.assertEqual(
            mock_run.call_args_list,
            [
                call(
                    [
                        "docker",
                        "ps",
                        "-aq",
                        "--filter",
                        f"label={utils.SWEBENCH_SESSION_LABEL}=session-1",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ),
            ],
        )

    def test_list_swebench_container_ids_without_session_is_empty(self):
        with patch.object(utils.subprocess, "run", MagicMock()) as mock_run:
            container_ids = utils.list_swebench_container_ids()

        self.assertEqual(container_ids, set())
        mock_run.assert_not_called()

    def test_cleanup_swebench_containers_removes_session_ids(self):
        with patch.object(
            utils, "list_swebench_container_ids", return_value={"session-a", "session-b"}
        ), patch.object(utils.subprocess, "run") as mock_run:
            utils.cleanup_swebench_containers(session_id="session-1")

        mock_run.assert_called_once_with(
            ["docker", "rm", "-f", "session-a", "session-b"],
            capture_output=True,
            timeout=30,
        )

    def test_cleanup_swebench_containers_without_recorded_or_session_ids_is_noop(self):
        with patch.object(utils.subprocess, "run", MagicMock()) as mock_run:
            utils.cleanup_swebench_containers()

        mock_run.assert_not_called()

    def test_add_swebench_session_label_to_run_args_preserves_existing_args(self):
        config = {"environment": {"run_args": ["--rm", "--network", "none"]}}

        utils.add_swebench_session_label_to_run_args(config, "session-1")

        self.assertEqual(
            config["environment"]["run_args"],
            [
                "--rm",
                "--network",
                "none",
                "--label",
                f"{utils.SWEBENCH_SESSION_LABEL}=session-1",
            ],
        )

    def test_docker_client_wrapper_adds_session_label_without_changing_name(self):
        client = MagicMock()
        wrapped = utils.add_swebench_session_label_to_docker_client(
            client,
            "session-1",
        )

        wrapped.containers.create(
            "image",
            name="default-name",
            labels={"existing": "value"},
        )

        client.containers.create.assert_called_once_with(
            "image",
            name="default-name",
            labels={
                "existing": "value",
                utils.SWEBENCH_SESSION_LABEL: "session-1",
            },
        )


if __name__ == "__main__":
    unittest.main()
