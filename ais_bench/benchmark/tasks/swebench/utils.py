import subprocess
import uuid
from typing import Callable, Iterable, Optional, Set, TypeVar

from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import SWEB_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchRuntimeError

_logger = AISLogger(name="swebench.utils")

DEFAULT_LITELLM_TIMEOUT = 200

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "SWE-bench/SWE-bench_Multilingual",
}


SWEBENCH_SESSION_LABEL = "ais_bench.swebench.session"


def make_swebench_session_id() -> str:
    return uuid.uuid4().hex


def _merge_docker_labels(labels, session_id: str) -> dict:
    """Merge session label into Docker labels dict.

    Docker SDK ``containers.create/run(labels=...)`` expects a mapping
    (label key -> value).  Always returns a dict.
    """
    if isinstance(labels, dict):
        merged = dict(labels)
    elif isinstance(labels, (list, tuple)):
        # Convert ``["k=v", ...]`` to dict
        merged = {}
        for item in labels:
            if isinstance(item, str) and "=" in item:
                k, v = item.split("=", 1)
                merged[k] = v
    else:
        merged = {}
    merged[SWEBENCH_SESSION_LABEL] = session_id
    return merged


class _DockerContainersWithSessionLabel:
    def __init__(self, containers, session_id: str):
        self._containers = containers
        self._session_id = session_id

    def create(self, *args, **kwargs):
        kwargs["labels"] = _merge_docker_labels(
            kwargs.get("labels"),
            self._session_id,
        )
        return self._containers.create(*args, **kwargs)

    def run(self, *args, **kwargs):
        kwargs["labels"] = _merge_docker_labels(
            kwargs.get("labels"),
            self._session_id,
        )
        return self._containers.run(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._containers, name)


class _DockerClientWithSessionLabel:
    def __init__(self, client, session_id: str):
        self._client = client
        self.containers = _DockerContainersWithSessionLabel(
            client.containers,
            session_id,
        )

    def __getattr__(self, name):
        return getattr(self._client, name)


def add_swebench_session_label_to_docker_client(client, session_id: str):
    """Return a Docker client wrapper that labels containers it creates."""
    return _DockerClientWithSessionLabel(client, session_id)


def list_swebench_container_ids(session_id: Optional[str] = None) -> Set[str]:
    """Return Docker container IDs tagged for one SWE-bench task session."""
    if not session_id:
        return set()

    container_ids: Set[str] = set()
    try:
        r = subprocess.run(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label={SWEBENCH_SESSION_LABEL}={session_id}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            container_ids.update(
                x.strip() for x in r.stdout.strip().splitlines() if x.strip()
            )
    except subprocess.TimeoutExpired:
        _logger.debug("Timed out listing containers for session %s", session_id)
    except FileNotFoundError:
        _logger.debug("Docker not available; cannot list containers")
    except Exception:
        _logger.debug("Unexpected error listing containers", exc_info=True)
    return container_ids


def cleanup_swebench_containers(
    *,
    container_ids: Optional[Iterable[str]] = None,
    session_id: Optional[str] = None,
):
    """Stop and remove containers created by the current SWE-bench task."""
    targets = set(container_ids or [])
    targets.update(list_swebench_container_ids(session_id))
    targets = sorted(targets)
    if not targets:
        return
    try:
        subprocess.run(
            ["docker", "rm", "-f"] + targets,
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        _logger.warning("Timed out removing containers: %s", targets)
    except FileNotFoundError:
        _logger.debug("Docker not available; cannot remove containers")
    except Exception:
        _logger.warning("Unexpected error removing containers", exc_info=True)


def add_swebench_session_label_to_run_args(config: dict, session_id: str) -> None:
    """Add this task's Docker label to mini-swe-agent Docker run args."""
    environment = config.setdefault("environment", {})
    run_args = list(environment.get("run_args", ["--rm"]))
    label_flag = f"{SWEBENCH_SESSION_LABEL}={session_id}"
    if label_flag not in run_args:
        run_args.extend(["--label", label_flag])
    environment["run_args"] = run_args


def docker_image_exists_locally(image: str) -> bool:
    r = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return r.returncode == 0


def docker_pull_image(image: str, logger: AISLogger) -> bool:
    logger.info("Pulling Docker image: %s", image)
    r = subprocess.run(["docker", "pull", image])
    return r.returncode == 0


_T = TypeVar("_T")


def ensure_swebench_docker_images(
    items: Iterable[_T],
    logger: AISLogger,
    get_image_name: Callable[[_T], str],
    *,
    task_label: str = "infer",
) -> None:
    """Ensure each item's SWE-bench Docker image exists locally; pull if missing.

    Raises RuntimeError if any required image is still unavailable after pull
    (so tasks are not started with guaranteed-to-fail environments).
    """
    ordered_unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = get_image_name(item)
        if name not in seen:
            seen.add(name)
            ordered_unique.append(name)

    failed: list[str] = []
    for image in ordered_unique:
        if docker_image_exists_locally(image):
            logger.debug("Docker image already present: %s", image)
            continue
        if docker_pull_image(image, logger):
            if docker_image_exists_locally(image):
                continue
        failed.append(image)

    if failed:
        raise AISBenchRuntimeError(
            SWEB_CODES.DOCKER_IMAGE_UNAVAILABLE,
            "Required SWE-bench Docker image(s) missing or pull failed; "
            f"aborting {task_label}. Images: {failed}",
        )
