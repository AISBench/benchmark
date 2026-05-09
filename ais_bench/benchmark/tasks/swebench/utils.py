import subprocess
from typing import Callable, Iterable, TypeVar

from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import SWEB_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchRuntimeError

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "SWE-bench/SWE-bench_Multilingual",
}


def cleanup_swebench_containers():
    """Stop and remove Docker containers created by SWE-bench tasks.

    Cleans both:
    - miniswe-agent inference containers (names start with 'minisweagent-')
    - SWE-bench eval harness containers (names contain 'sweb.eval')
    """
    name_filters = ["minisweagent-", "sweb.eval"]
    for name_filter in name_filters:
        try:
            r = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"name={name_filter}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode != 0 or not (r.stdout or "").strip():
                continue
            ids = [x.strip() for x in r.stdout.strip().splitlines() if x.strip()]
            if not ids:
                continue
            subprocess.run(
                ["docker", "rm", "-f"] + ids,
                capture_output=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass


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
            f"aborting {task_label}. Images: {failed}"
        )
