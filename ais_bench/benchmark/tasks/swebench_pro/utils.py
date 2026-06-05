import os
import subprocess
import re
from typing import Callable, Iterable, TypeVar
import json

from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import SWEBP_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchRuntimeError


def cleanup_swebench_pro_containers():
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


def list_swebench_pro_images(client) -> set[str]:
    """List all current SWE-bench Pro images (jefzda/sweap-images:*)"""
    existing_images = set()
    try:
        for image in client.images.list(all=True):
            for tag in image.tags:
                if tag.startswith("jefzda/sweap-images:"):
                    existing_images.add(tag)
    except Exception:
        pass
    return existing_images


def remove_swebench_pro_image(client, image_tag: str, logger: AISLogger):
    """Remove a single SWE-bench Pro image"""
    try:
        client.images.remove(image_tag, force=True)
        logger.debug(f"Removed image: {image_tag}")
    except Exception as e:
        logger.warning(f"Failed to remove image {image_tag}: {e}")


def clean_swebench_pro_images(client, prior_images: set[str], logger: AISLogger):
    """Clean up new images pulled during evaluation (not in the original list)"""
    current_images = list_swebench_pro_images(client)
    new_images = current_images - prior_images
    
    if new_images:
        logger.info("Cleaning up %d new SWE-bench Pro images...", len(new_images))
        for image_tag in new_images:
            remove_swebench_pro_image(client, image_tag, logger)
        logger.info("Image cleanup completed.")
    else:
        logger.debug("No new images to clean up.")


def docker_image_exists_locally(image: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return r.returncode == 0
    except Exception:
        return False


def docker_pull_image(image: str, logger: AISLogger) -> bool:
    logger.info("Pulling Docker image: %s", image)
    r = subprocess.run(["docker", "pull", image])
    return r.returncode == 0


_T = TypeVar("_T")


def ensure_swebench_pro_docker_images(
    items: Iterable[_T],
    logger: AISLogger,
    get_image_name: Callable[[_T], str],
    *,
    task_label: str = "infer",
) -> None:
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
        logger.info("Docker image not present: %s, need pull!!!!", image)
        if docker_pull_image(image, logger):
            if docker_image_exists_locally(image):
                continue
        failed.append(image)

    if failed:
        raise AISBenchRuntimeError(
            SWEBP_CODES.DOCKER_IMAGE_UNAVAILABLE,
            "Required SWE-bench Pro Docker image(s) missing or pull failed; "
            f"aborting {task_label}. Images: {failed}"
        )


def build_problem_statement(row: dict) -> str:
    parts = [row["problem_statement"]]
    if row.get("requirements"):
        parts.append(f"\nRequirements:\n{row['requirements']}")
    if row.get("interface"):
        parts.append(f"\nNew interfaces introduced:\n{row['interface']}")
    return "\n".join(parts)


def get_dockerhub_image_uri(raw_instance: dict) -> str:
    uid = raw_instance["instance_id"]
    repo_name = raw_instance["repo"]
    repo_base, repo_name_only = repo_name.lower().split("/")
    hsh = uid.replace("instance_", "")

    if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = 'element-web'  # Keep full name for this one case
    elif 'element-hq' in repo_name.lower() and 'element-web' in repo_name.lower():
        repo_name_only = 'element'
        if hsh.endswith('-vnan'):
            hsh = hsh[:-5]
    # All other repos: strip -vnan suffix
    elif hsh.endswith('-vnan'):
        hsh = hsh[:-5]
    
    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]
    
    return f"jefzda/sweap-images:{tag}"


def load_base_docker(docker_dir_abs, iid):
    with open(f"{docker_dir_abs}/base_dockerfile/{iid}/Dockerfile") as fp:
        return fp.read()


def instance_docker(docker_dir_abs, iid):
    with open(f"{docker_dir_abs}/instance_dockerfile/{iid}/Dockerfile") as fp:
        return fp.read()


def create_entryscript(docker_dir_abs, sample):
    before_repo_set_cmd = sample["before_repo_set_cmd"].strip().split("\n")[-1]
    selected_test_files_to_run = ",".join(eval(sample["selected_test_files_to_run"]))
    base_commit = sample["base_commit"]
    base_dockerfile = load_base_docker(docker_dir_abs, sample["instance_id"])
    instance_dockerfile = instance_docker(docker_dir_abs, sample["instance_id"])
    
    # Extract ENV commands from dockerfiles
    env_cmds = []
    for dockerfile_content in [base_dockerfile, instance_dockerfile]:
        for line in dockerfile_content.split("\n"):
            line = line.strip()
            if line.startswith("ENV"):
                # Convert ENV commands to export statements
                env_cmd = line.replace("ENV", "export", 1)
                env_cmds.append(env_cmd)
    
    env_cmds = "\n".join(env_cmds)

    entry_script = f"""
{env_cmds}
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files_to_run} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
    return entry_script


def load_local_script(scripts_dir, instance_id, script_name):
    """Load a script file from local scripts directory."""
    script_path = os.path.join(scripts_dir, instance_id, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    with open(script_path, 'r') as f:
        return f.read()


def prepare_run(uid, output_dir, prefix, redo):
    uid_dir = os.path.join(output_dir, uid)
    os.makedirs(uid_dir, exist_ok=True)
    output_path = os.path.join(uid_dir, f"{prefix}_output.json")
    if not redo and os.path.exists(output_path):
        with open(output_path, "r") as f:
            return json.load(f), output_path, os.path.join(uid_dir, "workspace")
    workspace_dir = os.path.join(uid_dir, "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    return None, output_path, workspace_dir


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch."""
    if not patch:
        return patch

    sections = re.split(r'(?=^diff --git )', patch, flags=re.MULTILINE)

    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r'^Binary files .* differ$', section, re.MULTILINE):
            continue
        if re.search(r'^GIT binary patch$', section, re.MULTILINE):
            continue
        kept.append(section)

    return "".join(kept)


def assemble_workspace_files(uid, scripts_dir, docker_dir_abs, patch, sample):
    run_script = load_local_script(scripts_dir, uid, "run_script.sh")
    parser_script = load_local_script(scripts_dir, uid, "parser.py")
    entryscript_content = create_entryscript(docker_dir_abs, sample)

    cleaned_patch = strip_binary_hunks(patch)

    files = {
        "patch.diff": cleaned_patch,
        "run_script.sh": run_script,
        "parser.py": parser_script,
        "entryscript.sh": entryscript_content,
    }
    return files, entryscript_content


def write_files_local(workspace_dir, files):
    for rel_path, content in files.items():
        dst = os.path.join(workspace_dir, rel_path)
        with open(dst, "w") as f:
            f.write(content)


def write_patch_snapshot(output_dir, uid, prefix, patch):
    with open(os.path.join(output_dir, uid, f"{prefix}_patch.diff"), "w") as f:
        f.write(patch)


def collect_outputs_local(workspace_dir, output_dir, uid, prefix, logger):
    def _copy_safe(src_name, dest_name):
        src_path = os.path.join(workspace_dir, src_name)
        dest_path = os.path.join(output_dir, uid, dest_name)
        try:
            with open(src_path, "r") as f_in:
                content = f_in.read()
        except FileNotFoundError:
            content = ""
        with open(dest_path, "w") as f_out:
            f_out.write(content if content is not None else "")

    _copy_safe("stdout.log", f"{prefix}_stdout.log")
    _copy_safe("stderr.log", f"{prefix}_stderr.log")

    # Then try to read output.json
    try:
        with open(os.path.join(workspace_dir, "output.json"), "r") as f_in:
            output = json.load(f_in)
            with open(os.path.join(output_dir, uid, f"{prefix}_output.json"), "w") as f:
                json.dump(output, f)
            return output
    except FileNotFoundError:
        logger.error(
            SWEBP_CODES.HARNESS_RUNTIME_FAILED, 
            f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details"
        )
        return None


def save_entryscript_copy(output_dir, uid, prefix, entryscript_content):
    with open(os.path.join(output_dir, uid, f"{prefix}_entryscript.sh"), "w") as f:
        f.write(entryscript_content if entryscript_content is not None else "")


def eval_with_docker(patch, sample, output_dir, scripts_dir, docker_dir_abs, logger, prefix="", docker_client=None, timeout=7200):
    if docker_client is None:
        try:
            import docker
        except Exception:
            raise RuntimeError("docker SDK is not installed. Install via 'pip install docker' or run without --use_local_docker")
        docker_client = docker.from_env()
 
    uid = sample["instance_id"]
    redo = False
    existing_output, output_path, workspace_dir = prepare_run(uid, output_dir, prefix, redo)
    if existing_output is not None:
        return existing_output

    try:
        try:
            files, entryscript_content = assemble_workspace_files(uid, scripts_dir, docker_dir_abs, patch, sample)
        except FileNotFoundError as e:
            logger.error(SWEBP_CODES.HARNESS_RUNTIME_FAILED, f"Error loading scripts for {uid}: {e}")
            return None
        write_files_local(workspace_dir, files)
        write_patch_snapshot(output_dir, uid, prefix, patch)

        abs_workspace_dir = os.path.abspath(workspace_dir)
        volumes = {abs_workspace_dir: {"bind": "/workspace", "mode": "rw"}}
        run_kwargs = {
            "volumes": volumes,
            "detach": True,
            "remove": True,
            "entrypoint": "/bin/bash",  # Override image entrypoint
            "command": ["-c", "bash /workspace/entryscript.sh"],
        }

        dockerhub_image_uri = get_dockerhub_image_uri(sample)
        logger.debug(f"Using image: {dockerhub_image_uri}")
        container = docker_client.containers.run(
            dockerhub_image_uri,
            **run_kwargs,
        )

        try:
            result = container.wait(timeout=timeout)
            status_code = result.get("StatusCode", 1) if isinstance(result, dict) else 1
        except docker.errors.Timeout:
            logger.error(SWEBP_CODES.HARNESS_RUNTIME_FAILED, f"Container for {uid} timed out after {timeout}s, terminating...")
            try:
                container.stop(timeout=10)  # 10s graceful stop
            except Exception:
                container.kill()  # Force kill
            return {
                "tests": [],
                "error": "timeout",
                "message": f"Evaluation timed out after {timeout} seconds",
                "instance_id": uid,
            }

        if status_code != 0:
            logger.error(SWEBP_CODES.HARNESS_RUNTIME_FAILED, f"Entryscript failed for {uid} with return code: {status_code}")
        output = collect_outputs_local(workspace_dir, output_dir, uid, prefix, logger)
        if output is None:
            return None
        save_entryscript_copy(output_dir, uid, prefix, entryscript_content)

        return output
    except Exception as e:
        logger.error(SWEBP_CODES.HARNESS_RUNTIME_FAILED, f"Error in eval_with_docker for {uid}: {repr(e)}")
        logger.error(SWEBP_CODES.HARNESS_RUNTIME_FAILED, f"Error type: {type(e)}")
        return None


def merge_nested_dicts(d1: dict, d2: dict) -> dict:
    """Merge two nested dictionaries, updating d1 in place.
    If a key exists in both dictionaries, the value from d2 will be used.
    """
    for key, value in d2.items():
        if isinstance(value, dict):
            d1[key] = merge_nested_dicts(d1.get(key, {}), value)
        else:
            d1[key] = value
    return d1