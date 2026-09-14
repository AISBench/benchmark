"""L1/L2 baked image builder.

Renders Jinja2 Dockerfile templates and runs ``docker build`` to produce:
- L1: ``swebench/django-{case}-base:latest``
- L2: ``swebench/django-{case}-with-{agent}:latest``
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from rich.console import Console

from .config import (
    BUILD_LOG_DIR,
    DOCKERFILE_L1,
    DOCKERFILES_DIR,
    base_image_tag,
    dockerfile_for_agent,
    l2_image_tag,
    prebuilt_image,
)

console = Console()


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def image_exists(tag: str, *, in_orchestrator: bool = True) -> bool:
    """Check whether a docker image tag exists.

    SWE-bench DinD images live INSIDE the orchestrator container
    (``swebench-orchestrator``), not on the host. So by default we run
    ``docker images`` via ``docker exec`` on the orchestrator.

    Set ``in_orchestrator=False`` to check the host's image list instead.
    """
    if in_orchestrator:
        from .container import exec_in_orchestrator
        try:
            out = exec_in_orchestrator(
                "docker", "images", "--format", "{{.Repository}}:{{.Tag}}", check=False
            ).stdout
        except subprocess.CalledProcessError:
            return False
    else:
        out = _docker("images", "--format", "{{.Repository}}:{{.Tag}}", check=False).stdout
    return tag in out.splitlines()


def _render(template_name: str, **context: str) -> str:
    env = Environment(
        loader=FileSystemLoader(str(DOCKERFILES_DIR)),
        keep_trailing_newline=True,
        trim_blocks=False,
    )
    tmpl = env.get_template(template_name)
    return tmpl.render(**context)


def _render_to_tempfile(template_name: str, **context: str) -> tuple[Path, Path]:
    """Render the template and write to a temp dir alongside a placeholder
    context file. Returns (build_context_dir, dockerfile_path)."""
    rendered = _render(template_name, **context)
    ctx_dir = Path(tempfile.mkdtemp(prefix="swebench-dind-build-"))
    (ctx_dir / "Dockerfile").write_text(rendered)
    return ctx_dir, ctx_dir / "Dockerfile"


def _build_image(
    *,
    tag: str,
    dockerfile: Path,
    context: Path,
    build_args: dict[str, str],
    log_path: Path,
    platform: str = "linux/amd64",
) -> bool:
    BUILD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "build",
        "--platform", platform,
        *[f"--build-arg={k}={v}" for k, v in build_args.items()],
        "-f", str(dockerfile),
        "-t", tag,
        str(context),
    ]
    console.print(f"  [docker build] {tag}")
    with log_path.open("w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    return result.returncode == 0


def build_l1(case: str, *, skip_existing: bool = True, force: bool = False) -> bool:
    """Build the L1 case-base image for ``case``.

    Returns True on success (or if skipped because already present).
    """
    tag = base_image_tag(case)
    if skip_existing and image_exists(tag) and not force:
        console.print(f"  [L1 skip] {tag} already exists")
        return True

    ctx, dockerfile = _render_to_tempfile(DOCKERFILE_L1, BASE_IMAGE=prebuilt_image(case))
    log = BUILD_LOG_DIR / f"l1-{case}.log"
    return _build_image(
        tag=tag,
        dockerfile=dockerfile,
        context=ctx,
        build_args={"BASE_IMAGE": prebuilt_image(case)},
        log_path=log,
    )


def build_l2(case: str, agent: str, *, skip_existing: bool = True, force: bool = False) -> bool:
    """Build the L2 agent-baked image for (case, agent)."""
    template_name = dockerfile_for_agent(agent)
    if template_name is None:
        console.print(f"  [L2 skip] no Dockerfile template for agent={agent!r}")
        return False
    tag = l2_image_tag(case, agent)
    if skip_existing and image_exists(tag) and not force:
        console.print(f"  [L2 skip] {tag} already exists")
        return True
    ctx, dockerfile = _render_to_tempfile(template_name, AGENT=agent)
    log = BUILD_LOG_DIR / f"l2-{case}-{agent}.log"
    return _build_image(
        tag=tag,
        dockerfile=dockerfile,
        context=ctx,
        build_args={
            "BASE_IMAGE": base_image_tag(case),
            "AGENT": agent,
        },
        log_path=log,
    )


def build_l1_all(cases: list[str], *, skip_existing: bool = True, force: bool = False) -> dict[str, bool]:
    """Build L1 images for all cases (sequentially)."""
    results = {}
    for c in cases:
        results[c] = build_l1(c, skip_existing=skip_existing, force=force)
    return results


def build_l2_all(
    cases: list[str], agents: list[str], *, skip_existing: bool = True, force: bool = False,
) -> dict[tuple[str, str], bool]:
    """Build L2 images for all (case, agent) pairs (sequentially)."""
    results = {}
    for c in cases:
        for a in agents:
            results[(c, a)] = build_l2(c, a, skip_existing=skip_existing, force=force)
    return results