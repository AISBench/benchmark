"""Single source of truth for SWE-bench DinD pipeline parameters.

All hardcoded values that were scattered across scripts (model name, image
tags, multipliers, API endpoints, etc.) live here so that adding a new
case/agent requires changing exactly one file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


# === Paths ===
CLI_ROOT = Path(__file__).resolve().parent.parent  # /home/zengziyu/mini_matrix/cli
ROOT = CLI_ROOT.parent  # /home/zengziyu/mini_matrix

CONTAINER_NAME = os.environ.get("CONTAINER_NAME", "swebench-orchestrator")
DATA_CONTAINER_NAME = "swebench-data-3demo"

ORCHESTRATOR_IMAGE = os.environ.get(
    "ORCHESTRATOR_IMAGE",
    "swebench/orchestrator:v0.1-2026-08-04-patched-v11",
)
DATA_IMAGE = "swebench/swebench-data:v0.1-2026-07-30-3demo"

JOBS_DIR = ROOT / "jobs"
TASKS_DIR = ROOT / "tasks"
CONFIG_DIR = ROOT / "config"
LOGS_DIR = ROOT / "logs"
API_KEY_ENV = ROOT / "scripts" / "api_key.env"

ORCH_DIR = ROOT / "orchestrator"
ENTRYPOINT_SH = ORCH_DIR / "entrypoint.sh"
AGENT_PATCHES_DIR = ORCH_DIR / "agent-patches"
IMAGE_CACHE_DIR = ORCH_DIR / "image-cache"

# Inside the orchestrator container
CONTAINER_TASKS_DIR = "/opt/swebench/data/tasks"
CONTAINER_JOBS_DIR = "/opt/swebench/jobs"
CONTAINER_AGENT_PATCHES_DIR = "/opt/swebench/agent-patches"


# === Model & API ===
DEFAULT_MODEL = "openai/Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_API_BASE = "https://api.siliconflow.cn/v1"
DEFAULT_BARE_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


# === Default cases / agents ===
DEFAULT_CASES: list[str] = ["11099", "12308", "13741"]
NEW_CASES: list[str] = ["10097", "10554", "10880"]
ALL_CASES: list[str] = DEFAULT_CASES + NEW_CASES

DEFAULT_AGENTS: list[str] = ["aider", "mini-swe-agent", "qwen-code"]
ALL_AGENTS: list[str] = ["aider", "mini-swe-agent", "qwen-code", "openhands-sdk"]


# === Harbor CLI plumbing ===
# What we pass to each `harbor jobs start` as `--ae KEY=VAL` flags.
COMMON_AE_KEYS = [
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "LITELLM_API_KEY",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
]

# Per-agent extra env (key without value, value substituted by `substitute_ae`)
AGENT_AE: dict[str, list[str]] = {
    "aider": ["AIDER_API_KEY=openai=${OPENAI_API_KEY}"],
    "mini-swe-agent": ["MSWEA_API_KEY=${OPENAI_API_KEY}"],
    "qwen-code": [
        "OPENAI_BASE_URL=https://api.siliconflow.cn/v1",
        "OPENAI_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct",
    ],
    "openhands-sdk": [
        "OPENAI_BASE_URL=https://api.siliconflow.cn/v1",
        "LLM_MODEL=openai/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    ],
}

# Harbor CLI flag for picking the agent
HARBOR_AGENT_FLAG: dict[str, str] = {
    "aider": "aider",
    "mini-swe-agent": "mini-swe-agent",
    "qwen-code": "qwen-coder",
    "openhands-sdk": "openhands-sdk",
}

# Default timeout multipliers (matches scripts/launch_*.sh)
DEFAULT_MULTIPLIERS = {
    "agent_setup": 4,
    "agent": 4,
    "verifier": 4,
    "environment_build": 4,
}


# === Image tags ===
# Base SWE-bench prebuilt image template.
# e.g. docker.1ms.run/swebench/sweb.eval.x86_64.django_1776_django-11099:latest
BASE_PREBUILT_TEMPLATE = (
    "docker.1ms.run/swebench/sweb.eval.x86_64.django_1776_django-{case}:latest"
)


def prebuilt_image(case: str) -> str:
    """Return the upstream SWE-bench prebuilt image for ``case``."""
    return BASE_PREBUILT_TEMPLATE.format(case=case)


def base_image_tag(case: str) -> str:
    """L1 case-base image tag, e.g. ``swebench/django-11099-base:latest``."""
    return f"swebench/django-{case}-base:latest"


def l2_image_tag(case: str, agent: str) -> str:
    """L2 agent-baked image tag, e.g. ``swebench/django-11099-with-aider:latest``."""
    return f"swebench/django-{case}-with-{agent}:latest"


# === Dockerfile template names ===
DOCKERFILE_L1 = "Dockerfile.l1-base.j2"
DOCKERFILE_L2_BY_AGENT: dict[str, str] = {
    "aider": "Dockerfile.l2-agent-aider.j2",
    "mini-swe-agent": "Dockerfile.l2-agent-msa.j2",
    "qwen-code": "Dockerfile.l2-agent-qwen.j2",
    "openhands-sdk": "Dockerfile.l2-agent-oh.j2",
}


# === Build ===
# Bundled templates + context dir for `docker build`
DOCKERFILES_DIR = Path(__file__).parent / "dockerfiles"
BUILD_LOG_DIR = Path("/tmp/swebench-dind-builds")


# === Helpers ===
def task_dir_name(case: str, agent: str) -> str:
    """Return the task directory name, e.g. ``django__django-11099-aider``."""
    return f"django__django-{case}-{agent}"


def container_task_path(case: str, agent: str) -> str:
    """Return the container-side task path passed to ``harbor jobs start``."""
    return f"{CONTAINER_TASKS_DIR}/{task_dir_name(case, agent)}"


def substitute_ae(values: Iterable[str], api_key: str, model: str, api_base: str) -> list[str]:
    """Replace ``${OPENAI_API_KEY}`` etc. in agent env strings."""
    subs = {"${OPENAI_API_KEY}": api_key, "${MODEL}": model, "${API_BASE}": api_base}
    out = []
    for v in values:
        for k, sub in subs.items():
            v = v.replace(k, sub)
        out.append(v)
    return out


def dockerfile_for_agent(agent: str) -> str | None:
    """Map agent → L2 Dockerfile template name."""
    return DOCKERFILE_L2_BY_AGENT.get(agent)