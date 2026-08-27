"""Idempotent-install probe patcher for harbor agent modules.

Patches the installed harbor source under
``/usr/local/lib/python3.12/dist-packages/harbor/agents/installed/<agent>.py``
inside the orchestrator container so that the ``install()`` method skips
work when the agent is already pre-installed in the baked image.

Generalizes the legacy ``scripts/patch_qwen_code.py``.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from rich.console import Console

from .container import exec_in_orchestrator

console = Console()

HARBOR_PKG_DIR = Path("/usr/local/lib/python3.12/dist-packages/harbor")

# (agent, harbor_module_filename)
AGENT_TO_MODULE: dict[str, str] = {
    "aider": "aider.py",
    "mini-swe-agent": "mini_swe_agent.py",
    "qwen-code": "qwen_code.py",
    "openhands-sdk": "openhands_sdk.py",
}

# CLI binary name per agent (used for `command -v`)
AGENT_CLI: dict[str, str] = {
    "aider": "aider",
    "mini-swe-agent": "mini-swe-agent",
    "qwen-code": "qwen",
    "openhands-sdk": "openhands",
}


def _exec_python_in_container(code: str) -> str:
    """Run a Python script inside the orchestrator container.

    We write the code to a temp file on the host, then ``docker cp`` it
    into the container and run it. Avoids shell-escaping nightmares with
    nested f-strings and quotes.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        local = Path(f.name)
    try:
        remote = f"/tmp/{local.name}"
        subprocess.run(
            ["docker", "cp", str(local), f"swebench-orchestrator:{remote}"],
            check=True,
        )
        result = exec_in_orchestrator("python3", remote)
        return result.stdout
    finally:
        local.unlink(missing_ok=True)
        exec_in_orchestrator("rm", "-f", remote, check=False)


def _install_path(module_file: str) -> str:
    return str(HARBOR_PKG_DIR / "agents" / "installed" / module_file)


def _build_patch_script(path: str, cli: str) -> str:
    """Return the in-container Python script that does the actual patching.

    Built as a plain triple-quoted string with ``{path}`` / ``{cli}``
    placeholders filled via ``str.format``. (We avoid f-strings because the
    embedded raw-strings + escapes are a parsing nightmare.)
    """
    template = """
import re, os, glob, sys

PATH = __PATH__
CLI = __CLI__
MARKER = 'Idempotent probe (PATCHED by swebench-dind CLI)'

content = open(PATH).read()
if MARKER in content:
    print('ALREADY PATCHED')
    sys.exit(0)

pattern = re.compile(
    r'(@override\\s*\\n\\s*async def install\\(self, environment: BaseEnvironment\\) -> None:\\n)'
    r'(.*?)'
    r'(\\n    (?:async )?def \\w+|\\nclass \\w+)',
    re.DOTALL,
)
m = pattern.search(content)
if m is None:
    print('install() method not found')
    sys.exit(1)

header, body, tail = m.group(1), m.group(2), m.group(3)
not_found = CLI + '-not-found'
probe_lines = [
    '        # Idempotent probe (PATCHED by swebench-dind CLI): skip install if baked.',
    '        probe = await environment.exec(',
    '            command="command -v ' + CLI + ' >/dev/null 2>&1 && ' + CLI + ' --version || { echo \\'' + not_found + '\\'; exit 1; }"',
    '        )',
    '        if probe.return_code == 0:',
    '            return',
    '',
]
probe = '\\n'.join(probe_lines) + '\\n'
new_content = content[:m.start()] + header + probe + body + tail + content[m.end():]
open(PATH, 'w').write(new_content)

removed = []
base = os.path.basename(PATH).replace('.py', '')
for pyc in glob.glob(os.path.dirname(PATH) + '/__pycache__/' + base + '.cpython-*.pyc'):
    os.remove(pyc)
    removed.append(pyc)

print('PATCHED')
print('Removed pyc:', removed)
"""
    return template.replace('__PATH__', repr(path)).replace('__CLI__', repr(cli))


def patch_agent(agent: str) -> bool:
    """Patch the harbor agent module to add the idempotent install probe.

    Idempotent: re-running on an already-patched module is a no-op.
    Returns True if patched (or already patched), False on error.
    """
    module_file = AGENT_TO_MODULE.get(agent)
    cli = AGENT_CLI.get(agent)
    if module_file is None or cli is None:
        console.print(f"[red]Unknown agent: {agent}[/red]")
        return False

    path = _install_path(module_file)
    console.print(f"[bold]patch[/bold] {path}")
    py_script = _build_patch_script(path, cli)
    out = _exec_python_in_container(py_script)
    console.print(out)
    return "PATCHED" in out or "ALREADY PATCHED" in out


def patch_all(agents: list[str]) -> dict[str, bool]:
    """Patch a list of agents. Returns {agent: success}."""
    return {a: patch_agent(a) for a in agents}