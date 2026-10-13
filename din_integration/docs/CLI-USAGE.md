# CLI Usage Reference

Full reference for every `swebench-dind` subcommand. The CLI is built with
[Typer](https://typer.tiangolo.com/), so `swebench-dind <command> --help`
always shows the live, in-sync option list.

## Top-level

```text
swebench-dind [--version] [--help]
swebench-dind <subcommand> ...
```

Subcommands:

| Command | Purpose |
|---|---|
| `orchestrator` | DinD container lifecycle (start / stop / status) |
| `build` | Build L1 / L2 baked images |
| `launch` | Launch trials (single, 3×3, or N×M matrix) |
| `watch` | Block until a named job's `result.json` is finished |
| `summarize` | Aggregate `jobs/*/result.json` into md / csv / json |
| `patch` | Patch Harbor for idempotent installs |
| `aisbench` | AISBench integration (P1) |

---

## `orchestrator`

Manages the `swebench-orchestrator` DinD container. Equivalent to the legacy
`scripts/start_orchestrator.sh` and `scripts/stop_orchestrator.sh`.

### `orchestrator start`

```text
swebench-dind orchestrator start [--recreate]
```

- Idempotent: if the container is already running with `dockerd` ready,
  prints status and exits.
- `--recreate` removes the existing container first (data is preserved on
  the host because we bind-mount `jobs/`, `tasks/`, `logs/`, etc.).
- Wait up to 120 s for `dockerd` inside the container to come up.
- The OpenAI API key is read from `mini_matrix/scripts/api_key.env`.

### `orchestrator stop`

```text
swebench-dind orchestrator stop [--remove]
```

- `docker stop` by default. Job data on the host is untouched.
- `--remove` also `docker rm -f`s the container.

### `orchestrator status`

```text
swebench-dind orchestrator status
```

Reports:

- container name
- whether the container exists
- whether it's running
- whether `docker info` inside the container succeeds (i.e. DinD dockerd ready)

Example:

```
swebench-orchestrator
  exists:    True
  running:   True
  dockerd:   ✅ ready
```

---

## `build`

Builds the L1 case-base and L2 agent-baked images. Both live **inside**
the orchestrator's Docker daemon (DinD), not on the host, so
`image_exists()` shells out via `docker exec`.

### `build l1`

```text
swebench-dind build l1 --case 11099 --case 12308 [--force]
```

- Required: one or more `--case` numbers.
- Skips if `swebench/django-{case}-base:latest` already exists
  (idempotent by default).
- `--force` rebuilds even if the image is present.
- Renders `Dockerfile.l1-base.j2` with `BASE_IMAGE=prebuilt_image(case)`
  and tags the result.

### `build l2`

```text
swebench-dind build l2 --agent qwen-code [--case 11099] [--force]
```

- Required: one or more `--agent` names.
- `--case` defaults to `DEFAULT_CASES` = `[11099, 12308, 13741]`.
- Builds `swebench/django-{case}-with-{agent}:latest` on top of the
  matching L1 base.
- Each agent has its own Jinja2 template under
  `swebench_dind/dockerfiles/Dockerfile.l2-agent-*.j2`.

### `build all`

```text
swebench-dind build all [--case ...] [--agent ...] [--force]
```

- Defaults: all 6 cases (3 default + 3 new) and the 3 default agents.
- Sequentially builds every L1 + every L2 in the cartesian product.

---

## `launch`

Drives `harbor jobs start` inside the orchestrator. Every trial becomes
one `docker exec` invocation, and the job directory is cleaned of stale
state first (so re-runs work).

### `launch trial`

```text
swebench-dind launch trial --case 11099 --agent aider \
    [--job-name custom-name] [-n 1] [-m MODEL] [--api-base URL] \
    [--wait] [--timeout-min 120]
```

- `--job-name` defaults to `"{agent}-{case}"` (e.g. `aider-11099`).
- `-n` is the harbor trial-count (almost always `1` for SWE-bench).
- `-m` defaults to `openai/Qwen/Qwen3-Coder-30B-A3B-Instruct`.
- `--api-base` defaults to `https://api.siliconflow.cn/v1`.
- `--wait` blocks until `result.json` reports `finished_at`.
- Per-stage timeout multipliers are set to **4×** by default (so the
  heavy QEMU x86_64 emulation has enough headroom).
- Agent-specific env (e.g. `AIDER_API_KEY`, `OPENAI_BASE_URL`) is set
  automatically via `--ae` flags; see `config.AGENT_AE`.

### `launch 3x3`

```text
swebench-dind launch 3x3 [--wait] [--timeout-min 120]
```

- Hard-coded to `DEFAULT_CASES` × `DEFAULT_AGENTS` = 3 × 3 = 9 trials.
- All jobs are launched via `subprocess.Popen`, so they run in parallel
  inside the orchestrator.
- With `--wait`, each one is polled in order until it finishes.

### `launch matrix`

```text
swebench-dind launch matrix [--case ...] [--agent ...] [--wait] [--timeout-min 120]
```

- Same as `launch 3x3` but the case/agent lists are configurable.

---

## `watch`

```text
swebench-dind watch <job_name>
```

Polls `jobs/<job_name>/result.json` inside the orchestrator every 30 s
until it reports `finished_at`, or times out at 24 h. Useful for
re-attaching to a trial that you launched in another shell.

---

## `summarize`

```text
swebench-dind summarize [--jobs-dir JOBS_DIR] [--output-dir OUTPUT_DIR] [--include SUBSTR,...]
```

- Scans `JOBS_DIR` (default `/home/zengziyu/mini_matrix/jobs`) for
  `*/result.json`.
- Reads `stats.evals[*].metrics[0].mean` for pass@1.
- Outputs `summary-<ts>.{md,csv,json}` into `OUTPUT_DIR`
  (default `/home/zengziyu/mini_matrix/logs`).
- `--include` accepts a comma-separated substring filter on job names,
  e.g. `--include aider,qwen-code` keeps only those agents.

Output schema mirrors the legacy `scripts/summarize.py` — the md table
is keyed by `(case, agent)` with a count + mean column.

---

## `patch`

### `patch harbor`

```text
swebench-dind patch harbor [--agent qwen-code] [--agent aider]
```

- Injects an idempotent install probe into Harbor's installed agent
  module inside the orchestrator:

  ```python
  probe = await environment.exec(
      command="command -v <cli> >/dev/null 2>&1 && <cli> --version || echo not-found"
  )
  if probe.return_code == 0:
      return  # already baked; skip install
  ```

- This means **re-runs of a trial on an L2-baked image skip the slow
  `pip install` / `npm install`** — the heavy lifting happens once at
  bake time.
- Idempotent: the patch checks for the
  `"Idempotent probe (PATCHED by swebench-dind CLI)"` marker and is a
  no-op if already present.
- Cleans up the matching `.pyc` cache after writing.

| `--agent` key | Harbor module | CLI binary probed |
|---|---|---|
| `aider` | `aider.py` | `aider` |
| `mini-swe-agent` | `mini_swe_agent.py` | `mini-swe-agent` |
| `qwen-code` | `qwen_code.py` | `qwen` |
| `openhands-sdk` | `openhands_sdk.py` | `openhands` |

---

## `aisbench` (P1)

### `aisbench install`

```text
swebench-dind aisbench install
```

- Symlinks `swebench_dind/aisbench_adapter/` →
  `~/aisbench/runtime/swebench_dind/` (falls back to a recursive copy
  if symlink fails).
- Writes an example config at `~/aisbench/configs/swebench_dind_3x3.py`.
- Idempotent: existing symlinks are reported and left in place.

### `aisbench run`

```text
swebench-dind aisbench run --config configs/swebench_dind_3x3.py
```

Thin wrapper that calls `ais_bench <config>` after the adapter has
been installed.

### `aisbench result-format`

```text
swebench-dind aisbench result-format
```

Prints the AISBench-required schema docstring (see
`swebench_dind/aisbench_adapter/result_writer.py::SCHEMA_DOC`) for
quick reference.

---

## Environment & paths

These come from `swebench_dind/config.py`:

| Variable / path | Default | Used by |
|---|---|---|
| `CONTAINER_NAME` | `swebench-orchestrator` | `docker exec`, `docker rm` |
| `ORCHESTRATOR_IMAGE` | `swebench/orchestrator:v0.1-2026-08-04-patched-v11` | `orchestrator start` |
| `JOBS_DIR` | `/home/zengziyu/mini_matrix/jobs` | bind mount, `summarize` |
| `TASKS_DIR` | `/home/zengziyu/mini_matrix/tasks` | bind mount |
| `LOGS_DIR` | `/home/zengziyu/mini_matrix/logs` | bind mount, `summarize` output |
| `API_KEY_ENV` | `/home/zengziyu/mini_matrix/scripts/api_key.env` | `OPENAI_API_KEY` source |
| `DEFAULT_MODEL` | `openai/Qwen/Qwen3-Coder-30B-A3B-Instruct` | launch trial default |
| `DEFAULT_API_BASE` | `https://api.siliconflow.cn/v1` | launch trial default |
| `DEFAULT_CASES` | `[11099, 12308, 13741]` | launch 3x3 / matrix default |
| `DEFAULT_AGENTS` | `[aider, mini-swe-agent, qwen-code]` | launch 3x3 / matrix default |

---

## Typical end-to-end session

```bash
cd /home/zengziyu/mini_matrix/cli
source .venv/bin/activate

# Bring up the orchestrator (idempotent)
swebench-dind orchestrator start

# One-time: bake images (idempotent; skips existing)
swebench-dind build all

# One-time: patch Harbor so trial setup skips re-installing baked agents
swebench-dind patch harbor --agent qwen-code --agent aider --agent mini-swe-agent

# Run the default 3×3 matrix and wait for completion
swebench-dind launch 3x3 --wait --timeout-min 60

# Aggregate results
swebench-dind summarize
```

To add a new case or agent: edit `swebench_dind/config.py` only. All
downstream commands pick it up.