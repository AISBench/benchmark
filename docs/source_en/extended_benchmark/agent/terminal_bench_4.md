# Terminal-Bench 4.0 Adaptation Guide

Terminal-Bench 4.0 (official release `v4.0.0`) contains **66 tasks (63 CPU + 3 GPU)** with **157 task images + 2 external images (159 in total)**. This document describes how to evaluate Terminal-Bench 4.0 with AISBench via Harbor, covering both the **socket (DooD)** and **dind (DinD)** run modes.

> - For general Agent evaluation usage (parameters, live monitoring, result layout), see 📚 [Harbor-based Agent Evaluation](../../base_tutorials/scenes_intro/agent_benchmark.md); this page only covers the TB4.0-specific adaptation steps.
> - The steps are distilled from the full TB4.0 adaptation record `TB4_RUNTIME_SETUP_FULL_FLOW.md`; the section mapping is listed in the appendix at the end of this page.

## 1. Resource Checklist

| Resource | Link | Description |
| --- | --- | --- |
| Dataset source code | https://github.com/harbor-framework/terminal-bench/archive/refs/tags/v4.0.0.zip | v4.0.0 source (tasks/tools), for verifying task structure; not used at run time |
| Prebuilt task package | https://github.com/harbor-framework/terminal-bench/releases/download/v4.0.0/terminal-bench-prebuilt-v4.0.0.tar.gz | 66 tasks' `task.toml`/`instruction.md`/`tests/`; **extracting it yields the Harbor-format dataset** |
| Official image manifest | https://github.com/harbor-framework/terminal-bench/releases/download/v4.0.0/prebuilt-images.json | digest manifest of the 157 images, for material validation |
| Dataset images | https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-4-images/terminal-bench-4.0-images-x86_64.tar | 159 images packaged (~177G, x86_64) |
| Dependency docs | https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-4-images/dep.zip | full-flow scripts and configs (release validation, agent-deps patching, payments verifier, run scripts, etc.) |

Download the agent offline dependency packages for the corresponding OS from [Agent dependency preparation](../../base_tutorials/scenes_intro/agent_benchmark.md) and put them into an `agent-deps/` directory; the TB4.0 task images cover `ubuntu:20.04/22.04/24.04`, `debian:12/13`, and `fedora:42`. Validated agents: `claude-code` (socket mode) and `mini-swe-agent` (dind mode).

**Environment requirements**: x86_64 (the image package does not support aarch64); disk >= 400G; docker compose > 2.0.0 and buildx > 0.18.0 (dind mode additionally requires docker > 20.0.0); offline environments need PyPI/apt mirrors (related patch scripts are included in dep.zip).

## 2. The Two Run Modes

| Aspect | socket mode (default, recommended) | dind mode (strong isolation) |
| --- | --- | --- |
| Principle | Mounts the host's `/var/run/docker.sock` (DooD); docker commands inside the container operate the host daemon directly | `--privileged` starts an independent dockerd inside the container (DinD); trial containers only exist in the inner daemon |
| Task images | `docker load` on the **host** in advance; immediately visible inside the container | Passed via `--case-images` (required); the launch script loads them into the **inner** daemon automatically |
| Data mounts | Dataset / agent-deps / output directories are mounted through at the **same host paths** (so the trial containers' bind mounts resolve on the host daemon) | The original host paths are mounted, with aliases at `/opt/data/terminal-bench-2`, `/opt/agent-resources/deps`, and `/benchmark/outputs`. The dataset and agent-deps are read-only. |
| Docker version | No special requirement | > 20.0.0 (the launch script uses `--cgroupns=host`; not directly usable on very old host Docker such as 18.09) |
| Use case | The host daemon can be used directly; images and troubleshooting are the easiest | Strong isolation needed, or the host daemon should not be operated by the evaluation |
| Security | Holding docker.sock inside the container equals host root-level daemon control; do not run untrusted commands in it | Trial containers never touch the host daemon; better isolation |

## 3. Common Preparation Steps (for both modes)

### Step 1: Validate materials and extract the task package

Run these commands on the host from the directory containing the downloaded files. Keep these variables for the later host commands.

```bash
TB4_WORKDIR="$(pwd -P)"
export TB4_WORKDIR
export TB4_DEP_DIR="$TB4_WORKDIR/dep"
export TB4_TASKS="$TB4_WORKDIR/prebuilt/tasks"
export TB4_AGENT_DEPS="$TB4_WORKDIR/agent-deps"
export TB4_TRIALS="$TB4_WORKDIR/trials"

# 1) After downloading the 5 materials from section 1, compare SHA256 first (against the GitHub release page)
sha256sum terminal-bench-prebuilt-v4.0.0.tar.gz prebuilt-images.json

# 2) Extract the dependency docs (validation/patch/run scripts)
unzip dep.zip -d "$TB4_DEP_DIR"

# 3) Validate the release: release_tag=v4.0.0, 66 tasks, 157 images, all digests pinned
python3 "$TB4_DEP_DIR/01-release-validation/validate_release.py"

# 4) Extract the task package (prebuilt/tasks/ is the dataset, containing 66 task directories)
mkdir -p prebuilt
tar -xzf terminal-bench-prebuilt-v4.0.0.tar.gz -C prebuilt/

# 5) Cross-check consistency among task package / manifest / compose
python3 "$TB4_DEP_DIR/01-release-validation/check_package_consistency.py"
```

> The validation scripts read `prebuilt-images.json` and `prebuilt/tasks` from the current directory by default. The package consistency check requires PyYAML. See `dep/README.md` for environment-specific settings.

### Step 2: Strip digests from task.toml (mandatory when using the image package)

The image package is a `docker save` product, and `docker load` **does not preserve RepoDigests**; meanwhile `task.toml` references images as `docker_image = "repo:tag@sha256:..."`, so after loading, Compose cannot resolve them and tries to pull, which fails. Harbor actually reads each task's `task.toml` (patching only `prebuilt-images.json` has no effect).

```bash
# Before: should be 132 references with @sha256
grep -R 'docker_image.*@sha256' prebuilt/tasks/*/task.toml | wc -l

# Strip digests in batch (keep repo:tag)
sed -i -E 's/(docker_image[[:space:]]*=[[:space:]]*"[^"]+)@sha256:[0-9a-f]+"/\1"/' prebuilt/tasks/*/task.toml

# After: should be 0
grep -R 'docker_image.*@sha256' prebuilt/tasks/*/task.toml | wc -l
```

> If you apply the task patches in step 7, that script also removes the digests, so you can skip this step.

### Step 3: Prepare agent offline dependencies

1. Download the offline packages of the chosen agent for the corresponding OS from the table in [Agent dependency preparation](../../base_tutorials/scenes_intro/agent_benchmark.md) and put them into the `agent-deps/` directory.
2. If a bundle installs the agent under `/root/.local`, task containers running as non-root (`nobody`) cannot find the executable: patch the bundle with the `post.sh` approach from `dep/05-agent-deps/` (move it to `/opt` and generate a wrapper in `/usr/local/bin`).

## 4. Mode 1: socket (DooD) Steps

### Step 4a: Load task images on the host

```bash
docker load -i terminal-bench-4.0-images-x86_64.tar   # 159 images, ~177G, takes a while

# Two external images in the package have wrong repo prefixes; re-tag them:
docker tag harborframework/terminal-bench:4.3.1 apache/kafka-native:4.3.1
docker tag harborframework/terminal-bench:7-alpine redis:7-alpine
```

### Step 5a: Start the runtime container

```bash
wget -O start_agent_runtime.sh https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/scripts/start_agent_runtime.sh

bash ./start_agent_runtime.sh \
    --image ghcr.io/aisbench/agent-runtime:v3.1-20261013-master-ubuntu24.04-py312-x86_64 \
    --dataset "$TB4_TASKS" \
    --agent-deps "$TB4_AGENT_DEPS" \
    --agent mini-swe-agent \
    --name tb4-agent-runtime \
    --trials-dir "$TB4_TRIALS"
```

- In socket mode the task images are already on the host, so `--case-images` is not needed.
- Use the latest runtime image tag from the [Agent runtime images list](https://github.com/AISBench/benchmark/wiki/Agent-runtime-images).


## 5. Mode 2: dind (DinD) Steps

### Step 4b: Start the runtime container (images loaded automatically)

```bash
bash "$TB4_DEP_DIR/03-runtime-container/start_tb4_agent_runtime.sh" --mode dind \
    --name tb4-agent-runtime \
    --image ghcr.io/aisbench/agent-runtime:v3.1-20261013-master-ubuntu24.04-py312-x86_64 \
    --dataset "$TB4_TASKS" \
    --case-images "$TB4_WORKDIR/terminal-bench-4.0-images-x86_64.tar" \
    --agent-deps "$TB4_AGENT_DEPS" \
    --agent mini-swe-agent \
    --trials-dir "$TB4_TRIALS"
```

Use the TB4 launcher from `dep.zip` for DinD. It starts a separate dockerd and loads the image archive (~177G) into it. It also mounts the dataset, agent dependencies, and trials at their original host paths. The task patch script uses the dataset mount to access the verifier build context inside the runtime.

### Step 5b: Verify the inner daemon and images

```bash
docker exec tb4-agent-runtime docker info \
  --format 'version={{.ServerVersion}} driver={{.Driver}}'

# Re-tag the 2 external images (inner daemon)
docker exec tb4-agent-runtime docker tag harborframework/terminal-bench:4.3.1 apache/kafka-native:4.3.1
docker exec tb4-agent-runtime docker tag harborframework/terminal-bench:7-alpine redis:7-alpine

# Verify: 157 task images + 2 external images
docker exec tb4-agent-runtime docker images --format '{{.Repository}}:{{.Tag}}' \
  | grep -c '^harborframework/terminal-bench:'               # expect 157
docker exec tb4-agent-runtime docker images --format '{{.Repository}}:{{.Tag}}' \
  | grep -cE '^(apache/kafka-native:4.3.1|redis:7-alpine)$'   # expect 2
```


## 6. Evaluation Run

### Step 6: Install the payments verifier

The verifier for `payments-pipeline-fix` runs pip with `env -i`, which clears the pip mirror environment variables. `PipConfVerifier` writes the mirror to `/etc/pip.conf` before verification.

Run on the host after starting the runtime. Re-run this installer each time you recreate the container.

```bash
bash "$TB4_DEP_DIR/07-payments-verifier/install_payments_verifier.sh"
```

### Step 7: Apply runtime and task patches

Run these commands on the host. Install the runtime Python patches before running a model. The installer normalizes `mmengine.ConfigDict` before passing the nested mini-swe-agent configuration to `yaml.safe_dump` and adds `Logger.exception()` to AISBench's logger. Re-run it after recreating the container.

```bash
bash "$TB4_DEP_DIR/runtime-patches/install_runtime_patches.sh"
```

The task patch removes pinned digests from `task.toml` and applies three fixes:

- `sglang-qwen-burst` uses `GIT_CONFIG_*` environment variables for the verifier's Git configuration.
- `photonic-waveguide-routing` increases the verifier timeout from 300 to 600 seconds.
- `retro-console-soc` increases the verifier's nextpnr timeout from 300 to 600 seconds.

The script builds local verifier images for `sglang-qwen-burst` and `retro-console-soc` and points their task configs at those images.

```bash
# MODE=dind or MODE=dood
MODE=dind
BASE="$TB4_WORKDIR" bash "$TB4_DEP_DIR/runtime-patches/task-fixes/apply_task_fixes.sh" \
    all --mode "$MODE" --tasks "$TB4_TASKS"
BASE="$TB4_WORKDIR" bash "$TB4_DEP_DIR/runtime-patches/task-fixes/apply_task_fixes.sh" \
    check --mode "$MODE" --tasks "$TB4_TASKS"
```

For DooD, set `MODE=dood` to build on the host daemon. For DinD, use `MODE=dind` to build on the inner daemon. Run `check` after applying the patches. Use `--force-build` to rebuild existing images or `all --dry-run` to preview the changes. Backups are written under `dep/runtime-patches/task-fixes/backups/<timestamp>/`.

### Step 8: Check the account balance and model parameters

Before starting or resuming a manual DeepSeek run, run the following checks on the host. The DinD launcher in step 11 runs both checks automatically.

```bash
read -rsp "DeepSeek API key: " AIS_AGENT_API_KEY; echo
export AIS_AGENT_API_KEY
export API_BASE=https://api.deepseek.com/v1
export MODEL=openai/deepseek-chat
python3 -I "$TB4_DEP_DIR/run/probe_deepseek_balance.py" &&
    python3 -I "$TB4_DEP_DIR/run/probe_model_params.py" "$API_BASE" "$MODEL"
```

The balance probe calls the [DeepSeek balance API](https://api-docs.deepseek.com/api/get-user-balance/). Check that `is_available=true` and that the reported `total_balance` by currency covers your planned usage. Use `GET /models` for connectivity, authentication, and model discovery. The parameter probe sends one inference request with the example mini-swe-agent settings. For another provider, use its billing tools and verify inference with the parameters configured for the run.

### Step 9: Enter the runtime container

```bash
docker exec -it -e TB4_TASKS -e TB4_AGENT_DEPS -e TB4_TRIALS tb4-agent-runtime bash
agent_env harbor
cd "$TB4_TRIALS"
```

Both modes can use the host paths passed into the container above. Section 2 lists the additional DinD mount paths. Run the canary, manual full run, and manual resume in this shell. Exit the shell before using the host launcher.

### Step 10: Run the oracle canary

```bash
# Run inside the runtime. Oracle does not call the model.
ais_bench /benchmark/ais_bench/configs/agent_example/harbor_agent_task.py --mode agent \
    -a oracle --model openai/any --api-base http://127.0.0.1:1/v1 --agent-api-key unused \
    -p "$TB4_TASKS" --agent-deps "$TB4_AGENT_DEPS" \
    --include-task-name batched-eval-parity -n 1 -k 1 -y
# Expect reward 1.0. This exercises Compose and sidecars and takes about 10 minutes.
# Repeat with --include-task-name payments-pipeline-fix to verify step 6. Expect reward 1.0.
```

### Step 11: Full run (63 CPU tasks)

#### Configure the model inside the runtime

Edit the matching `models` entry in `/benchmark/ais_bench/configs/agent_example/harbor_agent_task.py`. Match its `agent_name`, `model_names`, and `api_base` to the agent, model, and endpoint used on the command line. The baseline mini-swe-agent settings are:

```python
llm_kwargs = {
    "config": {
        "model": {
            "model_class": "litellm",
            "model_kwargs": {
                "temperature": 1.0,
                "top_p": 1.0,
                "reasoning_effort": "max",
                "allowed_openai_params": ["reasoning_effort"],
                "max_tokens": 256000,
            },
        },
    },
}
```

Set `model_info.max_output_tokens` to the same value as `max_tokens`. Set `model_info.max_input_tokens` using the provider's published input limit. Check that the endpoint accepts the example settings before using them. Oracle does not require this configuration.

#### Option A: Run manually inside the container (DooD or DinD)

```bash
read -rsp "Model API key: " AIS_AGENT_API_KEY; echo
export API_BASE=https://api.deepseek.com/v1
export MODEL=openai/deepseek-chat
ais_bench /benchmark/ais_bench/configs/agent_example/harbor_agent_task.py --mode agent \
    -a mini-swe-agent \
    --model "$MODEL" \
    --api-base "$API_BASE" \
    --agent-api-key "$AIS_AGENT_API_KEY" \
    -p "$TB4_TASKS" \
    --agent-deps "$TB4_AGENT_DEPS" \
    --exclude-task-name fp8-rmsnorm-gemm jax-speedrun-gpu math-eval-grader \
    -n 6 -k 1 \
    --no-force-build --delete -y
```

#### Option B: Launch from the host (DinD)

After configuring the model inside the runtime, return to the host shell and run the launcher below. It starts the evaluation via `docker exec`. Use either option A or option B for a full run.

```bash
BASE="$TB4_WORKDIR" bash "$TB4_DEP_DIR/run/run_tb4_full_eval.sh"
```

The default concurrency is `N=6`. The launcher checks the dataset, patches, images, and disk space, then runs the balance and parameter probes before starting evaluation. Set `CHECK_ONLY=1` to check the environment only. The older wrapper also works with `BASE="$TB4_WORKDIR" N=6 bash "$TB4_DEP_DIR/run/run_mini_swe_full_n8.sh"`.

- The full CPU run excludes `fp8-rmsnorm-gemm`, `jax-speedrun-gpu`, and `math-eval-grader`. **Pass@1 = passed count / 63**.
- For `claude-code` with the Anthropic-format endpoint, the validated options are `--ak reasoning_effort=max --ak max_turns=500`.
- For a first run, use `--include-task-name <single task>` or `--n-tasks 1` with the manual command.

### Step 12: Resume and rerun

Resolve the failure before resuming. After HTTP 402, top up the account and repeat the balance check in step 8. To retry only the trials with exceptions, run inside the runtime with the same parameters as the manual command in step 11 and append `--reuse <timestamp> --purge-exception-cases`.

For verifier timeouts, `harbor trial regrade` reruns the verifier. Use `--verifier-timeout-multiplier` if needed.

### Step 13: Read the results

- Summary: `outputs/default/<timestamp>/summary/summary_<timestamp>.csv` (`correct` / `wrong` / `exception` / `avg_score`).
- Details: `stats.evals` in `results/<model>/<dataset>/details/result.json`, plus each trial's `trial_*/verifier/reward.json` and `ctrf.json`.
- See [Execution Results and On-Disk Files](../../base_tutorials/scenes_intro/agent_benchmark.md#execution-results-and-on-disk-files).

## Appendix: Mapping to the full-flow record

| Steps in this document | `TB4_RUNTIME_SETUP_FULL_FLOW.md` |
| --- | --- |
| Material validation and task extraction | §1, §3.1 |
| Image loading and digest stripping | §3.2, §10.2 |
| socket mode | §3.4, §6 |
| dind mode | §0.1, §10 |
| payments verifier | §9.1 |
| canary / balance probe / full run / resume | §9.2~9.5, §9.7 |
