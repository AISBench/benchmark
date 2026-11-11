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
| Data mounts | Dataset / agent-deps / output directories are mounted through at the **same host paths** (so the trial containers' bind mounts resolve on the host daemon) | Dataset mounted read-only at `/opt/data/`, agent-deps at `/opt/agent-resources/deps`, outputs written to `--trials-dir` |
| Docker version | No special requirement | > 20.0.0 (the launch script uses `--cgroupns=host`; not directly usable on very old host Docker such as 18.09) |
| Use case | The host daemon can be used directly; images and troubleshooting are the easiest | Strong isolation needed, or the host daemon should not be operated by the evaluation |
| Security | Holding docker.sock inside the container equals host root-level daemon control; do not run untrusted commands in it | Trial containers never touch the host daemon; better isolation |

## 3. Common Preparation Steps (for both modes)

### Step 1: Validate materials and extract the task package

```bash
# 1) After downloading the 5 materials from section 1, compare SHA256 first (against the GitHub release page)
sha256sum terminal-bench-prebuilt-v4.0.0.tar.gz prebuilt-images.json

# 2) Extract the dependency docs (validation/patch/run scripts)
unzip dep.zip -d dep

# 3) Validate the release: release_tag=v4.0.0, 66 tasks, 157 images, all digests pinned
python3 dep/01-release-validation/validate_release.py

# 4) Extract the task package (prebuilt/tasks/ is the dataset, containing 66 task directories)
mkdir -p prebuilt
tar -xzf terminal-bench-prebuilt-v4.0.0.tar.gz -C prebuilt/

# 5) Cross-check consistency among task package / manifest / compose
python3 dep/01-release-validation/check_package_consistency.py
```

> Scripts inside dep.zip are written against fixed working directories; on a new machine, first replace the three adaptation points from `dep/README.md` (paths / proxy address / image name) in batch.

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
wget https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/scripts/start_agent_runtime.sh

./start_agent_runtime.sh \
    --image ghcr.io/aisbench/agent-runtime:v3.1-20261013-master-ubuntu24.04-py312-x86_64 \
    --dataset /abs/path/prebuilt/tasks \
    --agent-deps /abs/path/agent-deps \
    --agent mini-swe-agent \
    --name agent-runtime \
    --trials-dir /abs/path/trials
```

- In socket mode the task images are already on the host, so `--case-images` is not needed.
- Use the latest runtime image tag from the [Agent runtime images list](https://github.com/AISBench/benchmark/wiki/Agent-runtime-images).

### Step 6a: Enter the container

```bash
docker exec -it agent-runtime /bin/bash
agent_env harbor        # activate the harbor venv
cd /abs/path/trials     # output directory (same-path pass-through, directly visible on the host)
```

## 5. Mode 2: dind (DinD) Steps

### Step 4b: Start the runtime container (images loaded automatically)

```bash
./start_agent_runtime.sh --mode dind \
    --image ghcr.io/aisbench/agent-runtime:v3.1-20261013-master-ubuntu24.04-py312-x86_64 \
    --dataset /abs/path/prebuilt/tasks \
    --case-images /abs/path/terminal-bench-4.0-images-x86_64.tar \
    --agent-deps /abs/path/agent-deps \
    --agent mini-swe-agent \
    --trials-dir /abs/path/trials
```

The script starts an independent dockerd inside the container and automatically `docker load`s the image package into the inner daemon (~177G, takes several minutes).

### Step 5b: Verify the inner daemon and images

```bash
docker exec agent-runtime docker info \
  --format 'version={{.ServerVersion}} driver={{.Driver}}'

# Re-tag the 2 external images (inner daemon)
docker exec agent-runtime docker tag harborframework/terminal-bench:4.3.1 apache/kafka-native:4.3.1
docker exec agent-runtime docker tag harborframework/terminal-bench:7-alpine redis:7-alpine

# Verify: 157 task images + 2 external images
docker exec agent-runtime docker images --format '{{.Repository}}:{{.Tag}}' \
  | grep -c '^harborframework/terminal-bench:'               # expect 157
docker exec agent-runtime docker images --format '{{.Repository}}:{{.Tag}}' \
  | grep -cE '^(apache/kafka-native:4.3.1|redis:7-alpine)$'   # expect 2
```

### Step 6b: Enter the container

```bash
docker exec -it agent-runtime /bin/bash
agent_env harbor
# Inside the container: dataset at /opt/data/, agent-deps at /opt/agent-resources/deps,
# outputs at /benchmark/outputs (the RUN_HINT printed at login contains the full command template)
```

## 6. Evaluation Run (identical for both modes)

### Step 7: Install the payments verifier (score-critical)

The verifier of `payments-pipeline-fix` runs pip with `env -i`, which clears the pip mirror environment variables and always fails; `PipConfVerifier` writes the mirror into `/etc/pip.conf` before verification (the file survives `env -i`).

```bash
# Run on the host (idempotent; re-run after each rebuild of the runtime container;
# adjust the working directory / container name inside the script to your environment)
bash dep/07-payments-verifier/install_payments_verifier.sh
```

### Step 8: oracle canary smoke test (no model cost)

```bash
# Run inside the container; oracle makes no model calls, the key just needs to be non-empty
ais_bench /benchmark/ais_bench/configs/agent_example/harbor_agent_task.py --mode agent \
    -a oracle --model openai/any --api-base http://127.0.0.1:1/v1 --agent-api-key unused \
    -p <dataset path> --agent-deps <agent-deps path> \
    --include-task-name ctr-optimization -n 1 -k 1
# Expect reward 1.0 (covers the compose/sidecar chain);
# then run once more with --include-task-name payments-pipeline-fix to verify step 7; expect reward 1.0
```

### Step 9: Balance probe (mandatory before every run)

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://api.deepseek.com/v1/models \
  -H "Authorization: Bearer sk-..."
# 200 = sufficient balance; 402 = top up first (an exhausted balance records trials as UnknownApiError and truncates the whole batch)
```

### Step 10: Full run (63 CPU tasks)

```bash
ais_bench /benchmark/ais_bench/configs/agent_example/harbor_agent_task.py --mode agent \
    -a mini-swe-agent \
    --model openai/deepseek-chat \
    --api-base https://api.deepseek.com/v1 \
    --agent-api-key sk-... \
    -p <dataset path> \
    --agent-deps <agent-deps path> \
    --exclude-task-name fp8-rmsnorm-gemm jax-speedrun-gpu math-eval-grader \
    -n 8 -k 1 \
    --no-force-build --delete
```

- Among the 66 tasks, 3 GPU tasks (`fp8-rmsnorm-gemm`, `jax-speedrun-gpu`, `math-eval-grader`) are excluded by default; **the Pass@1 score = passed count / 63**.
- When switching to `claude-code`, validated parameters are: `--ak reasoning_effort=max --ak max_turns=500` (the model service uses the Anthropic-format API).
- For the first run, start with `--include-task-name <single task>` or `--n-tasks 1` to verify the pipeline.

### Step 11: Resume and rerun

```bash
# For 402/UnknownApiError etc.: after topping up -> probe returns 200, rerun only the failed trials
# (passed trials are not billed again)
ais_bench /benchmark/ais_bench/configs/agent_example/harbor_agent_task.py --mode agent \
    ... (same parameters as step 10) ... --reuse <timestamp> --purge-exception-cases
```

- For verifier-timeout failures, use `harbor trial regrade` to rerun only the verifier (optionally with `--verifier-timeout-multiplier`); do not rerun the agent.

### Step 12: Read the results

- Summary: `outputs/default/<timestamp>/summary/summary_<timestamp>.csv` (`correct` / `wrong` / `exception` / `avg_score`).
- Details: `stats.evals` in `results/<model>/<dataset>/details/result.json`; per-case `trial_*/verifier/reward.json` and `ctrf.json`.
- For the on-disk layout see 📚 [Execution Results and On-Disk Files](../../base_tutorials/scenes_intro/agent_benchmark.md#execution-results-and-on-disk-files).

## Appendix: Mapping to the full-flow record

| Steps in this document | `TB4_RUNTIME_SETUP_FULL_FLOW.md` |
| --- | --- |
| Material validation and task extraction | §1, §3.1 |
| Image loading and digest stripping | §3.2, §10.2 |
| socket mode | §3.4, §6 |
| dind mode | §0.1, §10 |
| payments verifier | §9.1 |
| canary / balance probe / full run / resume | §9.2~9.5, §9.7 |
