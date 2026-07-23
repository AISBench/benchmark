# AISBench Agent Runtime

Provides images and scripts for the runtime container used by AISBench Agent evaluations (Harbor Terminal-Bench, SWE-bench, SWE-bench Pro, etc.).

> This repository directory is a community runtime supplement to AISBench/benchmark and is not part of the core benchmark evaluation logic. After preparing the runtime container with the scripts and images in this directory, users still run evaluations with the native `ais_bench` command; the principles are identical to those in each benchmark's documentation.

## What Problem Does It Solve

Environment preparation for agent evaluation has three major pain points:

1. **Dependency conflicts**: harbor forces datasets to upgrade to 4.0+; SWE-bench and SWE-bench Pro each need a different fork of `mini-swe-agent`, with the same package name overwriting each other.
2. **Error-prone container configuration**: DinD modes A/B, `--cgroupns=host`, `daemon.json`, seccomp — any missed step only surfaces as an error at evaluation time.
3. **Frequent dataset / case image version changes**: datasets and case images have many versions; baking them into the runtime image causes rapid expiration.

This package solves these problems in layers:

| Layer | Content | Solves |
|---|---|---|
| `Dockerfile.agent-runtime` | Adds 3 isolated venv layers on top of the `aisbench_benchmark` base image (harbor / swebench / swebench_pro) | Dependency conflicts |
| `bootstrap.sh` | One-click runtime container start: auto-selects DinD/Socket mode + mounts datasets + loads case tar | Error-prone container config + dataset injection |
| `doctor.sh` | Static self-check (L1, seconds) — validates docker / venv / pack / resources | Pre-run runtime validation |
| `packs/<name>.yaml` | Metadata for each benchmark (venv name / native config / docs) | Decouples toolchain from benchmarks |

## Who Prepares Datasets / Case Images

**What is deliberately not done**: This solution does **not** pre-bake agent benchmark datasets and case sandbox images into the runtime image. The reason is that both change versions frequently, and baking them in means:

- Every dataset update requires rebuilding the runtime image — a maintenance burden and a large download for users
- A full case sandbox image set is ~71GB and cannot be baked into a base image

**Who is responsible for what**:

| Item | Who prepares | How to integrate into the runtime container |
|---|---|---|
| runtime image | AISBench maintainers | `docker pull ghcr.io/aisbench/agent-runtime:latest-...` (or `--runtime-tar` offline) |
| Dataset (task.toml, etc.) | User prepares on the host | `bootstrap.sh --datasets <full dataset path>` (mounted at the same path inside the container + injected as env var) |
| case sandbox image | User prepares tar on the host | `bootstrap.sh --case-tar <tar>` (auto `docker cp` into container + `docker load`) |
| Model call parameters (api_base / model_names) | User edits native config | Inside the container: `vim ais_bench/configs/agent_example/...` |

## Quick Start (Harbor Terminal-Bench as example, aarch64)
The quick start targets host machines with Docker version below 20.0.0. For other environments, refer to the corresponding agent evaluation documentation.

```bash
# 1. Prepare dataset and image tar on the host (skip if already present)

git clone https://modelers.cn/AISBench/terminal-bench-2-offline-mini.git # Dataset preparation
wget https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/agent_runtime_image_v3.1-20260701-master-ubuntu24.04-py312-aarch64.tar.gz # Runtime image preparation (optional; if omitted, the latest is fetched automatically)
wget https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-offline-prepared-images-selected-0.10_aarch64.tar # Case image preparation; get the link from the corresponding agent evaluation doc as needed
mkdir /path/to/test_wkp/ # Create an empty working directory on the host

# 2. Start the runtime container with one click on the host (auto-selects DinD/Socket mode, auto-mounts datasets; if the environment has no external network, fetch ais_bench_agent_bootstrap.sh elsewhere first and run with bash)
#    Auto-copies case image tar into the container and runs docker load
curl -fsSL https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/ais_bench_agent_bootstrap.sh \
    | bash -s -- \
        --datasets /path/to/terminal-bench-2-offline-mini/terminal-bench-2-offline-selected_0.10/ \
        --runtime-tar /path/to/agent_runtime_image_v3.1-20260701-master-ubuntu24.04-py312-aarch64.tar.gz \
        --case-tar /path/to/terminal-bench-2-offline-prepared-images-selected-0.10.tar \
        --host-path /path/to/test_wkp/ \
        --container-name test_agent_run
# --datasets must point to a directory structure consistent with the terminal-bench-2-offline-selected_0.10/ subdirectory of the terminal-bench-2-offline-mini repo
# --runtime-tar (optional) pre-downloaded runtime image
# --case-tar must point to a tar structure consistent with the case image tar described in the corresponding agent evaluation doc
# --host-path must be an empty directory; a same-named directory will be created inside the container to mount datasets and case images
# --container-name must be unique; otherwise the old container will be overwritten

# 3. Enter the container (case images are already loaded and ready to use)
docker exec -it test_agent_run bash

# 4. (No vim needed) The native config path is read automatically from AISBENCH_AGENT_DATASET_PATH
#    Only vim model_names / api_base
vim ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py

# 5. Verify runtime is ready
ais_bench_agent_doctor.sh harbor

# 6. Run evaluation
agent_env harbor
ais_bench ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py --debug
```

Switch datasets: destroy the old container + restart bootstrap (update dataset path / case tar together):

```bash
docker rm -f test_agent_run
bash ais_bench_agent_bootstrap.sh \
    --datasets /data/datasets/harbor/full/terminal-bench-2 \
    --case-tar /data/cases/terminal-bench-2-prepared-images_x86_64.tar.gz
```

## Directory Structure

```
agent_runtime/
├── README.md                       # This file
├── Dockerfile.agent-runtime        # runtime image build file (BASE_IMAGE passed via --build-arg, not hardcoded)
├── build_image_agent_runtime.sh    # build script (supports --base-tag/--push/--upload/--multi-arch)
├── ais_bench_agent_bootstrap.sh                    # one-click runtime container start (user-side entry point, must be uploaded to OBS)
├── doctor.sh                       # runtime readiness verification (inside the container; only validates docker/venv/config, not datasets/cases)
├── packs/                          # per-benchmark manifests (name/runtime_venv/native_config/native_doc)
│   ├── harbor.yaml                 # Harbor Terminal-Bench
│   ├── swebench.yaml               # SWE-bench (mini_swe_agent + SWE-bench harness)
│   └── swebench_pro.yaml           # SWE-bench Pro (scaleapi adapted version)
└── patches/                        # patch scripts used at build / startup time
    └── harbor_compose_patch.py     # adds seccomp=unconfined + network_mode=host to harbor's docker-compose-base.yaml
```

## ais_bench_agent_bootstrap.sh Usage

```bash
# Minimal call: mount a single dataset directory
bash ais_bench_agent_bootstrap.sh --datasets /data/datasets

# Mount multiple directories
bash ais_bench_agent_bootstrap.sh --datasets /data/datasets --datasets /data/extra

# Force mode A/B
bash ais_bench_agent_bootstrap.sh --mode A --datasets /data/datasets

# Custom container name (to distinguish when running multiple runtimes on one machine)
bash ais_bench_agent_bootstrap.sh --container-name my_eval_1 --datasets /data/datasets

# Custom runtime image (recommended: pass an explicit tag for reproducibility)
bash ais_bench_agent_bootstrap.sh \
    --runtime-image ghcr.io/aisbench/agent-runtime:v3.1-20260522-master-ubuntu24.04-py312-x86_64 \
    --datasets /data/datasets

# Mode B + custom /benchmark extraction target (use only when /opt is not writable)
bash ais_bench_agent_bootstrap.sh --mode B --host-path /data/ais_bench_host --datasets /data/datasets

# Offline mode (intranet/isolated environment): load runtime image from a tar already downloaded on the host
#   Completely skips docker pull / OBS download
#   Use case: intranet deployment machines that cannot reach ghcr.io or external OBS
bash ais_bench_agent_bootstrap.sh --datasets /data/datasets --runtime-tar /opt/aisbench/agent-runtime-ubuntu24-py312-x86_64.tar.gz

# Fully offline: runtime tar + case image tar passed together
#   After container start, case tar is auto-copied into the container and docker loaded
bash ais_bench_agent_bootstrap.sh \
    --datasets /data/datasets \
    --runtime-tar /opt/aisbench/agent-runtime.tar.gz \
    --case-tar /opt/aisbench/case-tb2-mini-0.10.tar.gz

# Load multiple case images at once (--case-tar can be repeated, or a directory can be passed)
bash ais_bench_agent_bootstrap.sh \
    --datasets /data/datasets \
    --runtime-tar /opt/aisbench/agent-runtime.tar.gz \
    --case-tar /opt/aisbench/case-tb2-mini-0.10.tar.gz \
    --case-tar /opt/aisbench/case-tb2-mini-0.14.tar.gz \
    --case-tar /opt/aisbench/case-tars/         # all .tar/.tar.gz/.tgz under the directory will be loaded
```

`--datasets` / `--host-path` / `--runtime-tar` / `--case-tar` must be **absolute paths** and **must exist on the host** (the script validates them). The in-container path is the same as the host path.

The full path passed to `--datasets` is injected verbatim into the container as the environment variable `AISBENCH_AGENT_DATASET_PATH`, and native ais_bench configs (e.g. `harbor_terminal_bench_2_task.py`) use this env var directly as the dataset `path` field — **no concatenation, no conversion, completely identical**. Therefore:

- **Recommended**: pass `--datasets` as the full path of your prepared harbor benchmark dataset (e.g. `/data/datasets/harbor/mini-0.10/terminal-bench-2-offline-selected_0.10`); the config then needs no vim for path
- **Multiple directories**: `--datasets` can be passed multiple times, but the env var uses only the first; users can manually `export AISBENCH_AGENT_DATASET_PATH=...` to override

### Command-Line Parameter Reference

| Parameter | Default | Description |
|---|---|---|
| `--datasets <HOST_PATH>` | None (not mounted) | Dataset directory; can be passed multiple times |
| `--runtime-tar <HOST_PATH>` | None (pull) | runtime image tar; for offline scenarios |
| `--case-tar <HOST_PATH>` | None (prepare manually in container) | case image tar; file or directory; can be passed multiple times |
| `--mode A\|B` | Auto-detected | Force DinD (A) or Socket passthrough (B) |
| `--container-name <NAME>` | `ais_bench_agent` | runtime container name |
| `--runtime-image <TAG>` | `ghcr.io/aisbench/agent-runtime:latest-ubuntu24.04-py312-${ARCH}` | runtime image tag |
| `--host-path <ABS_PATH>` | `/opt/ais_bench_agent` | Extraction target for `/benchmark` in mode B |

### Environment Variables

Only one env variable is kept (use CLI parameters for other configs):

| env | Default | Description |
|---|---|---|
| `OBS_RUNTIME_TAR_BASE` | `https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/runtime` | OBS runtime tar download base URL (usually no need to change) |

### Offline Scenarios

Deployment machines in intranet/isolated environments cannot reach `ghcr.io/aisbench/agent-runtime` or OBS, but have obtained the runtime tar package and case image tar via USB stick, intranet proxy, etc.:

1. **Obtain tar package** (either):
   - Ask a maintainer to run `build_image_agent_runtime.sh --upload 1` to upload to OBS; intranet users download from OBS
   - On a machine with internet access, run `docker save ghcr.io/aisbench/agent-runtime:<tag> -o agent-runtime.tar.gz` and copy it in
2. **Run on the deployment machine**:
   ```bash
   # Minimal: pass only the runtime tar (case images still need manual pull / load inside the container)
   bash ais_bench_agent_bootstrap.sh --datasets /data/datasets --runtime-tar /path/to/agent-runtime.tar.gz

   # Fully offline: pass both runtime + case tar
   bash ais_bench_agent_bootstrap.sh \
       --datasets /data/datasets \
       --runtime-tar /path/to/agent-runtime.tar.gz \
       --case-tar /path/to/case-tb2-mini-0.10.tar.gz
   ```
3. **Behavior**:
   - `--runtime-tar`: completely skips `docker pull` and OBS `curl` download; runs `docker load -i <tar>` and auto-detects the tag (grep `agent-runtime`), preferring `RUNTIME_IMAGE`; gives a precise error on detection failure
   - `--case-tar <PATH>`: supports a single tar or a directory (a directory recursively loads all `.tar` / `.tar.gz` / `.tgz`). The script `docker cp`s it into the container, then runs `docker load -i`. **Supports both A/B modes** (mode A loads into the in-container DinD; mode B loads into the container, but since the socket is shared with the host, it is effectively loaded onto the host as well). For details on modes A/B, see [OVERVIEW.en.md](../OVERVIEW.en.md#running-agent--sandbox-benchmarks-docker-inside-the-container)
   - `--case-tar` can be repeated

Mode B (Socket passthrough) by default extracts `/benchmark` to the host's `/opt/ais_bench_agent`. If your environment's `/opt` is not writable (e.g. some read-only root containers/sandboxes), use `--host-path` to redirect to a writable path:

```bash
bash ais_bench_agent_bootstrap.sh --datasets /data/datasets --host-path /data/ais_bench_host
```

Normal host machines do not need to set this variable. For other configurable environment variables, see the header comments of `bootstrap.sh`.

## Image Build

The runtime image is built on top of the `aisbench_benchmark` base image; the base image tag is passed via a parameter, not hardcoded:

```bash
# Basic build (local, current architecture)
bash docker/agent_runtime/build_image_agent_runtime.sh \
    --base-tag v3.1-20260522-master

# Specify OS/Python (default ubuntu24.04 + py312)
bash docker/agent_runtime/build_image_agent_runtime.sh \
    --base-tag v3.1-20260522-master --os ubuntu24.04 --py-version py312

# Build and push to a remote registry
bash docker/agent_runtime/build_image_agent_runtime.sh \
    --base-tag v3.1-20260522-master --push 1

# Multi-arch build and push
bash docker/agent_runtime/build_image_agent_runtime.sh \
    --base-tag v3.1-20260522-master --multi-arch 1 --push 1

# Build, push, and upload offline package to OBS (for ais_bench_agent_bootstrap.sh fallback download)
bash docker/agent_runtime/build_image_agent_runtime.sh \
    --base-tag v3.1-20260522-master --push 1 --upload 1
```

The build script automatically validates (4 items):
1. ais_bench is available
2. All 3 venvs (harbor / swebench / swebench_pro) are complete + all 3 venvs contain the ais_bench wrapper + both swebench venvs can import minisweagent
3. doctor.sh / packs are in place
4. The harbor compose template has been patched with `seccomp=unconfined`

## Supported Packs

| pack name | runtime_venv | Documentation | Description |
|---|---|---|---|
| `harbor` | harbor | [harbor_bench.md](../../docs/source_en/extended_benchmark/agent/harbor_bench.md) | Harbor Terminal-Bench 2.0 |
| `swebench` | swebench | [swe_bench.md](../../docs/source_en/extended_benchmark/agent/swe_bench.md) | SWE-bench (lite/verified/full/multilingual, etc.) |
| `swebench_pro` | swebench_pro | [swe_bench_pro.md](../../docs/source_en/extended_benchmark/agent/swe_bench_pro.md) | SWE-bench Pro (x86 only) |

pack.yaml does not declare dataset paths or case image acquisition methods — these are entirely under user control:
- Dataset path: explicitly specified by the user via `bootstrap.sh --datasets <full dataset path>` (run whichever you want)
- Case image: the user performs `docker pull` or `docker load` according to the document pointed to by pack.yaml's `native_doc`

If you want to support more benchmarks in the future, just add a `packs/<name>.yaml`.

> Common harbor dataset directory names (for reference only, unrelated to the tool):
> - `/data/datasets/harbor/full/terminal-bench-2` (89 cases, including a few external-network tasks)
> - `/data/datasets/harbor/mini-0.10/terminal-bench-2-offline-selected_0.10` (7 cases)
> - `/data/datasets/harbor/mini-0.14/terminal-bench-2-offline-selected_0.14` (10 cases)
> - `/data/datasets/harbor/mini-0.20/terminal-bench-2-offline-selected_0.20` (14 cases)
>
> Whichever path the user passes to `bootstrap.sh --datasets` is injected into the container by the env var `AISBENCH_AGENT_DATASET_PATH`.
> The mini-* series is K-means sampled from `terminal-bench-2-offline` (70 cases after removing external-network tasks) and runs fully offline.
>
> For SWE-bench dataset notes, see [swe_bench.md](../../docs/source_en/extended_benchmark/agent/swe_bench.md) (HF download); for SWE-bench Pro, see [swe_bench_pro.md](../../docs/source_en/extended_benchmark/agent/swe_bench_pro.md).

## Detailed Solution

For the full design and parameter descriptions of each script, see the header comments of each script.
