# Docker Image Overview

## Quick Reference
- AISBench Benchmark is maintained by the [AISBench AI System Performance Benchmark Committee](https://www.aisbench.com/about).

- Image Overview

| Item | Description |
| --- | --- |
| Default image registry | `ghcr.io/aisbench/aisbench_benchmark` |
| Build script | `build_image.sh` |
| Supported OS | Ubuntu 22.04 / 24.04, openEuler 22.03 / 24.03 |
| Supported Python | 3.10, 3.11, 3.12 |
| Build strategy | Multi-stage build (builder → runtime) |
| Working directory | `/benchmark` |


- Where to get help
    + [📖AISBench Benchmark Documentation](https://ais-bench-benchmark.readthedocs.io/en/latest/)
    + [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AISBench/benchmark)
    + [🤔Report an Issue](https://github.com/AISBench/benchmark/issues/new/choose)

### AISBench Benchmark
AISBench Benchmark is a model evaluation tool built on [OpenCompass](https://github.com/open-compass/opencompass). It is compatible with OpenCompass's configuration system, dataset structure, and model backend implementation, and extends support for service-based models.
> ⚠️Note: AISBench Benchmark images focus on service-based model evaluation and do not support offline inference models. Built-in pipelines for sandbox-isolated benchmarks (SWE-Bench, terminal-bench 2, etc.) are NOT provided — but the image ships with Docker Engine (>= 20.0) and Docker Compose v2 (>= 2.0.0), so these benchmarks can be run manually inside the container. See [Running Agent / Sandbox Benchmarks](#running-agent--sandbox-benchmarks-docker-inside-the-container).

## Image Tag Convention & Dockerfile Archive Paths

Image tag format:

```
{hub_repo}:{TAG}-{OS}-{py_version}-{arch}
```

Example: `ghcr.io/aisbench/aisbench_benchmark:v3.1-20260522-master-ubuntu22.04-py310-x86_64`

Where:
- `v3.1-20260522-master` is the version number, formatted as `v{major}.{minor}-{date}-{branch}`
- `ubuntu22.04` is the OS version
- `py310` is the Python version
- `x86_64` is the architecture

### Dockerfile Inventory

| Dockerfile | Base Image | Python | Path |
| --- | --- | --- | --- |
| [Dockerfile.py310.ubuntu22.04](ubuntu/Dockerfile.py310.ubuntu22.04) | `ubuntu:22.04` | 3.10 | `docker/ubuntu/` |
| [Dockerfile.py312.ubuntu24.04](ubuntu/Dockerfile.py312.ubuntu24.04) | `ubuntu:24.04` | 3.12 | `docker/ubuntu/` |
| [Dockerfile.py310.openeuler22.03](openeuler/Dockerfile.py310.openeuler22.03) | `openeuler/openeuler:22.03-lts` | 3.10 | `docker/openeuler/` |
| [Dockerfile.py311.openeuler24.03](openeuler/Dockerfile.py311.openeuler24.03) | `openeuler/openeuler:24.03-lts` | 3.11 | `docker/openeuler/` |

Dockerfile naming convention: `Dockerfile.{py_version}.{os}`

## Quick Start

### Running Existing Images

#### Official Image Registry

All images are archived on GHCR: https://github.com/orgs/AISBench/packages/container/package/aisbench_benchmark

To obtain the Docker image with tag `v3.1-20260522-master-openeuler24.03-py311-aarch64`, there are two main approaches:

1. Pull via `docker pull`

```bash
docker pull ghcr.io/aisbench/aisbench_benchmark:v3.1-20260522-master-openeuler24.03-py311-aarch64
```

2. Import from an image archive

```bash
# Download the Docker image archive
wget https://aisbench.obs.cn-north-4.myhuaweicloud.com/images/benchmark/github/aisbench_benchmark_v3.1-20260522-master-openeuler24.03-py311-aarch64.tar.gz
# Import the image from the archive
docker load -i aisbench_benchmark_v3.1-20260522-master-openeuler24.03-py311-aarch64.tar.gz
```

#### Starting a Docker Container from the Image

Use the following command as a reference:

```bash
# docker run --name ${your_container_name} -it -d --net=host \
#  -w /benchmark \
#  --ipc=host \
#  -v ${host_dataset_path}:${container_dataset_path} \
#  ${IMAGE_ID} \
#  bash

docker run --name ais_bench_container -it -d --net=host \
 -w /benchmark \
 --ipc=host \
 -v /data/datasets:/datasets \
 81a36d90beed \
 bash
```

Run `docker ps` to verify the container is running.

#### Using AISBench Evaluation Tools Inside the Container

Enter the container:

```bash
# docker exec -it ${your_container_name} /bin/bash
docker exec -it ais_bench_container /bin/bash
```

Once inside the container, create symbolic links under `/benchmark/ais_bench/datasets` pointing to the datasets in `/datasets` (which maps to the host directory `/data/datasets`):

```bash
# Batch create symlinks for all files/directories under /datasets
for dir in /datasets/*; do name=$(basename "$dir"); ln -s "$dir" "/benchmark/ais_bench/datasets/$name"; done
```

Navigate to `/benchmark` and run the following command to verify the AISBench evaluation tools are functional:

```
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen_string --search
```

### Local Build

Use the `build_image.sh` script to build:

```bash
# Basic build
bash docker/build_image.sh --tag v3.1-20260522-master

# Specify OS and Python version
bash docker/build_image.sh --tag v3.1-20260522-master --os ubuntu22.04 --py-version py310

# Build and push to remote registry
bash docker/build_image.sh --tag v3.1-20260522-master --push 1

# Build, push, and upload offline package to OBS
bash docker/build_image.sh --tag v3.1-20260522-master --push 1 --upload 1

# Build with cache (faster rebuilds)
bash docker/build_image.sh --tag v3.1-20260522-master --use-cache 1

# Specify a custom image registry
bash docker/build_image.sh --tag v3.1-20260522-master --hub-repo docker.io/myuser/myimage
```

### Build Script Parameter Reference

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `--tag` | Yes | - | Image tag name |
| `--os` | No | `ubuntu22.04` | Operating system type |
| `--py-version` | No | `py310` | Python version |
| `--hub-repo` | No | `ghcr.io/aisbench/aisbench_benchmark` | Image registry URL |
| `--image-output-dir` | No | `/home/ais_bench_ci/release_images` | Offline package output directory |
| `--obs-path` | No | `/home/ais_bench_ci/obsutil_linux_arm64_5.7.9/` | OBS tool path |
| `--push` | No | `0` | Push to remote registry (1=yes) |
| `--upload` | No | `0` | Upload to OBS bucket (1=yes) |
| `--use-cache` | No | `0` | Use build cache (1=yes) |

### Custom Development

To customize a Dockerfile, follow these steps:

1. Create or modify a Dockerfile under `docker/ubuntu/` or `docker/openeuler/`, following the naming convention `Dockerfile.{py_version}.{os}`
2. All Dockerfiles use a multi-stage build pattern:
   - **builder stage**: clone the repository, install dependencies, compile and install
   - **runtime stage**: copy artifacts from builder, producing a slim runtime image
3. Pass the target version tag via `--build-arg GIT_TAG=${TAG}` during build
4. Build using `build_image.sh` or directly with `docker build`:

```bash
docker build \
    --network host \
    --build-arg GIT_TAG=v1.0.0 \
    -f docker/ubuntu/Dockerfile.py310.ubuntu22.04 \
    -t myimage:latest \
    docker/
```

## Running Agent / Sandbox Benchmarks (Docker Inside the Container)

> ⚠️ This section only applies when running agent benchmarks that require isolated sandboxes (SWE-Bench, terminal-bench 2, etc.). It is **not** required for regular service-based model evaluation.

The image ships with Docker Engine (≥ 20.0) and Docker Compose v2 (≥ 2.0.0). There are two modes for running Docker inside the container — pick one based on your host Docker version and isolation needs.

### Mode A — Docker-in-Docker (recommended, true nested containers, requires host Docker ≥ 20.10 + cgroup v2)
```bash
┌────────────────────────── Host ──────────────────────────┐
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────────────┐       │
│  │  Host Kernel     │         │  Host dockerd            │       │
│  │  (cgroup v2)     │◀───────▶│  /var/run/docker.sock    │       │
│  └──────────────────┘         │  manages host's own      │       │
│                               │  containers              │       │
│                               └──────────────────────────┘       │
│                                                                  │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ │
│  ┃  AISBench Container  (--privileged --cgroupns=host)         ┃ │
│  ┃                                                             ┃ │
│  ┃   ┌──────────────────────┐     ┌──────────────────────┐     ┃ │
│  ┃   │   Docker CLI         │────▶│  Inner dockerd       │     ┃ │
│  ┃   │   (user runs)        │     │  (independent proc    │     ┃ │
│  ┃   └──────────────────────┘     │   inside container)  │     ┃ │
│  ┃                                │  daemon.json:        │     ┃ │
│  ┃                                │  native.cgroupdriver │     ┃ │
│  ┃                                │    =cgroupfs          │     ┃ │
│  ┃                                │  storage=vfs          │     ┃ │
│  ┃                                └──────────┬───────────┘     ┃ │
│  ┃                                           │ spawn           ┃ │
│  ┃                                           ▼                ┃ │
│  ┃                                ┌──────────────────────┐     ┃ │
│  ┃                                │   containerd         │     ┃ │
│  ┃                                │   (embedded in       │     ┃ │
│  ┃                                │    inner dockerd)    │     ┃ │
│  ┃                                └──────────┬───────────┘     ┃ │
│  ┃                                           │                 ┃ │
│  ┃                                           ▼                 ┃ │
│  ┃                                ┌──────────────────────┐     ┃ │
│  ┃                                │  Nested child         │     ┃ │
│  ┃                                │  container            │     ┃ │
│  ┃                                │  (truly isolated      │     ┃ │
│  ┃                                │   namespaces)         │     ┃ │
│  ┃                                │  ┌────────────────┐   │     ┃ │
│  ┃                                │  │ Agent process  │   │     ┃ │
│  ┃                                │  │ OpenBLAS       │   │     ┃ │
│  ┃                                │  │ Python deps    │   │     ┃ │
│  ┃                                │  └────────────────┘   │     ┃ │
│  ┃                                └──────────────────────┘     ┃ │
│  ┃                                                             ┃ │
│  ┃   —— Isolation boundary: independent PID/IPC/Net/Mount/User ┃ │
│  ┃      namespaces ——                                         ┃ │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛ │
│                                                                  │
│  ✦ Nested child containers do NOT appear in host `docker ps`     │
│  ✦ Child containers inherit Docker's official default seccomp    │
│    profile (no clone3 blocking)                                  │
│  ✦ Host dockerd restart does NOT affect inner dockerd            │
└──────────────────────────────────────────────────────────────────┘
```

A standalone `dockerd` is started inside the container, so child containers are fully isolated from the host. This is the **preferred mode** for agent benchmarks: child containers inherit Docker's official default seccomp profile, so the `pthread_create` / `clone3` block triggered by openEuler / RHEL hardened profiles does not occur; and there is no stale-socket issue when the host's `dockerd` restarts.

**Prerequisites (verify on the host before proceeding):**

```bash
docker version --format '{{.Server.Version}}'    # must be >= 20.10
uname -r                                          # recommended >= 5.10
stat -fc %T /sys/fs/cgroup                        # must print "cgroup2fs"
```

**Step 1 — Start the container:**

```bash
# --privileged + --cgroupns=host are BOTH required on cgroup v2 hosts.
# Without --cgroupns=host, nested containers fail with:
#   "cannot enter cgroupv2 /sys/fs/cgroup/docker with domain controllers -- it is in an invalid state"
docker run --name ais_bench_container -it -d \
    --net=host \
    --ipc=host \
    --privileged \
    --cgroupns=host \
    -w /benchmark \
    -v /data/datasets:/datasets \
    ghcr.io/aisbench/aisbench_benchmark:v3.1-20260522-master-openeuler24.03-py311-aarch64 \
    bash
```

**Step 2 — Configure and start dockerd inside the container:**

```bash
docker exec -it --privileged ais_bench_container /bin/bash

# Force cgroupfs driver — required for DinD on cgroup v2 hosts.
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<'EOF'
{
  "exec-opts": ["native.cgroupdriver=cgroupfs"],
  "storage-driver": "vfs"
}
EOF

nohup dockerd > /tmp/dockerd.log 2>&1 &

# Wait for the daemon socket to be ready
for i in $(seq 1 30); do
    [ -S /var/run/docker.sock ] && break
    sleep 1
done

# Verify
docker info
docker --version
docker compose version
```

**Notes (Mode A):**

- `--privileged` is mandatory; without it `dockerd` will fail to start.
- `--cgroupns=host` is mandatory on cgroup v2 hosts.
- `cgroupfs` cgroup driver (set via `daemon.json`) is mandatory for DinD. Docker 27.x defaults to `systemd`, which is unavailable inside a container.
- `vfs` is the safest storage driver for DinD. Use `overlay2` only when the host kernel and the container's rootfs support it (still requires `--privileged`).
- For very long-running DinD workloads, consider adding `"default-runtime": "runc"`, `"log-driver": "json-file"`, and `"data-root"` overrides to `/etc/docker/daemon.json`.

### Mode B — Socket Passthrough (works with any Docker version ≥ 1.0)
```bash
┌────────────────────────── Host ──────────────────────────┐
│                                                                  │
│  ┌──────────────────┐                                            │
│  │  Host Kernel     │                                            │
│  └──────────────────┘                                            │
│                                                                  │
│  ┌──────────────────────────────────────────┐  bind mount         │
│  │  Host dockerd                            │  /var/run/docker    │
│  │  /var/run/docker.sock ──────────────────────────┐             │
│  │  (manages host's own containers)         │       │             │
│  └────────────┬─────────────────────────────┘       │             │
│               │                                     │             │
│               │  actually creates/manages           │             │
│               ▼                                     │             │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━┓ │
│  ┃  AISBench Container  (--privileged, shares PID/IPC w/ host)┃ │
│  ┃                                                            ┃ │
│  ┃   ┌──────────────────────┐                                 ┃ │
│  ┃   │   Docker CLI         │── HTTP/Unix socket call ──────┘ ┃ │
│  ┃   │   (user runs)        │  no dockerd process inside       ┃ │
│  ┃   └──────────────────────┘                                 ┃ │
│  ┃                                                            ┃ │
│  ┃   —— Shares kernel, PID, IPC namespaces with host ——       ┃ │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛ │
│                                                                  │
│  ┌──────────────────────────────────────────┐                    │
│  │  Child container (actually on the host)  │                    │
│  │  ┌────────────────┐                      │                    │
│  │  │ Agent process  │  ← created/collected  │                    │
│  │  │ OpenBLAS       │    by host dockerd    │                    │
│  │  │ Python deps    │                      │                    │
│  │  └────────────────┘                      │                    │
│  └──────────────────────────────────────────┘                    │
│                                                                  │
│  ✦ Child containers appear in host `docker ps`                  │
│  ✦ Child containers inherit host dockerd's seccomp profile      │
│    (openEuler/RHEL hardened profile → clone3 blocking)          │
│  ✦ Host dockerd restart → socket inode stale → `docker restart`│
└──────────────────────────────────────────────────────────────────┘
```

Mount the host's Docker socket so that `docker run` inside the container actually creates containers on the **host** daemon. Use this mode only when the host Docker is older than 20.10, or when cgroup v2 is unavailable and Mode A cannot be used.

**Step 1 — Start the container**

```bash
HOST_PATH=/path/to/benchmark_wkp

# 1. Create the working directory on the host and populate it with the image's /benchmark contents
mkdir -p $HOST_PATH
docker run -d --name tmp_extract ${image_id} bash
docker cp tmp_extract:/benchmark/. $HOST_PATH/
docker rm -f tmp_extract

docker run --name ais_bench_container -it -d \
    --net=host \
    --privileged \
    -w ${HOST_PATH} \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ${HOST_PATH}:${HOST_PATH} \
    -v /data/datasets:/datasets \
    ghcr.io/aisbench/aisbench_benchmark:v3.1-20260522-master-openeuler24.03-py311-aarch64 \
    bash
```

That's it — no `dockerd` to start, no `daemon.json` to write. The Docker CLI inside the container talks to the host daemon.
Note: `HOST_PATH` is a working directory that must be created on the **host**. Please make sure the working directory contains no other files or sub-directories.

**Step 2 — Enter the container and re-link ais_bench**

```bash
# Enter the container; you are now in $HOST_PATH
docker exec -it ais_bench_container /bin/bash
# Re-install ais_bench in editable mode inside $HOST_PATH
# (only changes the link target, does not pull dependencies)
pip3 install -e ./ --use-pep517 --no-deps --no-build-isolation --break-system-packages
```

Pros:
- Works with **any Docker version (1.0+)** — no version requirement on the host
- No `--cgroupns=host`, no kernel version check
- Simplest setup; this is how most CI platforms (GitHub Actions, Buildkite, GitLab CI) run Docker

Cons:
- Containers spawned inside the benchmark container appear on the host's `docker ps`
- No isolation — child containers share the host kernel, network, and PID namespace
- Image pulls must be reachable from the host
- ⚠️ **The container's socket handle goes stale whenever the host's `dockerd` restarts.** When the host daemon restarts (machine reboot, daemon upgrade, `systemctl restart docker`, etc.), the bind-mounted `/var/run/docker.sock` inside the container still points to the old (unlinked) inode, so subsequent `docker` commands fail with `Cannot connect to the Docker daemon`. In that case, run `docker restart <container>` to re-establish the bind mount. In CI pipelines, **start a fresh container for every task** to avoid state carried over from previous runs.

**Common issue: child container reports `pthread_create failed: Operation not permitted`**

Under Mode B, child containers are created by the **host dockerd** and inherit the **host daemon's default seccomp profile**. Distributions such as openEuler / RHEL ship a stricter default profile than Docker's upstream one, which blocks the `clone3` syscall used by OpenBLAS / NumPy when spawning worker threads. The symptom looks like:

```
OpenBLAS blas_thread_init: pthread_create failed for thread 52 of 64: Operation not permitted
OpenBLAS blas_thread_init: RLIMIT_NPROC 1048576 current, 1048576 max
```

Note that `RLIMIT_NPROC` is not exhausted (the call returns `EPERM`, not `EAGAIN`), which confirms a seccomp block rather than a resource limit. Mode A is unaffected because the inner dockerd uses Docker's official default profile.

**Fix**: explicitly relax seccomp in the `docker-compose.yml` that launches the child container:

```yaml
services:
  main:
    security_opt:
      - seccomp=unconfined
    # ... rest of the config unchanged
```

This only applies to that service; it does not affect the outer AISBench container or the host daemon. If Agent further spawns grandchild containers from `main`, those still inherit the host profile by default — add the same `security_opt` at that layer as well.

### Run a container workload (works in both modes)

```bash
# Inside the container
docker pull alpine:latest
docker run --rm alpine:latest echo "Hello from a nested container"

# Or with docker compose v2
cat > /tmp/docker-compose.yml <<'EOF'
services:
  hello:
    image: alpine:latest
    command: echo "compose v2 works"
EOF
docker compose -f /tmp/docker-compose.yml up
```

### Which mode should I choose?

- **Mode A (Docker-in-Docker)** is the recommended mode, covering terminal-bench 2, SWE-Bench, and most agent benchmarks. Child containers are isolated from the host and are not affected by the host's seccomp profile — the fewest pitfalls. Requires host Docker ≥ 20.10 + cgroup v2.
- **Mode B (Socket Passthrough)** is used only when the host Docker is older than 20.10, or when cgroup v2 is unavailable. It is the simplest to set up, but child containers inherit the host's seccomp profile (which may block `clone3`), and the container must be manually `docker restart`-ed after the host's `dockerd` restarts.

### Security Implications of `--privileged`

`--privileged` is a heavy flag that removes most container isolation. Use it only when needed (i.e., for Mode A / DinD).

**Risks**

- **Disables all Linux capability restrictions** — the container gets nearly all root capabilities (`CAP_SYS_ADMIN`, `CAP_NET_ADMIN`, `CAP_SYS_PTRACE`, `CAP_SYS_MODULE`, …).
- **Bypasses seccomp and AppArmor** — any syscall filter is removed, so rules like the ones that broke Python threading no longer apply.
- **Full device access** — `/dev/sda`, `/dev/mem`, `/dev/kvm`, etc. become accessible; a process inside the container can mount, read, or wipe host disks.
- **Writes to the host cgroup tree** — a privileged container can modify other containers' resource limits and freeze/kill them.
- **Kernel-module load/unload** — the container can load or unload host kernel modules (when `CAP_SYS_MODULE` is retained on the host).

**Mitigations when `--privileged` is unavoidable**

- Run the benchmark inside a dedicated VM (or a dedicated bare-metal machine) that holds no production data.
- Use a separate Docker daemon (a dedicated `dockerd` on a non-default socket) so the privileged container cannot reach production workloads.
- Audit any code that will run inside the container before granting the flag.
- If you only need to relax the seccomp filter (e.g., for `pip install` threading), prefer `--security-opt seccomp=unconfined` instead — it is a much smaller blast radius.

**Alternatives to `--privileged` for DinD**

- `--security-opt seccomp=unconfined` alone is **not** enough for DinD — dockerd still needs cgroup and device access, which only `--privileged` (or a custom runtime) provides.
- [Sysbox](https://github.com/nestybox/sysbox) — a container runtime that supports nested containers without `--privileged`, at the cost of installing a custom runtime on the host.
- Rootless Docker — runs `dockerd` as a non-root user; has its own limitations (no `overlay2` on most distros, network restrictions, etc.).

## Agent Evaluation One-Click Environment Preparation

For agent benchmarks such as Harbor Terminal-Bench, SWE-bench, and SWE-bench Pro, this repository provides a one-click environment preparation solution in [`docker/agent_runtime/`](agent_runtime/README.md). It consolidates the Mode A/B selection, `daemon.json` configuration, `--cgroupns=host`, and seccomp handling from the "Running Agent / Sandbox Benchmarks" section above into a single script, and additionally solves:

- **Dependency conflicts**: Dependencies of multiple agent benchmarks conflict with each other (e.g., harbor forces upgrading `datasets` to 4.0+, and two `mini-swe-agent` forks override each other with the same package name). Solved via multi-venv isolation inside the runtime image.
- **Huge case images**: SWE-bench full (~1TB) cannot be packaged as a whole, so it is **not** baked into the runtime image. Users `docker pull` / `docker load` themselves.
- **Frequent dataset versions**: Agent dataset versions change quickly, so they are **not** baked into the runtime image. Users prepare them on the host and mount them into the container via `bootstrap.sh --datasets <PATH>` (container path = host path).
- **No environment verification**: `doctor.sh` verifies the runtime configuration (L1 static) before running the benchmark + scans for case image presence (warning), and gives precise fix guidance on failure.

**Usage** (all three packs use the same workflow; choose `--datasets` and `agent_env` per benchmark):

```bash
# Host: prepare the dataset directory + start the runtime container with one click
#       (mounts the dataset directory into the container)
# Harbor:
mkdir -p /data/datasets/harbor/mini-0.10
# Prepare dataset (see harbor_bench.md) + case image (can be loaded offline via bootstrap --case-tar)
curl -fsSL https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/ais_bench_agent_bootstrap.sh \
    | bash -s -- \
        --datasets /data/datasets/harbor/mini-0.10/terminal-bench-2-offline-selected_0.10 \
        --case-tar /data/cases/case-tb2-mini-0.10.tar.gz
docker exec -it ais_bench_agent bash
# Inside the container:
ais_bench_agent_doctor.sh harbor
agent_env harbor
ais_bench ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py --debug

# SWE-bench / SWE-bench Pro work the same way:
#   --datasets <swebench dataset directory>
#   agent_env swebench | swebench_pro
#   ais_bench ais_bench/configs/swe_bench[_pro]_examples/... --debug
# For dataset and case image acquisition, see each benchmark's documentation.
```

For the solution design and per-script parameters, see [`docker/agent_runtime/README.md`](agent_runtime/README.md). Each agent benchmark doc ([harbor_bench.md](../docs/source_en/extended_benchmark/agent/harbor_bench.md), [swe_bench.md](../docs/source_en/extended_benchmark/agent/swe_bench.md), [swe_bench_pro.md](../docs/source_en/extended_benchmark/agent/swe_bench_pro.md)) also has a "Quick Start" section at the top.

> This section is the executable packaging of the "Running Agent / Sandbox Benchmarks" section. To understand the principles, still read the Mode A/B sections above; for quick start, just use the script in this section.

## License / Disclaimer

This project's images and build scripts are licensed under the [LICENSE file](https://github.com/AISBench/benchmark/blob/master/LICENSE) in the repository root.

**Disclaimer**: This Docker image is provided "as is", without warranty of any kind, express or implied. Users should evaluate whether the image meets their requirements and assume full responsibility for any consequences arising from its use. Third-party software packages installed in the image are governed by their respective license terms.
