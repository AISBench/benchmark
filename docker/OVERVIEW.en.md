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

### Mode A — Socket Passthrough (recommended, works with any Docker version ≥ 1.0)

Mount the host's Docker socket so that `docker run` inside the container actually creates containers on the **host** daemon.

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

### Mode B — Docker-in-Docker (true nested containers, requires host Docker ≥ 20.10 + cgroup v2)

Use this when you need full isolation between the benchmark container and the spawned child containers.

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

**Notes (Mode B):**

- `--privileged` is mandatory; without it `dockerd` will fail to start.
- `--cgroupns=host` is mandatory on cgroup v2 hosts.
- `cgroupfs` cgroup driver (set via `daemon.json`) is mandatory for DinD. Docker 27.x defaults to `systemd`, which is unavailable inside a container.
- `vfs` is the safest storage driver for DinD. Use `overlay2` only when the host kernel and the container's rootfs support it (still requires `--privileged`).
- For very long-running DinD workloads, consider adding `"default-runtime": "runc"`, `"log-driver": "json-file"`, and `"data-root"` overrides to `/etc/docker/daemon.json`.

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

- Use **Mode A** if your host Docker is older than 20.10, or if you don't need isolation between the benchmark container and the child containers. This covers terminal-bench 2, SWE-Bench, and most agent benchmarks.
- Use **Mode B** only when you specifically need the child containers to be isolated from the host (e.g., running untrusted workloads, or when you need a clean cgroup hierarchy per benchmark run).

### Security Implications of `--privileged`

`--privileged` is a heavy flag that removes most container isolation. Use it only when needed (i.e., for Mode B / DinD).

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

## License / Disclaimer

This project's images and build scripts are licensed under the [LICENSE file](https://github.com/AISBench/benchmark/blob/master/LICENSE) in the repository root.

**Disclaimer**: This Docker image is provided "as is", without warranty of any kind, express or implied. Users should evaluate whether the image meets their requirements and assume full responsibility for any consequences arising from its use. Third-party software packages installed in the image are governed by their respective license terms.
