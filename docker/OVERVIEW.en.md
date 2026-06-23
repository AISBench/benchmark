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
> ⚠️Note: AISBench Benchmark images only support evaluation of service-based models, not offline inference models. Evaluation scenarios that require launching independent sandbox environments (such as SWE-Bench, terminal-bench 2, etc.) are also not currently supported.

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

## License / Disclaimer

This project's images and build scripts are licensed under the [LICENSE file](https://github.com/AISBench/benchmark/blob/master/LICENSE) in the repository root.

**Disclaimer**: This Docker image is provided "as is", without warranty of any kind, express or implied. Users should evaluate whether the image meets their requirements and assume full responsibility for any consequences arising from its use. Third-party software packages installed in the image are governed by their respective license terms.
