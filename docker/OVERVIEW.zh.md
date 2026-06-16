# Docker 镜像概览

## 快速参考
- AISBench Benchmark由[AISBench人工智能系统性能评测基准委员会](https://www.aisbench.com/about)维护。

- 镜像简介

| 项目 | 说明 |
| --- | --- |
| 默认镜像仓库 | `ghcr.io/aisbench/aisbench_benchmark` |
| 构建脚本 | `build_image.sh` |
| 支持的 OS | Ubuntu 22.04 / 24.04, openEuler 22.03 / 24.03 |
| 支持的 Python | 3.10, 3.11, 3.12 |
| 构建方式 | 多阶段构建（builder → runtime） |
| 工作目录 | `/benchmark` |


- 从哪里获取帮助
    + [📖AISBench Benchmark 文档](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/)
    + [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AISBench/benchmark)
    + [🤔报告问题](https://github.com/AISBench/benchmark/issues/new/choose)

### AISBench Benchmark
AISBench Benchmark 是基于 [OpenCompass](https://github.com/open-compass/opencompass) 构建的模型评测工具，兼容 OpenCompass 的配置体系、数据集结构与模型后端实现，并在此基础上扩展了对服务化模型的支持能力。
> ⚠️注意：AISBench Benchmark的镜像只支持服务化模型的评测，不支持离线推理模型的评测。当前也不支持SWE-Bench、terminal-bench 2等测评过程中需要启动独立沙箱环境的测评常见

## 镜像 Tag 说明及 Dockerfile 归档路径

镜像 Tag 格式为：

```
{hub_repo}:{TAG}-{OS}-{py_version}-{arch}
```

例如：`ghcr.io/aisbench/aisbench_benchmark:v3.1-20260522-master-ubuntu22.04-py310-x86_64`
其中：
- `v3.1-20260522-master` 为版本号，格式为 `v{大版本号}.{小版本号}-{日期}-{分支}`
- `ubuntu22.04` 为操作系统版本
- `py310` 为 Python 版本
- `x86_64` 为架构

### Dockerfile 文件清单

| Dockerfile | 基础镜像 | Python | 路径 |
| --- | --- | --- | --- |
| [Dockerfile.py310.ubuntu22.04](ubuntu/Dockerfile.py310.ubuntu22.04) | `ubuntu:22.04` | 3.10 | `docker/ubuntu/` |
| [Dockerfile.py312.ubuntu24.04](ubuntu/Dockerfile.py312.ubuntu24.04) | `ubuntu:24.04` | 3.12 | `docker/ubuntu/` |
| [Dockerfile.py310.openeuler22.03](openeuler/Dockerfile.py310.openeuler22.03) | `openeuler/openeuler:22.03-lts` | 3.10 | `docker/openeuler/` |
| [Dockerfile.py311.openeuler24.03](openeuler/Dockerfile.py311.openeuler24.03) | `openeuler/openeuler:24.03-lts` | 3.11 | `docker/openeuler/` |

Dockerfile 命名规则：`Dockerfile.{py_version}.{os}`

## 快速开始

### 运行已有镜像
#### 官方镜像获取
所有镜像的ghcr归档：https://github.com/orgs/AISBench/packages/container/package/aisbench_benchmark

以tag为`v3.1-20260522-master-openeuler24.03-py311-aarch64`的docker 镜像获取主要有两种方式;
1. docker pull 命令拉取
```bash
docker pull ghcr.io/aisbench/aisbench_benchmark:v3.1-20260522-master-openeuler24.03-py311-aarch64
```

2. 从镜像打包文件中导入
```bash
# 下载docker镜像打包文件aisbench_benchmark_v3.1-20260522-master-openeuler24.03-py311-aarch64.tar.gz
wget https://aisbench.obs.cn-north-4.myhuaweicloud.com/images/benchmark/github/aisbench_benchmark_v3.1-20260522-master-openeuler24.03-py311-aarch64.tar.gz
# 从打包文件中导入镜像
docker load -i aisbench_benchmark_v3.1-20260522-master-openeuler24.03-py311-aarch64.tar.gz
```

#### 基于docker 镜像启动docker 容器
可以参考如下命令启动：
```bash
# docker run --name ${你的容器名称} -it -d --net=host \
#  -w /benchmark \
#  --ipc=host \
#  -v ${宿主机数据集路径}:${容器内数据集路径}
#  ${IMAGE ID} \
#  bash

docker run --name ais_bench_container -it -d --net=host \
 -w /benchmark \
 --ipc=host \
 -v /data/datasets:/datasets \
 81a36d90beed \
 bash
```
执行`docker ps`可以看到刚才创建的容器正在执行。

#### 进入docker容器中使用AISBench测评工具
执行命令
```bash
# docker exec -it ${你的容器名称} /bin/bash
docker exec -it ais_bench_container /bin/bash
```
进入容器后，需要在`/benchmark/ais_bench/datasets`内建立软链接，链接到`/datasets`内（物理机上存放所有数据集的文件夹`/data/datasets`）的数据集，可以执行如下命令达成：
```bash
# 批量创建软链接（/datasets 下的所有文件/目录）
for dir in /datasets/*; do name=$(basename "$dir"); ln -s "$dir" "/benchmark/ais_bench/datasets/$name"; done
```

进入 /benchmark，执行如下命令验证AISBench评测工具可用:
```
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen_string --search
```

### 本地构建

使用 `build_image.sh` 脚本构建：

```bash
# 基础构建
bash docker/build_image.sh --tag v3.1-20260522-master

# 指定 OS 和 Python 版本
bash docker/build_image.sh --tag v3.1-20260522-master --os ubuntu22.04 --py-version py310

# 构建并推送到远程仓库
bash docker/build_image.sh --tag v3.1-20260522-master --push 1

# 构建、推送并上传离线包到 OBS
bash docker/build_image.sh --tag v3.1-20260522-master --push 1 --upload 1

# 使用缓存构建（加速重复构建）
bash docker/build_image.sh --tag v3.1-20260522-master --use-cache 1

# 指定自定义镜像仓库
bash docker/build_image.sh --tag v3.1-20260522-master --hub-repo docker.io/myuser/myimage
```

### 构建脚本参数一览

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--tag` | 是 | - | 镜像 TAG 名称 |
| `--os` | 否 | `ubuntu22.04` | 操作系统类型 |
| `--py-version` | 否 | `py310` | Python 版本 |
| `--hub-repo` | 否 | `ghcr.io/aisbench/aisbench_benchmark` | 镜像仓库地址 |
| `--image-output-dir` | 否 | `/home/ais_bench_ci/release_images` | 离线包输出目录 |
| `--obs-path` | 否 | `/home/ais_bench_ci/obsutil_linux_arm64_5.7.9/` | OBS 工具路径 |
| `--push` | 否 | `0` | 是否推送到远程仓库（1=是） |
| `--upload` | 否 | `0` | 是否上传到 OBS 桶（1=是） |
| `--use-cache` | 否 | `0` | 是否使用缓存构建（1=是） |

### 二次开发

如需自定义 Dockerfile，按以下步骤操作：

1. 在 `docker/ubuntu/` 或 `docker/openeuler/` 下新建或修改 Dockerfile，遵循命名规则 `Dockerfile.{py_version}.{os}`
2. 所有 Dockerfile 均采用多阶段构建模式：
   - **builder 阶段**：克隆仓库、安装依赖、编译安装
   - **runtime 阶段**：从 builder 复制产物，生成精简运行镜像
3. 构建时通过 `--build-arg GIT_TAG=${TAG}` 传入目标版本标签
4. 使用 `build_image.sh` 或直接 `docker build` 构建：

```bash
docker build \
    --network host \
    --build-arg GIT_TAG=v1.0.0 \
    -f docker/ubuntu/Dockerfile.py310.ubuntu22.04 \
    -t myimage:latest \
    docker/
```

## 许可证 / 免责声明

本项目镜像及其构建脚本按仓库根目录的 [LICENSE 文件](https://github.com/AISBench/benchmark/blob/master/LICENSE) 授权。

**免责声明**：本 Docker 镜像按"原样"提供，不提供任何明示或暗示的保证。使用者应自行评估镜像是否满足其需求，并对使用本镜像所产生的任何后果负责。镜像中安装的第三方软件包遵循其各自的许可证条款。
