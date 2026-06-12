#!/bin/bash

usage() {
    echo "用法: $0 --tag <TAG> [--os <操作系统>] [--py-version <Python版本>] [--obs-path <OBS工具路径>] [--hub-repo <镜像仓库>] [--image-output-dir <输出目录>] [--push <0|1>] [--upload <0|1>] [--use-cache <0|1>] [--multi-arch <0|1>]"
    echo ""
    echo "参数说明:"
    echo "  --tag              必填，要更新的TAG名称，遵循AISBench仓库的tag命名规则，且必须是存在的tag，例如v3.1-20260522-master"
    echo "  --os               可选，操作系统类型，默认: ubuntu22.04"
    echo "  --py-version       可选，Python版本，默认: py310"
    echo "  --obs-path         可选，OBS工具路径，默认: /home/ais_bench_ci/obsutil_linux_arm64_5.7.9/"
    echo "  --hub-repo         可选，镜像仓库地址，默认: ghcr.io/aisbench/aisbench_benchmark"
    echo "  --image-output-dir 可选，离线包输出目录，默认: /home/ais_bench_ci/release_images"
    echo "  --push             可选，是否推送到远程仓库，默认: 0"
    echo "  --upload           可选，是否上传到OBS桶，默认: 0"
    echo "  --use-cache        可选，是否使用缓存构建，默认: 0"
    echo "  --multi-arch       可选，是否构建多架构镜像（amd64+arm64），默认: 0。启用后在各架构机器上分别构建并推送各自架构镜像，再用docker manifest合并为统一tag（不带架构后缀）"
    echo ""
    echo "示例:"
    echo "  $0 --tag v3.1-20260522-master"
    echo "  $0 --tag v3.1-20260522-master --os ubuntu22.04 --py-version py311"
    echo "  $0 --tag v3.1-20260522-master --hub-repo docker.io/myrepo/myimage"
    echo "  $0 --tag v3.1-20260522-master --image-output-dir /tmp/images"
    echo "  $0 --tag v3.1-20260522-master --push 1"
    echo "  $0 --tag v3.1-20260522-master --push 1 --upload 1"
    echo "  $0 --tag v3.1-20260522-master --multi-arch 1 --push 1"
    exit 1
}

TAG=""
OS="ubuntu22.04"
py_version="py310"
obsutils_path="/home/ais_bench_ci/obsutil_linux_arm64_5.7.9/"
hub_repo="ghcr.io/aisbench/aisbench_benchmark"
image_output_dir="/home/ais_bench_ci/release_images"
push=0
upload=0
use_cache=0
multi_arch=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --os)
            OS="$2"
            shift 2
            ;;
        --py-version)
            py_version="$2"
            shift 2
            ;;
        --obs-path)
            obsutils_path="$2"
            shift 2
            ;;
        --hub-repo)
            hub_repo="$2"
            shift 2
            ;;
        --image-output-dir)
            image_output_dir="$2"
            shift 2
            ;;
        --push)
            push="$2"
            shift 2
            ;;
        --upload)
            upload="$2"
            shift 2
            ;;
        --use-cache)
            use_cache="$2"
            shift 2
            ;;
        --multi-arch)
            multi_arch="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "错误：未知参数 $1"
            usage
            ;;
    esac
done

if [ -z "$TAG" ]; then
    echo "错误：缺少必需参数 --tag"
    usage
fi

arch=$(uname -m)

if [[ "$OS" == ubuntu* ]]; then
    dockerfile_dir="ubuntu"
elif [[ "$OS" == openeuler* ]]; then
    dockerfile_dir="openeuler"
else
    dockerfile_dir="ubuntu"
fi

dockerfile_path="$(dirname "$0")/${dockerfile_dir}/Dockerfile.${py_version}.${OS}"
image_name=${hub_repo}:${TAG}-${OS}-${py_version}-${arch}
manifest_image_name=${hub_repo}:${TAG}-${OS}-${py_version}

offline_pkg_name=ais_bench_benchmark_image_${TAG}-${OS}-${py_version}-${arch}.tar.gz
offline_pkg_full_path=${image_output_dir}/${offline_pkg_name}

if [ ! -f "${dockerfile_path}" ]; then
    echo "错误：Dockerfile不存在，路径：${dockerfile_path}"
    echo "提示：请确保文件存在于对应目录"
    exit 1
fi

echo "开始清理本地旧资源..."
if docker images -q ${image_name} > /dev/null 2>&1; then
    docker rmi -f ${image_name}
    echo "已删除本地旧镜像：${image_name}"
fi

if [ -f "${offline_pkg_full_path}" ]; then
    rm -f "${offline_pkg_full_path}"
    echo "已删除本地旧离线包：${offline_pkg_full_path}"
fi

if [ "$use_cache" == "1" ]; then
    echo "开始构建新镜像（使用缓存）..."
    echo "使用Dockerfile：${dockerfile_path}"
    docker build \
        --network host \
        --build-arg GIT_TAG=${TAG} \
        -f ${dockerfile_path} \
        -t ${image_name} \
        $(dirname "$0")
else
    echo "开始构建新镜像（强制不使用缓存，确保更新完整）..."
    echo "使用Dockerfile：${dockerfile_path}"
    docker build \
        --no-cache \
        --network host \
        --build-arg GIT_TAG=${TAG} \
        -f ${dockerfile_path} \
        -t ${image_name} \
        $(dirname "$0")
fi

if [ $? -ne 0 ]; then
    echo "错误：镜像构建失败，终止后续操作"
    exit 1
fi

echo "开始验证镜像功能..."
validation_output=$(docker run --rm ${image_name} ais_bench --models vllm_api_stream_chat --datasets synthetic_gen -s 2>&1)
validation_status=$?

if [ $validation_status -ne 0 ]; then
    echo "错误：镜像测试命令执行失败（退出码: ${validation_status}）"
    echo "测试输出："
    echo "${validation_output}"
    exit 1
fi

validation_checks=(
    "Task Type"
    "Task Name"
    "Config File Path"
    "vllm_api_stream_chat"
    "synthetic_gen"
    "/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
    "/benchmark/ais_bench/benchmark/configs/datasets/synthetic/synthetic_gen.py"
)

all_checks_passed=true
for check in "${validation_checks[@]}"; do
    if ! echo "${validation_output}" | grep -F "${check}" > /dev/null 2>&1; then
        echo "错误：验证失败，未找到预期内容：${check}"
        all_checks_passed=false
    fi
done

if [ "$all_checks_passed" = false ]; then
    echo "错误：镜像测试输出与预期不符，测试失败"
    echo "预期输出应包含以下关键内容："
    printf '  - %s\n' "${validation_checks[@]}"
    echo ""
    echo "实际测试输出："
    echo "${validation_output}"
    exit 1
fi

echo "镜像验证成功！"

if [ "$push" == "1" ]; then
    echo "开始推送镜像到远程仓库（覆盖已有同名镜像）..."
    docker push ${image_name}

    if [ $? -ne 0 ]; then
        echo "错误：镜像推送失败，终止后续操作"
        exit 1
    fi
fi

if [ "$multi_arch" == "1" ]; then
    if [ "$push" != "1" ]; then
        echo "提示：多架构模式下未开启推送，manifest合并需要已推送的镜像。跳过manifest合并。"
    else
        arch_image_amd64=${hub_repo}:${TAG}-${OS}-${py_version}-x86_64
        arch_image_arm64=${hub_repo}:${TAG}-${OS}-${py_version}-aarch64

        echo "开始创建多架构manifest list：${manifest_image_name}"
        echo "  - ${arch_image_amd64}"
        echo "  - ${arch_image_arm64}"

        docker buildx imagetools create \
            -t ${manifest_image_name} \
            ${arch_image_amd64} \
            ${arch_image_arm64}

        if [ $? -ne 0 ]; then
            echo "错误：多架构manifest创建失败"
            exit 1
        fi

        echo "多架构manifest list已更新：${manifest_image_name}"
        echo "  docker buildx imagetools inspect ${manifest_image_name}"
    fi
fi

echo "构建成功，镜像已更新：${image_name}"

if [ "$upload" == "1" ]; then
    echo "开始打包离线包..."
    mkdir -p ${image_output_dir}
    docker save ${image_name} | gzip -9 > ${offline_pkg_full_path}
    echo "离线包已生成：${offline_pkg_full_path}"
    chmod 640 ${offline_pkg_full_path}
    echo "开始上传离线包到OBS桶（强制覆盖已有文件）..."
    if [ ! -d "${obsutils_path}" ] || [ ! -x "${obsutils_path}/obsutil" ]; then
        echo "错误：obsutil路径不存在或不可执行，路径：${obsutils_path}"
        exit 1
    fi

    cd ${obsutils_path}
    ./obsutil cp ${offline_pkg_full_path} obs://aisbench/images/benchmark/github/${offline_pkg_name} -f

    if [ $? -eq 0 ]; then
        echo "全部操作完成！"
        echo "镜像已更新：${image_name}"
        if [ "$multi_arch" == "1" ] && [ "$push" == "1" ]; then
            echo "多架构manifest list：${manifest_image_name}"
        fi
        echo "离线包已更新并上传：${offline_pkg_name}"
    else
        echo "错误：OBS桶上传失败"
        exit 1
    fi
else
    echo "跳过上传到OBS桶（--upload 未设置为1）"
    echo "全部操作完成！"
    echo "镜像已构建：${image_name}"
    if [ "$multi_arch" == "1" ] && [ "$push" == "1" ]; then
        echo "多架构manifest list：${manifest_image_name}"
    fi
fi