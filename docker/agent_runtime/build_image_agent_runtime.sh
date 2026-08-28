#!/bin/bash
#
# 构建 AISBench Agent Runtime 镜像
#
# 在 aisbench_benchmark 基础镜像之上追加多 venv 隔离层，
# 用于 agent 测评（Harbor Terminal-Bench 等）的环境准备。
#
# 用法:
#   bash docker/agent_runtime/build_image_agent_runtime.sh --base-tag <TAG> [选项]
#
# 必填参数:
#   --base-tag <TAG>      基镜像（aisbench_benchmark）的 TAG，例如 v3.1-20260522-master
#                         会自动拼接为 <hub_repo>:<base-tag>-<os>-<py_version>-<arch>
#
# 可选参数:
#   --os <OS>             操作系统，默认 ubuntu24.04
#   --py-version <VER>    Python 版本，默认 py312（agent runtime 必须用 py312）
#   --hub-repo <REPO>     基镜像仓库地址，默认 ghcr.io/aisbench/aisbench_benchmark
#   --target-hub-repo <REPO>  目标镜像仓库地址（agent-runtime 镜像推到哪），默认 ghcr.io/aisbench/agent-runtime
#   --image-output-dir <DIR>  离线包输出目录，默认 /home/ais_bench_ci/release_images
#   --obs-path <PATH>     OBS 工具路径，默认 /home/ais_bench_ci/obsutil_linux_arm64_5.7.9/
#   --push <0|1>          是否推送到远程仓库，默认 0
#   --upload <0|1>        是否上传离线包到 OBS，默认 0
#   --use-cache <0|1>     是否使用缓存构建，默认 0
#   --multi-arch <0|1>    是否构建多架构镜像（amd64+arm64），默认 0
#   -h, --help            显示帮助
#
# 示例:
#   # 基础构建（本地，x86_64 或 aarch64 视当前机器）
#   bash docker/agent_runtime/build_image_agent_runtime.sh --base-tag v3.1-20260522-master
#
#   # 指定 OS/Python 版本
#   bash docker/agent_runtime/build_image_agent_runtime.sh --base-tag v3.1-20260522-master \
#       --os ubuntu24.04 --py-version py312
#
#   # 构建并推送
#   bash docker/agent_runtime/build_image_agent_runtime.sh --base-tag v3.1-20260522-master --push 1
#
#   # 多架构构建并推送（需在各自架构机器上分别执行 + manifest 合并）
#   bash docker/agent_runtime/build_image_agent_runtime.sh --base-tag v3.1-20260522-master \
#       --multi-arch 1 --push 1
#
#   # 构建、推送、并上传离线包到 OBS
#   bash docker/agent_runtime/build_image_agent_runtime.sh --base-tag v3.1-20260522-master \
#       --push 1 --upload 1

set -e

usage() {
    echo "用法: $0 --base-tag <TAG> [选项]"
    echo ""
    echo "必填参数:"
    echo "  --base-tag <TAG>          基镜像（aisbench_benchmark）的 TAG，例如 v3.1-20260522-master"
    echo ""
    echo "可选参数:"
    echo "  --os <OS>                 操作系统，默认: ubuntu24.04"
    echo "  --py-version <VER>        Python 版本，默认: py312（agent runtime 推荐 py312）"
    echo "  --hub-repo <REPO>         基镜像仓库地址，默认: ghcr.io/aisbench/aisbench_benchmark"
    echo "  --target-hub-repo <REPO>  目标镜像仓库，默认: ghcr.io/aisbench/agent-runtime"
    echo "  --image-output-dir <DIR>  离线包输出目录，默认: /home/ais_bench_ci/release_images"
    echo "  --obs-path <PATH>         OBS 工具路径，默认: /home/ais_bench_ci/obsutil_linux_arm64_5.7.9/"
    echo "  --push <0|1>              是否推送到远程仓库，默认: 0"
    echo "  --upload <0|1>            是否上传离线包到 OBS，默认: 0"
    echo "  --use-cache <0|1>         是否使用缓存构建，默认: 0"
    echo "  --multi-arch <0|1>        是否构建多架构镜像（amd64+arm64），默认: 0"
    echo "  -h, --help                显示本帮助"
    echo ""
    echo "示例:"
    echo "  $0 --base-tag v3.1-20260522-master"
    echo "  $0 --base-tag v3.1-20260522-master --push 1 --upload 1"
    echo "  $0 --base-tag v3.1-20260522-master --multi-arch 1 --push 1"
    exit 1
}

# ============ 默认配置 ============
BASE_TAG=""
OS="ubuntu24.04"
py_version="py312"
hub_repo="ghcr.io/aisbench/aisbench_benchmark"
target_hub_repo="ghcr.io/aisbench/agent-runtime"
image_output_dir="/home/ais_bench_ci/release_images"
obsutils_path="/home/ais_bench_ci/obsutil_linux_arm64_5.7.9/"
push=0
upload=0
use_cache=0
multi_arch=0
harbor_wheel=""

# ============ 参数解析 ============
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-tag)
            BASE_TAG="$2"
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
        --hub-repo)
            hub_repo="$2"
            shift 2
            ;;
        --target-hub-repo)
            target_hub_repo="$2"
            shift 2
            ;;
        --image-output-dir)
            image_output_dir="$2"
            shift 2
            ;;
        --obs-path)
            obsutils_path="$2"
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
        --harbor-wheel)
            harbor_wheel="$2"
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

# ============ 校验必填参数 ============
if [ -z "$BASE_TAG" ]; then
    echo "错误：缺少必需参数 --base-tag"
    echo "提示：--base-tag 是 aisbench_benchmark 基镜像的 TAG，例如 v3.1-20260522-master"
    usage
fi

# agent runtime 依赖 python3.12 的 venv，若用其他 py_version 给出警告
if [ "$py_version" != "py312" ]; then
    echo "警告：agent runtime 推荐 py312，当前指定 ${py_version}，可能因缺 python3.12 导致 venv 创建失败"
fi

# ============ 计算镜像名 ============
arch=$(uname -m)

# 基镜像全名: <hub_repo>:<base-tag>-<os>-<py_version>-<arch>
base_image="${hub_repo}:${BASE_TAG}-${OS}-${py_version}-${arch}"

# 目标镜像全名: <target_hub_repo>:<base-tag>-<os>-<py_version>-<arch>
# 用 base-tag 作为 agent-runtime 的 tag，便于追溯基镜像版本
image_name="${target_hub_repo}:${BASE_TAG}-${OS}-${py_version}-${arch}"

# 多架构 manifest 名（不带 arch 后缀）
manifest_image_name="${target_hub_repo}:${BASE_TAG}-${OS}-${py_version}"

# 离线包名
offline_pkg_name="agent_runtime_image_${BASE_TAG}-${OS}-${py_version}-${arch}.tar.gz"
offline_pkg_full_path="${image_output_dir}/${offline_pkg_name}"

# Dockerfile 路径
dockerfile_path="$(dirname "$0")/Dockerfile.agent-runtime"

if [ ! -f "${dockerfile_path}" ]; then
    echo "错误：Dockerfile 不存在：${dockerfile_path}"
    exit 1
fi

echo "============================================================"
echo " 构建 AISBench Agent Runtime 镜像"
echo "============================================================"
echo "  基镜像:       ${base_image}"
echo "  目标镜像:     ${image_name}"
echo "  Dockerfile:   ${dockerfile_path}"
echo "  arch:         ${arch}"
echo "  push:         ${push}"
echo "  upload:       ${upload}"
echo "  use_cache:    ${use_cache}"
echo "  multi_arch:   ${multi_arch}"
echo "============================================================"

# ============ 检查基镜像是否存在（本地或远程） ============
echo "检查基镜像 ${base_image} ..."
if ! docker image inspect "${base_image}" >/dev/null 2>&1; then
    echo "  本地不存在，尝试 pull ..."
    if ! docker pull "${base_image}" 2>/dev/null; then
        echo "错误：基镜像拉取失败：${base_image}"
        echo "提示：请确认 --base-tag / --os / --py-version / --hub-repo 参数正确"
        exit 1
    fi
fi
echo "  ✓ 基镜像就绪"

# ============ 清理本地旧资源 ============
echo "清理本地旧资源..."
if docker images -q "${image_name}" >/dev/null 2>&1; then
    docker rmi -f "${image_name}" >/dev/null 2>&1 || true
    echo "  已删除本地旧镜像：${image_name}"
fi
if [ -f "${offline_pkg_full_path}" ]; then
    rm -f "${offline_pkg_full_path}" || true
    echo "  已删除本地旧离线包：${offline_pkg_full_path}"
fi

# ============ [v3 wrapper] harbor wheel 占位文件 ============
# Dockerfile.agent-runtime 期望 repo root 有 .harbor_wheel_cache.whl
# 传了 --harbor-wheel → cp 真实 wheel；未传 → touch 空文件（回退到 harbor==0.20.0）
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WHEEL_PLACEHOLDER="${REPO_ROOT}/.harbor_wheel_cache.whl"

if [ -n "${harbor_wheel}" ]; then
    if [ ! -f "${harbor_wheel}" ]; then
        echo "错误：--harbor-wheel 指定的文件不存在: ${harbor_wheel}"
        exit 1
    fi
    echo "  使用本地 harbor wheel: ${harbor_wheel}"
    cp "${harbor_wheel}" "${WHEEL_PLACEHOLDER}"
else
    touch "${WHEEL_PLACEHOLDER}"
fi
# 构建后无论成功或失败都清理占位文件
trap "rm -f ${WHEEL_PLACEHOLDER}" EXIT

# ============ 构建镜像 ============
BUILD_ARGS="--build-arg BASE_IMAGE=${base_image}"

if [ "$use_cache" == "1" ]; then
    echo "开始构建（使用缓存）..."
    docker build \
        --network host \
        ${BUILD_ARGS} \
        -f "${dockerfile_path}" \
        -t "${image_name}" \
        "$(dirname "$0")/../../"
else
    echo "开始构建（强制不使用缓存）..."
    docker build \
        --no-cache \
        --network host \
        ${BUILD_ARGS} \
        -f "${dockerfile_path}" \
        -t "${image_name}" \
        "$(dirname "$0")/../../"
fi

if [ $? -ne 0 ]; then
    echo "错误：镜像构建失败"
    exit 1
fi
echo "✓ 镜像构建成功：${image_name}"

# ============ 验证镜像 ============
echo "开始验证镜像..."

# 1. ais_bench 可用
echo "  [1/4] ais_bench 可用性..."
# ais_bench CLI 不支持 --version，用 --help 第一行即可
validation_output=$(docker run --rm "${image_name}" ais_bench --help 2>&1 | head -1) || {
    echo "错误：ais_bench 不可用"
    echo "${validation_output}"
    exit 1
}
echo "    ✓ ais_bench: ${validation_output}"

# 2. venv 完整性
echo "  [2/4] venv 完整性..."
# 单独运行 venv 检查，每项独立 + set +e + 单独捕获 stderr，
# 让具体哪一项失败、错误信息是什么都能完整打印，方便排障
docker run --rm "${image_name}" bash -c '
    set +e  # 单个子命令失败不中止整体
    failures=()

    # 2.1 三个 venv 目录 + python 可执行
    for v in harbor swebench swebench_pro; do
        if [ -d /opt/venvs/$v ] && [ -x /opt/venvs/$v/bin/python ]; then
            ver=$(/opt/venvs/$v/bin/python --version 2>&1)
            echo "    ✓ $v venv: ${ver}"
        else
            echo "    ✗ $v venv: 缺失或 python 不可执行"
            failures+=("$v venv 缺失或 python 不可执行")
        fi
    done

    # 2.2 三个 venv 都需有 ais_bench wrapper
    #     harbor 的 wrapper 让 subprocess 启动 harbor venv 的 python；
    #     swebench / swebench_pro 的 wrapper 让主进程 import venv-local 的 minisweagent
    for v in harbor swebench swebench_pro; do
        if [ -x /opt/venvs/$v/bin/ais_bench ]; then
            echo "    ✓ $v-ais_bench-wrapper: OK"
        else
            echo "    ✗ $v-ais_bench-wrapper: 缺失"
            failures+=("$v-ais_bench-wrapper 缺失")
        fi
    done

    # 2.3 swebench / swebench_pro 各自能 import 对应 fork 的 mini-swe-agent
    #     PyPI 包名 mini-swe-agent（带横杠），但 Python 模块名是 minisweagent（连写无下划线）
    #     单独捕获 stderr，方便看到 ModuleNotFoundError / 内部 ImportError 等真实错误
    for v in swebench swebench_pro; do
        out=$(/opt/venvs/$v/bin/python -c "import minisweagent; print(minisweagent.__file__)" 2>&1)
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "    ✓ $v minisweagent: ${out}"
        else
            echo "    ✗ $v minisweagent: import 失败（python exit=$rc）"
            echo "      错误信息:"
            echo "$out" | sed "s/^/        /"
            failures+=("$v minisweagent import 失败")
        fi
    done

    echo ""
    if [ ${#failures[@]} -gt 0 ]; then
        echo "  失败项汇总 (${#failures[@]}):"
        for f in "${failures[@]}"; do
            echo "    - $f"
        done
        echo ""
        echo "  排查建议:"
        echo "    1. 进入镜像手动检查:"
        echo "       docker run --rm -it ${image_name} bash"
        echo "       ls -la /opt/venvs/"
        echo "       /opt/venvs/swebench/bin/python --version"
        echo "       /opt/venvs/swebench/bin/python -c \"import minisweagent\""
        echo "    2. 若 import 报 ModuleNotFoundError：Dockerfile 步骤 8/9 的 pip install 未生效；"
        echo "       检查 gh-proxy.com / 华为云 pypi 镜像可达性；可加 --no-cache 强制重建步骤 8/9。"
        echo "    3. 若 import 报内部 ImportError：minisweagent 自身依赖在当前 venv 内不可见；"
        echo "       检查 mini-swe-agent setup.py 的 install_requires 在 venv 内是否齐全。"
        exit 1
    fi
' || {
    echo "错误：venv 检查失败（详见上方失败项汇总与排查建议）"
    exit 1
}
echo "    ✓ harbor / swebench / swebench_pro venv 都完整"

# 3. doctor.sh / packs 就位
echo "  [3/4] 脚本就位..."
scripts_output=$(docker run --rm "${image_name}" bash -c '
    [ -x /usr/local/bin/ais_bench_agent_doctor.sh ] && echo "doctor.sh: OK" || echo "doctor.sh: 缺失"
    ls /opt/agent-resources/packs/*.yaml 2>/dev/null | while read f; do
        echo "pack: $(basename $f .yaml)"
    done
') || {
    echo "错误：脚本检查失败"
    exit 1
}
echo "${scripts_output}" | sed 's/^/    /'

# 4. harbor 的 docker-compose-base.yaml 已 patch
echo "  [4/4] harbor compose 模板 patch 校验..."
patch_output=$(docker run --rm "${image_name}" bash -c '
    /opt/venvs/harbor/bin/python -c "
import harbor, os, yaml
p = os.path.dirname(harbor.__file__) + \"/environments/docker/docker-compose-base.yaml\"
cfg = yaml.safe_load(open(p))
svc = cfg.get(\"services\", {}).get(\"main\", {})
opts = svc.get(\"security_opt\", [])
print(\"seccomp=unconfined:\", \"seccomp=unconfined\" in opts)
print(\"network_mode=host:\", svc.get(\"network_mode\") == \"host\")
" 2>&1
') || {
    echo "错误：harbor compose 模板校验失败"
    echo "${patch_output}"
    exit 1
}
echo "${patch_output}" | sed 's/^/    /'
if ! echo "${patch_output}" | grep -q "seccomp=unconfined: True"; then
    echo "错误：harbor compose 模板未正确 patch seccomp"
    exit 1
fi
echo "    ✓ harbor compose patch OK"

echo "✓ 镜像验证通过"

# ============ 推送镜像 ============
if [ "$push" == "1" ]; then
    echo "推送镜像到远程仓库..."
    docker push "${image_name}"
    if [ $? -ne 0 ]; then
        echo "错误：镜像推送失败"
        exit 1
    fi
    echo "✓ 已推送：${image_name}"

    # 同时打 latest tag 并推送，供 bootstrap.sh 默认拉取
    latest_image_name="${target_hub_repo}:latest-${OS}-${py_version}-${arch}"
    echo "  打 latest tag: ${latest_image_name}"
    docker tag "${image_name}" "${latest_image_name}"
    docker push "${latest_image_name}"
    if [ $? -ne 0 ]; then
        echo "错误：latest tag 推送失败"
        exit 1
    fi
    echo "✓ 已推送 latest：${latest_image_name}"
fi

# ============ 多架构 manifest 合并 ============
if [ "$multi_arch" == "1" ]; then
    if [ "$push" != "1" ]; then
        echo "提示：多架构模式下未开启推送，manifest 合并需要已推送的镜像。跳过 manifest 合并。"
    else
        arch_image_amd64="${target_hub_repo}:${BASE_TAG}-${OS}-${py_version}-x86_64"
        arch_image_arm64="${target_hub_repo}:${BASE_TAG}-${OS}-${py_version}-aarch64"
        latest_arch_image_amd64="${target_hub_repo}:latest-${OS}-${py_version}-x86_64"
        latest_arch_image_arm64="${target_hub_repo}:latest-${OS}-${py_version}-aarch64"
        latest_manifest_image_name="${target_hub_repo}:latest-${OS}-${py_version}"

        echo "创建多架构 manifest list：${manifest_image_name}"
        echo "  - ${arch_image_amd64}"
        echo "  - ${arch_image_arm64}"

        docker buildx imagetools create \
            -t "${manifest_image_name}" \
            "${arch_image_amd64}" \
            "${arch_image_arm64}"

        if [ $? -ne 0 ]; then
            echo "错误：多架构 manifest 创建失败"
            exit 1
        fi
        echo "✓ 多架构 manifest list 已更新：${manifest_image_name}"

        echo "创建 latest 多架构 manifest list：${latest_manifest_image_name}"
        echo "  - ${latest_arch_image_amd64}"
        echo "  - ${latest_arch_image_arm64}"

        docker buildx imagetools create \
            -t "${latest_manifest_image_name}" \
            "${latest_arch_image_amd64}" \
            "${latest_arch_image_arm64}"

        if [ $? -ne 0 ]; then
            echo "错误：latest 多架构 manifest 创建失败"
            exit 1
        fi
        echo "✓ latest 多架构 manifest list 已更新：${latest_manifest_image_name}"
        echo "  docker buildx imagetools inspect ${manifest_image_name}"
        echo "  docker buildx imagetools inspect ${latest_manifest_image_name}"
    fi
fi

# ============ 打包离线包并上传 OBS ============
if [ "$upload" == "1" ]; then
    echo "打包离线包..."
    mkdir -p "${image_output_dir}"
    docker save "${image_name}" | gzip -9 > "${offline_pkg_full_path}"
    echo "  离线包已生成：${offline_pkg_full_path}"
    chmod 640 "${offline_pkg_full_path}"

    echo "上传离线包到 OBS 桶..."
    if [ ! -d "${obsutils_path}" ] || [ ! -x "${obsutils_path}/obsutil" ]; then
        echo "错误：obsutil 路径不存在或不可执行：${obsutils_path}"
        exit 1
    fi

    cd "${obsutils_path}"
    ./obsutil cp "${offline_pkg_full_path}" "obs://aisbench/images/agent/runtime/${offline_pkg_name}" -f

    if [ $? -eq 0 ]; then
        echo "✓ 离线包已上传：obs://aisbench/images/agent/runtime/${offline_pkg_name}"
    else
        echo "错误：OBS 桶上传失败"
        exit 1
    fi
fi

# ============ 完成 ============
echo ""
echo "============================================================"
echo "✓ 构建完成"
echo "============================================================"
echo "  镜像: ${image_name}"
if [ "$push" == "1" ]; then
    echo "  latest: ${target_hub_repo}:latest-${OS}-${py_version}-${arch}"
fi
if [ "$multi_arch" == "1" ] && [ "$push" == "1" ]; then
    echo "  manifest:         ${manifest_image_name}"
    echo "  latest manifest:  ${target_hub_repo}:latest-${OS}-${py_version}"
fi
if [ "$upload" == "1" ]; then
    echo "  离线包: obs://aisbench/images/agent/runtime/${offline_pkg_name}"
fi
echo ""
echo "用户使用："
echo "  curl -fsSL https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/ais_bench_agent_bootstrap.sh | bash"
echo "  # 或本地测试："
echo "  bash docker/agent_runtime/bootstrap.sh --mode A"
