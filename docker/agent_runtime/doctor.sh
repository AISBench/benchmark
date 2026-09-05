#!/bin/bash
# ais_bench_agent_doctor.sh
#
# 验证指定 pack 的 runtime 是否就绪（仅 runtime，不含数据集 / case 镜像）
#
# 本脚本在 runtime 容器内执行。它是"runtime 就绪"的可执行证明，失败时给出
# 精确修复指引，不让用户盲调。
#
# 边界（刻意不做的事）：
#   - 不准备 case 镜像（用户负责 pull / load）
#   - 不下载 / 校验数据集（用户在 bootstrap.sh --datasets 时已挂载）
#   - 不模拟跑评测
#   - 不区分 benchmark 的不同子集/采样；医生只看 runtime 配齐了没
#
# 用法:
#   ais_bench_agent_doctor.sh <pack>
#
# 参数:
#   pack              pack 名称，如 harbor
#   -h, --help        显示帮助
#
# 验证内容（一次性 L1 静态检查，秒级）:
#   - docker daemon 可用
#   - venv 完整性（pack.runtime_venv 对应路径存在 + python 可执行）
#   - pack.yaml 一致性（pack.native_config 存在）
#   - 磁盘/内存余量

set -e

# ============ 默认配置 ============
PACKS_ROOT="${AGENT_PACKS_ROOT:-/opt/agent-resources/packs}"
PACK=""

# ============ 工具函数 ============
log()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "[错误] $*" >&2; exit 1; }

yq() {
    python3.12 -c "
import yaml, sys
try:
    c = yaml.safe_load(open('$PACK_FILE'))
except FileNotFoundError:
    sys.exit('pack 文件不存在: $PACK_FILE')
$1
"
}

usage() {
    cat <<EOF
用法: ais_bench_agent_doctor.sh <pack>

  pack              pack 名称，如 harbor

验证内容（一次性 L1 静态检查，秒级）:
  - docker daemon / venv 完整性 / pack.yaml 一致性 / 磁盘内存

可用 pack:
$(ls "${PACKS_ROOT}"/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | sed 's/^/  /')

示例:
  ais_bench_agent_doctor.sh harbor
EOF
    exit 1
}

# ============ 参数解析 ============
[ $# -lt 1 ] && usage
PACK="$1"; shift
[ "$PACK" = "-h" ] || [ "$PACK" = "--help" ] && usage
[ $# -gt 0 ] && { echo "未知参数: $1" >&2; usage; }

PACK_FILE="${PACKS_ROOT}/${PACK}.yaml"
[ ! -f "$PACK_FILE" ] && fail "pack 不存在: ${PACK}（查找路径 ${PACKS_ROOT}）"

VENV=$(yq "print(c.get('runtime_venv',''))")
NATIVE_CONFIG=$(yq "print(c.get('native_config',''))")
NATIVE_DOC=$(yq "print(c.get('native_doc',''))")

echo "============================================================"
echo " AISBench Agent runtime 验证: ${PACK}"
echo "============================================================"

# ============ L1 静态自检 ============
echo ""
echo "[L1] 静态自检"
L1_FAIL=0

echo "  [1/3] docker daemon..."
if docker info >/dev/null 2>&1; then
    echo "    ✓ $(docker --version)"
else
    echo "    ✗ docker daemon 不可用"
    L1_FAIL=1
fi

echo "  [2/3] venv 完整性: ${VENV}"
VENV_PATH="/opt/venvs/${VENV}"
if [ -n "$VENV" ] && [ -d "$VENV_PATH" ] && [ -x "$VENV_PATH/bin/python" ]; then
    echo "    ✓ ${VENV} python: $($VENV_PATH/bin/python --version 2>&1)"
else
    echo "    ✗ venv 不存在: $VENV_PATH"
    L1_FAIL=1
fi

echo "  [3/3] pack.yaml 一致性..."
if [ -n "$NATIVE_CONFIG" ] && [ -f "$NATIVE_CONFIG" ]; then
    echo "    ✓ native_config: $NATIVE_CONFIG"
else
    echo "    ✗ native_config 不存在: ${NATIVE_CONFIG:-<空>}"
    L1_FAIL=1
fi

echo ""
echo "  [资源] 磁盘/内存余量..."
AVAIL_GB=$(df -BG /var/lib/docker 2>/dev/null | awk 'NR==2{print $4}' | tr -d G || echo 0)
if [ "$AVAIL_GB" -gt 20 ] 2>/dev/null; then
    echo "    ✓ docker 数据盘剩余 ${AVAIL_GB}GB"
else
    echo "    ⚠ docker 数据盘仅剩 ${AVAIL_GB}GB（建议 ≥20GB）"
fi
MEM_GB=$(free -g 2>/dev/null | awk '/Mem:/{print $7}' || echo 0)
if [ "$MEM_GB" -gt 4 ] 2>/dev/null; then
    echo "    ✓ 可用内存 ${MEM_GB}GB"
else
    echo "    ⚠ 可用内存 ${MEM_GB}GB（建议 ≥4GB）"
fi

# ============ 总结 ============
echo ""
echo "============================================================"
if [ "$L1_FAIL" = "1" ]; then
    echo " ✗ ${PACK} runtime 验证未通过"
    echo "============================================================"
    echo ""
    echo "修复指引:"
    echo "  - docker daemon 不可用: 检查容器内 dockerd 是否启动（模式 A）；或宿主 docker.sock 是否挂载（模式 B）"
    echo "  - venv 缺失: runtime 镜像损坏，重新拉取或重建容器"
    echo "    docker pull ghcr.io/aisbench/agent-runtime:latest-ubuntu24.04-py312-\${ARCH}"
    echo "  - native_config 不存在: 检查 /benchmark 路径下是否有 ais_bench 仓库"
    exit 1
fi
echo " ✓ ${PACK} runtime 就绪"
echo "============================================================"
cat <<EOF

接下来你需要做的事（都是你的事，doctor 不替你做）:
  1. 数据集：bootstrap.sh --datasets 已挂载，AISBENCH_AGENT_DATASET_PATH 已就绪。
     原生配置的 path 字段会自动从该 env var 读，无需改。
  2. case 沙箱镜像：bootstrap.sh --case-tar 已加载到容器内 docker daemon，
     或由你自行 docker pull / docker load。缺镜像时原生 ais_bench 会报错。
     详细获取方式见各 benchmark 文档：
       ${NATIVE_DOC:-<未在 pack.yaml 中指定 native_doc>}
  3. 修改原生配置中的 model_names / api_base
  4. 跑真实测评

  agent_env ${VENV}
  vim ${NATIVE_CONFIG}             # 仅改 model_names / api_base；path 已自动
  ais_bench ${NATIVE_CONFIG} --debug

断点续跑:
  ais_bench ${NATIVE_CONFIG} --debug --reuse <timestamp>

切换不同的数据集 / case 集:
  物理机（销毁旧容器 + 重新起）:
    docker rm -f ais_bench_agent
    bash bootstrap.sh --datasets <新数据集路径> --case-tar <新 case tar>
  容器内：不再跑 doctor，直接跑 ais_bench
EOF
