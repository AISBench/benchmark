#!/bin/bash
# ============================================================================
# ais_bench_agent_entrypoint.sh — Agent Runtime Container Entrypoint (v3 A4)
#
# 职责（**仅 setup，不启 dockerd**）：
#   1. ARM64 host → 注册 binfmt x86_64 ELF → /usr/bin/qemu-x86_64-static
#   2. 写 /etc/docker/daemon.json（cgroupfs + vfs + 可选 registry-mirrors 从
#      AIS_BENCH_AGENT_REGISTRY_MIRROR 读，逗号分隔多 mirror）
#   3. ARM64 → export DOCKER_DEFAULT_PLATFORM=linux/amd64
#      （让 dockerd 选 amd64 manifest，否则 ARM64 host 拉 x86_64 image 会
#       "no matching manifest for linux/arm64"）
#   4. /opt/swebench/agent-patches 注入到 harbor 的 installed agents
#   5. 打印 banner
#   6. exec "$@"
#
# 设计要点：
#   - **幂等**：可重复执行（不会重复写 daemon.json / 重复注册 binfmt）
#   - **不启 dockerd**：dockerd 启动逻辑在 bootstrap.sh（line 343）的
#     `docker exec ... bash -c '... nohup dockerd ...'`，避免双启冲突
#   - **不引入新依赖**：用 python3 写 JSON（不依赖 jq）
#   - **fail-open**：写 daemon.json 失败、binfmt 注册失败均不阻塞（仅 warn）
# ============================================================================
set -u  # 注意：不加 -e，部分步骤 fail-open

echo "[A4 entrypoint] agent-runtime container starting (PID $$)"

ARCH="$(uname -m)"

# ---------- 1. ARM64 host: register binfmt ----------
if [ "${ARCH}" = "aarch64" ] || [ "${ARCH}" = "arm64" ]; then
    echo "[A4 entrypoint] ARM64 host detected, registering binfmt for x86_64 emulation"

    # binfmt_misc 必须先 mount（容器启动时可能没自动 mount）
    if ! mount | grep -q binfmt_misc; then
        mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc 2>/dev/null || \
            echo "[A4 entrypoint] [warn] could not mount binfmt_misc"
    fi

    # qemu-x86_64-static 由 A1 build-time 装（仅 TARGETARCH=arm64 时）
    if [ ! -x /usr/bin/qemu-x86_64-static ]; then
        echo "[A4 entrypoint] [warn] /usr/bin/qemu-x86_64-static not found"
        echo "[A4 entrypoint] [hint] image may have been built without TARGETARCH=arm64"
    elif [ -f /proc/sys/fs/binfmt_misc/qemu-x86_64 ]; then
        echo "[A4 entrypoint] binfmt already registered"
    else
        # 用 heredoc 写（printf 在 /proc/sys/fs/binfmt_misc/register 会 I/O error）
        cat > /proc/sys/fs/binfmt_misc/register << "BINFMT_EOF"
:qemu-x86_64:M::\x7f\x45\x4c\x46\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x3e\x00:\xff\xff\xff\xff\xff\xfe\xfe\xfc\xff\xff\xff\xff\xff\xff\xff\xff\xfe\xff\xff\xff:/usr/bin/qemu-x86_64-static:OCF
BINFMT_EOF
        echo "[A4 entrypoint] binfmt registered: $(cat /proc/sys/fs/binfmt_misc/qemu-x86_64 2>/dev/null | head -1)"
    fi

    # 让 dockerd 选 amd64 manifest（镜像 list 只有 amd64，arm64 不选会报错）
    export DOCKER_DEFAULT_PLATFORM=linux/amd64
fi

# ---------- 2. Write /etc/docker/daemon.json ----------
mkdir -p /etc/docker

write_daemon_json() {
    python3 << 'PYEOF'
import json
import os

mirror = os.environ.get('AIS_BENCH_AGENT_REGISTRY_MIRROR', '').strip()
cfg = {
    'exec-opts': ['native.cgroupdriver=cgroupfs'],
    'storage-driver': 'vfs',
}
if mirror:
    cfg['registry-mirrors'] = [u.strip() for u in mirror.split(',') if u.strip()]

target = '/etc/docker/daemon.json'

# 幂等：若已存在且内容一致，不重写（避免 dockerd reload）
if os.path.exists(target):
    try:
        with open(target) as f:
            existing = json.load(f)
        if existing == cfg:
            print(f"[A4 entrypoint] daemon.json already up-to-date")
            raise SystemExit(0)
    except (json.JSONDecodeError, OSError):
        pass  # 文件损坏或不存在 → 重写

with open(target, 'w') as f:
    json.dump(cfg, f, indent=2)
print(f"[A4 entrypoint] daemon.json written: {cfg}")
PYEOF
}
write_daemon_json || echo "[A4 entrypoint] [warn] daemon.json write failed (continuing)"

# ---------- 3. ARM64: DOCKER_DEFAULT_PLATFORM 已 export ----------
# 已在上方 export；这里仅 echo 一下方便用户 verify
if [ "${ARCH}" = "aarch64" ] || [ "${ARCH}" = "arm64" ]; then
    echo "[A4 entrypoint] DOCKER_DEFAULT_PLATFORM=${DOCKER_DEFAULT_PLATFORM:-<unset>}"
fi

# ---------- 4. Inject /opt/swebench/agent-patches ----------
if [ -d /opt/swebench/agent-patches ]; then
    HARBOR_INSTALLED="$(python3 -c "
try:
    import harbor, os
    print(os.path.join(os.path.dirname(harbor.__file__), 'agents', 'installed'))
except ImportError:
    print('')
" 2>/dev/null)"

    if [ -n "${HARBOR_INSTALLED}" ] && [ -d "${HARBOR_INSTALLED}" ]; then
        echo "[A4 entrypoint] Injecting agent patches into ${HARBOR_INSTALLED}..."
        patched=0
        skipped=0
        for f in /opt/swebench/agent-patches/*.py; do
            [ -f "$f" ] || continue
            name="$(basename "$f")"
            if [ -f "${HARBOR_INSTALLED}/${name}" ]; then
                cp "$f" "${HARBOR_INSTALLED}/${name}"
                echo "  patched: ${name}"
                patched=$((patched + 1))
            else
                skipped=$((skipped + 1))
            fi
        done
        echo "[A4 entrypoint] agent patches done: ${patched} patched, ${skipped} skipped (no matching installed agent)"
    else
        echo "[A4 entrypoint] [warn] harbor installed agents dir not found: ${HARBOR_INSTALLED:-<none>}"
    fi
fi

# ---------- 5. Banner ----------
echo "[A4 entrypoint] ENV summary:"
echo "  ARCH=${ARCH}"
echo "  DOCKER_DEFAULT_PLATFORM=${DOCKER_DEFAULT_PLATFORM:-<unset>}"
echo "  AIS_BENCH_AGENT_REGISTRY_MIRROR=${AIS_BENCH_AGENT_REGISTRY_MIRROR:-<unset>}"
HARBOR_VER="$(harbor --version 2>&1 | head -1 || echo '<unavailable>')"
echo "  harbor: ${HARBOR_VER}"

# ---------- 6. exec ----------
echo "[A4 entrypoint] setup done; exec \$@"
exec "$@"
