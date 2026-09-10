#!/usr/bin/env python3
# Patch harbor 的 docker compose 模板（适配 harbor 0.20.0 多模板布局）：
#   1. 给 services.main 追加 security_opt: ["seccomp=unconfined"]
#      openEuler/RHEL 宿主默认 seccomp profile 拦截 clone3，导致 trial 容器中
#      OpenBLAS/NumPy 初始化线程报 pthread_create failed: Operation not permitted
#   2. 给 services.main 设置 network_mode: "host"（仅当其未显式声明 network_mode 时，
#      例如 docker-compose-no-network.yaml 的 main 已是 none，则保留不动）
#
# harbor 0.20.0 的 compose 模板（python3.12 site-packages/harbor/environments/docker/）：
#   docker-compose-prebuilt.yaml   main 用预构建镜像（trial 默认路径之一）
#   docker-compose-build.yaml      main 从 Dockerfile 构建（trial 默认路径之一）
#   其余（no-network / egress-control / windows-keepalive）按上面规则自然跳过或仅加 seccomp
# （harbor 0.6.1 的旧布局是单文件 docker-compose-base.yaml，本脚本同样兼容）
#
# 用法：harbor_compose_patch.py <compose.yaml> [更多 compose.yaml ...]
#
# 与 Dockerfile.agent-runtime 的 RUN 段配合：从 build context COPY 进镜像后调用。
# 用独立脚本而不是 RUN 内联 heredoc，是为了绕开 BuildKit 对 RUN 内嵌多行引号字符串的
# 解析限制（Dockerfile 1.0 起就不支持引号内裸换行 + 续行符的混合写法）。

import sys

import yaml


def patch_file(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    services = cfg.setdefault("services", {})
    if "main" not in services:
        print(f"skip (no main service): {path}")
        return 0

    svc = services["main"]

    opts = svc.setdefault("security_opt", [])
    if "seccomp=unconfined" not in opts:
        opts.append("seccomp=unconfined")

    # 尊重显式 network_mode（如 no-network 模板的 none），其余补 host 直连
    if not svc.get("network_mode"):
        svc["network_mode"] = "host"

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print("patched:", path)
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: harbor_compose_patch.py <path/to/compose.yaml> [more.yaml ...]",
              file=sys.stderr)
        return 2
    for path in sys.argv[1:]:
        ret = patch_file(path)
        if ret != 0:
            return ret
    return 0


if __name__ == "__main__":
    sys.exit(main())
