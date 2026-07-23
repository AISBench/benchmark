#!/usr/bin/env python3
# Patch harbor 0.6.1 的 docker-compose-base.yaml：
#   1. 给 main service 追加 security_opt: ["seccomp=unconfined"]
#   2. 把 main service 的 network_mode 设为 "host"
# 两个改动都是 harbor_bench.md 第 2.2 节所要求的 agent-side patch。
#
# 用法：harbor_compose_patch.py <compose_base.yaml 路径>
#
# 与 Dockerfile.agent-runtime 的 RUN 段配合：从 build context COPY 进镜像后调用。
# 用独立脚本而不是 RUN 内联 heredoc，是为了绕开 BuildKit 对 RUN 内嵌多行引号字符串的
# 解析限制（Dockerfile 1.0 起就不支持引号内裸换行 + 续行符的混合写法）。

import sys
import yaml


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: harbor_compose_patch.py <path/to/docker-compose-base.yaml>",
              file=sys.stderr)
        return 2

    path = sys.argv[1]

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    svc = cfg.setdefault("services", {}).setdefault("main", {})
    opts = svc.setdefault("security_opt", [])
    if "seccomp=unconfined" not in opts:
        opts.append("seccomp=unconfined")

    # harbor_bench.md 第 2.2 节：agent 测评需要 main 直连宿主网络
    svc["network_mode"] = "host"

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print("patched:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
