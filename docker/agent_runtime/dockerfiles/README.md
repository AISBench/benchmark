# L1/L2 baked image Jinja2 模板

mini_matrix 5 层 DinD 的 L1 (case-base) / L2 (agent) baked image 构建模板。

模板语义：
- `Dockerfile.l1-base.j2` — FROM 官方 SWE-bench prebuilt → WORKDIR /testbed + mkdir /logs
- `Dockerfile.l2-agent-aider.j2` — FROM L1 → pip install aider-chat
- `Dockerfile.l2-agent-msa.j2`   — FROM L1 → pip install mini-swe-agent[litellm_proxy]
- `Dockerfile.l2-agent-oh.j2`    — FROM L1 → openhands-sdk 禁用占位（QEMU 不稳定，详见注释）
- `Dockerfile.l2-agent-qwen.j2`  — FROM L1 → Node 22 + npm install @qwen-code/qwen-code

build 入口：`build_l2_baked_image.sh`（v3 D1 提交）。模板在运行时容器内也可见，
镜像构建时 COPY 进 /opt/swebench/dockerfiles/，runtime 容器内用户可一键 build。

build 产物 tag 规范（与 mini_matrix 一致）：
- L1: `swebench/<dataset>-<case>-base:latest`
- L2: `swebench/<dataset>-<case>-with-<agent>:latest`

trial 容器使用 L2 baked image 时，harbor CLI 通过 `--l2-image <tag>` 跳过 trial 内 install
（节省 pip/npm 网络 + QEMU 时间）。