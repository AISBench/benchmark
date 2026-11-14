---
name: demo-port-planner
description: 演示用最小 SKILL：读取集群规划 JSON，校验节点数与每节点端口数，输出总端口数。仅用于验证测评框架可迁移到其他仓。
---

# Demo Port Planner

## 用法

```sh
python scripts/check.py --plan plan.json
```

`plan.json` 形如 `{"nodes": 2, "ports_per_node": 8}`。

- 合法时打印 `PLAN OK nodes=<N> ports=<N*P>`，退出码 0。
- 非法时在 stderr 打印 `PLAN INVALID: <原因>`，退出码 2。

## 约束

- `nodes` 必须是 >= 1 的整数。
- `ports_per_node` 必须在 1..64 之间。
