"""演示用 SKILL 脚本：严格的参数校验 + 明确退出码。"""
import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    try:
        plan = json.loads(open(args.plan, encoding="utf-8").read())
    except Exception as exc:
        print(f"PLAN INVALID: cannot read plan: {exc}", file=sys.stderr)
        return 2

    nodes = plan.get("nodes")
    ports = plan.get("ports_per_node")
    if not isinstance(nodes, int) or nodes < 1:
        print("PLAN INVALID: nodes must be >= 1", file=sys.stderr)
        return 2
    if not isinstance(ports, int) or not 1 <= ports <= 64:
        print("PLAN INVALID: ports_per_node must be within 1..64", file=sys.stderr)
        return 2

    print(f"PLAN OK nodes={nodes} ports={nodes * ports}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
