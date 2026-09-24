#!/usr/bin/env python3
"""
filter_matrix.py — 根据 --datasets / --agents 过滤 matrix.yaml,生成子集 yaml。

不修改 matrix.yaml,只读它、过滤、写到 --output。
"""
import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML not installed. pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Source matrix.yaml")
    p.add_argument("--output", required=True, help="Filtered output yaml")
    p.add_argument("--datasets", default="", help="Comma-separated dataset path basenames to keep")
    p.add_argument("--agents", default="", help="Comma-separated agent names to keep")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.input).read_text())

    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(",") if d.strip()}
        cfg["datasets"] = [
            ds for ds in cfg.get("datasets", [])
            if Path(ds["path"]).name in keep
        ]

    if args.agents:
        keep = {a.strip() for a in args.agents.split(",") if a.strip()}
        cfg["agents"] = [
            a for a in cfg.get("agents", [])
            if a["name"] in keep
        ]

    Path(args.output).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    n_d = len(cfg.get("datasets", []))
    n_a = len(cfg.get("agents", []))
    print(f"[filter_matrix] {n_d} datasets × {n_a} agents = {n_d * n_a} trials")
    print(f"[filter_matrix] Written: {args.output}")


if __name__ == "__main__":
    main()