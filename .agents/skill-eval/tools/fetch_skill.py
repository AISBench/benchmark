"""按 commit 拉取被测 SKILL 目录（只取 skill 子树，避免整仓克隆）。

用法：
    python tools/fetch_skill.py --repo https://github.com/AISBench/benchmark.git \
        --path .agents/skills/ascend-llm-deploy-eval --dest upstream [--commit <sha>]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(*command: str, cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--path", required=True, help="仓内 skill 目录，如 .agents/skills/xxx")
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--commit", default="", help="固定 commit；留空则取默认分支最新")
    args = parser.parse_args()

    dest = args.dest.resolve()
    if not (dest / ".git").exists():
        if dest.exists() and any(dest.iterdir()):
            print(f"{dest} 已存在且不是 git 仓库，请换目录或清空", file=sys.stderr)
            return 2
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", args.repo, str(dest))
    run("git", "sparse-checkout", "set", args.path, cwd=dest)
    if args.commit:
        run("git", "fetch", "--depth", "1", "origin", args.commit, cwd=dest)
        run("git", "checkout", args.commit, cwd=dest)

    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    skill_dir = dest / Path(args.path)
    if not (skill_dir / "SKILL.md").is_file():
        print(f"未找到 {skill_dir}/SKILL.md", file=sys.stderr)
        return 2
    print(f"skill : {skill_dir}")
    print(f"commit: {head}")
    print("把该 commit 填进 suite.json 的 skill.commit，测评结果即可复现到具体版本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
