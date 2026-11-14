"""测评集自检（mutation test）：故意改坏 SKILL，验证对应用例会失败。

没有这一步，「全部通过」可能只是因为断言太松。每次改动 SKILL 后也应重跑它。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skill_eval.harness import Runner, Suite  # noqa: E402

# 每个 mutation：编号、说明、[(相对文件, 原文, 替换或 None=删除)], 期望失败的用例
MUTATIONS = [
    (
        "M1",
        "稳态正式请求数从 10× 退回 4×",
        [("scripts/task_model.py",
          "'formal_requests':case['concurrency']*(10 if case.get('steady_state',False) else 4),'dataset_id'",
          "'formal_requests':case['concurrency']*(4 if case.get('steady_state',False) else 4),'dataset_id'")],
        ["C06"],
    ),
    (
        "M2",
        "去掉 ready 任务的 unresolved 拦截",
        [("scripts/task_model.py", "if spec['state']=='ready' and spec['unresolved']:", "if False:")],
        ["C02"],
    ),
    (
        "M3",
        "关闭开关时仍然注入 DEV_MODE（静默改参）",
        [("scripts/task_model.py", "if e['enable_cache_reset_api']:", "if True:")],
        ["C08"],
    ),
    (
        "M4",
        "删除参考脚本环境与绑定的一致性校验（生成期 + 编译期两处）",
        [("scripts/run.py",
          "if key in expected and str(expand(value,context))!=str(expected[key]):", "if False:"),
         ("scripts/topology.py",
          "if key in env and '{{' not in str(env[key]) and env[key]!=value:", "if False:")],
        ["C10"],
    ),
    (
        "M5",
        "代理入口标记消失（依赖检查会读不到命令）",
        [("scripts/run.py", "('# @proxy-command\\n' if item['role']=='proxy' else '')",
          "('' if item['role']=='proxy' else '')")],
        ["C16"],
    ),
    (
        "M6",
        "模型目录挂载丢掉只读标记",
        [("scripts/run.py",
          "argv += ['-v', mount['source']+':'+mount['target']+(':ro' if mount.get('read_only') else '')]",
          "argv += ['-v', mount['source']+':'+mount['target']]")],
        ["C15"],
    ),
    (
        "M7",
        "AISBench retry 语义被改成 3（工具内部重试）",
        [("scripts/aisbench_cli.py", "retry=1,", "retry=3,")],
        ["C17"],
    ),
    (
        "M8",
        "挂载模板参考文档被删除，SKILL.md 链接失效",
        [("references/mount-profiles.md", None, None)],
        ["S02"],
    ),
]


def apply(edits, target: Path) -> str | None:
    """返回错误说明；None 表示应用成功。"""
    for relative, old, new in edits:
        path = target / relative
        if old is None:
            path.unlink()
            continue
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            return f"{relative} 原文匹配 {count} 次"
        path.write_text(text.replace(old, new), encoding="utf-8")
    return None


def run(suite_dir: Path, skill_dir: Path, work: Path, python: str, deps: str | None, md: Path | None = None) -> int:
    suite = Suite(suite_dir, skill_dir, deps)
    by_id = {case["id"]: case for case in suite.cases}
    failures = 0
    rows = []
    print(f"{'编号':<4}{'改动':<50}{'期望失败':<10}{'实际':<26}结果")
    for name, description, edits, expected in MUTATIONS:
        target = work / "mutants" / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(skill_dir, target)
        error = apply(edits, target)
        if error:
            print(f"{name:<4}{description:<48}{','.join(expected):<10}{error}")
            rows.append((name, description, expected, error, "跳过"))
            failures += 1
            continue

        mutant_suite = Suite(suite_dir, target.resolve(), deps)
        runner = Runner(mutant_suite, work / "runs" / name, python)
        results = {case_id: runner.run_case(by_id[case_id]) for case_id in expected}
        detected = [cid for cid, res in results.items() if not res.passed]
        ok = set(detected) == set(expected)
        failures += 0 if ok else 1
        detail = ", ".join(f"{cid}={'FAIL' if not r.passed else 'PASS'}" for cid, r in results.items())
        print(f"{name:<4}{description:<48}{','.join(expected):<10}{detail:<26}{'检出' if ok else '未检出（断言过松）'}")
        rows.append((name, description, expected, detail, "检出" if ok else "未检出"))

    detected_count = len(MUTATIONS) - failures
    print()
    print(f"mutation 检出：{detected_count}/{len(MUTATIONS)}")
    if md:
        md.parent.mkdir(parents=True, exist_ok=True)
        try:
            shown = Path(skill_dir).resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            shown = Path(skill_dir).name
        lines = [
            "# 测评集自检（mutation test）",
            "",
            f"被测 SKILL：`{shown}`；执行解释器：`{Path(python).name}`。",
            "",
            "做法：把 SKILL 复制一份，逐条注入「真实可能发生的退化」，再跑对应用例。",
            "用例必须由 PASS 变成 FAIL，否则说明断言太松，测评集本身不可信。",
            "",
            f"**检出 {detected_count}/{len(MUTATIONS)}。**",
            "",
            "| 编号 | 注入的退化 | 期望失败用例 | 实际 | 结论 |",
            "|---|---|---|---|---|",
        ]
        for name, description, expected, detail, verdict in rows:
            lines.append(f"| {name} | {description} | {','.join(expected)} | {detail} | {verdict} |")
        lines += ["", "复现：`python tools/mutation_check.py --python <py> --deps <deps> --md reports/mutation.md`", ""]
        md.write_text("\n".join(lines), encoding="utf-8")
        print(f"report : {md}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("suites/ascend-llm-deploy-eval"))
    parser.add_argument("--skill-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--deps", default=None)
    parser.add_argument("--work", type=Path, default=Path(".eval-work/mutation"))
    parser.add_argument("--md", type=Path, default=None)
    args = parser.parse_args()

    suite_dir = args.suite.resolve()
    skill_dir = args.skill_dir or Path(
        json.loads((suite_dir / "suite.json").read_text(encoding="utf-8-sig"))["skill"]["local_dir"]
    )
    skill_dir = (skill_dir if skill_dir.is_absolute() else (suite_dir / skill_dir)).resolve()
    raise SystemExit(run(suite_dir, skill_dir, args.work, args.python, args.deps, args.md))
