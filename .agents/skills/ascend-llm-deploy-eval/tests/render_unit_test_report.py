"""Render the current validation record directly from machine-readable reports."""
import argparse
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    reports = [json.loads(p.read_text(encoding='utf-8')) for p in args.reports]
    hashes = {r['source']['tested_source_sha256'] for r in reports}
    if len(hashes) != 1:
        raise ValueError('Reports must cover identical tested source')
    lines = ['# 当前单元测试验证记录', '',
             '由 tests/render_unit_test_report.py 根据原始 JSON 报告生成；不要手改计数。', '',
             '测试内容 SHA256：`'+next(iter(hashes))+'`。文本换行统一为 LF 后计算，避免 Git 的 CRLF/LF 转换产生伪差异。摘要覆盖 scripts、tests、schemas、examples 的 Python/JSON 文件及 requirements.txt；不含本文和其他说明文档。', '',
             '| UTC 时间 | 系统 | Python | 总数 | 通过 | 失败 | 错误 | 跳过 | 结果 |',
             '|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in reports:
        lines.append(f"| {r['recorded_at_utc']} | {r['system']} | {r['python']} | {r['tests_run']} | {r['passed']} | {r['failures']} | {r['errors']} | {len(r['skipped'])} | {'PASS' if r['successful'] else 'FAIL'} |")
    lines += ['', '## 平台及跳过原因', '']
    for r in reports:
        lines += ['### '+r['system'], '', '`'+r['platform']+'`', '']
        if not r['skipped']:
            lines += ['无跳过。', '']
        for skip in r['skipped']:
            lines.append('- `'+skip['test']+'`：'+skip['reason'].replace('\n', ' '))
        lines.append('')
    lines += ['本报告仅记录离线回归，不启动真实容器或模型，也不证明真实硬件性能；是否需要实机复验须按具体修改影响另行评估。', '']
    args.output.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
