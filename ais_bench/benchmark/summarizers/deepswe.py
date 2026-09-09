# flake8: noqa
# yapf: disable
"""DeepSWE 专属 Summarizer。

与通用 HarborSummarizer 的区别：
- 直接扫描 trial 级 result.json，按 DeepSWE 口径汇总指标：
  resolved 率（reward>=1）、avg F2P / P2P / partial（verifier rewards 分项）、
  token 用量、耗时、异常分布。

输入目录结构:
    <work_dir>/results/<agent>/<dataset>.json                    汇总分数
    <work_dir>/results/<agent>/<dataset>/details/<trial>/result.json   trial 级结果
"""
import csv
import os
import os.path as osp
from datetime import datetime

import mmengine
import tabulate

from ais_bench.benchmark.utils.logging.logger import AISLogger


def _parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except ValueError:
        return None


class DeepSWESummarizer:
    """DeepSWE agent benchmark summarizer.

    正确解析并显示 result.json 中的 metrics 指标：
    - reward 字段（0/1）作为 trial 得分，resolved 率 = resolved / n_trials
    - f2p / p2p / partial 作为分项指标取均值（仅在含 verifier 结果的 trial 上）
    - token / 耗时 / 异常为累加或均值统计
    """

    COLUMNS = [
        'agent', 'model_name', 'dataset',
        'n_trials', 'resolved', 'resolved_rate',
        'avg_f2p', 'avg_p2p', 'avg_partial',
        'exceptions', 'input_tokens', 'output_tokens', 'avg_time_sec',
    ]

    def __init__(self, config) -> None:
        self.cfg = config
        self.logger = AISLogger()
        self.model_cfgs = config['models']
        self.dataset_cfgs = config['datasets']
        self.work_dir = config['work_dir']

    def summarize(self, time_str=None):
        rows = []
        for model in self.model_cfgs:
            for dataset in self.dataset_cfgs:
                row = self._build_row(model, dataset['abbr'])
                if row:
                    rows.append(row)
        if not rows:
            self.logger.warning('No deep-swe results found to summarize.')
            return

        table = [list(self.COLUMNS)] + [[str(r.get(c, '-')) for c in self.COLUMNS]
                                        for r in rows]
        print(tabulate.tabulate(table, headers='firstrow', tablefmt='grid'))

        time_str = time_str or datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_dir = osp.join(self.work_dir, 'summary')
        mmengine.mkdir_or_exist(summary_dir)
        csv_path = osp.join(summary_dir, f'summary_deepswe_{time_str}.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            for r in rows:
                writer.writerow([r.get(c, '-') for c in self.COLUMNS])
        self.logger.info(f'write summary csv to {osp.abspath(csv_path)}')


    def _collect_trials(self, model_abbr, dataset_abbr):
        """扫描 trial 级 result.json，返回原始 trial 指标列表。"""
        details_dir = osp.join(self.work_dir, 'results', model_abbr,
                               dataset_abbr, 'details')
        if not osp.isdir(details_dir):
            return None
        trials = []
        for name in sorted(os.listdir(details_dir)):
            tdir = osp.join(details_dir, name)
            if not osp.isdir(tdir):
                continue
            result_path = osp.join(tdir, 'result.json')
            if not osp.exists(result_path):
                continue
            trials.append(mmengine.load(result_path))
        return trials

    def _build_row(self, model_cfg, dataset_abbr):
        model_abbr = model_cfg['abbr']
        trials = self._collect_trials(model_abbr, dataset_abbr)
        if trials is None:
            return None

        n_trials = len(trials)
        resolved = 0
        exceptions = 0
        f2ps, p2ps, partials = [], [], []
        tok_in = tok_out = 0
        durations = []
        for tr in trials:
            exc = tr.get('exception_info')
            vr = (tr.get('verifier_result') or {})
            rewards = vr.get('rewards') or {}
            if exc is not None:
                exceptions += 1
                continue
            if rewards.get('reward', 0) >= 1:
                resolved += 1
            if 'f2p' in rewards:
                f2ps.append(rewards['f2p'])
            if 'p2p' in rewards:
                p2ps.append(rewards['p2p'])
            if 'partial' in rewards:
                partials.append(rewards['partial'])
            ar = tr.get('agent_result') or {}
            tok_in += ar.get('n_input_tokens') or 0
            tok_out += ar.get('n_output_tokens') or 0
            ta = _parse_ts(tr.get('started_at'))
            tb = _parse_ts(tr.get('finished_at'))
            if ta and tb:
                durations.append((tb - ta).total_seconds())

        def _avg(xs):
            return round(sum(xs) / len(xs), 4) if xs else '-'

        model_names = model_cfg.get('model_names') or []
        return {
            'agent': model_cfg.get('agent_name', model_abbr),
            'model_name': ', '.join(model_names) or '-',
            'dataset': dataset_abbr,
            'n_trials': n_trials,
            'resolved': resolved,
            'resolved_rate': round(resolved / n_trials, 4) if n_trials else '-',
            'avg_f2p': _avg(f2ps),
            'avg_p2p': _avg(p2ps),
            'avg_partial': _avg(partials),
            'exceptions': exceptions,
            'input_tokens': tok_in,
            'output_tokens': tok_out,
            'avg_time_sec': round(sum(durations) / len(durations), 1)
            if durations else '-',
        }
