# flake8: noqa
# yapf: disable
import csv
import os
import os.path as osp
from datetime import datetime

import mmengine
import tabulate

from ais_bench.benchmark.utils.logging.logger import AISLogger


class HarborSummarizer:
    """Harbor agent benchmark summarizer: prints one table, writes one csv.

    Columns are data-driven via ``COLUMNS`` (key = short english header). To
    add a column, append its key to ``COLUMNS`` and fill it in ``_build_row``.
    Reporting columns are selected by configuration, never by result contents.
    """

    COLUMNS = [
        'agent', 'model_name', 'dataset', 'avg_score', 'correct', 'wrong',
        'exception',
    ]

    NORMALIZED_COLUMNS = ["accuracy", "integrity", "acceptance", "evidence"]

    def __init__(self, config) -> None:
        self.cfg = config
        self.logger = AISLogger()
        self.model_cfgs = config['models']
        self.dataset_cfgs = config['datasets']
        self.work_dir = config['work_dir']
        self.normalized_columns = config.get('normalized_columns', any(
            d.get('reporting') is not None for d in self.dataset_cfgs
        ))

    def summarize(self, time_str=None):
        rows = []
        for model in self.model_cfgs:
            for dataset in self.dataset_cfgs:
                row = self._build_row(model, dataset['abbr'])
                if row:
                    rows.append(row)
        if not rows:
            self.logger.warning('No harbor results found to summarize.')
            return

        columns = list(self.COLUMNS)
        if self.normalized_columns:
            columns.extend(self.NORMALIZED_COLUMNS)
        table = [columns] + [[str('-' if r.get(c) is None else r.get(c, '-')) for c in columns]
                                        for r in rows]
        print(tabulate.tabulate(table, headers='firstrow', tablefmt='grid'))

        time_str = time_str or datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_dir = osp.join(self.work_dir, 'summary')
        mmengine.mkdir_or_exist(summary_dir)
        csv_path = osp.join(summary_dir, f'summary_{time_str}.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for r in rows:
                writer.writerow([r.get(c, '-') for c in columns])
        self.logger.info(f'write summary csv to {osp.abspath(csv_path)}')

    def _build_row(self, model_cfg, dataset_abbr):
        model_abbr = model_cfg['abbr']
        result_path = osp.join(self.work_dir, 'results', model_abbr,
                               f'{dataset_abbr}.json')
        if not osp.exists(result_path):
            return None
        result = mmengine.load(result_path)
        dist = result.get('reward_distribution') or []
        correct = sum(d.get('count', 0) for d in dist if d.get('score', 0) >= 1.0)
        wrong = sum(d.get('count', 0) for d in dist
                    if 0 <= d.get('score', 0) < 1.0)
        model_names = model_cfg.get('model_names') or []
        row = {
            'agent': model_cfg.get('agent_name', model_abbr),
            'model_name': ', '.join(model_names) or '-',
            'dataset': dataset_abbr,
            'avg_score': result.get('avg_score', '-'),
            'correct': correct,
            'wrong': wrong,
            'exception': result.get('n_errors', 0),
        }
        if 'normalized_result' in result or 'reporting_error' in result:
            from pathlib import Path
            from ais_bench.benchmark.utils.harbor_reporting import read_completed_report

            if 'reward_distribution' not in result:
                row.update(correct='-', wrong='-')
            if 'n_errors' not in result:
                row['exception'] = '-'
            reference = result.get('normalized_result')
            row.update(accuracy=None, integrity='invalid', acceptance='blocked',
                       evidence=reference)
            try:
                if 'reporting_error' in result:
                    raise ValueError(result['reporting_error'])
                if not isinstance(reference, (str, os.PathLike)) or not reference:
                    raise ValueError('normalized_result must be a nonempty path')
                path = Path(result_path).parent / reference
                row['evidence'] = osp.relpath(path.resolve(), Path(self.work_dir).resolve())
                report = read_completed_report(path)
                row.update(accuracy=report['metrics']['accuracy'],
                           integrity=report['integrity']['status'],
                           acceptance=report['acceptance']['status'])
            except (ValueError, OSError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
                self.logger.warning(f'Cannot consume normalized report {reference!r}: {exc}')
        return row
