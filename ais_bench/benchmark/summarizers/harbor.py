# flake8: noqa
# yapf: disable
import csv
import os.path as osp
from datetime import datetime

import mmengine
import tabulate

from ais_bench.benchmark.utils.logging.logger import AISLogger


class HarborSummarizer:
    """Harbor agent benchmark summarizer: prints one table, writes one csv.

    Columns are data-driven via ``COLUMNS`` (key = short english header). To
    add a column, append its key to ``COLUMNS`` and fill it in ``_build_row``.
    """

    COLUMNS = [
        'model', 'dataset', 'avg_score', 'correct', 'wrong', 'exception',
        'total_time',
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
                row = self._build_row(model['abbr'], dataset['abbr'])
                if row:
                    rows.append(row)
        if not rows:
            self.logger.warning('No harbor results found to summarize.')
            return

        table = [list(self.COLUMNS)] + [[str(r.get(c, '-')) for c in self.COLUMNS]
                                        for r in rows]
        print(tabulate.tabulate(table, headers='firstrow', tablefmt='grid'))

        time_str = time_str or datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_dir = osp.join(self.work_dir, 'summary')
        mmengine.mkdir_or_exist(summary_dir)
        csv_path = osp.join(summary_dir, f'summary_{time_str}.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            for r in rows:
                writer.writerow([r.get(c, '-') for c in self.COLUMNS])
        self.logger.info(f'write summary csv to {osp.abspath(csv_path)}')

    def _build_row(self, model_abbr, dataset_abbr):
        result_path = osp.join(self.work_dir, 'results', model_abbr,
                               f'{dataset_abbr}.json')
        if not osp.exists(result_path):
            return None
        result = mmengine.load(result_path)
        dist = result.get('reward_distribution') or []
        correct = sum(d.get('count', 0) for d in dist if d.get('score', 0) >= 1.0)
        wrong = sum(d.get('count', 0) for d in dist
                    if 0 <= d.get('score', 0) < 1.0)
        return {
            'model': model_abbr,
            'dataset': dataset_abbr,
            'avg_score': result.get('avg_score', '-'),
            'correct': correct,
            'wrong': wrong,
            'exception': result.get('n_errors', 0),
            'total_time': self._total_time(model_abbr, dataset_abbr),
        }

    def _total_time(self, model_abbr, dataset_abbr):
        path = osp.join(self.work_dir, 'results', model_abbr, dataset_abbr,
                        'details', 'result.json')
        if not osp.exists(path):
            return '-'
        data = mmengine.load(path) or {}
        try:
            start = datetime.fromisoformat(data['started_at'])
            finish = datetime.fromisoformat(data['finished_at'])
        except (TypeError, KeyError, ValueError):
            return '-'
        return f'{(finish - start).total_seconds():.1f}s'
