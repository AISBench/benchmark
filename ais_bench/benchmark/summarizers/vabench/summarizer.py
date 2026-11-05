"""Official means, fixed 0–100 display scores and the AISBench VABench total."""
import os.path as osp

from ais_bench.benchmark.summarizers.default import (
    DefaultSummarizer, model_abbr_from_cfg_used_in_summarizer)
from ais_bench.benchmark.utils.core.abbr import get_infer_output_path
from ais_bench.benchmark.utils.vabench.data import read_json
from ais_bench.benchmark.summarizers.vabench.results import display_report


class VABenchSummarizer(DefaultSummarizer):
    def _pick_up_results(self):
        raw, parsed, metrics, modes = {}, {}, {}, {}
        display = []
        for model in self.model_cfgs:
            name = model_abbr_from_cfg_used_in_summarizer(model)
            raw[name], parsed[name] = {}, {}
            for dataset in self.dataset_cfgs:
                file = get_infer_output_path(model, dataset, osp.join(self.work_dir, 'results'))
                if not osp.isfile(file):
                    continue
                report = display_report(read_json(file))
                for dimension, data in report['metrics'].items():
                    row = f'{dataset["abbr"]}/{dimension}'
                    values = {k: data[k] for k in (
                        'mean', 'normalized_score', 'expected', 'scored', 'omitted', 'missing',
                        'failed', 'coverage_percent', 'status')}
                    values['mean'] = values['mean'] if values['mean'] is not None else '—'
                    values['normalized_score'] = (values['normalized_score']
                                                  if values['normalized_score'] is not None else '—')
                    values['direction'] = report['directions'][dimension]
                    raw[name][row] = dict(values, scenes=data['scenes'],
                                          out_of_bounds_count=data.get('out_of_bounds_count', 0))
                    parsed[name][row] = values
                    metrics[row] = ['mean', 'normalized_score', 'scored', 'expected', 'coverage_percent', 'status']
                    modes[row] = 'gen'
                    if row not in display:
                        display.append(row)
                total = report['total']
                if total['score'] is None:
                    # Do not print an unusable total row; it stays in report.json/total.csv.
                    continue
                row = f'{dataset["abbr"]}/total'
                values = dict(mean='—', normalized_score=total['score'] if total['score'] is not None else '—',
                              expected=total['expected_dimensions'], scored=total['valid_dimensions'],
                              direction='higher', omitted='—', missing=len(total['missing_dimensions']),
                              failed='—', coverage_percent='—', status=total['label'])
                parsed[name][row] = values
                raw[name][row] = dict(values, aggregate=total)
                metrics[row], modes[row] = ['normalized_score', 'scored', 'expected', 'status'], 'gen'
                if row not in display:
                    display.append(row)
        self.dataset_abbrs = display
        return raw, parsed, metrics, modes

    def _format_table(self, parsed_results, dataset_metrics, dataset_eval_mode,
                      required_dataset_abbrs=None, skip_all_slash=False):
        table = [['model', 'dataset/dimension', 'raw direction', 'raw mean', 'score/100',
                  'expected', 'scored', 'LSA omitted', 'missing', 'failed', 'coverage %', 'status']]
        for model, rows in parsed_results.items():
            for name, values in rows.items():
                cells = [values[k] for k in ('direction', 'mean', 'normalized_score', 'expected',
                                             'scored', 'omitted', 'missing', 'failed', 'coverage_percent', 'status')]
                for index, digits in ((1, 5), (2, 2), (8, 1)):
                    if isinstance(cells[index], (int, float)):
                        cells[index] = f'{cells[index]:.{digits}f}'
                # DefaultSummarizer's Markdown/CSV writers join string cells.
                table.append([str(model), str(name)] + [str(cell) for cell in cells])
        return table

    def _calculate_group_metrics(self, raw_results, parsed_results, dataset_metrics, dataset_eval_mode):
        # The VABench total is calculated inside each selected dataset only.
        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode
