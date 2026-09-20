import importlib
import os.path as osp
from importlib.metadata import entry_points
from typing import Callable, List, Optional, Type, Union

from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import Registry as OriginalRegistry


def load_class(class_path):
    """动态加载类路径并返回类对象"""
    try:
        parts = class_path.split('.')
        module_name = '.'.join(parts[:-1])
        class_name = parts[-1]

        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"无法加载类 {class_path}: {e}") from e


def get_locations(module_dir):
    """返回核心模块位置;插件位置由 get_plugin_locations 延迟探测。

    插件模块通常要 ``from ais_bench.benchmark.registry import MODELS`` 来
    注册类,而 registry.py 本身会在 ``ais_bench.benchmark.utils.config``
    初始化期间被导入(见文件末尾延迟探测循环的说明),所以在 Registry 构造
    或插件探测阶段以任何形式执行插件代码都会形成循环导入。
    """
    return [f'ais_bench.benchmark.{module_dir}']


def _submodule_exists(pkg, module_dir):
    """判断 pkg 下是否存在 module_dir 子包或模块(纯文件系统检查,不导入)。"""
    rel = module_dir.replace('.', osp.sep)
    for base in getattr(pkg, '__path__', None) or []:
        if osp.isdir(osp.join(base, rel)) or osp.isfile(osp.join(base, rel + '.py')):
            return True
    return False


def get_plugin_locations(module_dir):
    """探测提供 module_dir 子包的插件,返回其位置列表。

    只做文件系统级存在性检查,不执行任何插件代码。位置字符串交给 mmengine
    Registry 在首次 ``get()`` 未命中时通过 ``import_from_location()`` 惰性
    导入(自带 ``_imported`` 守卫),那时循环导入的各方都已初始化完成。
    """
    locations = []
    try:
        # 使用 .select() 方法替代已弃用的 .get() 方法
        for entry_point in entry_points().select(group='ais_bench.benchmark_plugins'):
            try:
                pkg = entry_point.load()
                if _submodule_exists(pkg, module_dir):
                    locations.append(f'{pkg.__name__}.{module_dir}')
            except Exception:
                continue
    except Exception:
        pass
    return locations


class Registry(OriginalRegistry):

    # override the default force behavior
    def register_module(
            self,
            name: Optional[Union[str, List[str]]] = None,
            force: bool = True,
            module: Optional[Type] = None) -> Union[type, Callable]:
        return super().register_module(name, force, module)


PARTITIONERS = Registry('partitioner', locations=get_locations('partitioners'))
RUNNERS = Registry('runner', locations=get_locations('runners'))
TASKS = Registry('task', locations=get_locations('tasks') + get_locations('tasks.custom_tasks'))
MODELS = Registry('model', locations=get_locations('models'))
# TODO: LOAD_DATASET -> DATASETS
LOAD_DATASET = Registry('load_dataset', locations=get_locations('datasets'))
TEXT_POSTPROCESSORS = Registry(
    'text_postprocessors', locations=get_locations('utils.postprocess.text_postprocessors'))

EVALUATORS = Registry('evaluators', locations=get_locations('evaluators'))

ICL_INFERENCERS = Registry('icl_inferencers',
                           locations=get_locations('openicl.icl_inferencer'))
ICL_RETRIEVERS = Registry('icl_retrievers',
                          locations=get_locations('openicl.icl_retriever'))
ICL_DATASET_READERS = Registry(
    'icl_dataset_readers',
    locations=get_locations('openicl.icl_dataset_reader'))
ICL_PROMPT_TEMPLATES = Registry(
    'icl_prompt_templates',
    locations=get_locations('openicl.icl_prompt_template'))
ICL_EVALUATORS = Registry('icl_evaluators',
    locations=get_locations('openicl.icl_evaluator'))
METRICS = Registry('metric',
                   parent=MMENGINE_METRICS,
                   locations=get_locations('metrics'))
TOT_WRAPPER = Registry('tot_wrapper', locations=get_locations('datasets'))

CLIENTS = Registry('client', locations=get_locations('clients'))

PERF_METRIC_CALCULATORS = Registry('perf_metric_calculator', locations=get_locations('calculators'))


# 延迟探测插件位置:在全部 Registry 构造完成、registry.py 执行到末尾之后,
# 把插件子包的位置字符串追加到各 Registry 的 _locations。
#
# 这里刻意不导入插件代码。registry.py 会被 utils/config/build.py 在
# ``ais_bench.benchmark.utils.config`` 初始化期间导入,此刻 utils.config 尚未
# 初始化完成;一旦探测过程执行插件代码(插件子包 -> benchmark.datasets ->
# openicl.icl_inferencer -> icl_base_inferencer 的
# ``from ais_bench.benchmark.utils.config import build_model_from_cfg``),就会
# 命中半初始化的 utils.config 抛 ImportError,并被下游的 except ImportError
# 静默吞掉,最终表现为与真实原因完全无关的报错(例如 "Failed to import
# GSM8KDataset from ais_bench.benchmark.datasets")。导入交给 mmengine Registry
# 在首次 get() 未命中时通过 import_from_location() 惰性完成。
for _registry, _module_dir in (
    (PARTITIONERS, 'partitioners'),
    (RUNNERS, 'runners'),
    (TASKS, 'tasks'),
    (TASKS, 'tasks.custom_tasks'),
    (MODELS, 'models'),
    (LOAD_DATASET, 'datasets'),
    (TEXT_POSTPROCESSORS, 'utils.postprocess.text_postprocessors'),
    (EVALUATORS, 'evaluators'),
    (ICL_INFERENCERS, 'openicl.icl_inferencer'),
    (ICL_RETRIEVERS, 'openicl.icl_retriever'),
    (ICL_DATASET_READERS, 'openicl.icl_dataset_reader'),
    (ICL_PROMPT_TEMPLATES, 'openicl.icl_prompt_template'),
    (ICL_EVALUATORS, 'openicl.icl_evaluator'),
    (METRICS, 'metrics'),
    (TOT_WRAPPER, 'datasets'),
    (CLIENTS, 'clients'),
    (PERF_METRIC_CALCULATORS, 'calculators'),
):
    _registry._locations.extend(get_plugin_locations(_module_dir))


def build_from_cfg(cfg):
    """A helper function that builds object with MMEngine's new config."""
    return PARTITIONERS.build(cfg)
