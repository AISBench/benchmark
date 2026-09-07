import importlib
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
    """返回核心模块位置;插件位置通过 get_plugin_locations 延迟探测导入。

    插件模块通常要 ``from ais_bench.benchmark.registry import MODELS`` 来
    注册类,如果在 Registry 构造期间(例如 MODELS 尚未完成赋值)急切导入
    插件子包,会形成循环导入。因此插件探测延迟到本模块全部 Registry
    构造完成之后统一执行,见文件末尾的延迟探测循环。
    """
    return [f'ais_bench.benchmark.{module_dir}']


def get_plugin_locations(module_dir):
    """探测并导入提供 module_dir 子包的插件,返回其位置列表。"""
    locations = []
    try:
        # 使用 .select() 方法替代已弃用的 .get() 方法
        for entry_point in entry_points().select(group='ais_bench.benchmark_plugins'):
            try:
                pkg = entry_point.load()
                pkg_dir = pkg.__name__
                custom_loc = f'{pkg_dir}.{module_dir}'
                try:
                    _ = __import__(custom_loc, fromlist=["*"])
                    locations.append(custom_loc)
                except ImportError:
                    continue
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


# 延迟探测插件位置:在全部 Registry 构造完成、registry.py 执行到末尾之后
# 再导入插件子包。此时 MODELS/LOAD_DATASET 等均已赋值,插件注册类不会再
# 触发 "partially initialized module" 循环导入错误。
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
