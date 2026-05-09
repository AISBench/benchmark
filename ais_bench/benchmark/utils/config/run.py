# flake8: noqa
# yapf: disable
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import OpenICLEvalTask, OpenICLApiInferTask
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator

logger = AISLogger()


def get_config_type(obj) -> str:
    if isinstance(obj, str):
        return obj
    return f"{obj.__module__}.{obj.__name__}"


def try_fill_in_custom_cfgs(config):
    if "datasets" not in config:
        return config

    for dataset_cfg in config["datasets"]:
        if "infer_cfg" not in dataset_cfg:
            logger.debug(f"Filling in infer config for dataset {dataset_cfg['abbr']}")
            dataset_cfg["infer_cfg"] = dict(
            prompt_template=dict(type=get_config_type(PromptTemplate), template="{dummy}"),
            retriever=dict(type=get_config_type(ZeroRetriever)),
            inferencer=dict(type=get_config_type(GenInferencer)),
            )
        if "reader_cfg" not in dataset_cfg:
            logger.debug(f"Filling in reader config for dataset {dataset_cfg['abbr']}")
            dataset_cfg["reader_cfg"] = dict(input_columns=["dummy"], output_column="dummy")
        if "eval_cfg" not in dataset_cfg:
            logger.debug(f"Filling in eval config for dataset {dataset_cfg['abbr']}")
            dataset_cfg["eval_cfg"] = dict(
            evaluator=dict(type=get_config_type(AccEvaluator)),
            )

    return config

def get_models_attr(cfg):
    logger.debug(f"Checking model attributes for {len(cfg['models'])} models")
    attr_list = []
    for model_cfg in cfg['models']:
        attr = model_cfg.get('attr', 'service') # default service
        if attr not in ['local', 'service']:
            raise AISBenchConfigError(UTILS_CODES.ILLEGAL_MODEL_ATTR, f"Model config contain illegal attr, model abbr is {model_cfg.get('abbr')}")
        if attr not in attr_list:
            attr_list.append(attr)

    if len(attr_list) != 1:
        raise AISBenchConfigError(UTILS_CODES.MIXED_MODEL_ATTRS, "Cannot run local and service model together! Please check parameters of --models!")

    logger.debug(f"All models have consistent attr: {attr_list[0]}")
    return attr_list[0]


def fill_infer_cfg(cfg, args):
    logger.debug(f"Filling inference config with max_num_workers={args.max_num_workers}, max_workers_per_gpu={args.max_workers_per_gpu}, debug={args.debug}")
    new_cfg = dict(infer=dict(
        partitioner=dict(type=get_config_type(NaivePartitioner)),
        runner=dict(
            max_num_workers=args.max_num_workers,
            max_workers_per_gpu=args.max_workers_per_gpu,
            debug=args.debug,
            task=dict(type=get_config_type(OpenICLApiInferTask)),
            type=get_config_type(LocalRunner),
        )), )
    for data_config in cfg['datasets']:
        retriever_cfg = data_config['infer_cfg']['retriever']
        infer_cfg = data_config['infer_cfg']
        if "prompt_template" in infer_cfg:
            retriever_cfg["prompt_template"] = infer_cfg["prompt_template"]
        if "ice_template" in infer_cfg:
            retriever_cfg["ice_template"] = infer_cfg["ice_template"]

    cfg.merge_from_dict(new_cfg)
    logger.debug("Inference config filled successfully")


def fill_eval_cfg(cfg, args):
    logger.debug(f"Filling evaluation config with max_num_workers={args.max_num_workers}, max_workers_per_gpu={args.max_workers_per_gpu}, debug={args.debug}")
    new_cfg = dict(eval=dict(
        partitioner=dict(type=get_config_type(NaivePartitioner)),
        runner=dict(
            max_num_workers=args.max_num_workers,
            debug=args.debug,
            task=dict(type=get_config_type(OpenICLEvalTask)),
        )), )

    new_cfg['eval']['runner']['type'] = get_config_type(LocalRunner)
    new_cfg['eval']['runner']['max_workers_per_gpu'] = args.max_workers_per_gpu
    cfg.merge_from_dict(new_cfg)
    logger.debug("Evaluation config filled successfully")
