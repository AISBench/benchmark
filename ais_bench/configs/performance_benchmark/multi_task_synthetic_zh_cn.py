import copy
from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import synthetic_datasets as base_synthetic_datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as base_vllm_api_stream_chat

# 关键：统一收束 batch_size / request_rate / request_count / input_range / output_range 五个参数
# 用户希望配几个任务，就在对应列表中追加几个元素（同一分组内的列表长度需保持一致）
# 注意：models 与 datasets 的列表长度需保持一致，二者会按下标一一配对，而非笛卡尔积
tasks_params = {
    "models": {
        "batch_size": [1, 2, 4, 8, 16, 32],
        "request_rate": [0, 0, 0, 0, 0, 0],
    },
    "datasets": {
        "request_count": [100, 100, 100, 100, 100, 100],
        "input_range":  [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)],
        "output_range": [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)],
    },
}

# 关键：通过 deepcopy 复制同一个基础模型配置，按 tasks_params["models"] 批量覆盖 batch_size / request_rate
models = []
for idx, (batch_size, request_rate) in enumerate(zip(tasks_params["models"]["batch_size"],
                                                    tasks_params["models"]["request_rate"])):
    model_cfg = copy.deepcopy(base_vllm_api_stream_chat[0])
    model_cfg["abbr"] = f"vllm-api-stream-chat-bs{batch_size}-rr{request_rate}"
    model_cfg["host_ip"] = "localhost"
    model_cfg["host_port"] = 8080
    model_cfg["max_out_len"] = 512
    model_cfg["batch_size"] = batch_size
    model_cfg["request_rate"] = request_rate
    # 关键：每个模型任务使用独立的 generation_kwargs
    model_cfg["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)
    models.append(model_cfg)

# 关键：按 tasks_params["datasets"] 批量构建合成数据集任务，名称按索引自动生成
datasets = []
for idx, (request_count, input_range, output_range) in enumerate(
    zip(tasks_params["datasets"]["request_count"],
        tasks_params["datasets"]["input_range"],
        tasks_params["datasets"]["output_range"])
):
    ds = dict(base_synthetic_datasets[0])
    ds["abbr"] = f"synthetic-string-{idx}"
    ds["config"] = {
        "Type": "string",
        "RequestCount": request_count,
        "StringConfig": {
            "Input": {
                "Method": "uniform",
                "Params": {"MinValue": input_range[0], "MaxValue": input_range[1]},
            },
            "Output": {
                "Method": "uniform",
                "Params": {"MinValue": output_range[0], "MaxValue": output_range[1]},
            },
        },
    }
    datasets.append(ds)

# 关键：按索引一一配对 models[i] 与 datasets[i]，避免笛卡尔积
# 例如 models[0](batch_size=1) 仅与 datasets[0](input_range=(1,2)) 配对，而非与所有数据集交叉组合
model_dataset_combinations = [
    dict(models=[models[idx]], datasets=[datasets[idx]])
    for idx in range(min(len(models), len(datasets)))
]

work_dir = "outputs/default/"

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLApiInferTask)),
)