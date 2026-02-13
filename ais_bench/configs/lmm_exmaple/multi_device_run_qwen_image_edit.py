from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.models.lmm_models.qwen_image_edit import models as qwen_image_edit_models
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gedit.gedit_gen import gedit_datasets

device_list = [0, 1, 2, 3]

datasets = []
models = []
model_dataset_combinations = []

for i in device_list:
    model_config = {k: v for k, v in qwen_image_edit_models[0].items()}
    model_config['abbr'] = f"{model_config['abbr']}-{i}"
    model_config['device_kwargs'] = dict(model_config['device_kwargs'])
    model_config['device_kwargs']['device_id'] = i
    models.append(model_config)

    dataset_config = {k: v for k, v in gedit_datasets[0].items()}
    dataset_config['abbr'] = f"{dataset_config['abbr']}-{i}"
    dataset_config['split_count'] = len(device_list)
    dataset_config['split_index'] = i
    datasets.append(dataset_config)

    # 关键：为每个设备创建一个独立的 model-dataset 组合
    model_dataset_combinations.append({
        'models': [model_config],      # 只包含当前模型
        'datasets': [dataset_config]   # 只包含当前数据集
    })