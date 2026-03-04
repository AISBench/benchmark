from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.models.lmm_models.qwen_image_edit import models as qwen_image_edit_models
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gedit.gedit_gen_0_shot_llmjudge import gedit_datasets

# ====== User configuration parameters =========
qwen_image_edit_models[0]["path"] = "/path/to/Qwen-Image-Edit-2509/" # Please modify the weight path according to actual situation
qwen_image_edit_models[0]["infer_kwargs"]["num_inference_steps"] = 50 # Please modify the inference steps according to actual situation
device_list = [0] # [0, 1, 2, 3]
# ====== User configuration parameters =========

datasets = []
models = []
model_dataset_combinations = []

for i, device_id in enumerate(device_list):
    model_config = {k: v for k, v in qwen_image_edit_models[0].items()}
    model_config['abbr'] = f"{model_config['abbr']}-{i}"
    model_config['device_kwargs'] = dict(model_config['device_kwargs'])
    model_config['device_kwargs']['device_id'] = device_id
    models.append(model_config)

    dataset_configs = []
    for dataset in gedit_datasets:
        dataset_config = {k: v for k, v in dataset.items()}
        dataset_config['abbr'] = f"{dataset_config['abbr']}-{i}"
        dataset_config['split_count'] = len(device_list)
        dataset_config['split_index'] = i
        dataset_configs.append(dataset_config)
    datasets.extend(dataset_configs)

    # Key: Create an independent model-dataset combination for each device
    model_dataset_combinations.append({
        'models': [model_config],      # Only include current model
        'datasets': dataset_configs   # Only include current datasets
    })