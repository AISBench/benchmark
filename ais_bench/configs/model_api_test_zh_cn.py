from mmengine.config import read_base

with read_base():
# 模型任务，选择其中一个，其他模型任务参考：https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/all_params/models.html 获取更多数据集任务
    # vllm_api_general 是基础模型，仅支持文本生成
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general
    # vllm_api_general_chat 是对话模型，支持对话
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat
    # vllm_api_stream_chat 是流式对话模型，支持流式对话
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat
    # vllm_api_general_stream 是流式模型，支持流式生成
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as vllm_api_general_stream
# 数据集任务，参考：https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/get_started/datasets.html 获取更多数据集任务
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets

# datasets =
models = vllm_api_general_chat

models[0]["path"] = ""  # 指定模型序列化词表文件的绝对路径（精度测试场景一般不需要配置）
models[0]["model"] = "" # 指定服务端加载的模型名称，根据 VLLM 推理服务实际拉取的模型名称配置（配置为空字符串则自动获取）
models[0]["request_rate"] = 0 # 请求发送频率：每 1/request_rate 秒向服务端发送 1 条请求；小于 0.001 时一次性发送所有请求
models[0]["api_key"] = "" # 自定义 API key，默认为空字符串
models[0]["host_ip"] = "localhost" # 指定推理服务的 IP
models[0]["host_port"] = 8080 # 指定推理服务的端口
models[0]["url"] = "" # 自定义访问推理服务的 URL 路径（当基础 URL 不是 http://host_ip:host_port 的组合时需要配置；配置后 host_ip 和 host_port 将被忽略）
models[0]["max_out_len"] = 512 # 推理服务输出的最大 token 数
models[0]["batch_size"] = 1 # 发送请求的最大并发数
models[0]["trust_remote_code"] = False # tokenizer 是否信任远程代码，默认为 False
models[0]["generation_kwargs"] = dict( # 模型推理参数，参考 VLLM 文档配置；AISBench 评测工具不做处理，直接附加到发送的请求中
    temperature=0.01,
    ignore_eos=False,
)

# datasets[0]["path"] = ais_bench/datasets/gsm8k # 指定数据集目录的绝对路径（精度测试场景需要配置）

work_dir = 'outputs/default/'  # 指定任务结果和日志的保存工作目录（默认为 outputs/default/）
