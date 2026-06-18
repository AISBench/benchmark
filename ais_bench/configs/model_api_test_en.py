from mmengine.config import read_base

with read_base():
# model tasks, choose one of them, other model tasks refer: https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/all_params/models.html
    # vllm_api_general is the base model, it only support text generation
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general
    # vllm_api_general_chat is the chat model, it support chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat
    # vllm_api_stream_chat is the stream chat model, it support stream chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat
    # vllm_api_general_stream is the stream model, it support stream generation
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as vllm_api_general_stream

# dataset task, get from https://ais-bench-benchmark-rf.readthedocs.io/en/latest/get_started/datasets.html
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets

models = vllm_api_general_chat

models[0]["path"] = ""  # Specify the absolute path of the model serialized vocabulary file (generally not required for accuracy testing scenarios)
models[0]["model"] = "" # Specify the name of the model loaded on the server, configured according to the actual model name pulled by the VLLM inference service (configure as an empty string to get it automatically)
models[0]["request_rate"] = 0 # Request sending frequency: send 1 request to the server every 1/request_rate seconds; if less than 0.001, all requests are sent at once
models[0]["api_key"] = "" # Custom API key, default is an empty string
models[0]["host_ip"] = "localhost" # Specify the IP of the inference service
models[0]["host_port"] = 8080 # Specify the port of the inference service
models[0]["url"] = "" # Custom URL path for accessing the inference service (needs to be configured when the base URL is not a combination of http://host_ip:host_port; after configuration, host_ip and host_port will be ignored)
models[0]["max_out_len"] = 512 # Maximum number of tokens output by the inference service
models[0]["batch_size"] = 1 # Maximum concurrency for sending requests
models[0]["trust_remote_code"] = False # Whether the tokenizer trusts remote code, default is False;
models[0]["generation_kwargs"] = dict( # Model inference parameters, configured with reference to the VLLM documentation; the AISBench evaluation tool does not process them and attaches them to the sent request
    temperature=0.01,
    ignore_eos=False,
)

# datasets[0]["path"] = ais_bench/datasets/gsm8k # Specify the absolute path of the dataset directory (required for accuracy testing scenarios)

work_dir = 'outputs/default/'  # Specify the working directory for saving task results and logs (default is outputs/default/)
