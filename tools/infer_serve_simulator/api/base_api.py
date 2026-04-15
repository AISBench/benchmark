import os
import yaml
import random
from abc import ABC, abstractmethod
import random
import string

# 生成指定长度的随机字母数字字符串
def random_string(length=8):
    chars = string.ascii_letters + string.digits  # 包含大小写字母和数字
    return " ".join(random.choices(chars, k=length)) + " "

def load_yaml_config():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(cur_dir, 'api_config.yaml')
    if not os.path.exists(config_path):
        raise FileExistsError(f"Can't find api_config.yaml in {cur_dir}")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class BaseAPI(ABC):
    def __init__(self):
        self.config = load_yaml_config()
        self.ttft = self.config['stream_latency']['ttft']
        self.tpot = self.config['stream_latency']['tpot']
        self.e2el = self.config['text_latency']['e2el']
        self.random_conf = self.config['random_dataset']

    @abstractmethod
    def generate_text(self):
        pass

    @abstractmethod
    def generate_stream(self):
        pass

    @abstractmethod
    def text_request_body(self, content):
        pass

    @abstractmethod
    def stream_request_body(self, output_list):
        pass

    def gen_output_list(self, max_tokens, ignore_eos=False):
        if ignore_eos:
            min_tokens = max_tokens
        else:
            min_tokens = self.random_conf["min_tokens"] if self.random_conf["min_tokens"] < max_tokens else max_tokens
        output_length = random.randint(min_tokens, max_tokens)

        if self.config["general"]["enable_mtp"]:
            tokens_per_chunk = self.random_conf["tokens_per_chunk"]
            token_length_list = [tokens_per_chunk for _ in range(output_length // tokens_per_chunk)]
            if output_length % tokens_per_chunk != 0:
                token_length_list.append(output_length % tokens_per_chunk)
        else:
            token_length_list = [1 for _ in range(output_length)]

        if self.random_conf["random_content"]: # 随机生成token, 后续再拓展不同分布的随机方式
            return [random_string(token_len) for token_len in token_length_list], output_length
        else:
            return ['A ' * token_len for token_len in token_length_list], output_length







