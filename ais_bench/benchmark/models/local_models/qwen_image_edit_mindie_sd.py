# flake8: noqa
# yapf: disable
import os
import time
from typing import Dict, List, Optional, Union

import torch
import torch_npu
import base64
import io
from PIL import Image

from ais_bench.benchmark.models.local_models.base import BaseLMModel
from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES
from ais_bench.benchmark.models.local_models.huggingface_above_v4_33 import (_convert_chat_messages,
                                                                            _get_meta_template,
                                                                            )

# 解决 diffuser 0.35.1 torch2.1 报错
def custom_op(
    name,
    fn=None,
    /,
    *,
    mutates_args,
    device_types=None,
    schema=None,
    tags=None,
):
    def decorator(func):
        return func

    if fn is not None:
        return decorator(fn)

    return decorator

def register_fake(
    op,
    fn=None,
    /,
    *,
    lib=None,
    _stacklevel: int = 1,
    allow_override: bool = False,
):
    def decorator(func):
        return func

    if fn is not None:
        return decorator(fn)

    return decorator

if hasattr(torch, 'library'):
    torch.library.custom_op = custom_op
    torch.library.register_fake = register_fake

# 导入 qwen_image_edit 相关模块
try:
    from ais_bench.third_party.mindie_sd.qwenimage_edit.transformer_qwenimage import QwenImageTransformer2DModel
    from ais_bench.third_party.mindie_sd.qwenimage_edit.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
    from mindiesd import CacheConfig, CacheAgent
except ImportError as e:
    raise ImportError(f"请确保 qwenimage_edit 模块在 Python 路径中: {e}")

PromptType = Union[PromptList, str]

# 模型推理相关配置常量
DEFAULT_MODEL_PATH = "/home/yanhe/models/Qwen-Image-Edit-2509/"
DEFAULT_TORCH_DTYPE = "bfloat16"
DEFAULT_DEVICE = "npu"
DEFAULT_DEVICE_ID = 0
DEFAULT_NUM_INFERENCE_STEPS = 1 # 40
DEFAULT_TRUE_CFG_SCALE = 4.0
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_SEED = 0
DEFAULT_NUM_IMAGES_PER_PROMPT = 1
DEFAULT_QUANT_DESC_PATH = None

# 缓存配置开关
COND_CACHE = bool(int(os.environ.get('COND_CACHE', 0)))
UNCOND_CACHE = bool(int(os.environ.get('UNCOND_CACHE', 0)))


@MODELS.register_module()
class QwenImageEditModel(BaseLMModel):
    """Model wrapper for Qwen-Image-Edit-2509 models.

    Args:
        path (str): The path to the model.
        model_kwargs (dict): Additional model arguments.
        sample_kwargs (dict): Additional sampling arguments.
        vision_kwargs (dict): Additional vision arguments.
        meta_template (Optional[Dict]): The model's meta prompt template.
    """

    def __init__(self,
                 path: str = DEFAULT_MODEL_PATH,
                 device_kwargs: dict = dict(),
                 infer_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 **other_kwargs):
        self.logger = AISLogger()
        self.path = path
        self.max_out_len = other_kwargs.get('max_out_len', None)
        self.template_parser = _get_meta_template(meta_template)

        # 设备配置
        self.device = device_kwargs.get('device', DEFAULT_DEVICE)
        #self.device_id = device_kwargs.get('device_id', DEFAULT_DEVICE_ID)
        # 在这里声明环境变量
        self.logger.debug(f"device id from kwargs: {device_kwargs.get('device_id', DEFAULT_DEVICE_ID)}")
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = f"{device_kwargs.get('device_id', DEFAULT_DEVICE_ID)}"
        self.device_id = DEFAULT_DEVICE_ID
        self.device_str = f"{self.device}:{DEFAULT_DEVICE_ID}"
        self.logger.debug(f"device_str: {self.device_str};  device_id: {self.device_id}")
        self.logger.debug(f"ASCEND_RT_VISIBLE_DEVICES: {os.getenv('ASCEND_RT_VISIBLE_DEVICES')}")

        # 模型配置
        self.torch_dtype = other_kwargs.get('torch_dtype', DEFAULT_TORCH_DTYPE)
        self.torch_dtype = torch.bfloat16 if self.torch_dtype == "bfloat16" else torch.float32

        # 推理配置
        self.num_inference_steps = infer_kwargs.get('num_inference_steps', DEFAULT_NUM_INFERENCE_STEPS)
        self.true_cfg_scale = infer_kwargs.get('true_cfg_scale', DEFAULT_TRUE_CFG_SCALE)
        self.guidance_scale = infer_kwargs.get('guidance_scale', DEFAULT_GUIDANCE_SCALE)
        self.seed = infer_kwargs.get('seed', DEFAULT_SEED)
        self.num_images_per_prompt = infer_kwargs.get('num_images_per_prompt', DEFAULT_NUM_IMAGES_PER_PROMPT)
        self.quant_desc_path = infer_kwargs.get('quant_desc_path', DEFAULT_QUANT_DESC_PATH)

        # 加载模型
        self._load_model()

        # 缓存配置
        if COND_CACHE or UNCOND_CACHE:
            # 保守cache
            cache_config = CacheConfig(
                method="dit_block_cache",
                blocks_count=60,
                steps_count=self.num_inference_steps,
                step_start=10,
                step_interval=3,
                step_end=35,
                block_start=10,
                block_end=50
            )
            self.pipeline.transformer.cache_cond = CacheAgent(cache_config) if COND_CACHE else None
            self.pipeline.transformer.cache_uncond = CacheAgent(cache_config) if UNCOND_CACHE else None
            self.logger.info("启用缓存配置")

    def _load_model(self):
        """加载模型"""
        self.logger.info(f"从 {self.path} 加载模型...")

        # 设置设备
        if self.device == "npu":
            torch.npu.set_device(self.device_id)

        # 加载 transformer
        transformer = QwenImageTransformer2DModel.from_pretrained(
            os.path.join(self.path, 'transformer'),
            torch_dtype=self.torch_dtype,
            device_map=None,               # 禁用自动设备映射
            low_cpu_mem_usage=True         # 启用CPU低内存模式
        )

        # 量化配置
        if self.quant_desc_path:
            from mindiesd import quantize
            self.logger.info("Quantizing Transformer (单独量化核心组件)...")
            quantize(
                model=transformer,
                quant_des_path=self.quant_desc_path,
                use_nz=True,
            )
            if self.device == "npu":
                torch.npu.empty_cache()  # 清理NPU显存缓存

        # 加载 pipeline
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self.path,
            transformer=transformer,
            torch_dtype=self.torch_dtype,
            device_map=None,
            low_cpu_mem_usage=True
        )

        # VAE优化配置（避免显存溢出）
        self.pipeline.vae.use_slicing = True
        self.pipeline.vae.use_tiling = True

        # 移动模型到目标设备
        self.pipeline.to(self.device_str)
        self.pipeline.set_progress_bar_config(disable=None)  # 显示进度条

    def _get_meta_template(self, meta_template):
        """获取元模板"""
        class DummyTemplateParser:
            def parse_template(self, prompt_template, mode):
                return prompt_template
        return DummyTemplateParser()

    def _generate(self, input) -> List[Image]:
        """Generate result given a input.

        Args:
            input (PromptType): A string or PromptDict.
                The PromptDict should be organized in AISBench'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        # 处理输入格式
        images = []
        prompts = []
        neg_prompts = []
        print(f"in _generate")
        #self.logger.info(f"输入: {input}")
        if isinstance(input, str):
            prompts.append(input)
            neg_prompts.append("")
        elif isinstance(input, list):
            # 处理包含图像的输入
            for item in input[0]["prompt"]:
                if item["type"] == "image_url":
                    base64_url = item["image_url"]["url"].split(",")[1]
                    img = Image.open(io.BytesIO(base64.b64decode(base64_url))).convert("RGB")
                    images.append(img)
                elif item["type"] == "text":
                    prompts.append(item["text"])
                    neg_prompts.append("")
                else:
                    prompts.append("")
                    neg_prompts.append("")

        # 如果没有图像输入，使用默认图像
        if not images:
            raise ValueError("QwenImageEditModel requires image input")

        # 执行推理
        results = []
        for prompt, neg_prompt in zip(prompts, neg_prompts):
            # 准备输入参数
            print("in _generate loop")
            inputs = {
                "image": images,
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "generator": torch.Generator(device=self.device_str).manual_seed(self.seed),
                "true_cfg_scale": self.true_cfg_scale,
                "guidance_scale": self.guidance_scale,
                "num_inference_steps": self.num_inference_steps,
                "num_images_per_prompt": self.num_images_per_prompt,
            }

            # 执行推理并计时
            if self.device == "npu":
                torch.npu.synchronize()  # 昇腾设备同步
            start_time = time.time()

            with torch.inference_mode():
                output = self.pipeline(**inputs)

            if self.device == "npu":
                torch.npu.synchronize()
            end_time = time.time()
            infer_time = end_time - start_time
            self.logger.info(f"推理完成，耗时: {infer_time:.2f}秒")

        return output

    def encode(self, prompt: str) -> torch.Tensor:
        """Encode prompt to tokens. Not necessary for most cases.

        Args:
            prompt (str): Input string.

        Returns:
            torch.Tensor: Encoded tokens.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not implement `encode` method.')

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode tokens to text. Not necessary for most cases.

        Args:
            tokens (torch.Tensor): Input tokens.

        Returns:
            str: Decoded text.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not implement `decode` method.')

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        # 对于图像编辑模型，token长度计算可能不同，这里返回一个默认值
        return len(prompt.split())

    def generate(self, inputs, outputs, **kwargs):
        """Generate completion from inputs.

        Args:
            inputs: Inputs for generation.
            max_out_lens: Maximum output lengths.
            **kwargs: Additional keyword arguments.

        Returns:
            List[str]: Generated completions.
        """
        #self.logger.info(f"model {inputs=}")
        if not isinstance(inputs, list):
            inputs = [inputs]

        for i, input in enumerate(inputs):
            result = self._generate(input)
            # result is QwenImagePipelineOutput with 'images' attribute
            if hasattr(result, 'images') and result.images:
                outputs[i].success = True
                outputs[i].content = result.images  # 将图像列表赋值给 content
            else:
                outputs[i].success = False
                outputs[i].content = [""]
