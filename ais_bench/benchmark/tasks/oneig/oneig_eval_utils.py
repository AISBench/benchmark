"""OneIG 评估器公共工具函数模块

提供多个子评估任务共享的工具函数和 Judge 推理器。
"""
import os
import stat
import sys


# ONEIG_DTYPE_MAP 延迟初始化缓存
_ONEIG_DTYPE_MAP = None


def _get_dtype_map():
    """获取数据类型映射（延迟导入 torch）"""
    global _ONEIG_DTYPE_MAP
    if _ONEIG_DTYPE_MAP is None:
        import torch
        _ONEIG_DTYPE_MAP = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }
    return _ONEIG_DTYPE_MAP


# PEP 562: 模块级 __getattr__ 实现 ONEIG_DTYPE_MAP 延迟加载
def __getattr__(name):
    if name == 'ONEIG_DTYPE_MAP':
        return _get_dtype_map()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def rm_error(func, path, exc_info):
    """Windows 权限错误处理：修改只读文件后重试删除"""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def split_image_grid(image_path, grid_size, cache_dir):
    """图像网格切分（公共工具函数，供多个 Evaluator 共享）"""
    from PIL import Image

    with Image.open(image_path) as grid_image:
        grid_image.load()  # 强制立即读取全部像素数据，避免延迟读取问题

        width, height = grid_image.size
        individual_width = width // grid_size[0]
        individual_height = height // grid_size[1]

        image_list = []

        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                box = (
                    j * individual_width,
                    i * individual_height,
                    (j + 1) * individual_width,
                    (i + 1) * individual_height
                )
                individual_image = grid_image.crop(box)

                if not _is_black_image(individual_image):
                    image_list.append(individual_image)

    image_path_list = []
    for idx, image in enumerate(image_list):
        img_path = os.path.join(cache_dir, f"{idx}.jpg")
        image.save(img_path)
        image_path_list.append(img_path)

    return image_path_list


def _is_black_image(image):
    """检测是否为黑色图像（公共工具函数）"""
    pixels = image.load()
    for i in range(image.width):
        for j in range(image.height):
            if pixels[i, j] != (0, 0, 0):
                return False
    return True


def ensure_oneig_path(oneig_root):
    """将 OneIG 项目根目录加入 sys.path

    OneIG 不是可安装的 Python 包，其内部使用 from scripts.utils.xxx 的
    绝对导入方式。这种方式要求 scripts 的父目录在 sys.path 中。
    运行时需要手动将其根目录加入 sys.path，且全程保留（因为内部存在延迟导入）。

    Args:
        oneig_root: str - OneIG 项目根目录

    Raises:
        ImportError: oneig_root 未设置或路径不存在
    """
    if not oneig_root:
        raise ImportError(
            "oneig_root is not set. "
            "Please pass oneig_root in the evaluator config."
        )

    if not os.path.isdir(oneig_root):
        raise ImportError(
            f"oneig_root does not exist: {oneig_root}"
        )

    if oneig_root not in sys.path:
        sys.path.insert(0, oneig_root)


class OneIGJudgeInferencer:
    """OneIG Judge 推理器

    使用 transformers 本地推理，支持 batch_inference。

    Args:
        model_path: str - 模型路径（本地路径或 HuggingFace 模型名）
        device: str - 运行设备（'cuda' 或 'cpu'）
        dtype: torch.dtype - 模型数据类型
        use_flash_attention: bool - 是否使用 Flash Attention
        batch_size: int - 批量推理的批次大小
        seed: int - 随机种子（默认 42）
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        dtype=None,
        use_flash_attention: bool = True,
        batch_size: int = 8,
        seed: int = 42,
    ):
        try:
            import torch
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        except ImportError as e:
            raise ImportError(f"Failed to import required modules: {e}, please install transformers >= 4.57.0.")

        if dtype is None:
            dtype = torch.bfloat16

        # 设置随机种子确保推理结果可复现
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        attn_impl = "flash_attention_2" if use_flash_attention else "eager"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = torch.device(device)
        self.batch_size = batch_size

    def batch_inference(self, messages, max_new_tokens=128):
        """批量推理

        Args:
            messages: list of list of dict - 每个元素是一个对话消息列表
            max_new_tokens: int - 最大生成 token 数

        Returns:
            list of str: 每个消息的推理结果
        """
        import torch
        from qwen_vl_utils import process_vision_info

        all_outputs = []

        # 分批处理
        for batch_start in range(0, len(messages), self.batch_size):
            batch_messages = messages[batch_start:batch_start + self.batch_size]

            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages
            ]
            image_inputs, video_inputs = process_vision_info(batch_messages)

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            all_outputs.extend(output_texts)

        return all_outputs

    def infer_semantic(self, images_path: list, question: str):
        """对齐评估推理

        对一组图片提出 Yes/No 问题。

        Args:
            images_path: list of str - 图片路径列表
            question: str - 问题文本

        Returns:
            list of str: 每个图片的推理结果
        """
        messages = []
        for image_path in images_path:
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": f"{question}. Please answer 'Yes' or 'No' only."}
                    ],
                }
            ])
        return self.batch_inference(messages)

    def infer_ocr(self, images_path: list, max_new_tokens: int = 128):
        """文本评估推理

        对一组图片进行 OCR 识别。

        Args:
            images_path: list of str - 图片路径列表
            max_new_tokens: int - 最大生成 token 数

        Returns:
            list of str: 每个图片的 OCR 结果
        """
        TEXT_PROMPT = (
            "Recognize the text in the image, only reply with the text content, "
            "but avoid repeating previously mentioned content. "
            "If no text is recognized, please reply with 'No text recognized'."
        )
        messages = []
        for image_path in images_path:
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": TEXT_PROMPT}
                    ],
                }
            ])
        return self.batch_inference(messages, max_new_tokens=max_new_tokens)
