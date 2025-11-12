import os

from transformers import AutoTokenizer
from typing import List, Tuple

def load_tokenizer(tokenizer_path: str):
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer path {tokenizer_path} does not exist")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer from {tokenizer_path}: {e}")
    return tokenizer


class AISTokenizer:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = load_tokenizer(tokenizer_path)

    def encode(self, prompt: list) -> Tuple[float, List[int]]:
        """Encode a string into tokens, measuring processing time."""
        if isinstance(prompt, list):
            messages = self.tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=False
            )
        elif isinstance(prompt, str):
            messages = prompt
        else:
            # self.logger.error(f"Prompt: {prompt} is not a list or string.")
            return []
        tokens = self.tokenizer.encode(messages)
        return tokens

    def decode(self, tokens: List[int]) -> Tuple[List[float], str]:
        return self.tokenizer.decode(tokens)