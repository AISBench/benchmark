import os

from transformers import AutoTokenizer
from typing import List, Tuple

from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import FileOperationError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES

__all__ = ['load_tokenizer', 'AISTokenizer']

logger = AISLogger()

def load_tokenizer(tokenizer_path: str):
    """Load a tokenizer from the specified path.
    
    Args:
        tokenizer_path: Path to the tokenizer directory or model identifier
        
    Returns:
        AutoTokenizer: Loaded tokenizer instance
        
    Raises:
        FileOperationError: If tokenizer path doesn't exist or loading fails
    """
    logger.debug(f"Attempting to load tokenizer from: {tokenizer_path}")
    
    if not os.path.exists(tokenizer_path):
        logger.warning(f"Tokenizer path does not exist: {tokenizer_path}")
        raise FileOperationError(
            UTILS_CODES.TOKENIZER_PATH_NOT_FOUND,
            f"Tokenizer path '{tokenizer_path}' does not exist"
        )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Successfully loaded tokenizer from: {tokenizer_path}")
        return tokenizer
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from {tokenizer_path}: {type(e).__name__}: {e}")
        raise FileOperationError(
            UTILS_CODES.TOKENIZER_LOAD_FAILED,
            f"Failed to load tokenizer from {tokenizer_path}: {type(e).__name__}: {e}"
        ) from e
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
