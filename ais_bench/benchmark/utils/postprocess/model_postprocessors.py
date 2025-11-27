import re

from ais_bench.benchmark.registry import TEXT_POSTPROCESSORS
from ais_bench.benchmark.utils.logging import AISLogger

logger = AISLogger()

def list_decorator(func):
    """Decorator: make the function able to handle list input"""
    def wrapper(text_or_list, *args, **kwargs):
        if isinstance(text_or_list, list):
            logger.debug(
                f"list_decorator({func.__name__}): processing list of {len(text_or_list)} item(s)"
            )
            return [func(text, *args, **kwargs) for text in text_or_list]
        logger.debug(
            f"list_decorator({func.__name__}): processing single item"
        )
        return func(text_or_list, *args, **kwargs)
    return wrapper


@TEXT_POSTPROCESSORS.register_module('extract-non-reasoning-content')
@list_decorator
def extract_non_reasoning_content(
    text: str,
    think_start_token: str = '<think>',
    think_end_token: str = '</think>',
) -> str:
    """Extract content after the last reasoning tag from text.

    When only end token is present, returns content after the end token.
    When both tokens are present, removes all content between start and end tokens.

    Args:
        text (str): Input text containing reasoning tags.
        think_start_token (str, optional): Start token for reasoning section. Defaults to '<think>'.
        think_end_token (str, optional): End token for reasoning section. Defaults to '</think>'.

    Returns:
        str: Processed text after removing reasoning sections.

    Examples:
        >>> # When only end token exists
        >>> text = "This is a test.</think> How are you?"
        >>> extract_non_reasoning_content(text)
        'How are you?'

        >>> # When both tokens exist
        >>> text = "Start<think>reasoning here</think> End"
        >>> extract_non_reasoning_content(text)
        'Start End'
        
        >>> # When input is a list
        >>> texts = ["Start<think>reasoning</think> End", "Test</think> Result"]
        >>> extract_non_reasoning_content(texts)
        ['Start End', 'Result']
    """
    logger.debug(
        f"extract_non_reasoning_content: start_token='{think_start_token}', end_token='{think_end_token}'"
    )
    # If text contains only end token, split by end token and take the last part
    if not isinstance(text, str):
        return text
    if think_start_token not in text and think_end_token in text:
        result = text.split(think_end_token)[-1].strip()
        logger.debug(
            f"extract_non_reasoning_content: only end token present -> length={len(result)}"
        )
        return result

    # Original behavior for complete tag pairs
    reasoning_regex = re.compile(rf'{think_start_token}(.*?){think_end_token}',
                                 re.DOTALL)
    non_reasoning_content = reasoning_regex.sub('', text).strip()
    logger.debug(
        f"extract_non_reasoning_content: removed reasoning sections -> length={len(non_reasoning_content)}"
    )
    return non_reasoning_content
