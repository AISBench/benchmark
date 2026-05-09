import re

from ais_bench.benchmark.registry import TEXT_POSTPROCESSORS
from ais_bench.benchmark.utils.logging.logger import AISLogger


logger = AISLogger()
logger.warning("ais_bench.benchmark.utils.model_postprocessors is deprecated, please use ais_bench.benchmark.utils.postprocess.model_postprocessors instead.")


@TEXT_POSTPROCESSORS.register_module('extract-non-reasoning-content')
def extract_non_reasoning_content(
    text: str | list,
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

    def _process_single(item: str):
        # Keep historical behavior for non-string input.
        if not isinstance(item, str):
            return item
        if think_start_token not in item and think_end_token in item:
            result = item.split(think_end_token)[-1].strip()
            logger.debug(
                f"extract_non_reasoning_content: only end token present -> length={len(result)}"
            )
            return result

        reasoning_regex = re.compile(rf'{think_start_token}(.*?){think_end_token}',
                                     re.DOTALL)
        non_reasoning_content = reasoning_regex.sub('', item).strip()
        logger.debug(
            f"extract_non_reasoning_content: removed reasoning sections -> length={len(non_reasoning_content)}"
        )
        return non_reasoning_content

    if isinstance(text, list):
        logger.debug(
            f"extract_non_reasoning_content: processing list of {len(text)} item(s)"
        )
        return [_process_single(item) for item in text]

    logger.debug("extract_non_reasoning_content: processing single item")
    return _process_single(text)