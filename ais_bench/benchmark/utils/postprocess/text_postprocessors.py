import re
from typing import Callable, Optional, Union
from ast import literal_eval

from ais_bench.benchmark.registry import TEXT_POSTPROCESSORS
from ais_bench.benchmark.utils.logging import AISLogger


logger = AISLogger()


@TEXT_POSTPROCESSORS.register_module('general')
def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    # Remove article
    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()

    logger.debug(f"general_postprocess: result length={len(cleaned_text)}")
    return cleaned_text


@TEXT_POSTPROCESSORS.register_module('general_cn')
def general_cn_postprocess(text: str) -> str:
    logger.debug("general_cn_postprocess: start")
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*', category=UserWarning)
        import jieba
    cleaned_text = ' '.join(jieba.cut(text))
    logger.debug(f"general_cn_postprocess: result length={len(cleaned_text)}")
    return cleaned_text


@TEXT_POSTPROCESSORS.register_module('first-capital')
def first_capital_postprocess(text: str) -> str:
    for t in text:
        if t.isupper():
            logger.debug(f"first_capital_postprocess: found '{t}'")
            return t
    logger.debug("first_capital_postprocess: no capital found")
    return ''


@TEXT_POSTPROCESSORS.register_module('last-capital')
def last_capital_postprocess(text: str) -> str:
    for t in text[::-1]:
        if t.isupper():
            logger.debug(f"last_capital_postprocess: found '{t}'")
            return t
    logger.debug("last_capital_postprocess: no capital found")
    return ''


def first_option_postprocess_v1(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text, prioritizing the latest match in the text."""
    cut_length = 200
    cut_text = text[-cut_length:]
    result = first_option_postprocess(cut_text, options, cushion)
    logger.debug(
        f"first_option_postprocess_v1: options='{options}', found='{result or ''}'"
    )
    return result

def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        rf'答案是?\s*([{options}])',
        rf'答案是?\s*：\s*([{options}])',
        rf'答案是?\s*:\s*([{options}])',
        rf'答案选项应?该?是\s*([{options}])',
        rf'答案选项应?该?为\s*([{options}])',
        rf'答案应该?是\s*([{options}])',
        rf'答案应该?选\s*([{options}])',
        rf'答案选项为?\s*：\s*([{options}])',
        rf'答案选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        rf'答案选项是?\s*:\s*([{options}])',
        rf'答案为\s*([{options}])',
        rf'答案选\s*([{options}])',
        rf'选择?\s*([{options}])',
        rf'故选?\s*([{options}])'
        rf'只有选?项?\s?([{options}])\s?是?对',
        rf'只有选?项?\s?([{options}])\s?是?错',
        rf'只有选?项?\s?([{options}])\s?不?正确',
        rf'只有选?项?\s?([{options}])\s?错误',
        rf'说法不?对选?项?的?是\s?([{options}])',
        rf'说法不?正确选?项?的?是\s?([{options}])',
        rf'说法错误选?项?的?是\s?([{options}])',
        rf'([{options}])\s?是正确的',
        rf'([{options}])\s?是正确答案',
        rf'选项\s?([{options}])\s?正确',
        rf'所以答\s?([{options}])',
        rf'所以\s?([{options}][.。$]?$)',
        rf'所有\s?([{options}][.。$]?$)',
        rf'[\s，：:,]([{options}])[。，,\.]?$',
        rf'[\s，,：:][故即]([{options}])[。\.]?$',
        rf'[\s，,：:]因此([{options}])[。\.]?$',
        rf'[是为。]\s?([{options}])[。\.]?$',
        rf'因此\s?([{options}])[。\.]?$',
        rf'显然\s?([{options}])[。\.]?$',
        r'答案是\s?(\S+)(?:。|$)',
        r'答案应该是\s?(\S+)(?:。|$)',
        r'答案为\s?(\S+)(?:。|$)',
        rf'(?i)ANSWER\s*:\s*([{options}])',
        rf'[Tt]he answer is:?\s+\(?([{options}])\)?',
        rf'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        rf'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        rf'^选项\s?([{options}])',
        rf'^([{options}])\s?选?项',
        rf'(\s|^)[{options}][\s。，,：:\.$]',
        r'1.\s?(.*?)$',
        rf'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        rf'([{options}]):',
        rf'([{options}])',
    ]
    # flake8: noqa
    # yapf: enable

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if match.group(1) is not None and match.group(1) != '':
                outputs = match.group(1)
            else:
                outputs = match.group(0)
            for i in options:
                if i in outputs:
                    logger.debug(
                        f"first_option_postprocess: options='{options}', found='{i}'"
                    )
                    return i
    logger.debug("first_option_postprocess: no option matched")
    return ''


@TEXT_POSTPROCESSORS.register_module('first-capital-multi')
def first_capital_postprocess_multi(text: str) -> str:
    match = re.search(r'([A-D]+)', text)
    if match:
        logger.debug(f"first_capital_postprocess_multi: found '{match.group(1)}'")
        return match.group(1)
    logger.debug("first_capital_postprocess_multi: no match")
    return ''


def last_option_postprocess(text: str, options: str) -> str:
    match = re.findall(rf'([{options}])', text)
    if match:
        logger.debug(f"last_option_postprocess: found '{match[-1]}'")
        return match[-1]
    logger.debug("last_option_postprocess: no match")
    return ''


def first_number_postprocess(text: str) -> float:
    """Return the first number in a string."""
    # regex pattern to match numbers (both integers and decimals)
    pattern = r'(-?\d*\.?\d+)'

    # search the string for the pattern
    match = re.search(pattern, text)

    # if a match is found, return it. Otherwise, return None.
    result = float(match.group(1)) if match else None
    logger.debug(f"first_number_postprocess: found '{result}'")
    return result


@TEXT_POSTPROCESSORS.register_module('multiple-select')
def multiple_select_postprocess(text: str) -> str:
    # Extracts all unique uppercase letters, sorts them alphabetically, and joins them.
    # For example, 'Select A and B' will yield 'AB'.
    ret = set([t for t in text if t.isupper()])
    result = ''.join(sorted(ret))
    logger.debug(f"multiple_select_postprocess: result='{result}'")
    return result


@TEXT_POSTPROCESSORS.register_module('specific-xml-tag')
def xml_tag_postprocessor(text, tag):
    """Extracts content enclosed within a specified XML-style tag from a
    string.

    Args:
        texts: The input string containing XML-style tags.
        tag: The XML-style tag to extract content from (e.g., "<conclude>").  Must include the angle brackets.

    Returns:
        The content enclosed within the specified tag, or None if the tag is not found.
    """

    # Use a regular expression to find the content within the specified tag.  This handles cases where the tag might appear multiple times.
    matches = re.findall(
        rf'{tag}(.*?)</{tag[1:-1]}>', text,
        re.DOTALL)  # re.DOTALL allows . to match newline characters

    if matches:
        # Only keep the last one
        output = matches[-1].strip(
        )  # Extract the content and remove leading/trailing whitespace
        logger.debug(
            f"xml_tag_postprocessor: tag={tag}, matches={len(matches)}, returning last length={len(output)}"
        )
    else:
        output = 'NO ANSWER FOUND'
        logger.debug(f"xml_tag_postprocessor: tag={tag}, no match -> 'NO ANSWER FOUND'")

    return output


def general_eval_wrapper_postprocess(text: str,
                                     postprocess: Optional[Union[
                                         str, Callable]] = None,
                                     **kwargs) -> str:
    """Wrapper for eval text repr. Especially for chatglmpro.

    Args:
        text(str): Text to be postprocessed.
        postprocess(Callable, optional): Original post processing function.
            Defaults to None.
        **kwargs: Other necessary kwargs for post processing function.
    """
    try:
        text = literal_eval(text)
    except Exception:
        # in case empty input or other error, skip eval
        logger.warning(
            f"general_eval_wrapper_postprocess: literal_eval failed, using raw text (truncated): {text[:100]}{'...' if len(text) > 100 else ''}"
        )

    if postprocess:
        if isinstance(postprocess, str):
            postprocess_name = postprocess
            postprocess = TEXT_POSTPROCESSORS.get(postprocess)
            logger.debug(
                f"general_eval_wrapper_postprocess: resolved postprocess='{postprocess_name}' -> {postprocess}"
            )
        result = postprocess(text, **kwargs)
        logger.debug("general_eval_wrapper_postprocess: postprocess applied")
        return result
    logger.debug("general_eval_wrapper_postprocess: no postprocess provided")
    return text


def match_answer_pattern(response_text: str, answer_pattern: str):
    match = re.search(answer_pattern, response_text)
    extracted_answer = match.group(1) if match else ''
    logger.debug(
        f"match_answer_pattern: pattern='{answer_pattern}', matched={bool(match)}, length={len(extracted_answer)}"
    )
    return extracted_answer
