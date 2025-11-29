from datasets import Dataset, load_dataset

from ais_bench.benchmark.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import DSET_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError

from ais_bench.benchmark.datasets.base import BaseDataset

logger = AISLogger()


@TEXT_POSTPROCESSORS.register_module('gsm100_dataset')
def gsm100_dataset_postprocess(text: str) -> str:
    logger.debug(
        f"Applying gsm100_dataset_postprocess: input='{text[:50]}...'")
    result = text.replace(',', '')
    logger.debug(f"gsm100_dataset_postprocess result: '{result[:50]}...'")
    return result


@TEXT_POSTPROCESSORS.register_module('gsm100')
def gsm100_postprocess(text: str) -> str:
    logger.debug(f"Applying gsm100_postprocess: input='{text[:50]}...'")
    segs = text.split('The answer is')
    EXPECT_GSM100_ANSWER_SEPARATE_LEN = 2
    if len(segs) < EXPECT_GSM100_ANSWER_SEPARATE_LEN:
        logger.warning(
            "gsm100_postprocess: 'The answer is' not found in text, returning empty string")
        return ''
    text = segs[1]
    text = text.split(' ')
    flag = False
    ret = ''
    for word in text:
        for char in word:
            if char.isdigit():
                flag = True
                ret = word
                break
        if flag:
            break
    ret1 = ''
    for char in ret:
        if char.isdigit():
            ret1 += char
    logger.debug(f"gsm100_postprocess result: '{ret1}'")
    return ret1


@LOAD_DATASET.register_module()
class LEvalGSM100Dataset(BaseDataset):
    """
    LEval GSM100 dataset loader.

    Loads and processes the L-Eval GSM100 benchmark dataset, which contains
    grade school math problems in long-context format. Each sample consists of
    a mathematical problem, context, and numerical answer.

    The dataset is flattened from a nested structure where each context may have
    multiple question-answer pairs into individual samples with (question, context, answer).
    """

    @staticmethod
    def load(**kwargs):
        if 'path' not in kwargs:
            raise AISBenchConfigError(DSET_CODES.INVALID_DATASET_CONFIG, "The 'path' argument is required to load the dataset.")

        path = kwargs['path']
        logger.info(f"Loading LEval GSM100 dataset from path: {path}")
        full_path = get_data_path(path, local_mode=True)
        logger.debug(f"Resolved full path: {full_path}")
        split = 'test'
        dataset = load_dataset('json', data_files={split: full_path})
        raw_data = []
        total_items = len(dataset[split])
        logger.info(f"Processing {total_items} items from {split} split")
        for i in range(total_items):
            instructions = dataset[split]['instructions'][i]
            outputs = dataset[split]['outputs'][i]
            context = dataset[split]['input'][i]
            for question, answer in zip(instructions, outputs):
                raw_data.append({
                    'question': question,
                    'context': context,
                    'answer': answer
                })
        dataset[split] = Dataset.from_list(raw_data)
        logger.info(
            f"Dataset loading completed: {len(raw_data)} processed items")
        return dataset
