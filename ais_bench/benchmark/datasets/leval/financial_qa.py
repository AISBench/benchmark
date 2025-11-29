from datasets import Dataset, load_dataset

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import DSET_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError

from ..base import BaseDataset

logger = AISLogger()


@LOAD_DATASET.register_module()
class LEvalFinancialQADataset(BaseDataset):
    """
    LEval Financial QA dataset loader.
    Loads and processes the L-Eval Financial QA benchmark dataset, which contains
    financial documents with associated question-answer pairs.

    The dataset is flattened from a nested structure where each document context
    may have multiple question-answer pairs into individual samples with
    (question, context, answer). Additionally, the length of each answer is
    computed and included in the sample.
    """

    @staticmethod
    def load(**kwargs):
        if 'path' not in kwargs:
            raise AISBenchConfigError(DSET_CODES.INVALID_DATASET_CONFIG, "The 'path' argument is required to load the dataset.")
        path = kwargs['path']
        logger.info(f"Loading LEval Financial QA dataset from path: {path}")
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

            logger.debug(f"Item {i}: {len(instructions)} Q/A pairs")

            for question, answer in zip(instructions, outputs):
                raw_data.append({
                    'question': question,
                    'context': context,
                    'length': len(answer.split()),
                    'answer': answer
                })

        logger.info(
            f"Dataset loading completed: {len(raw_data)} flattened Q/A pairs")
        dataset[split] = Dataset.from_list(raw_data)
        return dataset
