from datasets import Dataset, load_dataset

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.logging.error_codes import DSET_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class LEvalQualityDataset(BaseDataset):
    """
    LEval Quality dataset loader.

    Loads and processes the L-Eval QuALITY benchmark dataset, which contains
    long articles and stories with multiple-choice questions for reading comprehension.

    The dataset is flattened from a nested structure where each context may have
    multiple question-answer pairs into individual samples with (question, context, answer).
    Note: answer[1] is used to extract the correct answer option from the output.
    """

    @staticmethod
    def load(**kwargs):
        if 'path' not in kwargs:
            raise AISBenchConfigError(DSET_CODES.INVALID_DATASET_CONFIG, "The 'path' argument is required to load the dataset.")

        path = kwargs['path']
        full_path = get_data_path(path, local_mode=True)
        split = 'test'
        dataset = load_dataset('json', data_files={split: full_path})
        raw_data = []
        for i in range(len(dataset[split])):
            instructions = dataset[split]['instructions'][i]
            outputs = dataset[split]['outputs'][i]
            context = dataset[split]['input'][i]
            for question, answer in zip(instructions, outputs):
                raw_data.append({
                    'question': question,
                    'context': context,
                    'answer': answer[1]
                })
        dataset[split] = Dataset.from_list(raw_data)
        return dataset
