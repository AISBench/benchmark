"""Simple Dataset Reader."""

import random
from typing import List, Optional, Union

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.registry import ICL_DATASET_READERS
from ais_bench.benchmark.utils.core.types import (check_dataset, check_str, check_type_list)
from ais_bench.benchmark.utils.logging.logger import AISLogger


logger = AISLogger()


@ICL_DATASET_READERS.register_module()
class DatasetReader:
    """In-conext Learning Dataset Reader Class Generate an DatasetReader
    instance through 'dataset'.

    Attributes:
        dataset (:obj:`Dataset` or :obj:`DatasetDict`): The dataset to be read.
        input_columns (:obj:`List[str]` or :obj:`str`): A list of column names
            (a string of column name) in the dataset that represent(s) the
            input field.
        output_column (:obj:`str`): A column name in the dataset that
            represents the prediction field.
        input_template (:obj:`PromptTemplate`, optional): An instance of the
            :obj:`PromptTemplate` class, used to format the input field
            content during the retrieval process. (in some retrieval methods)
        output_template (:obj:`PromptTemplate`, optional): An instance of the
            :obj:`PromptTemplate` class, used to format the output field
            content during the retrieval process. (in some learnable retrieval
            methods)
        train_split (str): The name of the training split. Defaults to 'train'.
        train_range (int or float or str, optional): The size of the partial
            training dataset to load.
            If None, the entire training dataset will be loaded.
            If int or float, the random partial dataset will be loaded with the
            specified size.
            If str, the partial dataset will be loaded with the
            specified index list (e.g. "[:100]" for the first 100 examples,
            "[100:200]" for the second 100 examples, etc.). Defaults to None.
        test_split (str): The name of the test split. Defaults to 'test'.
        test_range (int or float or str, optional): The size of the partial
            test dataset to load.
            If None, the entire test dataset will be loaded.
            If int or float, the random partial dataset will be loaded with the
            specified size.
            If str, the partial dataset will be loaded with the
            specified index list (e.g. "[:100]" for the first 100 examples,
            "[100:200]" for the second 100 examples, etc.). Defaults to None.
    """
    dataset = None
    input_template = None
    output_template = None

    def __init__(self,
                 dataset: Union[Dataset, DatasetDict, str],
                 input_columns: Union[List[str], str],
                 output_column: Optional[str],
                 max_tokens_column: Optional[str] = None,
                 input_template: Optional[PromptTemplate] = None,
                 output_template: Optional[PromptTemplate] = None,
                 train_split: str = 'train',
                 train_range: Optional[Union[int, float, str]] = None,
                 test_split: str = 'test',
                 test_range: Optional[Union[int, float, str]] = None) -> None:
        self.input_columns = check_type_list(input_columns, [List, str])
        if isinstance(self.input_columns, str):
            self.input_columns = self.input_columns.split()
        self.output_column = None
        if output_column:
            self.output_column = check_str(output_column)
        self.max_tokens_column = None
        if max_tokens_column:
            self.max_tokens_column = check_str(max_tokens_column)
        train_range = check_type_list(train_range, [None, int, float, str])
        test_range = check_type_list(test_range, [None, int, float, str])

        self.dataset = check_dataset(dataset)
        if isinstance(self.dataset, Dataset):
            self.dataset = DatasetDict({
                'train': self.dataset,
                'test': self.dataset
            })

        # Normalize the dataset so that it has only "train" and "test" splits.
        for origin_split, mapped_split, split_range in [[
                train_split, 'train', train_range
        ], [test_split, 'test', test_range]]:
            logger.debug(f"Loading {mapped_split} split with range {split_range}")
            self.dataset[mapped_split] = load_partial_dataset(
                self.dataset[origin_split], size=split_range)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __repr__(self):
        return (f'DatasetReader({{\n    dataset: {self.dataset},'
                f'\n    input_columns: {self.input_columns},\n'
                f'    output_columns: {self.output_column}\n}})')

    def get_max_out_len(self):
        if "max_out_len" in self.input_columns and "max_out_len" in self.dataset['test'].features:
            return self.dataset['test']['max_out_len']
        else:
            return None


def load_partial_dataset(
        dataset: Dataset,
        size: Optional[Union[int, float, str]] = None) -> Dataset:
    """Load a partial dataset.

    Args:
        dataset (Dataset): A :obj:`datasets.Dataset` instance.
        size (int or float or (int, int), optional): The size of the partial
            dataset to load. If None, the entire dataset will be loaded.
            If int or float, the random partial dataset will be loaded with the
            specified size. If str, the partial dataset will be loaded with the
            specified index list (e.g. "[:100]" for the first 100 examples,
            "[100:200]" for the second 100 examples, etc.). Defaults to None.
    """
    total_size = len(dataset)
    index_list = list(range(total_size))
    if isinstance(size, (int, float)):
        if size >= total_size or size <= 0:
            logger.debug(f"Size is out of range, loaded entire dataset")
            return dataset
        if size > 0 and size < 1:
            size = int(size * total_size)
        rand = random.Random(x=size)
        rand.shuffle(index_list)
        dataset = dataset.select(index_list[:size])
        logger.debug(f"Loaded {size} random examples from dataset")
    elif isinstance(size, str):
        try:
            dataset = dataset.select(eval(f'index_list{size}'))
        except Exception as e:
            logger.warning(f"Cannot parse size string: {size}, use entire dataset instead")
    else:
        if size is not None:
            logger.warning(f"Invalid size type: {type(size)}, use entire dataset instead")
    return dataset
