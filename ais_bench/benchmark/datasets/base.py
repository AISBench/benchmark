from abc import abstractmethod
from typing import List, Dict, Optional, Union

from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets.utils.logging import disable_progress_bar

from ais_bench.benchmark.openicl.icl_dataset_reader import DatasetReader

disable_progress_bar() # disable mapping progress bar, preventing terminal interface contamination


class BaseDataset:

    def __init__(self,
                 reader_cfg: Optional[Dict] = {},
                 k: Union[int, List[int]] = 1,
                 n: int = 1,
                 **kwargs):
        # maybe duplicate
        assert (max(k) if isinstance(k, List) else
                k) <= n, 'Maximum value of `k` must less than or equal to `n`'
        self.abbr = kwargs.pop('abbr', 'dataset')
        
        self.dataset = self.load(**kwargs)
        self._init_reader(**reader_cfg)
        self.repeated_dataset(self.abbr, n) # this process will update self.dataset and self.reader.dataset


    def _init_reader(self, **kwargs):
        self.reader = DatasetReader(self.dataset, **kwargs)
    

    def repeated_dataset(self, abbr: str, n: int):
        # Create repeated indices in batches to avoid generating an oversized index list at once
        def create_repeated_indices(length: int, n: int, batch_size: int = 10000) -> List[int]:
            """Generate repeated indices in batches to prevent memory peaks"""
            indices = []
            for start in range(0, length, batch_size):
                end = min(start + batch_size, length)
                batch_indices = [i for i in range(start, end) for _ in range(n)]
                indices.extend(batch_indices)
            return indices
        
        if isinstance(self.reader.dataset, Dataset):
            # Add metadata fields (use batching for efficiency)
            base_size = len(self.reader.dataset)
            writer_batch_size = max(min(base_size // 100, 1000), 16) # Batch size for dataset mapping writes (optimize memory usage).
            index_gen_batch_size = max(min(base_size // 10, 50000), 1000) # Batch size for index generation (prevent memory peaks).

            dataset = self.reader.dataset.map(
                lambda x, idx: {'subdivision': abbr, 'idx': idx},
                with_indices=True,
                writer_batch_size=writer_batch_size,
                load_from_cache_file=False
            )
            
            # Safely generate indices
            orig_len = len(dataset)
            indices = create_repeated_indices(orig_len, n, batch_size=index_gen_batch_size)
            
            # Achieve sample duplication through index selection
            self.reader.dataset = dataset.select(indices)
            
        else:
            # Handle DatasetDict cases
            new_dict = DatasetDict()
            for key in self.reader.dataset:
                # Add metadata fields (using batching)
                base_size = len(self.reader.dataset[key])
                writer_batch_size = max(min(base_size // 100, 1000), 16)
                index_gen_batch_size = max(min(base_size // 10, 50000), 1000)

                mapped_ds = self.reader.dataset[key].map(
                    lambda x, idx: {'subdivision': f'{abbr}_{key}', 'idx': idx},
                    with_indices=True,
                    writer_batch_size=writer_batch_size,
                    load_from_cache_file=False
                )
                
                orig_len = len(mapped_ds)
                indices = create_repeated_indices(orig_len, n, batch_size=index_gen_batch_size)
                
                new_dict[key] = mapped_ds.select(indices)
                
            self.reader.dataset = new_dict
        
        self.dataset = self.reader.dataset


    @property
    def train(self):
        return self.reader.dataset['train']

    @property
    def test(self):
        return self.reader.dataset['test']

    @abstractmethod
    def load(**kwargs) -> Union[Dataset, DatasetDict]:
        pass
