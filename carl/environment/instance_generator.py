# pylint: disable=abstract-method
import os
from abc import ABC
from collections.abc import Iterator
from itertools import chain
from typing import Any

import loguru
from joblib import load
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset


class InstanceGenerator(ABC):
    """This class is used to generate samples from a given generator."""
    def __init__(self, generator: Any, batch_size: int) -> None:
        self.generator = generator
        self.batch_size = batch_size

    def reset_dataloader(self) -> DataLoader:
        """
        This method is to produce a dataloader from the generator. Can be used (dataloader) to iterate over the data,
        i.e. with "for" loop.
        """
        raise NotImplementedError


class GeneralIterableDataLoader(IterableDataset):
    """
    This class is used to load data from a folder with pickle files and return an iterator over the data.
    """
    def __init__(self, path_to_folder_with_data: str) -> None:
        super().__init__()
        self.path_to_folder_with_data = path_to_folder_with_data
        loguru.logger.debug(f'path: {path_to_folder_with_data}')
        if path_to_folder_with_data is not None and not os.path.exists(self.path_to_folder_with_data):
            loguru.logger.debug(
                f'Path {self.path_to_folder_with_data} does not exist. Trying to find it in the parent folder.')
            base_folder = path_to_folder_with_data.split('/')[-1]
            self.path_to_folder_with_data = base_folder

    @staticmethod
    def process_data(data: str) -> Iterator[Any]:
        loguru.logger.info(f'Processing data from {data}')
        data: list[Any] = load(data)
        yield from data

    def get_stream(self):
        data_list: list[str] = [
            os.path.join(self.path_to_folder_with_data, f)
            for f in os.listdir(self.path_to_folder_with_data)
            if f.endswith(('.joblib', '.pkl'))
        ]
        loguru.logger.debug(f'Found {len(data_list)} files in {self.path_to_folder_with_data}')
        return chain.from_iterable(map(self.process_data, data_list))

    def __iter__(self):
        return self.get_stream()


class GeneralIterableDataLoaderFromFile(GeneralIterableDataLoader):
    """
    Same as above but for a single file.
    """
    def __init__(self, path_to_file_with_data: str) -> None:
        super().__init__(path_to_file_with_data)

    def get_stream(self):
        if os.path.isfile(self.path_to_folder_with_data):    # in that case we have a single file
            with open(self.path_to_folder_with_data, 'rb') as f:
                data: list[Any] = load(
                    f)    # for instance a list of nd arrays or a list of strings, any problem instance state
                return iter(data)
        loguru.logger.error(f'File {self.path_to_folder_with_data} does not exist.')
        return None


class BasicInstanceGenerator(InstanceGenerator):
    def __init__(self, generator: GeneralIterableDataLoader, batch_size: int) -> None:
        super().__init__(generator, batch_size)

    def reset_dataloader(self) -> DataLoader:
        return DataLoader(self.generator, self.batch_size)
