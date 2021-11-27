from utils.utils import read_config
from typing import TypeVar, Iterable, Iterator, Sequence, List, Optional, Dict, Tuple
from src.preprocess import SoundPreprocess


class IterableDataset:
    def __iter__(self):
        raise NotImplementedError

    def __add__(self):
        raise NotImplementedError

    def __getattr__(self):
        raise NotImplementedError


class SoundPreprocessIter(IterableDataset):
    def __init__(self):
        super(SoundPreprocessIter).__init__()
        self.dataset = self.get_dataset()
        self.start = 0
        self.end = len(self.dataset.files)

    def __iter__(self):
        return iter(self.dataset[ix] for ix in range(self.start, self.end))

    def get_dataset(self) -> "SoundPreprocess":
        return SoundPreprocess(config_path="config/config.yaml")
