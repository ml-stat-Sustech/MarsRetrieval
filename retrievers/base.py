from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class RetrieverBase(ABC):
    """Base retriever interface."""

    @abstractmethod
    def search(self, query) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def to_dataframe(self, results: Dict) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def save_results(self, output_dir: str, df_results, timestamp: str) -> None:
        raise NotImplementedError
