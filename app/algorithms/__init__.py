"""Label noise detection algorithms."""
import abc
from typing import List, Dict

from app.data.datasets import DataSet


class Algorithm:
    """Represents a label noise detection algorithm."""

    @abc.abstractmethod
    def run(self, data_set: DataSet) -> List[int]:
        """Returns the name of the algorithm."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_params(self) -> Dict[str, object]:
        raise NotImplementedError
