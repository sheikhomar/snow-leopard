"""Label noise detection algorithms."""
import abc
from typing import List

from app.data.datasets import DataSet


class Algorithm:
    """Represents a label noise detection algorithm."""

    @abc.abstractmethod
    def run(self, data_set: DataSet) -> List[int]:
        """Returns the name of the algorithm."""
        raise NotImplementedError
