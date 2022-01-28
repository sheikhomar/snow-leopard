import abc

from dataclasses import dataclass, field
from typing import Dict, Generator, List, Set

import pandas as pd
import numpy as np

from cleanlab.noise_generation import generate_noisy_labels
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import notebooks.feature as feature_engineering


@dataclass
class DataSplit:
    fold: int
    train_indices: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    test_indices: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


class DataSet(abc.ABC):

    @property
    @abc.abstractmethod
    def size(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_classes(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def label_attribute(self) -> str:
        """The name for the label attribute."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def y_given(self) -> str:
        """Returns the labels of all samples in the data set."""
        raise NotImplementedError

    @abc.abstractmethod
    def create_feature_transformer(self) -> TransformerMixin:
        raise NotImplementedError

    @abc.abstractmethod
    def get_raw_data_by_indices(self, indices: np.ndarray) -> pd.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def split(self, n_splits: int, shuffle: bool = True, random_state=None) -> Generator[DataSplit, None, None]:
        raise NotImplementedError


class IceCatOfficeDataSet(DataSet):
    def __init__(self, file_path: str, min_num_samples_per_category: int = 20) -> None:
        self._file_path = file_path
        self._min_num_samples_per_category = min_num_samples_per_category

        # Load data
        df_data = pd.read_csv(self._file_path, dtype=str, index_col=0)

        # Filter samples from small categories
        category_counts = df_data[self.label_attribute].value_counts()
        large_enough_categories = category_counts[category_counts >= min_num_samples_per_category].index.tolist()
        self._df_data = df_data[df_data.category_name.isin(large_enough_categories)]

        # Save a copy of the original labels for each sample.
        # This allows us to apply multiple label noise matrices.
        self._original_labels = list(self._df_data[self.label_attribute])

        # Encode labels
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(self._df_data[self.label_attribute])
        self._y_given = self._label_encoder.transform(self._df_data[self.label_attribute])
    
    @property
    def label_attribute(self) -> str:
        return "category_name"

    @property
    def size(self) -> int:
        return self._df_data.shape[0]

    @property
    def n_classes(self) -> int:
        return self._label_encoder.classes_.size

    @property
    def y_given(self) -> str:
        return self._y_given

    def create_feature_transformer(self) -> TransformerMixin:
        return feature_engineering.BasicIceCatFeatureTransformer(output_size=0.99)

    def get_raw_data_by_indices(self, indices: np.ndarray) -> pd.DataFrame:
        return self._df_data.iloc[indices]

    def split(self, n_splits: int, shuffle: bool = True, random_state=None):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        X = self._df_data
        y = self._df_data[self.label_attribute]
        for fold, (train_index, test_index) in enumerate(skf.split(X=X, y=y)):
            df_train = self._df_data.iloc[train_index]
            df_test = self._df_data.iloc[test_index]

            feateure_transformer = self.create_feature_transformer()
            feateure_transformer.fit(df_train)

            X_train = feateure_transformer.transform(df_train)
            X_test = feateure_transformer.transform(df_test)
            y_train = self._label_encoder.transform(df_train[self.label_attribute])
            y_test = self._label_encoder.transform(df_test[self.label_attribute])

            yield DataSplit(
                fold=fold,
                train_indices=train_index,
                X_train=X_train,
                y_train=y_train,
                test_indices=test_index,
                X_test=X_test,
                y_test=y_test
            )

    def apply_label_noise(self, noise_matrix: np.ndarray) -> None:
        # Reload original labels before applying a new noise matrix.
        # This resets earlier noise matrices.
        self._df_data[self.label_attribute] = self._original_labels
        y_original = self._label_encoder.transform(self._original_labels)

        # Generate noisy labels from the labels.
        y_modified = generate_noisy_labels(y=y_original, noise_matrix=noise_matrix)
        
        # Determine which sample indices have been modified.
        modified_indices, = np.where(y_modified != y_original)

        # Modify the internal dataframe.
        new_labels = [self._label_encoder.classes_[l] for l in y_modified]
        self._df_data[self.label_attribute] = new_labels

        # Transform to label indices.
        self._y_given = self._label_encoder.transform(self._df_data[self.label_attribute])

        return modified_indices       


def get_dataset(file_path: str) -> DataSet:
    if "ice-cat-office" in file_path:
        return IceCatOfficeDataSet(file_path=file_path)
    
    raise Exception(f"Do not know how to parse file {file_path}")
