from typing import List

import numpy as np

from cleanlab.pruning import get_noise_indices
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from app.algorithms import Algorithm
from app.data.datasets import DataSet


class CleanLabDetector(Algorithm):
    def __init__(self, *, n_folds: int = 3, n_repetitions: int = 5, verbose: bool = True) -> None:
        super().__init__()
        self._n_folds = n_folds
        self._n_repetitions = n_repetitions
        if verbose:
            self._progress_bar = tqdm(total=self._n_folds * self._n_repetitions)

    def run(self, data_set: DataSet) -> List[int]:
        all_mislabelled = []
        for i in range(self._n_repetitions):
            psx = self._compute_out_of_sample_predicted_probabilities(data_set=data_set)
            likely_mislabelled_indices = get_noise_indices(
                s=data_set.y_given,
                psx=psx,
                sorted_index_method='normalized_margin',
            )
            all_mislabelled.append(likely_mislabelled_indices)
        
        # Find the indices of the samples that are mislabelled in every iteration
        mislabelled_multiple_iterations = list(set.intersection(*map(set, all_mislabelled)))
        return mislabelled_multiple_iterations

    def _create_learning_algorithm(self):
        return LogisticRegression(max_iter=1000)
        # return GaussianProcessClassifier(kernel=1*RBF(1.0))

    def _compute_out_of_sample_predicted_probabilities(self, data_set: DataSet):
        print(f"Number of classes: {data_set.n_classes}")
        pred_probas = np.zeros(shape=(data_set.size, data_set.n_classes))
        for split in data_set.split(n_splits=self._n_folds):
            model = self._create_learning_algorithm()
            model.fit(split.X_train, split.y_train)
            y_pred_proba = model.predict_proba(split.X_test)

            print(f"Number of classes in the train set: {np.unique(split.y_train).shape}")
            print(f"Number of classes in the test set: {np.unique(split.y_test).shape}")
            
            pred_probas[split.test_indices] = y_pred_proba
            if self._progress_bar:
                self._progress_bar.update()
        return pred_probas


def create() -> Algorithm:
    return CleanLabDetector()
