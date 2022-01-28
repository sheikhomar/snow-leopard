
import json, shutil

from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Dict, List

import click
import pandas as pd
import numpy as np

from cleanlab.noise_generation import noise_matrix_is_valid

from app.algorithms import Algorithm
from app.data.datasets import DataSet, get_dataset

def type_1_error_rate(
        data_set_size: int,
        known_mislabelled_indices: List[int],
        detected_mislabelled_indices: List[int]
    ) -> float:
    all_indices = list(range(data_set_size))
    correctly_labelled_indices = list(set(all_indices) - set(known_mislabelled_indices))

    # Find samples which are correctly labelled but are detected as mislabelled.
    type_1_error_indices = [
        index
        for index in detected_mislabelled_indices
        if index in correctly_labelled_indices
    ]

    # Type 1 errors are correctly labelled instances that
    # are erroneously identified as mislabelled.
    er1 = len(type_1_error_indices) / len(correctly_labelled_indices)

    return er1


def type_2_error_rate(
        data_set_size: int,
        known_mislabelled_indices: List[int],
        detected_mislabelled_indices: List[int]
    ) -> float:
    # Type 2 errors are known mislabeled instances which are not detected.
    type_2_indices = [
        index
        for index in known_mislabelled_indices
        if index not in detected_mislabelled_indices
    ]
    er2 = len(type_2_indices) / len(known_mislabelled_indices)
    return er2


def noise_elimination_precision_score(
        data_set_size: int,
        known_mislabelled_indices: List[int],
        detected_mislabelled_indices: List[int]
    ) -> float:
    # Noise elimination precision (NEP) is the percentage of
    # detected instances that are known to be mislabelled.
    detected_and_known_indices = [
        index
        for index in known_mislabelled_indices
        if index in detected_mislabelled_indices
    ]
    nep = len(detected_and_known_indices) / len(detected_mislabelled_indices)
    return nep


class Evaluator:
    def __init__(self, *, algorithm_name: str, data_path: str, noise_path: str, n_repetitions: int, output_dir: str) -> None:
        self._algorithm_name = algorithm_name
        self._data_path = data_path
        self._noise_path = Path(noise_path)
        self._n_repetitions = n_repetitions
        self._output_dir = Path(output_dir)
        if self._output_dir.name != self._algorithm_name:
            self._output_dir = self._output_dir / self._algorithm_name
        experiment_no = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._output_dir = self._output_dir / experiment_no

    def run(self) -> None:
        for i in range(self._n_repetitions):
            self.run_iteration(i)

    def run_iteration(self, iteration: int) -> None:
        start_time = datetime.now()

        # Load label noise detection algorithm.
        detector = self._load_algorithm()

        # Load data
        data_set = get_dataset(self._data_path)

        # Load noise matrix.
        noise_matrix = self._get_noise_matrix(data_set=data_set)

        # Apply noise matrix.
        known_mislabelled_indices = data_set.apply_label_noise(noise_matrix=noise_matrix)

        # Run detection algorithm
        detected_mislabelled_indices = detector.run(data_set=data_set)

        scores = self._compute_scores(
            data_set_size=data_set.size,
            known_mislabelled_indices=known_mislabelled_indices,
            detected_mislabelled_indices=detected_mislabelled_indices,
        )

        end_time = datetime.now()

        self._output_dir.mkdir(exist_ok=True, parents=True)
        results = {
            "algorithm_name": self._algorithm_name,
            "algorithm_params": detector.get_params(),
            "start_time": start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "end_time": end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "scores": [{
                "iteration": iteration,
                **scores
            }],
            "data": {
                "path": self._data_path,
                "size": data_set.size,
                "class_names": list(data_set.class_names),
            },
            "known_mislabelled_indices": list(known_mislabelled_indices),
            "detected_mislabelled_indices": list(detected_mislabelled_indices),
            
        }
        iteration_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(self._output_dir / f"results-iter{iteration}-{iteration_id}.json", "w") as fp:
            json.dump(results, fp, indent=2)
        shutil.copy(self._noise_path, self._output_dir / self._noise_path.name)

        np.savez_compressed(self._output_dir / "noise-matrix.npz", noise_matrix=noise_matrix)

    def _compute_scores(
            self,
            data_set_size: int,
            known_mislabelled_indices: List[int],
            detected_mislabelled_indices: List[int]
        ) -> Dict[str, object]:
        scorers = {
            "type_1_error_rate": type_1_error_rate,
            "type_2_error_rate": type_2_error_rate,
            "noise_elimination_precision_score": noise_elimination_precision_score,
        }
        scores = dict()
        for name, scorer_fn in scorers.items():
            score = scorer_fn(
                data_set_size=data_set_size,
                known_mislabelled_indices=known_mislabelled_indices,
                detected_mislabelled_indices=detected_mislabelled_indices,
            )
            scores[name] = score
        return scores

    def _load_algorithm(self) -> Algorithm:
        try:
            m = import_module(f"app.algorithms.{self._algorithm_name}")
            algorithm = m.create()  # type: ignore
            return algorithm
        except ModuleNotFoundError:
            raise ValueError(f"Algorithm {self._algorithm_name} not found.")

    def _load_noise_matrix_from_file(self, file_path: str, classes: List[str]) -> np.ndarray:
        classes = np.array(classes)
        n_classes = classes.shape[0]
        noise_matrix = np.zeros(shape=(n_classes, n_classes))
        
        # Parse noise transition specification file.
        with open(file_path, mode="r") as fp:
            noise_config = json.load(fp)
        if noise_config["type"] == "fixed":
            for transition in noise_config["transitions"]:
                from_index = np.where(classes == transition["from"])[0][0]
                to_index = np.where(classes == transition["to"])[0][0]
                noise_matrix[from_index, to_index] = transition["rate"]
                if transition["symmetric"]:
                    noise_matrix[to_index, from_index] = transition["rate"]
        else:
            raise Exception("Unknown type: " + str(noise_config["type"]))
        
        # Ensure that the noise matrix is a column stochastic matrix i.e., 
        # sum of entries in each column must be 1.
        noise_matrix = noise_matrix + np.diag(1 - noise_matrix.sum(axis=1))
        
        return noise_matrix

    def _validate_noise_matrix(self, noise_matrix: np.ndarray, data_set: DataSet) -> None:
        _, count_per_class = np.unique(data_set.y_given, return_counts=True)
        py = count_per_class / count_per_class.sum()
        if not noise_matrix_is_valid(noise_matrix=noise_matrix, py=py):
            print(noise_matrix)
            raise Exception("Noise matrix is not valid!")

    def _get_noise_matrix(self, data_set: DataSet) -> np.ndarray:
        noise_matrix = self._load_noise_matrix_from_file(
            file_path=self._noise_path,
            classes=data_set.class_names
        )

        self._validate_noise_matrix(noise_matrix=noise_matrix, data_set=data_set)

        return noise_matrix


@click.command(help="Evaluates a noise detector.")
@click.option(
    "-a",
    "--algorithm-name",
    type=click.STRING,
    required=True,
)
@click.option(
    "-i",
    "--input-path",
    type=click.STRING,
    required=True,
)
@click.option(
    "-n",
    "--noise-path",
    type=click.STRING,
    required=True,
)
@click.option(
    "-r",
    "--n-repetitions",
    type=click.INT,
    required=False,
    default=5
)
@click.option(
    "-o",
    "--output-dir",
    type=click.STRING,
    required=True,
)
def main(
    algorithm_name: str,
    input_path: str,
    noise_path: str,
    n_repetitions: int,
    output_dir: str,
):
    Evaluator(
        algorithm_name=algorithm_name,
        data_path=input_path,
        noise_path=noise_path,
        n_repetitions=n_repetitions,
        output_dir=output_dir,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
