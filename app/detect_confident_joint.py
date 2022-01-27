from email.policy import default
import warnings

from typing import List
from datetime import datetime
from pathlib import Path
from pprint import pprint

import click
import pandas as pd
import numpy as np

from cleanlab.latent_estimation import compute_confident_joint, estimate_latent
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import notebooks.feature as feature_engineering


class DetectWithConfidentJoint:
    def __init__(self, data_path: str, n_repetitions: int, output_path: str) -> None:
        self._data_path = data_path
        self._n_repetitions = n_repetitions
        self._output_path = output_path

    def run(self) -> None:
        df_data = pd.read_csv(self._data_path, dtype=str, index_col=0)
        category_counts = df_data["category_name"].value_counts()
        large_enough_categories = category_counts[category_counts > 20].index.tolist()
        df_data = df_data[df_data.category_name.isin(large_enough_categories)]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="sklearn.preprocessing._label")
            psx, label_encoder = self._compute_out_of_sample_predicted_probabilities(
                df=df_data,
                n_repetitions=self._n_repetitions,
                fn_create_model=lambda: LogisticRegression(max_iter=1000),
                # fn_create_model=lambda: GaussianProcessClassifier(kernel=1*RBF(1.0)),
            )
        y_given = label_encoder.transform(df_data["category_name"])
        
        all_mislabelled = []
        for i in range(psx.shape[0]):
            confident_joint, likely_mislabelled_indices = compute_confident_joint(
                s=y_given,
                psx=psx[i,:],
                calibrate=False,
                return_indices_of_off_diagonals=True
            )
            all_mislabelled.append(likely_mislabelled_indices)
        
        # Find the indices of the samples that are mislabelled in every iteration
        mislabelled_multiple_repetitions = list(set.intersection(*map(set, all_mislabelled)))

        # Output samples that are most likely mislabelled
        df_mislabelled = df_data.iloc[mislabelled_multiple_repetitions].copy()
        cols = ["id", "supplier_name", "category_name", "title", "model_name", "description_short"]
        df_mislabelled[cols].to_excel(self._output_path, index=False)
        
    def _compute_out_of_sample_predicted_probabilities(
            self,
            df: pd.DataFrame,
            fn_create_model,
            label_attr_name: str = "category_name",
            n_folds: int = 3,
            n_repetitions: int = 2,
        
        ):
        pbar = tqdm(total=n_folds*n_repetitions)

        n_data = df.shape[0]
        n_classes = df[label_attr_name].unique().shape[0]
        pred_probas = np.zeros(shape=(n_repetitions, n_data, n_classes))
        
        label_encoder = LabelEncoder()
        label_encoder.fit(df[label_attr_name])

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        for repetition in range(n_repetitions):
            for fold, (train_index, test_index) in enumerate(skf.split(X=df, y=df[label_attr_name])):
                pbar.set_description(f"Iteration {repetition+1} fold {fold+1}")
                
                df_train = df.iloc[train_index]
                df_test = df.iloc[test_index]

                feateure_transformer = feature_engineering.BasicIceCatFeatureTransformer(output_size=0.99)
                feateure_transformer.fit(df_train)
                
                X_train = feateure_transformer.transform(df_train)
                X_test = feateure_transformer.transform(df_test)
                y_train = label_encoder.transform(df_train[label_attr_name])
                y_test = label_encoder.transform(df_test[label_attr_name])

                model = fn_create_model()
                model.fit(X_train, y_train)

                y_pred_proba = model.predict_proba(X_test)

                pred_probas[repetition, test_index] = y_pred_proba
                
                pbar.update()
        return pred_probas, label_encoder



@click.command(help="Detect noisy samples via Confident Joint.")
@click.option(
    "-i",
    "--input-path",
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
    "--output-path",
    type=click.STRING,
    required=True,
)
def main(
    input_path: str,
    n_repetitions: int,
    output_path: str,
):
    DetectWithConfidentJoint(
        data_path=input_path,
        n_repetitions=n_repetitions,
        output_path=output_path,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
