import click

from app.algorithms.cl import CleanLabDetector
from app.data.datasets import get_dataset


class DetectWithConfidentJoint:
    def __init__(self, data_path: str, n_repetitions: int, output_path: str) -> None:
        self._data_path = data_path
        self._n_repetitions = n_repetitions
        self._output_path = output_path

    def run(self) -> None:
        # Load data
        data_set = get_dataset(self._data_path)

        # Detect mislabelled samples
        detector = CleanLabDetector(n_folds=3, n_repetitions=self._n_repetitions)
        mislabelled_indices = detector.run(data_set=data_set)

        # Output samples that are most likely mislabelled
        df_mislabelled = data_set.get_raw_data_by_indices(mislabelled_indices).copy()
        cols = ["id", "supplier_name", "category_name", "title", "model_name", "description_short"]
        df_mislabelled[cols].to_excel(self._output_path, index=False)


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
