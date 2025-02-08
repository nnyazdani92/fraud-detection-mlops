"""
Downloads the MLG ULB Credit Card Fraud detection dataset and ssaves it as an artifact
"""
import os
import tempfile
import click
from kaggle.api.kaggle_api_extended import KaggleApi
import mlflow


def download_kaggle_dataset(dataset_name: str, dir_name: str) -> str:
    """
    Downloads a dataset from Kaggle to a temporary directory.

    Args:
        dataset_name (str): The name of the dataset in the format 'owner/dataset'.

    Returns:
        str: Path to the temporary directory where the dataset is downloaded.
    """
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset to the temporary directory
    api.dataset_download_files(dataset_name, path=dir_name, unzip=True)
    print(f"Dataset downloaded to directory: {dir_name}")


@click.command(
    help="Downloads the Credit card fraud dataset and saves it as an mlflow artifact "
    "called 'creditcard.csv'."
)
@click.option("--dataset-name", default="mlg-ulb/creditcardfraud")
def load_raw_data(dataset_name):
    """
    Downloads the Credit card fraud dataset and saves it as an mlflow artifact.
    """
    with mlflow.start_run():
        local_dir = tempfile.mkdtemp()
        download_kaggle_dataset(dataset_name, local_dir)
        dataset_file = os.path.join(local_dir, "creditcard.csv")
        print(f"Uploading dataset to MLflow: {dataset_file}")
        mlflow.log_artifact(dataset_file, "creditcard-csv-dir")


if __name__ == "__main__":
    load_raw_data()
