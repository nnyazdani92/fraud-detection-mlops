# src/download_data.py
from kaggle.api.kaggle_api_extended import KaggleApi

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

    return dir_name
