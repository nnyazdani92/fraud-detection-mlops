# src/download_data.py
import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_dataset(dataset_name: str):
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

    # Create a temporary directory
    temp_dir = "temp_download"
    os.makedirs(temp_dir, exist_ok=True)

    # Download the dataset to the temporary directory
    api.dataset_download_files(dataset_name, path=temp_dir, unzip=True)
    print(f"Dataset downloaded to temporary directory: {temp_dir}")

    return temp_dir
