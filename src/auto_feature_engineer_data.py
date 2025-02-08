"""
This script takes a parquet file with the data and preprocesses it. The preprocessing steps are:
1. Split the data into training, validation, and test sets.
2. Scale the features using StandardScaler.
"""
import os
import tempfile
import gc
import click
import pandas as pd
import numpy as np
import torch
import mlflow

from automated_feature_engineering.models import Autoencoder, Encoder
from automated_feature_engineering.engineer import FeatureEngineer
from automated_feature_engineering.trainer import AutoencoderTrainer


@click.command(
    help="""
    Given artifact dirs (see preprocess_data), train an autoencoder model for 
    feature engineering
    """
)
@click.option("--preprocessing-data-artifact-dir")
def auto_feature_engineer_data(preprocessing_data_artifact_dir):
    """
    Given artifact dirs (see preprocess_data), train an autoencoder model for feature engineering
    """
    with mlflow.start_run():
        mlflow.pytorch.autolog()
        torch.set_default_dtype(torch.float64)
        X_train = pd.read_parquet(os.path.join(
            preprocessing_data_artifact_dir, "X_train-parquet-dir", "X_train.parquet"))
        y_train = pd.read_csv(os.path.join(
            preprocessing_data_artifact_dir, "y_train-csv-dir", "y_train.csv"))
        X_val = pd.read_parquet(os.path.join(
            preprocessing_data_artifact_dir, "X_val-parquet-dir", "X_val.parquet"))
        y_val = pd.read_csv(os.path.join(
            preprocessing_data_artifact_dir, "y_val-csv-dir", "y_val.csv"))
        X_test = pd.read_parquet(os.path.join(
            preprocessing_data_artifact_dir, "X_test-parquet-dir", "X_test.parquet"))

        processed_data = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val
        }

        # Train an autoencoder model
        engineer = FeatureEngineer(processed_data)
        autoencoder = Autoencoder(
            input_dim=X_train.shape[1]).to(engineer.device)

        training_params = {
            "lr": 8e-5,
            "weight_decay": 1e-5,
            "patience": 15,
            "epochs": 200
        }

        trainer = AutoencoderTrainer(training_params, engineer.device)

        engineer.filter_non_fraud()
        train_loader, val_loader = engineer.create_loaders()
        _, _ = trainer.train(autoencoder, train_loader, val_loader)

        encoder = Encoder(autoencoder).eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            engineer.generate_features(encoder, tmpdir, "features")

        extra_files = ["best_model.pth"]

        encoder = encoder.to("cpu")

        mlflow.pytorch.log_model(
            pytorch_model=encoder,
            artifact_path="encoder",
            registered_model_name="FraudEncoder",
            extra_files=extra_files,
            input_example=X_train.iloc[0:1].values.astype(np.float64)
        )

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    auto_feature_engineer_data()
