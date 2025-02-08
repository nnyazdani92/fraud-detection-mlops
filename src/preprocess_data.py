"""
This script takes a parquet file with the data and preprocesses it. The preprocessing steps are:
1. Split the data into training, validation, and test sets.
2. Scale the features using StandardScaler.
"""

import os
import tempfile
import click
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import mlflow

@click.command(
    help="Given a parquet file (see etl_data), split the data, scale the features and save the result "
)
@click.option("--fraud-parquet")
@click.option("--target-column", default="Class")
@click.option("--test-size", default=0.25)
@click.option("--val-size", default=0.15)
@click.option("--random-state", default=42)
def preprocess_data(fraud_parquet, target_column, test_size, val_size, random_state):
    with mlflow.start_run():
        mlflow.sklearn.autolog()
        fraud_parquet_fp = os.path.join(fraud_parquet, "creditcard.parquet")
        df = pd.read_parquet(fraud_parquet_fp)
        X, y = df.drop(columns=[target_column]), df[target_column]
        X, y = X.astype(np.float64), y.astype(np.int64)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size,
            stratify=y_train_val,
            random_state=random_state
        )

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val), columns=X_val.columns)
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns)

        splits = {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test
        }

        tmpdir = tempfile.mkdtemp()

        scaler_fp = os.path.join(tmpdir, "scaler.pkl")
        joblib.dump(scaler, scaler_fp)
        mlflow.log_artifact(scaler_fp, "scaler")

        split_data: pd.DataFrame
        split_name: str

        for split_name, split_data in splits.items():
            if split_name.startswith("y"):
                split_fp = os.path.join(tmpdir, f"{split_name}.csv")
                split_data.to_csv(split_fp, index=False)
                mlflow.log_artifact(split_fp, f"{split_name}-csv-dir")
                continue
            split_fp = os.path.join(tmpdir, f"{split_name}.parquet")
            split_data.to_parquet(split_fp, index=False)
            mlflow.log_artifact(split_fp, f"{split_name}-parquet-dir")
            mlflow.log_metric(f"{split_name}_samples", split_data.shape[0])

        mlflow.log_param("target_column", target_column)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("val_size", val_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_dict(X_train_scaled.head().to_dict(),
                        "X_train_scaled_sample.json")


if __name__ == "__main__":
    preprocess_data()
