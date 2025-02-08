"""
This script takes the output of the auto_feature_engineer_data script and 
oversamples the minority class using SMOTE. 

The resampled data is saved as parquet and csv files in the mlflow artifact store.
"""

import os
import tempfile
import click
import pandas as pd
from imblearn.over_sampling import SMOTE
import mlflow


@click.command(
    help="""Given a parquet file (see auto_feature_engineer_data),
    oversample the minority class using SMOTE and save the result"""
)
@click.option("--selected-data-artifact-dir")
@click.option("--preprocessing-data-artifact-dir")
@click.option("--sampling-strategy", type=float)
def oversample_data(selected_data_artifact_dir, preprocessing_data_artifact_dir, sampling_strategy):
    """
    Given a parquet file (see auto_feature_engineer_data), 
    oversample the minority class using SMOTE and save the result
    """
    with mlflow.start_run():
        mlflow.sklearn.autolog()
        X_train = pd.read_parquet(os.path.join(
            selected_data_artifact_dir, "selected", "X_train_selected.parquet"))
        y_train = pd.read_csv(os.path.join(
            preprocessing_data_artifact_dir, "y_train-csv-dir", "y_train.csv"))

        # Oversample the minority class
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train, y_train)

        # Save the resampled data
        tmpdir = tempfile.mkdtemp()
        X_train_resampled_fp = os.path.join(
            tmpdir, "X_train_resampled.parquet")
        y_train_resampled_fp = os.path.join(tmpdir, "y_train_resampled.csv")

        X_train_resampled.to_parquet(X_train_resampled_fp, index=False)
        y_train_resampled.to_csv(y_train_resampled_fp, index=False)

        mlflow.log_param("sampling_strategy", sampling_strategy)
        mlflow.log_artifact(X_train_resampled_fp,
                            "X_train_resampled-parquet-dir")
        mlflow.log_artifact(y_train_resampled_fp, "y_train_resampled-csv-dir")


if __name__ == "__main__":
    oversample_data()
