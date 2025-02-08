"""
Automated feature selection pipeline component
"""

import os
import click
import pandas as pd
import mlflow

from automated_feature_selection.selectors import FeatureSelector


@click.command(
    help=""
)
@click.option("--engineering-data-artifact-dir")
@click.option("--preprocessing-data-artifact-dir")
def auto_feature_select(engineering_data_artifact_dir, preprocessing_data_artifact_dir):
    """
    Automated feature selection pipeline component
    """
    with mlflow.start_run():
        X_train = pd.read_parquet(os.path.join(
            engineering_data_artifact_dir, "features", "X_train_enriched.parquet"))
        X_val = pd.read_parquet(os.path.join(
            engineering_data_artifact_dir, "features", "X_val_enriched.parquet"))
        X_test = pd.read_parquet(os.path.join(
            engineering_data_artifact_dir, "features", "X_test_enriched.parquet"))
        y_train = pd.read_csv(os.path.join(
            preprocessing_data_artifact_dir, "y_train-csv-dir", "y_train.csv"))

        selector = FeatureSelector(X_train, X_val, X_test, y_train)
        selector.validate_data()
        individual_results, final_result = selector.run_selection()
        selector.log_selection_metrics(individual_results, final_result)
        selector.artifacts.log_features(final_result.selected_features)
        selected_data = selector.create_selected_datasets(
            final_result.selected_features)
        selector.artifacts.log_datasets(selected_data)


if __name__ == "__main__":
    auto_feature_select()
