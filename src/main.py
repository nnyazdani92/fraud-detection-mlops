"""
This script defines a workflow that chains together the mlflow pipeline.
"""
import os
import click
import mlflow


@click.command()
@click.option("--sampling-strategy", type=float, default=0.35)
def workflow(sampling_strategy):
    """
    This script defines a workflow that chains together the mlflow pipeline.
    """
    with mlflow.start_run():
        raw_data_run = mlflow.run(".", "load_raw_data", env_manager="local")
        raw_data_artifact_uri = mlflow.get_run(
            run_id=raw_data_run.run_id).info.artifact_uri
        fraud_csv_uri = os.path.join(
            raw_data_artifact_uri, "creditcard-csv-dir")

        etl_data_run = mlflow.run(".", "etl_data", parameters={
                                  "fraud_csv": fraud_csv_uri}, env_manager="local")
        etl_data_artifact_uri = mlflow.get_run(
            run_id=etl_data_run.run_id).info.artifact_uri
        fraud_parquet_uri = os.path.join(
            etl_data_artifact_uri, "creditcard-parquet-dir")

        preprocess_data_run = mlflow.run(".", "preprocess_data", parameters={
                                         "fraud_parquet": fraud_parquet_uri}, env_manager="local")

        preprocessing_data_artifact_dir = mlflow.get_run(
            run_id=preprocess_data_run.run_id).info.artifact_uri

        auto_feature_engineer_data_run = mlflow.run(".", "auto_feature_engineer_data", parameters={
            "preprocessing_data_artifact_dir": preprocessing_data_artifact_dir},
            env_manager="local"
        )

        engineering_data_artifact_dir = mlflow.get_run(
            run_id=auto_feature_engineer_data_run.run_id).info.artifact_uri

        auto_feature_select_run = mlflow.run(
            ".", "auto_feature_select",
            parameters={
                "engineering_data_artifact_dir": engineering_data_artifact_dir,
                "preprocessing_data_artifact_dir": preprocessing_data_artifact_dir,
            }, env_manager="local"
        )

        selected_data_artifact_dir = mlflow.get_run(
            run_id=auto_feature_select_run.run_id).info.artifact_uri

        oversample_data_run = mlflow.run(
            ".", "oversample_data",
            parameters={
                "selected_data_artifact_dir": selected_data_artifact_dir,
                "preprocessing_data_artifact_dir": preprocessing_data_artifact_dir,
                "sampling_strategy": sampling_strategy
            },
            env_manager="local"
        )

        oversampled_data_artifact_dir = mlflow.get_run(
            run_id=oversample_data_run.run_id).info.artifact_uri

        mlflow.run(
            ".", "train_model",
            parameters={
                "oversampled_data_artifact_dir": oversampled_data_artifact_dir,
                "selected_data_artifact_dir": selected_data_artifact_dir,
                "preprocessing_data_artifact_dir": preprocessing_data_artifact_dir
            },
            env_manager="local"
        )


if __name__ == "__main__":
    workflow()
