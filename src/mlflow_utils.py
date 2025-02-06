"""
This module contains auxiliary functions for using mlflow.
"""
import os
import json
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


def load_config():
    """Load configuration from JSON file"""
    with open(os.environ["JSON_CONFIG_PATH"], "r", encoding="utf-8") as f:
        return json.load(f)


def configure_mlflow(experiment_name: str) -> None:
    """Initialize MLflow tracking"""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)


def find_latest_run_id_by_experiment_and_stage(experiment_name: str, stage: str) -> str:
    """Find the latest successful run"""
    client = MlflowClient()
    experiment_id = mlflow.get_experiment_by_name(
        experiment_name).experiment_id
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.stage='{stage}' AND attributes.status='FINISHED'",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["attribute.start_time DESC"]
    )
    return runs[0].info.run_id if runs else None


def get_data(run_id: str, data_params: dict[list[str]], artifact_dir: str) -> dict:
    """Retrieve data from MLflow artifacts"""
    client = MlflowClient()
    data = {}

    for split_dir in data_params["split_dirs"]:
        artifacts = client.list_artifacts(
            run_id, os.path.join(artifact_dir, split_dir))
        for artifact in artifacts:
            if artifact.path.endswith(".parquet"):
                path = client.download_artifacts(run_id, artifact.path)
                split = artifact.path.split("/")[-1].split(".")[0]
                data[split] = pd.read_parquet(path)

    return data


def get_targets(preprocessing_run_id: str, config) -> dict[str, pd.Series]:
    """Retrieve target variables from preprocessing run"""
    client = MlflowClient()
    targets = {}
    for split, target_split in zip(config["split_dirs"], config["target_split_names"]):
        path = client.download_artifacts(
            preprocessing_run_id,
            f"data/processed/{split}/Target/{target_split}.parquet"
        )
        targets[f"{target_split}"] = pd.read_parquet(path).squeeze()
    return targets
