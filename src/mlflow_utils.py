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

def get_dataset(run_id: str, artifact_dir: str) -> pd.DataFrame:
    """Retrieve dataset from MLflow artifacts"""
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id, artifact_dir)
    for artifact in artifacts:
        if artifact.path.endswith(".csv"):
            path = client.download_artifacts(run_id, artifact.path)
            return pd.read_csv(path)
    return None


def get_data(run_id: str, dataset_params: dict, artifact_dir: str) -> dict:
    """Retrieve data from MLflow artifacts"""
    client = MlflowClient()
    data = {}

    for split_dir in dataset_params["split_dirs"]:
        artifacts = client.list_artifacts(
            run_id, os.path.join(artifact_dir, split_dir))
        for artifact in artifacts:
            if artifact.path.endswith(".parquet"):
                path = client.download_artifacts(run_id, artifact.path)
                split = artifact.path.split("/")[-1].split(".")[0]
                data[split] = pd.read_parquet(path)

    return data


def get_targets(preprocessing_run_id: str, dataset_params: dict) -> dict[str, pd.DataFrame]:
    """Retrieve target variables from preprocessing run"""
    client = MlflowClient()
    targets = {}
    for split_dir, split_name in zip(dataset_params["split_dirs"], dataset_params["split_names"]):
        path = client.download_artifacts(
            preprocessing_run_id,
            f"data/processed/{split_dir}/y_{split_name}.parquet"
        )
        targets[f"y_{split_name}"] = pd.read_parquet(path).squeeze()
    return targets
