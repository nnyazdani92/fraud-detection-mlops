"""
This module contains the function to initialize MLflow tracking.
"""
import os
import mlflow

def configure_mlflow(experiment_name: str) -> None:
    """Initialize MLflow tracking"""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)
