"""Feature engineering orchestrator"""
import os
from tempfile import TemporaryDirectory
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import mlflow


class FeatureEngineer:
    """Handles feature engineering pipeline"""

    def __init__(
        self,
        processed_data,
        batch_size=256,
        split_names=("train", "val", "test"),
        encoding_dim=16
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.processed_data = processed_data
        self.non_fraud_data = None
        self.batch_size = batch_size
        self.split_names = split_names
        self.encoding_dim = encoding_dim

    def filter_non_fraud(self) -> None:
        """Filter non-fraud samples for training"""
        self.non_fraud_data = {
            "train": self._filter_split("X_train", "y_train"),
            "val": self._filter_split("X_val", "y_val")
        }

    def _filter_split(self, features_key: str, target_key: str) -> pd.DataFrame:
        """Filter non-fraud samples from a single split"""
        non_fraud_mask = (self.processed_data[target_key] == 0).values.flatten()
        return self.processed_data[features_key][non_fraud_mask]

    def create_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders"""
        train_dataset = TensorDataset(
            torch.tensor(
                self.non_fraud_data["train"].values, dtype=torch.float64)
        )
        val_dataset = TensorDataset(
            torch.tensor(
                self.non_fraud_data["val"].values, dtype=torch.float64)
        )

        batch_size = self.batch_size
        return (
            DataLoader(train_dataset, batch_size=batch_size,
                       shuffle=True, num_workers=os.cpu_count()),
            DataLoader(val_dataset, batch_size=batch_size,
                       num_workers=os.cpu_count())
        )

    def generate_features(self, encoder: nn.Module, output_path, artifact_path) -> None:
        """Generate and save encoded features"""
        encoder.eval()
        split_names = self.split_names

        with torch.no_grad(), TemporaryDirectory() as tmp_dir:
            for split in split_names:
                features_key = f"X_{split}"
                data_tensor = torch.tensor(
                    self.processed_data[features_key].values,
                    dtype=torch.float64
                ).to(self.device)

                encoded = pd.DataFrame(
                    encoder(data_tensor).cpu().numpy(),
                    columns=[f"enc_{i}" for i in range(self.encoding_dim)]
                )

                enriched = pd.concat([
                    self.processed_data[features_key].reset_index(drop=True),
                    encoded.reset_index(drop=True)
                ], axis=1)

                output_path = Path(tmp_dir) / \
                    f"{features_key}_enriched.parquet"
                enriched.to_parquet(output_path)
                mlflow.log_artifact(
                    output_path,
                    artifact_path
                )
