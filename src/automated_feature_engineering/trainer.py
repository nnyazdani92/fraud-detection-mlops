"""Module for training autoencoder model"""
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import mlflow


@dataclass
class TrainingState:
    """State of the training process"""
    best_loss: float = float('inf')
    patience_counter: int = 0
    train_losses: list = None
    val_losses: list = None


class AutoencoderTrainer:
    """Handles autoencoder training process"""

    def __init__(self, training_params: dict, device: torch.device):
        self.device = device
        self.training_params = training_params
        self.state = TrainingState(
            train_losses=[],
            val_losses=[]
        )

    def configure_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Configure optimizer with parameters from config"""
        return torch.optim.Adam(
            model.parameters(),
            lr=self.training_params["lr"],
            weight_decay=self.training_params["weight_decay"]
        )

    def train_epoch(self, model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Single training epoch"""
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(loader, desc="Training", leave=False):
            data = batch[0].to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)

        return epoch_loss / len(loader.dataset)

    def validate_epoch(self, model: nn.Module, loader: DataLoader,
                       criterion: nn.Module) -> float:
        """Single validation epoch"""
        model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating", leave=False):
                data = batch[0].to(self.device)
                output = model(data)
                loss = criterion(output, data)
                epoch_loss += loss.item() * data.size(0)

        return epoch_loss / len(loader.dataset)

    def train(self, model: nn.Module, train_loader: DataLoader,
              val_loader: DataLoader) -> tuple[list, list]:
        """Full training loop with early stopping"""
        criterion = nn.SmoothL1Loss()
        optimizer = self.configure_optimizer(model)
        patience = self.training_params["patience"]

        with tqdm(range(self.training_params["epochs"]), desc="Training") as epoch_bar:
            for epoch in epoch_bar:
                train_loss = self.train_epoch(
                    model, train_loader, optimizer, criterion)
                val_loss = self.validate_epoch(model, val_loader, criterion)

                # Update state
                self.state.train_losses.append(train_loss)
                self.state.val_losses.append(val_loss)

                # Update progress bar
                epoch_bar.set_postfix({
                    'Train Loss': f'{train_loss:.4f}',
                    'Val Loss': f'{val_loss:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

                # Log metrics
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }, step=epoch)

                # Check early stopping
                if val_loss < self.state.best_loss:
                    self.state.best_loss = val_loss
                    self.state.patience_counter = 0
                    torch.save(model.state_dict(), "best_model.pth")
                else:
                    self.state.patience_counter += 1
                    if self.state.patience_counter >= patience:
                        break

        return self.state.train_losses, self.state.val_losses
