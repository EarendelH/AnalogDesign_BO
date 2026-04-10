import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
import os


class SurrogateNet(nn.Module):
    """
    Neural network architecture for surrogate model
    Provides mean prediction for Bayesian optimization

    Attributes:
        input_dim: Input dimension (number of parameters)
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability for regularization
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [256, 256, 128, 64],
                 dropout_rate: float = 0.1):
        """
        Initialize surrogate network

        Args:
            input_dim: Dimension of input parameters
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super().__init__()

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer for mean prediction
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network"""
        return self.network(x)


class SurrogateModel:
    """
    Surrogate model for Bayesian optimization
    Handles training, prediction, and uncertainty estimation

    Attributes:
        input_dim: Input dimension
        device: torch device (CPU/GPU)
        ensemble_size: Number of models in ensemble
        learning_rate: Learning rate for optimization
    """

    def __init__(self,
                 input_dim: int,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 ensemble_size: int = 5,
                 learning_rate: float = 1e-4):
        """
        Initialize surrogate model

        Args:
            input_dim: Dimension of input parameters
            device: Device to run model on
            ensemble_size: Number of models in ensemble
            learning_rate: Learning rate for optimization
        """
        self.input_dim = input_dim
        self.device = torch.device(device)
        self.ensemble_size = ensemble_size
        self.learning_rate = learning_rate

        # Initialize ensemble of networks
        self.networks = [
            SurrogateNet(input_dim).to(self.device)
            for _ in range(ensemble_size)
        ]

        # Initialize optimizers
        self.optimizers = [
            optim.Adam(net.parameters(), lr=learning_rate)
            for net in self.networks
        ]

        # Initialize scalers
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.loss_fn = nn.MSELoss()
        self.x_mean_tensor = None
        self.x_scale_tensor = None
        self.y_mean_tensor = None
        self.y_scale_tensor = None

    def _refresh_scaler_tensors(self):
        """Mirror fitted scaler statistics onto the model device."""
        if hasattr(self.x_scaler, "mean_") and hasattr(self.x_scaler, "scale_"):
            self.x_mean_tensor = torch.tensor(
                self.x_scaler.mean_, dtype=torch.float32, device=self.device
            )
            self.x_scale_tensor = torch.tensor(
                self.x_scaler.scale_, dtype=torch.float32, device=self.device
            )

        if hasattr(self.y_scaler, "mean_") and hasattr(self.y_scaler, "scale_"):
            self.y_mean_tensor = torch.tensor(
                self.y_scaler.mean_, dtype=torch.float32, device=self.device
            )
            self.y_scale_tensor = torch.tensor(
                self.y_scaler.scale_, dtype=torch.float32, device=self.device
            )

    def _ensure_scalers_ready(self):
        if self.x_mean_tensor is None or self.y_mean_tensor is None:
            self._refresh_scaler_tensors()

    def _scale_tensor_input(self, X_tensor: torch.Tensor) -> torch.Tensor:
        self._ensure_scalers_ready()
        return (X_tensor - self.x_mean_tensor) / self.x_scale_tensor

    def _ensemble_predict_from_scaled_tensor(
            self, X_scaled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = []
        for net in self.networks:
            net.eval()
            pred = net(X_scaled)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)
        mean_scaled = predictions.mean(dim=0)
        std_scaled = predictions.std(dim=0, unbiased=False)
        return mean_scaled, std_scaled

    def predict_tensor(self,
                       X_tensor: torch.Tensor,
                       requires_grad: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean/std directly on tensors, optionally preserving gradients.

        Args:
            X_tensor: Input tensor of shape [batch, input_dim]
            requires_grad: Whether gradients w.r.t. the input are needed

        Returns:
            mean_prediction: Tensor of shape [batch]
            std_prediction: Tensor of shape [batch]
        """
        if not torch.is_tensor(X_tensor):
            X_tensor = torch.as_tensor(X_tensor, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X_tensor.to(self.device, dtype=torch.float32)

        X_scaled = self._scale_tensor_input(X_tensor)

        if requires_grad:
            mean_scaled, std_scaled = self._ensemble_predict_from_scaled_tensor(X_scaled)
        else:
            with torch.no_grad():
                mean_scaled, std_scaled = self._ensemble_predict_from_scaled_tensor(X_scaled)

        mean_pred = mean_scaled * self.y_scale_tensor + self.y_mean_tensor
        std_pred = std_scaled * self.y_scale_tensor
        return mean_pred.squeeze(-1), std_pred.squeeze(-1)

    # def fit(self,
    #         X: np.ndarray,
    #         y: np.ndarray,
    #         save_dir: str,
    #         batch_size: int = 32,
    #         epochs: int = 100,
    #         validation_split: float = 0.2,
    #         save_freq: int = 10) -> dict:
    #     """
    #     Train the surrogate model with periodic saving

    #     Args:
    #         X: Input parameters
    #         y: Target values
    #         save_dir: Directory to save model checkpoints
    #         batch_size: Training batch size
    #         epochs: Number of training epochs
    #         validation_split: Fraction of data to use for validation
    #         save_freq: Frequency of model saving (epochs)

    #     Returns:
    #         history: Training history dictionary
    #     """
    #     # Create save directory if not exists
    #     os.makedirs(save_dir, exist_ok=True)

    #     # Scale data
    #     X_scaled = self.x_scaler.fit_transform(X)
    #     y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1))

    #     # Convert to tensors
    #     X_tensor = torch.FloatTensor(X_scaled).to(self.device)
    #     y_tensor = torch.FloatTensor(y_scaled).to(self.device)

    #     # Split data
    #     n_val = int(len(X) * validation_split)
    #     indices = torch.randperm(len(X))
    #     train_indices = indices[:-n_val]
    #     val_indices = indices[-n_val:]

    #     # Training history
    #     history = {'train_loss': [], 'val_loss': []}
    #     best_val_loss = float('inf')

    #     # Training loop
    #     for epoch in range(epochs):
    #         # Train each network in ensemble
    #         train_losses = []
    #         val_losses = []

    #         for net, opt in zip(self.networks, self.optimizers):
    #             net.train()
    #             # Mini-batch training
    #             for i in range(0, len(train_indices), batch_size):
    #                 batch_idx = train_indices[i:i + batch_size]
    #                 X_batch = X_tensor[batch_idx]
    #                 y_batch = y_tensor[batch_idx]

    #                 opt.zero_grad()
    #                 pred = net(X_batch)
    #                 loss = self.loss_fn(pred, y_batch)
    #                 loss.backward()
    #                 opt.step()

    #                 train_losses.append(loss.item())

    #             # Validation
    #             net.eval()
    #             with torch.no_grad():
    #                 val_pred = net(X_tensor[val_indices])
    #                 val_loss = self.loss_fn(val_pred, y_tensor[val_indices])
    #                 val_losses.append(val_loss.item())

    #         # Record metrics
    #         avg_train_loss = np.mean(train_losses)
    #         avg_val_loss = np.mean(val_losses)
    #         history['train_loss'].append(avg_train_loss)
    #         history['val_loss'].append(avg_val_loss)

    #         # Save model periodically
    #         if (epoch + 1) % save_freq == 0:
    #             checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pt')
    #             self.save_model(checkpoint_path)
    #             logging.info(f"Model saved at epoch {epoch + 1}: {checkpoint_path}")

    #         # Save best model
    #         if avg_val_loss < best_val_loss:
    #             best_val_loss = avg_val_loss
    #             best_model_path = os.path.join(save_dir, 'best_model.pt')
    #             self.save_model(best_model_path)
    #             logging.info(f"Best model saved with validation loss: {best_val_loss:.4f}")

    #         if epoch % 10 == 0:
    #             logging.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

    #     return history
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            save_dir: str,
            batch_size: int = 32,
            epochs: int = 100,
            validation_split: float = 0.2,
            save_freq: int = 10,
            return_history: bool = False) -> dict:
        """
        Train the surrogate model with periodic saving

        Args:
            X: Input parameters
            y: Target values
            save_dir: Directory to save model checkpoints
            batch_size: Training batch size
            epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            save_freq: Frequency of model saving (epochs)
            return_history: Whether to return training history

        Returns:
            history: Training history dictionary
        """
        # Create save directory if not exists
        os.makedirs(save_dir, exist_ok=True)

        # Scale data
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1))
        self._refresh_scaler_tensors()

        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=self.device)

        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        if n_samples > 1:
            n_val = max(1, n_val)
        else:
            n_val = 0

        indices = torch.randperm(n_samples, device=self.device)
        train_indices = indices[:-n_val] if n_val > 0 else indices
        val_indices = indices[-n_val:] if n_val > 0 else indices

        # Training history
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')

        # Training loop
        for epoch in range(epochs):
            # Train each network in ensemble
            train_losses = []
            val_losses = []

            for net, opt in zip(self.networks, self.optimizers):
                net.train()
                # Mini-batch training
                for i in range(0, len(train_indices), batch_size):
                    batch_idx = train_indices[i:i + batch_size]
                    if len(batch_idx) < 2:
                        continue
                    X_batch = X_tensor[batch_idx]
                    y_batch = y_tensor[batch_idx]

                    opt.zero_grad()
                    pred = net(X_batch)
                    loss = self.loss_fn(pred, y_batch)
                    loss.backward()
                    opt.step()

                    train_losses.append(loss.item())

                # Validation
                net.eval()
                if len(val_indices) > 0:
                    with torch.no_grad():
                        val_pred = net(X_tensor[val_indices])
                        val_loss = self.loss_fn(val_pred, y_tensor[val_indices])
                        val_losses.append(val_loss.item())

            # Record metrics
            avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            avg_val_loss = float(np.mean(val_losses)) if val_losses else avg_train_loss
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

            # Save model periodically
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pt')
                self.save_model(checkpoint_path)
                logging.info(f"Model saved at epoch {epoch + 1}: {checkpoint_path}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(save_dir, 'best_model.pt')
                self.save_model(best_model_path)
                logging.info(f"Best model saved with validation loss: {best_val_loss:.4f}")

            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        return history if return_history else None

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation

        Args:
            X: Input parameters

        Returns:
            mean_prediction: Mean of predictions
            std_prediction: Standard deviation of predictions
        """
        mean_pred, std_pred = self.predict_tensor(X, requires_grad=False)
        return mean_pred.detach().cpu().numpy().reshape(-1, 1), std_pred.detach().cpu().numpy().reshape(-1, 1)

    def save_model(self, path: str):
        """Save model state"""
        state = {
            'networks': [net.state_dict() for net in self.networks],
            'x_scaler': self.x_scaler,
            'y_scaler': self.y_scaler
        }
        torch.save(state, path)

    def load_model(self, path: str):
        """Load model state"""
        state = torch.load(path, map_location=self.device)
        for net, state_dict in zip(self.networks, state['networks']):
            net.load_state_dict(state_dict)
        self.x_scaler = state['x_scaler']
        self.y_scaler = state['y_scaler']
        self._refresh_scaler_tensors()
