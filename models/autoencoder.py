"""PyTorch Autoencoder Anomaly Detection Layer.

Trained exclusively on BENIGN flows to learn the compact manifold of normal traffic.
Flags zero-day and unseen attack classes via high reconstruction error (MSE).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from features.extract import FEATURE_COLS


class FlowAutoencoderNet(nn.Module):
    """Deep Autoencoder for tabular flow features.

    Architecture: 22 -> 16 -> 8 -> 4 (bottleneck) -> 8 -> 16 -> 22.
    The bottleneck forces the network to learn essential invariant relationships
    of benign traffic.
    """

    def __init__(self, input_dim: int = 22, bottleneck_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderAnomalyDetector:
    """End-to-end Autoencoder detector with preprocessing and threshold calibration."""

    def __init__(self, input_dim: int = 22, bottleneck_dim: int = 4):
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.scaler = StandardScaler()
        self.net = FlowAutoencoderNet(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
        self.threshold: float = 0.0

    def fit(
        self,
        X_benign: pd.DataFrame,
        epochs: int = 15,
        batch_size: int = 128,
        lr: float = 0.005,
        val_split: float = 0.2,
    ) -> AutoencoderAnomalyDetector:
        """Fit scaler, train autoencoder on benign flows, and calibrate threshold."""
        X_arr = X_benign[FEATURE_COLS].values
        self.scaler.fit(X_arr)
        X_scaled = self.scaler.transform(X_arr)

        n_val = int(len(X_scaled) * val_split)
        X_train_fold = X_scaled[:-n_val]
        X_val_fold = X_scaled[-n_val:]

        train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
        dataset = TensorDataset(train_tensor, train_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.net.train()
        for epoch in range(epochs):
            for batch_x, _ in loader:
                optimizer.zero_grad()
                recon = self.net(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

        # Calibrate threshold on benign validation set
        self.net.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
            val_recon = self.net(val_tensor)
            val_errors = torch.mean((val_tensor - val_recon) ** 2, dim=1).numpy()

        # Set threshold at the 99th percentile of benign validation errors (1% target FPR)
        self.threshold = float(np.percentile(val_errors, 99.0))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Compute reconstruction error (MSE) per flow."""
        self.net.eval()
        X_scaled = self.scaler.transform(X[FEATURE_COLS].values)
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            recon = self.net(x_tensor)
            errors = torch.mean((x_tensor - recon) ** 2, dim=1).numpy()
        return errors

    def score_normalized(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly score normalized to [0, 1] with 0.5 at the calibrated threshold."""
        raw = self.score(X)
        # Sigmoid-based scaling centered at threshold
        scaled = 1.0 / (1.0 + np.exp(-3.0 * (raw - self.threshold) / max(0.01, self.threshold)))
        return np.clip(scaled, 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary anomaly: 1 = anomaly (error >= threshold), 0 = normal."""
        raw_scores = self.score(X)
        return (raw_scores >= self.threshold).astype(int)

    def save(self, model_dir: Path) -> None:
        """Persist weights, scaler, and threshold metadata."""
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), model_dir / "autoencoder_weights.pt")
        joblib.dump(self.scaler, model_dir / "autoencoder_scaler.pkl")
        meta = {
            "input_dim": self.input_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "threshold": self.threshold,
        }
        with open(model_dir / "autoencoder_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, model_dir: Path) -> AutoencoderAnomalyDetector:
        """Load trained autoencoder detector from disk."""
        with open(model_dir / "autoencoder_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        detector = cls(input_dim=meta["input_dim"], bottleneck_dim=meta["bottleneck_dim"])
        detector.threshold = meta["threshold"]
        detector.scaler = joblib.load(model_dir / "autoencoder_scaler.pkl")
        weights = torch.load(model_dir / "autoencoder_weights.pt", map_location=torch.device("cpu"))
        detector.net.load_state_dict(weights)
        detector.net.eval()
        return detector
