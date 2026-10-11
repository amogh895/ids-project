"""Isolation Forest Anomaly Detection Baseline.

Trained exclusively on BENIGN flows to detect novel / zero-day anomalies
by measuring isolation path length.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features.extract import FEATURE_COLS


class IsolationForestAnomalyDetector:
    """Isolation Forest anomaly detection wrapper calibrated on benign data."""

    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=1,
        )
        self.threshold: float = 0.0

    def fit(self, X_benign: pd.DataFrame) -> IsolationForestAnomalyDetector:
        """Fit scaler and Isolation Forest on benign flows only."""
        X_scaled = self.scaler.fit_transform(X_benign[FEATURE_COLS])
        self.model.fit(X_scaled)

        # Raw scores: lower means more anomalous. Invert so higher = more anomalous.
        raw_scores = -self.model.score_samples(X_scaled)
        # Calibrate threshold at 99th percentile of benign training errors (1% FPR)
        self.threshold = float(np.percentile(raw_scores, 99.0))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Compute continuous anomaly score (higher = more anomalous)."""
        X_scaled = self.scaler.transform(X[FEATURE_COLS])
        raw_scores = -self.model.score_samples(X_scaled)
        return raw_scores

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary anomaly: 1 = anomaly, 0 = normal."""
        scores = self.score(X)
        return (scores >= self.threshold).astype(int)

    def save(self, model_dir: Path) -> None:
        """Save fitted model, scaler, and calibrated threshold."""
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_dir / "isolation_forest.pkl")
        joblib.dump(self.scaler, model_dir / "if_scaler.pkl")
        with open(model_dir / "if_threshold.json", "w", encoding="utf-8") as f:
            json.dump({"threshold": self.threshold}, f, indent=2)

    @classmethod
    def load(cls, model_dir: Path) -> IsolationForestAnomalyDetector:
        """Load fitted detector from disk."""
        detector = cls()
        detector.model = joblib.load(model_dir / "isolation_forest.pkl")
        detector.scaler = joblib.load(model_dir / "if_scaler.pkl")
        with open(model_dir / "if_threshold.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        detector.threshold = data["threshold"]
        return detector
