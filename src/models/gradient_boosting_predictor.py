"""Inference interface for the persisted Gradient Boosting pipeline."""

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("data/processed/hungary_2026/models/gradient_boosting_pipeline.joblib")


def load_gradient_boosting_model(model_path: Path = MODEL_PATH):
    """Load the fitted full sklearn pipeline used by the race simulator."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run `python -m src.models.train_gradient_boosting` first."
        )
    return joblib.load(model_path)


def predict_next_lap_time(model, features: pd.DataFrame) -> float:
    """Predict one next-lap time from a single-row feature DataFrame."""
    if len(features) != 1:
        raise ValueError("The simulator prediction interface expects exactly one row.")
    return float(model.predict(features)[0])
