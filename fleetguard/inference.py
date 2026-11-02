"""Load saved FleetGuard artifacts and predict RUL for one engine history."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

from .data import window_for_unit
from .model import WindowLSTM


def predict_history(history: pd.DataFrame, artifacts_dir: Path) -> float:
    """Predict the last-cycle RUL for a frame with cycle, settings, and sensor columns."""
    prep = joblib.load(artifacts_dir / "preprocessing_and_ridge.joblib")
    checkpoint = torch.load(artifacts_dir / "lstm.pt", map_location="cpu", weights_only=True)
    cols = prep["feature_columns"]
    required = {"cycle", *cols}
    if not required.issubset(history.columns):
        raise ValueError(f"history is missing columns: {sorted(required-set(history.columns))}")
    history = history.sort_values("cycle").copy()
    scaled = history.copy()
    scaled[cols] = scaled[cols].astype(np.float64)
    transformed = prep["scaler"].transform(scaled[cols].astype(np.float64))
    scaled[cols] = transformed.astype(np.float64, copy=False)
    scaled["unit"] = 1
    x = window_for_unit(scaled, cols, prep["window_length"])[None, ...]
    model = WindowLSTM(checkpoint["n_features"], checkpoint["hidden_size"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = float(model(torch.tensor(x, dtype=torch.float32)).item())
    return float(np.clip(prediction, 0, prep["rul_cap"]))
