"""Shared regression metrics helper."""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, R², MAE, and MAPE.
    Both arrays must be in original (non-log) scale.

    Returns
    -------
    dict with keys: RMSE, R2, MAE, MAPE_pct
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    # Avoid division by zero in MAPE
    mask = y_true > 1.0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    return {"RMSE": round(rmse, 2), "R2": round(r2, 4),
            "MAE": round(mae, 2), "MAPE_pct": round(mape, 2)}
