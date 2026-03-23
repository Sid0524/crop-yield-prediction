"""
AWS SageMaker Inference Entry Point
=====================================
Implements the four-function SageMaker contract for the XGBoost
crop yield predictor. Deployed via XGBoostModel container
(framework_version="2.0-1").

Contract:
  model_fn    — load model from model_dir (called once at startup)
  input_fn    — deserialise incoming request payload
  predict_fn  — run inference
  output_fn   — serialise response
"""

import os
import json
import logging
import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def model_fn(model_dir: str):
    """
    Load XGBoost model from model_dir.
    SageMaker calls this once when the endpoint starts.
    """
    model_path = os.path.join(model_dir, "xgboost_model.json")
    logger.info(f"Loading model from {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)
    logger.info("Model loaded successfully")
    return model


def input_fn(request_body: str, request_content_type: str):
    """
    Deserialise input payload into an xgb.DMatrix.

    Expected input format (application/json):
      [[avg_temp_c, total_rain_mm, gdd, soil_quality,
        irrigation_frac, fertilizer_kg_ha, year, ...crop_dummies, country_enc], ...]

    One row = one farm prediction request. Batch requests supported.
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)
        arr  = np.array(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        logger.info(f"Input shape: {arr.shape}")
        return xgb.DMatrix(arr)

    raise ValueError(
        f"Unsupported content type: {request_content_type}. "
        "Use 'application/json'."
    )


def predict_fn(input_data, model):
    """
    Run inference. Model predicts log1p(yield), so we back-transform.

    Returns numpy array of yield predictions in kg/ha.
    """
    log_preds = model.predict(input_data)
    yield_preds = np.expm1(log_preds)
    logger.info(f"Predicted {len(yield_preds)} yields, "
                f"range [{yield_preds.min():.0f}, {yield_preds.max():.0f}] kg/ha")
    return yield_preds


def output_fn(prediction: np.ndarray, accept: str):
    """
    Serialise predictions to JSON.

    Returns: (body_string, content_type)
    """
    response = {
        "predictions": prediction.tolist(),
        "unit": "kg/ha",
        "n": len(prediction),
    }
    return json.dumps(response), "application/json"
