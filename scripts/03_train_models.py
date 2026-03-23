"""
Phase 3 — Model Training
==========================
Trains three models on the 500K farm-scale dataset:
  1. XGBoost  (tree_method='hist', CPU-optimised)
  2. LightGBM (histogram-based gradient boosting)
  3. PyTorch MLP (4-layer, HuberLoss, AdamW)

Target variable: yield_kg_ha (log1p-transformed during training).
Temporal train/test split: train <= 2018, test >= 2019.

Outputs:
  models/xgboost_model.json
  models/lgbm_model.txt
  models/mlp_model.pth
  models/mlp_scaler.pkl
  models/feature_names.json
  outputs/metrics/model_results.json
  outputs/plots/model_comparison.png
  outputs/plots/mlp_loss_curve.png
  outputs/plots/residuals_xgboost.png
"""

import sys
import json
import pickle
import pathlib
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PARQ_PATH   = ROOT / "data" / "processed" / "farm_scale_500k.parquet"
MODELS_DIR  = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
PLOTS_DIR   = ROOT / "outputs" / "plots"
for d in [MODELS_DIR, METRICS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Feature config
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "avg_temp_c", "total_rain_mm", "gdd",
    "soil_quality", "irrigation_frac", "fertilizer_kg_ha",
    "year",
]
TARGET_COL = "yield_kg_ha"


# ===========================================================================
# Data Loading & Preprocessing
# ===========================================================================

def load_and_preprocess():
    print("Loading parquet ...")
    df = pd.read_parquet(PARQ_PATH)
    print(f"  {len(df):,} rows loaded")

    train_df = df[df["year"] <= 2018].copy()
    test_df  = df[df["year"] >  2018].copy()
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # One-hot encode crop (drop_first to avoid multicollinearity)
    train_df = pd.get_dummies(train_df, columns=["crop"], drop_first=True)
    test_df  = pd.get_dummies(test_df,  columns=["crop"], drop_first=True)
    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    # Target-encode country_code (avoids high-cardinality one-hot explosion)
    te = TargetEncoder(target_type="continuous", random_state=RANDOM_SEED)
    train_df["country_enc"] = te.fit_transform(
        train_df[["country_code"]], np.log1p(train_df[TARGET_COL])
    )
    test_df["country_enc"] = te.transform(test_df[["country_code"]])

    crop_dummy_cols = sorted([c for c in train_df.columns if c.startswith("crop_")])
    all_features = NUMERIC_FEATURES + crop_dummy_cols + ["country_enc"]

    X_train = train_df[all_features].values.astype("float32")
    y_train = np.log1p(train_df[TARGET_COL].values.astype("float64"))
    X_test  = test_df[all_features].values.astype("float32")
    y_test  = np.log1p(test_df[TARGET_COL].values.astype("float64"))
    y_test_orig = test_df[TARGET_COL].values.astype("float64")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_SEED
    )

    # Save feature names for downstream SHAP / SageMaker scripts
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(all_features, f, indent=2)

    return X_tr, X_val, X_test, y_tr, y_val, y_test, all_features, y_test_orig


# ===========================================================================
# Model 1: XGBoost
# ===========================================================================

def train_xgboost(X_tr, X_val, X_test, y_tr, y_val, y_test_orig):
    print("\n--- Training XGBoost ---")
    t0 = time.perf_counter()

    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cpu",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        early_stopping_rounds=50,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

    elapsed = time.perf_counter() - t0
    print(f"  Time: {elapsed:.1f}s  |  Best iteration: {model.best_iteration}")

    model.save_model(str(MODELS_DIR / "xgboost_model.json"))
    print(f"  Saved -> models/xgboost_model.json")

    y_pred = np.expm1(model.predict(X_test))
    metrics = compute_metrics(y_test_orig, y_pred)
    metrics["train_time_s"] = round(elapsed, 1)
    print(f"  Metrics: {metrics}")
    return model, metrics, y_pred


# ===========================================================================
# Model 2: LightGBM
# ===========================================================================

def train_lightgbm(X_tr, X_val, X_test, y_tr, y_val, y_test_orig):
    print("\n--- Training LightGBM ---")
    t0 = time.perf_counter()

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        num_leaves=127,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="regression",
        metric="rmse",
        n_jobs=-1,
        verbose=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    elapsed = time.perf_counter() - t0
    print(f"  Time: {elapsed:.1f}s")

    model.booster_.save_model(str(MODELS_DIR / "lgbm_model.txt"))
    print(f"  Saved -> models/lgbm_model.txt")

    y_pred = np.expm1(model.predict(X_test))
    metrics = compute_metrics(y_test_orig, y_pred)
    metrics["train_time_s"] = round(elapsed, 1)
    print(f"  Metrics: {metrics}")
    return model, metrics, y_pred


# ===========================================================================
# Model 3: PyTorch MLP
# ===========================================================================

class CropYieldMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, X_val, X_test, y_tr, y_val, y_test_orig):
    print("\n--- Training PyTorch MLP ---")
    t0 = time.perf_counter()

    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr).astype("float32")
    X_val_s = scaler.transform(X_val).astype("float32")
    X_te_s  = scaler.transform(X_test).astype("float32")

    Xt = torch.tensor(X_tr_s);  yt = torch.tensor(y_tr.astype("float32"))
    Xv = torch.tensor(X_val_s); yv = torch.tensor(y_val.astype("float32"))
    Xe = torch.tensor(X_te_s)

    # num_workers=0 is REQUIRED on Windows (no fork multiprocessing)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=2048,
                        shuffle=True, num_workers=0)

    model     = CropYieldMLP(X_tr_s.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=5, factor=0.5, min_lr=1e-6)
    criterion = nn.HuberLoss(delta=1.0)

    best_val, patience_ctr = float("inf"), 0
    train_losses, val_losses = [], []
    BEST_PATH = MODELS_DIR / "_best_mlp.pth"

    for epoch in range(50):
        model.train()
        ep_loss = sum(
            criterion(model(Xb), yb).item() * len(Xb)
            for Xb, yb in loader
        ) / len(Xt)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()

        scheduler.step(val_loss)
        train_losses.append(ep_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d} | train={ep_loss:.4f}  val={val_loss:.4f}"
                  f"  lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val, patience_ctr = val_loss, 0
            torch.save(model.state_dict(), BEST_PATH)
        else:
            patience_ctr += 1
            if patience_ctr >= 10:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(BEST_PATH, weights_only=True))
    BEST_PATH.rename(MODELS_DIR / "mlp_model.pth")

    with open(MODELS_DIR / "mlp_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    elapsed = time.perf_counter() - t0
    print(f"  Time: {elapsed:.1f}s")

    # Loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses,   label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.set_title("MLP Training Curve"); ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mlp_loss_curve.png", dpi=150)
    plt.close()

    model.eval()
    with torch.no_grad():
        y_pred = np.expm1(model(Xe).numpy())

    metrics = compute_metrics(y_test_orig, y_pred)
    metrics["train_time_s"] = round(elapsed, 1)
    print(f"  Metrics: {metrics}")
    return model, metrics, y_pred


# ===========================================================================
# Comparison Plots
# ===========================================================================

def plot_comparison(y_true, preds_dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = {"XGBoost": "steelblue", "LightGBM": "darkorange", "MLP": "seagreen"}

    for ax, (name, y_pred) in zip(axes, preds_dict.items()):
        idx = np.random.choice(len(y_true), min(8_000, len(y_true)), replace=False)
        ax.scatter(y_true[idx], y_pred[idx],
                   alpha=0.15, s=4, color=colors.get(name, "gray"))
        lim = np.percentile(np.concatenate([y_true, y_pred]), 99)
        ax.plot([0, lim], [0, lim], "r--", lw=1.2)
        ax.set_xlabel("Actual yield (kg/ha)")
        ax.set_ylabel("Predicted yield (kg/ha)" if ax == axes[0] else "")
        ax.set_title(name, fontsize=13)

    plt.suptitle("Predicted vs Actual — Test Set (2019-2022)", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> outputs/plots/model_comparison.png")


def plot_residuals(y_true, y_pred, model_name):
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=120, color="steelblue", edgecolor="none", alpha=0.75)
    ax.axvline(0, color="red", lw=1.5, label="Zero error")
    ax.set_xlabel("Residual (kg/ha)")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name} Residual Distribution (test set)")
    ax.legend()
    plt.tight_layout()
    fname = f"residuals_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(PLOTS_DIR / fname, dpi=150)
    plt.close()
    print(f"Saved -> outputs/plots/{fname}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    (X_tr, X_val, X_test, y_tr, y_val, y_test,
     feature_names, y_test_orig) = load_and_preprocess()

    all_metrics, all_preds = {}, {}

    xgb_model, xgb_m, xgb_pred = train_xgboost(X_tr, X_val, X_test, y_tr, y_val, y_test_orig)
    all_metrics["XGBoost"]  = xgb_m;  all_preds["XGBoost"]  = xgb_pred

    lgb_model, lgb_m, lgb_pred = train_lightgbm(X_tr, X_val, X_test, y_tr, y_val, y_test_orig)
    all_metrics["LightGBM"] = lgb_m;  all_preds["LightGBM"] = lgb_pred

    mlp_model, mlp_m, mlp_pred = train_mlp(X_tr, X_val, X_test, y_tr, y_val, y_test_orig)
    all_metrics["MLP"]      = mlp_m;  all_preds["MLP"]      = mlp_pred

    all_metrics["_feature_names"] = feature_names

    # Summary table
    print("\n" + "=" * 65)
    print(f"{'Model':<12} {'RMSE':>10} {'R2':>8} {'MAE':>10} {'MAPE%':>8} {'Time(s)':>9}")
    print("-" * 65)
    for name, m in all_metrics.items():
        if name.startswith("_"):
            continue
        print(f"{name:<12} {m['RMSE']:>10.1f} {m['R2']:>8.4f} "
              f"{m['MAE']:>10.1f} {m['MAPE_pct']:>8.2f} {m['train_time_s']:>9.1f}")
    print("=" * 65)

    with open(METRICS_DIR / "model_results.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved -> outputs/metrics/model_results.json")

    plot_comparison(y_test_orig, all_preds)
    plot_residuals(y_test_orig, xgb_pred, "XGBoost")


if __name__ == "__main__":
    main()
