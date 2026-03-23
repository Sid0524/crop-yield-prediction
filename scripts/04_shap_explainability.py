"""
Phase 4 — SHAP Explainability
================================
Generates three SHAP visualisations for the trained XGBoost model:
  1. Summary plot   — global feature importance (beeswarm)
  2. Force plot     — single-prediction explanation (HTML + static PNG)
  3. Dependence plots — avg_temp_c and total_rain_mm vs SHAP value

Also produces a 3-column feature importance comparison table:
  XGBoost gain | LightGBM split count | Mean |SHAP| (XGBoost)

Outputs:
  outputs/plots/shap_summary.png
  outputs/plots/shap_force_plot.html
  outputs/plots/shap_force_plot_static.png
  outputs/plots/shap_dependence_avg_temp_c.png
  outputs/plots/shap_dependence_total_rain_mm.png
  outputs/plots/feature_importance_comparison.png
"""

import sys
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

PARQ_PATH   = ROOT / "data" / "processed" / "farm_scale_500k.parquet"
MODELS_DIR  = ROOT / "models"
PLOTS_DIR   = ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
SHAP_SAMPLES = 5_000   # subsample for SHAP computation speed


def load_test_data():
    """Reproduce the exact preprocessing from 03_train_models.py."""
    print("Loading test data ...")
    df = pd.read_parquet(PARQ_PATH)
    test_df = df[df["year"] > 2018].copy()

    # One-hot encode crop
    test_df = pd.get_dummies(test_df, columns=["crop"], drop_first=True)

    # Target-encode country_code (fit on full df to match training)
    full_df = pd.get_dummies(df, columns=["crop"], drop_first=True)
    te = TargetEncoder(target_type="continuous", random_state=RANDOM_SEED)
    import numpy as _np
    te.fit(full_df[["country_code"]], _np.log1p(df["yield_kg_ha"].values))
    test_df["country_enc"] = te.transform(test_df[["country_code"]])

    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    # Align test columns to feature list
    for col in feature_names:
        if col not in test_df.columns:
            test_df[col] = 0.0

    X_test = test_df[feature_names].values.astype("float32")
    y_test = test_df["yield_kg_ha"].values.astype("float64")
    print(f"  Test set: {len(X_test):,} rows, {X_test.shape[1]} features")
    return X_test, y_test, feature_names


def load_models():
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(str(MODELS_DIR / "xgboost_model.json"))

    lgb_model = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_model.txt"))
    return xgb_model, lgb_model


def compute_shap_values(xgb_model, X_sample, feature_names):
    """Use TreeExplainer — fast, exact tree-native SHAP."""
    print(f"Computing SHAP values on {len(X_sample):,} samples ...")
    explainer  = shap.TreeExplainer(xgb_model)
    shap_vals  = explainer.shap_values(X_sample)
    print(f"  SHAP values shape: {shap_vals.shape}")
    return explainer, shap_vals


def plot_summary(shap_vals, X_sample, feature_names):
    print("Generating SHAP summary plot ...")
    X_df = pd.DataFrame(X_sample, columns=feature_names)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_df, max_display=15, show=False)
    plt.title("SHAP Summary — XGBoost (test set sample)", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved -> outputs/plots/shap_summary.png")


def plot_force(explainer, shap_vals, X_sample, feature_names, y_pred):
    """Force plot for the highest-predicted farm."""
    idx = int(np.argmax(y_pred))
    print(f"Generating force plot for farm index {idx} "
          f"(pred yield ~{np.expm1(y_pred[idx]):.0f} kg/ha) ...")
    X_df = pd.DataFrame(X_sample, columns=feature_names)

    # HTML interactive version
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_vals[idx],
        X_df.iloc[idx],
        feature_names=feature_names,
        matplotlib=False,
    )
    shap.save_html(str(PLOTS_DIR / "shap_force_plot.html"), force_plot)
    print("  Saved -> outputs/plots/shap_force_plot.html")

    # Static PNG for notebook embedding
    shap.force_plot(
        explainer.expected_value,
        shap_vals[idx],
        X_df.iloc[idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    plt.savefig(PLOTS_DIR / "shap_force_plot_static.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved -> outputs/plots/shap_force_plot_static.png")


def plot_dependence(shap_vals, X_sample, feature_names, features=None):
    if features is None:
        features = ["avg_temp_c", "total_rain_mm"]
    X_df = pd.DataFrame(X_sample, columns=feature_names)
    for feat in features:
        if feat not in feature_names:
            print(f"  Skipping {feat} (not in feature list)")
            continue
        print(f"Generating dependence plot: {feat} ...")
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            feat,
            shap_vals,
            X_df,
            interaction_index="auto",   # SHAP auto-detects strongest interactor
            feature_names=feature_names,
            show=False,
        )
        plt.title(f"SHAP Dependence — {feat}", fontsize=12)
        plt.tight_layout()
        safe = feat.replace("/", "_").replace(" ", "_")
        plt.savefig(PLOTS_DIR / f"shap_dependence_{safe}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved -> outputs/plots/shap_dependence_{safe}.png")


def plot_importance_comparison(xgb_model, lgb_model, shap_vals, feature_names):
    """
    3-column comparison:
      XGBoost gain | LightGBM split count | Mean |SHAP|

    Demonstrates that built-in importances disagree with each other
    and why SHAP is the most trustworthy measure.
    """
    print("Generating feature importance comparison ...")
    n = len(feature_names)

    # XGBoost gain importance
    xgb_imp = xgb_model.get_booster().get_score(importance_type="gain")
    xgb_arr = np.array([xgb_imp.get(f, 0.0) for f in feature_names])
    xgb_arr = xgb_arr / xgb_arr.sum() if xgb_arr.sum() > 0 else xgb_arr

    # LightGBM split importance
    lgb_arr = lgb_model.feature_importance(importance_type="split")
    # lgb uses its own feature ordering; map by position
    if len(lgb_arr) == n:
        lgb_arr = lgb_arr / lgb_arr.sum() if lgb_arr.sum() > 0 else lgb_arr
    else:
        lgb_arr = np.zeros(n)

    # Mean |SHAP|
    shap_arr = np.abs(shap_vals).mean(axis=0)
    shap_arr = shap_arr / shap_arr.sum() if shap_arr.sum() > 0 else shap_arr

    df_imp = pd.DataFrame({
        "Feature":        feature_names,
        "XGBoost_Gain":   xgb_arr,
        "LightGBM_Split": lgb_arr,
        "Mean_SHAP":      shap_arr,
    }).sort_values("Mean_SHAP", ascending=False).head(15)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    cols   = ["XGBoost_Gain", "LightGBM_Split", "Mean_SHAP"]
    titles = ["XGBoost (Gain)", "LightGBM (Split Count)", "Mean |SHAP| (XGBoost)"]
    colors = ["steelblue", "darkorange", "seagreen"]

    for ax, col, title, color in zip(axes, cols, titles, colors):
        sub = df_imp.sort_values(col, ascending=True)
        ax.barh(sub["Feature"], sub[col], color=color, alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Normalised importance")

    plt.suptitle("Feature Importance: Built-in vs SHAP", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved -> outputs/plots/feature_importance_comparison.png")
    return df_imp


def main():
    X_test, y_test, feature_names = load_test_data()
    xgb_model, lgb_model         = load_models()

    # Subsample for SHAP (TreeExplainer is fast but 500K is large)
    rng     = np.random.default_rng(RANDOM_SEED)
    idx_sub = rng.choice(len(X_test), SHAP_SAMPLES, replace=False)
    X_sub   = X_test[idx_sub]

    explainer, shap_vals = compute_shap_values(xgb_model, X_sub, feature_names)

    # Log-scale predictions for force plot index selection
    log_pred = xgb_model.predict(X_sub)

    plot_summary(shap_vals, X_sub, feature_names)
    plot_force(explainer, shap_vals, X_sub, feature_names, log_pred)
    plot_dependence(shap_vals, X_sub, feature_names)
    df_imp = plot_importance_comparison(xgb_model, lgb_model, shap_vals, feature_names)

    print("\nFeature importance comparison (top 10 by Mean |SHAP|):")
    print(df_imp.head(10).to_string(index=False))
    print("\nDone. All SHAP plots saved to outputs/plots/")


if __name__ == "__main__":
    main()
