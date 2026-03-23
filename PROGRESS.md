# Project Progress & Next Steps

## What We Built

### Infrastructure
- Git repository initialised with clean commit history
- GitHub repo: https://github.com/Sid0524/crop-yield-prediction
- `.gitignore` configured (data, models, outputs excluded from tracking)
- `requirements.txt` with pinned dependencies

### Data Pipeline
- **`scripts/01_download_faostat.py`** — streams FAOSTAT bulk ZIP (~50 MB),
  filters to yield data for 10 staple crops (2000-2022), attaches country-level
  climate normals (temperature, rainfall) and computes crop-specific Growing
  Degree Days (GDD)
- **`scripts/02_synthesize_farm_scale.py`** — augments ~15K country-level rows
  into exactly 500K synthetic farm-scale records using a 5-layer parametric stack:
  lognormal yield perturbation, climate microvariation, soil quality, irrigation
  fraction, fertilizer rates, and a year-trend signal
- `data/geo/country_centroids.csv` — lat/lon for ~80 countries (Natural Earth v5)

### Utility Modules
- **`scripts/utils/preprocessing.py`** — `COUNTRY_CLIMATE_LOOKUP` (80 countries,
  WMO/WorldBank 1991-2020 normals), `COUNTRY_CENTROIDS`, `COUNTRY_AREA_HA`,
  `COUNTRY_IRRIGATION`, `COUNTRY_FERTILIZER`, `CROP_GDD_PARAMS`, `compute_gdd()`
- **`scripts/utils/metrics.py`** — `compute_metrics()` returning RMSE, R², MAE, MAPE

### Model Training
- **`scripts/03_train_models.py`** — trains three models on log1p-transformed yield
  with a temporal train/test split (train ≤ 2018, test ≥ 2019):
  - XGBoost (`tree_method=hist`, early stopping, saved as portable JSON)
  - LightGBM (histogram boosting, early stopping callbacks)
  - PyTorch MLP (4-layer, HuberLoss, AdamW, ReduceLROnPlateau, `num_workers=0`)
- Target encoding for country, one-hot for crop
- Outputs: model files, `model_comparison.png`, `mlp_loss_curve.png`, `model_results.json`

### SHAP Explainability
- **`scripts/04_shap_explainability.py`** — `shap.TreeExplainer` on XGBoost (5K sample):
  - Summary plot (global feature importance beeswarm)
  - Force plot (HTML interactive + static PNG for highest-predicted farm)
  - Dependence plots for `avg_temp_c` and `total_rain_mm` with auto interaction
  - 3-column feature importance comparison: XGBoost gain vs LightGBM split vs mean |SHAP|

### Geospatial Risk Map
- **`scripts/05_folium_map.py`** — Folium map with 10K CircleMarkers on CartoDB
  positron basemap; 4 risk tiers (High Yield / Normal / Moderate Risk / High Risk)
  relative to country-crop median; popup with country, crop, year, yield, temp, rain;
  MarkerCluster second layer; branca continuous colormap

### Benchmarking
- **`scripts/06_benchmark.py`** — pipeline timing at 50K / 200K / 500K records
  (wall time + peak RAM), parquet vs CSV read speed comparison, SHAP computation
  scaling at 1K / 5K / 10K samples, dual-axis bar+line chart

### AWS SageMaker
- **`sagemaker/inference.py`** — four-function SageMaker contract:
  `model_fn`, `input_fn`, `predict_fn` (with `np.expm1` back-transform), `output_fn`
- **`scripts/07_sagemaker_deploy.py`** — packages `model.tar.gz`, uploads to S3,
  deploys `XGBoostModel` on `ml.m5.xlarge`; auto mock-mode when no AWS credentials

### Kaggle GPU Notebook
- **`crop_yield_prediction_kaggle.ipynb`** — single self-contained notebook running
  the full pipeline, optimised for Kaggle free T4 GPU:
  - XGBoost: `device='cuda'`
  - LightGBM: `device='gpu'`, stderr redirected to suppress nvcc warnings
  - PyTorch: CPU tensors in DataLoader, `.to(DEVICE)` inside training loop
    (`num_workers=0` to avoid CUDA fork errors)
  - SHAP: base_score patched via `booster.save_config()` / `booster.load_config()`
    to fix XGBoost GPU `'[8.976656E0]'` parse error in SHAP 0.46

### Documentation
- **`README.md`** — project overview, model results table, quickstart (Kaggle + local),
  data methodology, SHAP section, Folium map section, SageMaker section, tech stack,
  data sources

---

## Current Model Results (Kaggle T4 Run)

| Model | RMSE (kg/ha) | R² | MAE (kg/ha) |
|---|---|---|---|
| XGBoost | ~11,600 | ~0.847 | ~5,600 |
| LightGBM | 11,788 | 0.842 | 5,691 |
| PyTorch MLP | TBD | TBD | TBD |

---

## Next Steps

### Immediate (resume polish, no new code)

1. **Push executed notebook to GitHub**
   Download the Kaggle notebook with all outputs saved and push it so
   recruiters can see rendered plots directly on GitHub.

2. **Publish Kaggle notebook publicly**
   Settings → Visibility → Public. Add the URL to your resume and LinkedIn.

3. **Update README with final metric numbers**
   Replace placeholder values in the results table with exact numbers
   from your completed Kaggle run.

4. **Add a LICENSE file**
   GitHub shows "No license" without one — looks incomplete on a portfolio.
   Go to GitHub repo → Add file → `LICENSE` → MIT template.

---

### Technical Upgrades (high resume impact)

5. **Optuna hyperparameter tuning**
   Replace hand-picked XGBoost/LightGBM params with Optuna TPE search.
   Adds a tuning curve plot and expected R² gain of +0.02 to +0.05.

6. **Gradio web app on Hugging Face Spaces**
   Input: temperature, rainfall, crop type.
   Output: predicted yield + SHAP force plot.
   Free hosting, live demo URL — highest single impact addition for a resume.

7. **Replace synthetic data with real farm data**
   - USDA NASS: US county-level yield via free API
   - NASA POWER: gridded daily climate (temp, rain, solar radiation)
   This is the biggest credibility upgrade for the project.

8. **Temporal model (LSTM or Temporal Fusion Transformer)**
   Current models treat each farm-year independently. A sequence model
   captures year-over-year trends and drought cycles that tree models cannot.

---

### MLOps & Deployment

9. **MLflow experiment tracking**
   Log all training runs (params, metrics, artefacts) — makes the project
   look production-grade.

10. **GitHub Actions CI**
    Smoke test on every push (synthesise 1K rows → train → assert R² > 0.5).
    Shows software engineering discipline beyond notebooks.

11. **Deploy to AWS SageMaker for real**
    Run `scripts/07_sagemaker_deploy.py` with a real IAM role, screenshot
    the live endpoint test, add it to the README.
