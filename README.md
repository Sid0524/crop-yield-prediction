# Explainable Crop Yield Prediction with Geospatial Risk Mapping

End-to-end ML pipeline that predicts farm-scale crop yields globally, explains predictions with SHAP, and visualises risk geospatially — trained on 500K synthetic farm records derived from FAOSTAT country-level data.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.46-green)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

| Component | Details |
|---|---|
| **Data** | FAOSTAT crop yields (2000-2022) + synthesised 500K farm-scale records |
| **Models** | XGBoost · LightGBM · PyTorch MLP |
| **Explainability** | SHAP TreeExplainer — summary, force, and dependence plots |
| **Geospatial** | Folium interactive risk map (10K farm points, 4 risk tiers) |
| **Benchmarking** | Pipeline runtime at 50K / 200K / 500K records |
| **Deployment** | AWS SageMaker real-time endpoint (XGBoost) |

---

## Model Results (Test Set 2019–2022)

| Model | RMSE (kg/ha) | R² | MAE (kg/ha) | Train Time |
|---|---|---|---|---|
| **XGBoost** | 11,631 | **0.8462** | 5,627 | 3.6s (GPU) |
| **LightGBM** | 11,783 | 0.8422 | 5,692 | 14.9s (GPU) |
| **PyTorch MLP** | 13,114 | 0.8045 | 6,402 | 152s (GPU) |

> Note: High absolute RMSE is expected — the dataset spans 10 crops with yields ranging from ~2,500 kg/ha (barley) to ~65,000 kg/ha (sugar cane). R² > 0.84 across all models on unseen years (2019-2022) confirms strong generalisation.

---

## Project Structure

```
agriculture/
├── crop_yield_prediction_kaggle.ipynb  # Main notebook (Kaggle GPU-optimised)
├── requirements.txt
│
├── scripts/
│   ├── 01_download_faostat.py          # FAOSTAT bulk download + climate features
│   ├── 02_synthesize_farm_scale.py     # 5-layer 500K farm synthesis
│   ├── 03_train_models.py              # XGBoost / LightGBM / PyTorch MLP
│   ├── 04_shap_explainability.py       # SHAP summary, force, dependence plots
│   ├── 05_folium_map.py                # Geospatial yield risk map
│   ├── 06_benchmark.py                 # Pipeline scaling benchmark
│   ├── 07_sagemaker_deploy.py          # AWS SageMaker deployment
│   └── utils/
│       ├── preprocessing.py            # Country climate / centroid lookups
│       └── metrics.py                  # RMSE / R² / MAE / MAPE
│
├── sagemaker/
│   └── inference.py                    # SageMaker entry point script
│
└── data/  models/  outputs/            # Generated at runtime (git-ignored)
```

---

## Quickstart

### Option A — Kaggle (recommended, free GPU)

1. Go to [Kaggle](https://www.kaggle.com/code) → **New Notebook** → **File → Upload Notebook**
2. Upload `crop_yield_prediction_kaggle.ipynb`
3. Set **Accelerator → GPU T4 x2**
4. **Run All**

### Option B — Local

```bash
git clone https://github.com/Sid0524/crop-yield-prediction.git
cd crop-yield-prediction
pip install -r requirements.txt

python scripts/01_download_faostat.py
python scripts/02_synthesize_farm_scale.py
python scripts/03_train_models.py
python scripts/04_shap_explainability.py
python scripts/05_folium_map.py
python scripts/06_benchmark.py
python scripts/07_sagemaker_deploy.py   # mock mode without AWS credentials
```

---

## Data Methodology

FAOSTAT provides country-year-crop averages (~15K rows). To reach 500K farm-scale records, a **5-layer parametric synthesis stack** is applied:

| Layer | Method | Scientific Basis |
|---|---|---|
| Row replication | Weighted by country agricultural area | FAO Land Use data |
| Yield perturbation | Lognormal (σ=0.25) | USDA ERS within-country CV 20-30% |
| Climate microvariation | Normal temp · Lognormal rainfall | Published climate variability |
| Agronomic covariates | Beta-distributed soil quality, irrigation, fertilizer | FAO AQUASTAT |
| Year trend | +0.3%/year linear | Global yield trend literature |

Climate normals (temperature, rainfall) sourced from World Bank Climate Portal 1991-2020 averages for ~80 countries.

---

## SHAP Explainability

Three SHAP visualisations are generated for the XGBoost model:

- **Summary plot** — global feature importance ranked by mean |SHAP| across 5,000 test farms
- **Force plot** — single-prediction breakdown for the highest-predicted farm
- **Dependence plots** — `avg_temp_c` and `total_rain_mm` vs SHAP value with auto-detected interaction features

A 3-column comparison table contrasts XGBoost gain, LightGBM split count, and mean |SHAP| — demonstrating why built-in importances are unreliable and SHAP is the trustworthy measure.

---

## Geospatial Risk Map

Interactive Folium map with 10,000 farm points classified into 4 risk tiers relative to their country-crop median yield:

| Tier | Threshold | Colour |
|---|---|---|
| High Yield | ≥ 120% of median | Green |
| Normal | 85 – 120% | Blue |
| Moderate Risk | 60 – 85% | Orange |
| High Risk | < 60% | Red |

---

## AWS SageMaker Deployment

The trained XGBoost model is packaged as a `model.tar.gz` artifact and deployed to a real-time SageMaker endpoint using the pre-built XGBoost 2.x container.

```python
# Inference request (JSON)
payload = [[24.0, 1083.0, 1820.0, 70.0, 0.40, 130.0, 2021, ...]]
response = predictor.predict(payload)
# {"predictions": [4823.0], "unit": "kg/ha"}
```

Running `07_sagemaker_deploy.py` without AWS credentials prints a **mock mode** summary — safe for portfolio review.

> **Cost note:** `ml.m5.xlarge` costs ~$0.23/hr. Always call `predictor.delete_endpoint()` after testing.

---

## Tech Stack

- **Data** — pandas, pyarrow, requests
- **ML** — XGBoost 2.x, LightGBM 4.x, PyTorch 2.x
- **Explainability** — SHAP 0.46
- **Geospatial** — Folium, Branca
- **Visualisation** — Matplotlib, Seaborn
- **Cloud** — AWS SageMaker, boto3
- **Notebook** — Jupyter, optimised for Kaggle T4 GPU

---

## Data Sources

- **Yield data:** FAO – Food and Agriculture Organization, FAOSTAT Production / Crops and livestock products (domain QCL), 2000-2022. https://www.fao.org/faostat/
- **Climate normals:** World Bank Climate Portal, 1991-2020 averages. https://climateknowledgeportal.worldbank.org/
- **Irrigation:** FAO AQUASTAT country statistics
- **Fertilizer:** FAO fertilizer use by crop statistics
- **Centroids:** Natural Earth dataset v5

---

## License

MIT
