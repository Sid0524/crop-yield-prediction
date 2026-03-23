"""
Phase 2 — Farm-Scale Data Synthesis
======================================
Augments the country-level FAOSTAT yield data (~15K rows) into
500K synthetic farm-scale records using a 5-layer parametric
synthesis stack grounded in agricultural science.

Output: data/processed/farm_scale_500k.parquet
        data/geo/country_centroids.csv

Methodology reference:
  - Yield CV calibration: USDA ERS "Variability in U.S. Crop Yields"
  - Lognormal precipitation: Granger & Pomeroy (2001)
  - GDD formulation: McMaster & Wilhelm (1997)
"""

import sys
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils.preprocessing import (
    COUNTRY_CLIMATE_LOOKUP,
    COUNTRY_CENTROIDS,
    COUNTRY_AREA_HA,
    COUNTRY_ISO3,
    COUNTRY_IRRIGATION,
    COUNTRY_FERTILIZER,
    CROP_GDD_PARAMS,
    compute_gdd,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_CSV   = ROOT / "data" / "raw" / "faostat_yield.csv"
OUT_PARQ  = ROOT / "data" / "processed" / "farm_scale_500k.parquet"
CENT_CSV  = ROOT / "data" / "geo" / "country_centroids.csv"

(ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "geo").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_ROWS   = 500_000
RANDOM_SEED   = 42
# Minimum / maximum farms per country-crop-year row
MIN_FARMS_ROW = 5
MAX_FARMS_ROW = 8_000
# Scale factor: total farms per million hectares of agricultural land
FARMS_PER_M_HA = 3_500


def save_centroids() -> None:
    """Write country_centroids.csv from the lookup table."""
    rows = [
        {"country": k, "lat": v[0], "lon": v[1]}
        for k, v in COUNTRY_CENTROIDS.items()
        if k != "_DEFAULT"
    ]
    pd.DataFrame(rows).to_csv(CENT_CSV, index=False)
    print(f"Saved {len(rows)} country centroids → {CENT_CSV}")


def compute_farms_per_row(country: str) -> int:
    """
    Number of synthetic farms to generate for a single country-crop-year row.
    Proportional to country agricultural land area, clamped to [MIN, MAX].
    """
    area = COUNTRY_AREA_HA.get(country, COUNTRY_AREA_HA["_DEFAULT"])
    n = int(area / 1_000 * FARMS_PER_M_HA / 1_000)
    return max(MIN_FARMS_ROW, min(MAX_FARMS_ROW, n))


def synthesize_farms(row: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate synthetic farm-level records from a single FAOSTAT country-crop-year row.

    Layers:
      1. Row replication weighted by agricultural land area
      2. Lognormal yield perturbation (sigma=0.25, matching ~25% CV)
      3. Climate microvariation per farm
      4. Agronomic covariates (soil quality, irrigation, fertilizer)
      5. Year trend (+0.3%/year technology drift)
    """
    country  = row["country"]
    crop     = row["crop"]
    year     = int(row["year"])
    base_yield = float(row["yield_kg_ha"])

    n = compute_farms_per_row(country)

    # --- Climate lookup ---
    clim = COUNTRY_CLIMATE_LOOKUP.get(country, COUNTRY_CLIMATE_LOOKUP["_DEFAULT"])
    c_temp     = clim["temp"]
    c_rain     = clim["rain"]
    c_temp_std = clim["temp_std"]

    # --- Layer 3: Climate microvariation ---
    farm_temp = c_temp + rng.normal(0, c_temp_std * 0.3, n)
    # Lognormal rainfall: sigma calibrated so CoV ≈ 0.15
    farm_rain = c_rain * rng.lognormal(0.0, 0.15, n)
    farm_rain = np.clip(farm_rain, 10.0, 8000.0)

    # GDD recomputed per farm temperature
    params = CROP_GDD_PARAMS.get(crop, {"base_temp_c": 10, "season_days": 120})
    farm_gdd = np.maximum(0.0, farm_temp - params["base_temp_c"]) * params["season_days"]

    # --- Layer 5: Year trend ---
    year_multiplier = 1.0 + (year - 2000) * 0.003

    # --- Layer 2: Lognormal yield perturbation ---
    # sigma=0.25 gives CoV ≈ 25%, matching published farm-level yield variability
    farm_yield = base_yield * year_multiplier * rng.lognormal(0.0, 0.25, n)
    farm_yield = np.clip(farm_yield, 50.0, 200_000.0)

    # --- Layer 4: Agronomic covariates ---
    # Soil quality: Beta(2,2) centred around country mean ± 0.15
    soil_mean = min(max(0.4, 0.5 + (c_temp - 15) * 0.01), 0.7)
    soil_a = soil_mean * 4
    soil_b = (1 - soil_mean) * 4
    soil_quality = rng.beta(max(0.5, soil_a), max(0.5, soil_b), n) * 100

    irrig_mean = COUNTRY_IRRIGATION.get(country, COUNTRY_IRRIGATION["_DEFAULT"])
    irrig_a = max(0.5, irrig_mean * 5)
    irrig_b = max(0.5, (1 - irrig_mean) * 5)
    irrigation_frac = rng.beta(irrig_a, irrig_b, n)

    fert_mean = COUNTRY_FERTILIZER.get(country, COUNTRY_FERTILIZER["_DEFAULT"])
    # Lognormal fertilizer with sigma=0.4 → realistic right skew
    fertilizer_kg_ha = rng.lognormal(np.log(fert_mean + 1), 0.4, n)
    fertilizer_kg_ha = np.clip(fertilizer_kg_ha, 0.0, 1500.0)

    # --- Lat/Lon scatter around country centroid ---
    centroid = COUNTRY_CENTROIDS.get(country, COUNTRY_CENTROIDS["_DEFAULT"])
    farm_lat = centroid[0] + rng.normal(0, 3.0, n)
    farm_lon = centroid[1] + rng.normal(0, 4.0, n)
    farm_lat = np.clip(farm_lat, -90.0, 90.0)
    farm_lon = np.clip(farm_lon, -180.0, 180.0)

    iso3 = COUNTRY_ISO3.get(country, "UNK")

    return pd.DataFrame({
        "country":        country,
        "country_code":   iso3,
        "crop":           crop,
        "year":           np.int16(year),
        "lat":            farm_lat.astype("float32"),
        "lon":            farm_lon.astype("float32"),
        "avg_temp_c":     farm_temp.astype("float32"),
        "total_rain_mm":  farm_rain.astype("float32"),
        "gdd":            farm_gdd.astype("float32"),
        "soil_quality":   soil_quality.astype("float32"),
        "irrigation_frac":irrigation_frac.astype("float32"),
        "fertilizer_kg_ha": fertilizer_kg_ha.astype("float32"),
        "yield_kg_ha":    farm_yield.astype("float32"),
    })


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Run 01_download_faostat.py first."
        )

    # Save centroids CSV
    save_centroids()

    # Load FAOSTAT
    df_raw = pd.read_csv(RAW_CSV)
    print(f"Loaded FAOSTAT data: {len(df_raw):,} rows")

    rng = np.random.default_rng(RANDOM_SEED)
    chunks = []
    total_generated = 0

    for idx, row in df_raw.iterrows():
        chunk = synthesize_farms(row, rng)
        chunks.append(chunk)
        total_generated += len(chunk)
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1:,}/{len(df_raw):,} rows "
                  f"| Generated {total_generated:,} farms so far")

    print(f"\nTotal synthetic farms before sampling: {total_generated:,}")

    df_all = pd.concat(chunks, ignore_index=True)

    # Exact target
    if len(df_all) >= TARGET_ROWS:
        df_all = df_all.sample(TARGET_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        # If under target (unlikely), oversample with small noise
        shortage = TARGET_ROWS - len(df_all)
        print(f"  Under target by {shortage:,}; oversampling with noise ...")
        extra = df_all.sample(shortage, replace=True, random_state=RANDOM_SEED).copy()
        noise = rng.normal(0, 0.02, len(extra))
        extra["yield_kg_ha"] *= (1 + noise).astype("float32")
        df_all = pd.concat([df_all, extra], ignore_index=True)

    # Assign sequential farm IDs
    df_all.insert(0, "farm_id", np.arange(len(df_all), dtype=np.int32))

    print(f"\nFinal dataset shape: {df_all.shape}")
    print(df_all.dtypes)
    print(df_all.describe())

    df_all.to_parquet(OUT_PARQ, index=False)
    print(f"\nSaved → {OUT_PARQ}  ({OUT_PARQ.stat().st_size / 1e6:.1f} MB)")

    # Quick split summary
    train = df_all[df_all["year"] <= 2018]
    test  = df_all[df_all["year"] > 2018]
    print(f"Train (2000-2018): {len(train):,} rows")
    print(f"Test  (2019-2022): {len(test):,} rows")


if __name__ == "__main__":
    main()
