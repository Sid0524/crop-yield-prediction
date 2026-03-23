"""
Phase 1 — FAOSTAT Data Acquisition
====================================
Downloads the FAOSTAT Crops and Livestock Products bulk dataset,
filters to yield data for the top-10 staple crops (2000-2022),
attaches country-level climate normals and Growing Degree Days,
and saves the result to data/raw/faostat_yield.csv.

Data source:
  FAO – Food and Agriculture Organization of the United Nations
  FAOSTAT Production/Crops and livestock products (domain QCL)
  https://www.fao.org/faostat/en/#data/QCL
"""

import io
import os
import sys
import zipfile
import pathlib
import requests
import pandas as pd
import numpy as np

# Allow imports from sibling package when run as a script
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils.preprocessing import (
    COUNTRY_CLIMATE_LOOKUP,
    COUNTRY_ISO3,
    CROP_GDD_PARAMS,
    compute_gdd,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RAW_DIR / "faostat_yield.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAOSTAT_URL = (
    "https://bulks-faostat.fao.org/production/"
    "Production_Crops_Livestock_E_All_Data_(Normalized).zip"
)
# Fallback API endpoint if bulk ZIP download fails
FAOSTAT_API = (
    "https://fenixservices.fao.org/faostat/api/v1/en/data/QCL"
    "?area=all&element=5419&item=all&year=2000:2022&format=csv&show_codes=true"
)

TOP_CROPS = {
    "Wheat", "Rice, paddy", "Maize", "Soybeans",
    "Sugar cane", "Potatoes", "Cassava", "Sweet potatoes",
    "Sorghum", "Barley",
}

START_YEAR = 2000


def download_with_progress(url: str, desc: str = "Downloading") -> bytes:
    """Stream-download a URL and return raw bytes with a simple progress log."""
    print(f"{desc}: {url}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    chunks = []
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=1 << 16):  # 64 KB
        chunks.append(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB  ({pct:.0f}%)",
                  end="", flush=True)
    print()
    return b"".join(chunks)


def load_from_zip(raw_bytes: bytes) -> pd.DataFrame:
    """Extract the normalized CSV from the FAOSTAT ZIP archive."""
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        names = zf.namelist()
        # The normalized file contains 'Normalized' in the name
        csv_name = next((n for n in names if "Normalized" in n and n.endswith(".csv")), None)
        if csv_name is None:
            # Fall back to any CSV
            csv_name = next(n for n in names if n.endswith(".csv"))
        print(f"  Extracting: {csv_name}")
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, encoding="latin-1", low_memory=False)
    return df


def try_bulk_download() -> pd.DataFrame | None:
    """Attempt to download and parse the FAOSTAT bulk ZIP."""
    try:
        raw = download_with_progress(FAOSTAT_URL, "Downloading FAOSTAT bulk ZIP")
        return load_from_zip(raw)
    except Exception as e:
        print(f"  Bulk download failed: {e}")
        return None


def try_api_download() -> pd.DataFrame | None:
    """Fallback: download a subset via the FAOSTAT REST API."""
    try:
        print("Trying FAOSTAT API fallback ...")
        raw = download_with_progress(FAOSTAT_API, "  API")
        df = pd.read_csv(io.StringIO(raw.decode("utf-8")), low_memory=False)
        return df
    except Exception as e:
        print(f"  API fallback also failed: {e}")
        return None


def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to yield element, top-10 crops, year >= 2000."""
    print(f"Raw shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Normalise column names (FAOSTAT uses 'Element' for the measure type)
    df.columns = [c.strip() for c in df.columns]

    # Keep only Yield rows (kg/ha)
    if "Element" in df.columns:
        df = df[df["Element"].str.strip() == "Yield"].copy()
    elif "element" in df.columns:
        df = df[df["element"].str.strip() == "Yield"].copy()

    # Normalise crop/item column
    item_col = "Item" if "Item" in df.columns else "item"
    area_col = "Area" if "Area" in df.columns else "area"
    year_col = "Year" if "Year" in df.columns else "year"
    val_col  = "Value" if "Value" in df.columns else "value"

    df = df.rename(columns={item_col: "crop", area_col: "country",
                             year_col: "year",  val_col:  "yield_kg_ha"})

    # Filter crops
    df["crop"] = df["crop"].str.strip()
    df = df[df["crop"].isin(TOP_CROPS)].copy()

    # Filter years
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"] >= START_YEAR].copy()

    # Drop missing yields
    df["yield_kg_ha"] = pd.to_numeric(df["yield_kg_ha"], errors="coerce")
    df = df.dropna(subset=["yield_kg_ha"]).copy()
    df = df[df["yield_kg_ha"] > 0].copy()

    df["country"] = df["country"].str.strip()
    df = df[["country", "crop", "year", "yield_kg_ha"]].reset_index(drop=True)
    print(f"After filtering: {df.shape[0]} rows, {df['country'].nunique()} countries, "
          f"{df['crop'].nunique()} crops")
    return df


def attach_climate(df: pd.DataFrame) -> pd.DataFrame:
    """Attach climate normals and compute GDD for each row."""
    def _lookup(country: str, key: str) -> float:
        rec = COUNTRY_CLIMATE_LOOKUP.get(country, COUNTRY_CLIMATE_LOOKUP["_DEFAULT"])
        return rec[key]

    df["avg_temp_c"]    = df["country"].map(lambda c: _lookup(c, "temp")).astype("float32")
    df["total_rain_mm"] = df["country"].map(lambda c: _lookup(c, "rain")).astype("float32")
    df["temp_std"]      = df["country"].map(lambda c: _lookup(c, "temp_std")).astype("float32")
    df["rain_std"]      = df["country"].map(lambda c: _lookup(c, "rain_std")).astype("float32")

    df["gdd"] = df.apply(
        lambda r: compute_gdd(float(r["avg_temp_c"]), r["crop"]), axis=1
    ).astype("float32")

    # ISO3 code
    df["country_code"] = df["country"].map(
        lambda c: COUNTRY_ISO3.get(c, "UNK")
    )

    return df


def main() -> None:
    # 1. Download
    df_raw = try_bulk_download()
    if df_raw is None:
        df_raw = try_api_download()
    if df_raw is None:
        raise RuntimeError(
            "Could not download FAOSTAT data. Check your internet connection."
        )

    # 2. Filter
    df = filter_and_clean(df_raw)

    # 3. Climate features
    df = attach_climate(df)

    # 4. Save
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df):,} rows → {OUT_CSV}")
    print(df.head())
    print(df.dtypes)


if __name__ == "__main__":
    main()
