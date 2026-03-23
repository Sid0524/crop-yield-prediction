"""
Phase 5 — Folium Geospatial Risk Map
=======================================
Generates an interactive HTML choropleth-style map of predicted
crop yield risk across global farm locations.

Risk tiers (vs country-crop median yield):
  High Yield    (>=120%): green  #2ecc71
  Normal       (85-120%): blue   #3498db
  Moderate Risk (60-85%): orange #f39c12
  High Risk      (<60%):  red    #e74c3c

10K sample points are rendered as CircleMarkers with rich popups.
A MarkerCluster layer provides an alternative clustered view.

Output: outputs/maps/yield_risk_map.html
"""

import sys
import json
import pathlib
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from branca.colormap import LinearColormap
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

PARQ_PATH  = ROOT / "data" / "processed" / "farm_scale_500k.parquet"
MODELS_DIR = ROOT / "models"
MAPS_DIR   = ROOT / "outputs" / "maps"
MAPS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
MAP_SAMPLES  = 10_000   # max markers (>15K makes HTML unresponsive in browser)


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------
RISK_COLORS = {
    "High Yield":     "#2ecc71",
    "Normal":         "#3498db",
    "Moderate Risk":  "#f39c12",
    "High Risk":      "#e74c3c",
}

def assign_risk(yield_val: float, median_yield: float) -> str:
    if median_yield <= 0:
        return "Normal"
    ratio = yield_val / median_yield
    if ratio >= 1.20:
        return "High Yield"
    elif ratio >= 0.85:
        return "Normal"
    elif ratio >= 0.60:
        return "Moderate Risk"
    else:
        return "High Risk"


def load_data_with_predictions():
    print("Loading parquet ...")
    df = pd.read_parquet(PARQ_PATH)

    # Load feature names
    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    # Reproduce preprocessing: one-hot + target encode
    from sklearn.preprocessing import TargetEncoder
    df_enc = pd.get_dummies(df, columns=["crop"], drop_first=True)
    te = TargetEncoder(target_type="continuous", random_state=RANDOM_SEED)
    te.fit(df_enc[["country_code"]], np.log1p(df["yield_kg_ha"].values))
    df_enc["country_enc"] = te.transform(df_enc[["country_code"]])

    for col in feature_names:
        if col not in df_enc.columns:
            df_enc[col] = 0.0

    X = df_enc[feature_names].values.astype("float32")

    print("Loading XGBoost model ...")
    model = xgb.XGBRegressor()
    model.load_model(str(MODELS_DIR / "xgboost_model.json"))

    print("Running predictions on full dataset ...")
    log_pred = model.predict(X)
    df["predicted_yield"] = np.expm1(log_pred).astype("float32")

    # Restore crop column (was one-hot encoded above)
    # Re-derive from original df (crop column still in df)
    print(f"  Full dataset predicted. Shape: {df.shape}")
    return df


def add_risk_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute country-crop median and assign risk tier per farm."""
    medians = (
        df.groupby(["country", "crop"])["predicted_yield"]
        .median()
        .rename("median_yield")
        .reset_index()
    )
    df = df.merge(medians, on=["country", "crop"], how="left")
    df["risk_tier"] = df.apply(
        lambda r: assign_risk(r["predicted_yield"], r["median_yield"]), axis=1
    )
    tier_counts = df["risk_tier"].value_counts()
    print("Risk tier distribution:")
    for tier, cnt in tier_counts.items():
        print(f"  {tier:<18} {cnt:>8,} ({cnt/len(df)*100:.1f}%)")
    return df


def build_map(df_sample: pd.DataFrame) -> folium.Map:
    m = folium.Map(
        location=[20, 10],
        zoom_start=2,
        tiles="CartoDB positron",
    )

    # Continuous colormap (5th–95th percentile range)
    v_min = float(df_sample["predicted_yield"].quantile(0.05))
    v_max = float(df_sample["predicted_yield"].quantile(0.95))
    colormap = LinearColormap(
        colors=["#e74c3c", "#f39c12", "#2ecc71"],
        vmin=v_min,
        vmax=v_max,
        caption="Predicted Yield (kg/ha)",
    )
    colormap.add_to(m)

    # --- Layer 1: Individual CircleMarkers ---
    marker_group = folium.FeatureGroup(name="Farm Points", show=True)

    for _, row in df_sample.iterrows():
        color = RISK_COLORS.get(row["risk_tier"], "#95a5a6")
        popup_html = (
            f"<b>{row['country']}</b><br>"
            f"Crop: {row.get('crop', 'N/A')}<br>"
            f"Year: {int(row['year'])}<br>"
            f"Predicted: <b>{row['predicted_yield']:.0f} kg/ha</b><br>"
            f"Actual:    {row['yield_kg_ha']:.0f} kg/ha<br>"
            f"Risk: <span style='color:{color}'><b>{row['risk_tier']}</b></span><br>"
            f"Temp: {row['avg_temp_c']:.1f}°C  "
            f"Rain: {row['total_rain_mm']:.0f} mm"
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            weight=0.5,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{row['country']} | {row['risk_tier']}",
        ).add_to(marker_group)

    marker_group.add_to(m)

    # --- Layer 2: MarkerCluster (clustered view) ---
    cluster_group = folium.FeatureGroup(name="Clustered View", show=False)
    mc = MarkerCluster()

    for _, row in df_sample.iterrows():
        color = RISK_COLORS.get(row["risk_tier"], "#95a5a6")
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=0.5,
            tooltip=f"{row['country']} | {row['predicted_yield']:.0f} kg/ha",
        ).add_to(mc)

    mc.add_to(cluster_group)
    cluster_group.add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background-color: white; padding: 12px; border-radius: 6px;
                border: 1px solid #ccc; font-size: 13px; line-height: 1.8;">
    <b>Yield Risk Tier</b><br>
    <span style="color:#2ecc71">&#9679;</span> High Yield (&ge;120% of median)<br>
    <span style="color:#3498db">&#9679;</span> Normal (85–120%)<br>
    <span style="color:#f39c12">&#9679;</span> Moderate Risk (60–85%)<br>
    <span style="color:#e74c3c">&#9679;</span> High Risk (&lt;60%)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def main():
    df = load_data_with_predictions()
    df = add_risk_tiers(df)

    # Sample for map rendering performance
    df_sample = df.sample(MAP_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"\nBuilding map with {MAP_SAMPLES:,} sample points ...")

    m = build_map(df_sample)

    out_path = MAPS_DIR / "yield_risk_map.html"
    m.save(str(out_path))
    file_mb = out_path.stat().st_size / 1e6
    print(f"Saved -> {out_path}  ({file_mb:.1f} MB)")
    print("Open in a browser to explore the interactive map.")


if __name__ == "__main__":
    main()
