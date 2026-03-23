"""
Phase 6 — Pipeline Benchmarking
=================================
Benchmarks the full training pipeline (load → preprocess → XGBoost fit)
at three dataset scales: 50K, 200K, 500K records.

Also benchmarks:
  - Parquet vs CSV read speed
  - SHAP computation time at 1K / 5K / 10K sample sizes

Outputs:
  outputs/plots/benchmark_runtime.png
  outputs/metrics/benchmark_results.json
"""

import sys
import json
import time
import pathlib
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

PARQ_PATH   = ROOT / "data" / "processed" / "farm_scale_500k.parquet"
PLOTS_DIR   = ROOT / "outputs" / "plots"
METRICS_DIR = ROOT / "outputs" / "metrics"
for d in [PLOTS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SCALES      = [50_000, 200_000, 500_000]

NUMERIC_FEATURES = [
    "avg_temp_c", "total_rain_mm", "gdd",
    "soil_quality", "irrigation_frac", "fertilizer_kg_ha", "year",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame):
    df_enc = pd.get_dummies(df, columns=["crop"], drop_first=True)
    te = TargetEncoder(target_type="continuous", random_state=RANDOM_SEED)
    df_enc["country_enc"] = te.fit_transform(
        df_enc[["country_code"]], np.log1p(df["yield_kg_ha"].values)
    )
    crop_cols = sorted([c for c in df_enc.columns if c.startswith("crop_")])
    features  = NUMERIC_FEATURES + crop_cols + ["country_enc"]
    for col in features:
        if col not in df_enc.columns:
            df_enc[col] = 0.0
    X = df_enc[features].values.astype("float32")
    y = np.log1p(df["yield_kg_ha"].values.astype("float64"))
    return X, y


def quick_xgb_params():
    """Fewer estimators for benchmarking — representative but not full training."""
    return dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        device="cpu",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


# ---------------------------------------------------------------------------
# Benchmark 1: Pipeline scaling (50K / 200K / 500K)
# ---------------------------------------------------------------------------

def benchmark_pipeline(df_full: pd.DataFrame) -> list[dict]:
    results = []
    for n in SCALES:
        print(f"\nBenchmarking pipeline at {n:,} records ...")
        df_n = df_full.sample(n, random_state=RANDOM_SEED)

        t0 = time.perf_counter()
        X, y = preprocess(df_n)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.1, random_state=RANDOM_SEED
        )
        model = xgb.XGBRegressor(**quick_xgb_params())
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  verbose=False, early_stopping_rounds=20)
        elapsed = time.perf_counter() - t0

        # Peak RSS via psutil if available, else estimate
        try:
            import psutil, os
            mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
        except ImportError:
            mem_mb = n * X.shape[1] * 4 / 1e6 * 3  # rough estimate

        results.append({
            "n_records":      n,
            "train_time_s":   round(elapsed, 2),
            "peak_memory_mb": round(mem_mb, 1),
        })
        print(f"  n={n:>7,} | time={elapsed:.2f}s | mem~{mem_mb:.0f} MB")

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Parquet vs CSV read speed
# ---------------------------------------------------------------------------

def benchmark_file_formats(df_full: pd.DataFrame) -> dict:
    print("\nBenchmarking Parquet vs CSV read speed (full 500K dataset) ...")

    with tempfile.TemporaryDirectory() as tmp:
        parq_path = pathlib.Path(tmp) / "bench.parquet"
        csv_path  = pathlib.Path(tmp) / "bench.csv"

        df_full.to_parquet(parq_path, index=False)
        df_full.to_csv(csv_path, index=False)

        parq_size_mb = parq_path.stat().st_size / 1e6
        csv_size_mb  = csv_path.stat().st_size  / 1e6

        # Parquet
        t0 = time.perf_counter()
        for _ in range(3):
            pd.read_parquet(parq_path)
        parq_time = (time.perf_counter() - t0) / 3

        # CSV
        t0 = time.perf_counter()
        for _ in range(3):
            pd.read_csv(csv_path, low_memory=False)
        csv_time = (time.perf_counter() - t0) / 3

    result = {
        "parquet_read_s": round(parq_time, 3),
        "csv_read_s":     round(csv_time, 3),
        "parquet_size_mb": round(parq_size_mb, 1),
        "csv_size_mb":    round(csv_size_mb, 1),
        "speedup_x":      round(csv_time / parq_time, 1),
    }
    print(f"  Parquet: {parq_time:.3f}s ({parq_size_mb:.1f} MB)")
    print(f"  CSV:     {csv_time:.3f}s ({csv_size_mb:.1f} MB)")
    print(f"  Speedup: {result['speedup_x']}x faster with Parquet")
    return result


# ---------------------------------------------------------------------------
# Benchmark 3: SHAP computation scaling
# ---------------------------------------------------------------------------

def benchmark_shap(df_full: pd.DataFrame) -> list[dict]:
    import shap

    print("\nBenchmarking SHAP computation (XGBoost TreeExplainer) ...")

    # Train a quick model on 50K
    df_n = df_full.sample(50_000, random_state=RANDOM_SEED)
    X, y = preprocess(df_n)
    model = xgb.XGBRegressor(**quick_xgb_params())
    model.fit(X, y, verbose=False)

    explainer = shap.TreeExplainer(model)
    results   = []

    for n_shap in [1_000, 5_000, 10_000]:
        X_sub = X[:n_shap]
        t0    = time.perf_counter()
        _     = explainer.shap_values(X_sub)
        elapsed = time.perf_counter() - t0
        results.append({"n_shap_samples": n_shap, "shap_time_s": round(elapsed, 2)})
        print(f"  SHAP n={n_shap:>6,} | {elapsed:.2f}s")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_benchmark(pipeline_results: list[dict]):
    labels = [f"{r['n_records']//1000}K" for r in pipeline_results]
    times  = [r["train_time_s"]   for r in pipeline_results]
    mems   = [r["peak_memory_mb"] for r in pipeline_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax1.bar(x, times, color="steelblue", alpha=0.8, label="Train time (s)")
    ax1.set_xlabel("Dataset size")
    ax1.set_ylabel("Training time (seconds)", color="steelblue")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.tick_params(axis="y", labelcolor="steelblue")

    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{t:.1f}s", ha="center", va="bottom", fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(x, mems, "o-", color="darkorange", lw=2, ms=8, label="Peak RAM (MB)")
    ax2.set_ylabel("Peak memory (MB)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Pipeline Benchmark: Training Time & Memory vs Dataset Size", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "benchmark_runtime.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> outputs/plots/benchmark_runtime.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading full 500K dataset ...")
    df_full = pd.read_parquet(PARQ_PATH)
    print(f"  Loaded {len(df_full):,} rows")

    pipeline_results = benchmark_pipeline(df_full)
    format_results   = benchmark_file_formats(df_full)
    shap_results     = benchmark_shap(df_full)

    all_results = {
        "pipeline_scaling": pipeline_results,
        "file_format_comparison": format_results,
        "shap_scaling": shap_results,
    }

    with open(METRICS_DIR / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> outputs/metrics/benchmark_results.json")

    plot_benchmark(pipeline_results)

    print("\n=== Benchmark Summary ===")
    print(f"{'Scale':<10} {'Time(s)':>10} {'Mem(MB)':>10}")
    print("-" * 32)
    for r in pipeline_results:
        print(f"{r['n_records']:<10,} {r['train_time_s']:>10.2f} {r['peak_memory_mb']:>10.0f}")

    print(f"\nParquet is {format_results['speedup_x']}x faster to read than CSV")
    print(f"  and {format_results['csv_size_mb']/format_results['parquet_size_mb']:.1f}x smaller on disk")


if __name__ == "__main__":
    main()
