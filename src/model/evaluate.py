"""
Evaluate the trained autoencoder on held-out test engines.

Goal: confirm the model actually detects degradation, and pick an
anomaly threshold.

Logic:
1. Load the trained model + scaler + sensor list
2. Compute reconstruction error (MSE per cycle) on:
   - training-normal cycles  -> defines what "normal error" looks like
   - test engines' full life  -> should rise as engines degrade
3. Set the anomaly threshold from the normal error distribution
   (mean + 3*std, a common choice)
4. Produce two plots saved under docs/:
   - error distribution: normal vs degraded
   - reconstruction error over life, for a few sample test engines
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from autoencoder import Autoencoder
from preprocess import prepare_datasets

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


def reconstruction_error(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    """Per-row mean squared reconstruction error."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X)
        recon = model(X_tensor).numpy()
    return np.mean((X - recon) ** 2, axis=1)


def main(filepath: Path) -> None:
    data = prepare_datasets(filepath)
    sensor_cols = joblib.load(ARTIFACTS_DIR / "sensor_cols.pkl")

    model = Autoencoder(n_features=len(sensor_cols))
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "autoencoder.pth"))

    # Error on training-normal data -> "normal" reference distribution
    err_normal = reconstruction_error(model, data["X_train"])

    # Error on test engines (full life, includes degradation)
    err_test = reconstruction_error(model, data["X_test"])

    # Threshold: mean + 3 std of the normal error distribution
    threshold = err_normal.mean() + 3 * err_normal.std()
    print(f"Normal error   : mean={err_normal.mean():.4f} std={err_normal.std():.4f}")
    print(f"Test error     : mean={err_test.mean():.4f} max={err_test.max():.4f}")
    print(f"Anomaly threshold (mean+3std): {threshold:.4f}")

    flagged = (err_test > threshold).mean() * 100
    print(f"Share of test cycles flagged as anomalies: {flagged:.1f}%")

    DOCS_DIR.mkdir(exist_ok=True)

    # --- Plot 1: error distributions ---
    plt.figure(figsize=(8, 5))
    # Clip the x-axis for readability (a few extreme degraded cycles reach ~23,
    # which would otherwise squash the informative region near the threshold)
    x_max = np.percentile(err_test, 99)
    plt.hist(
        np.clip(err_normal, None, x_max), bins=50, alpha=0.6,
        label="Normal (train engines, early life)", density=True,
    )
    plt.hist(
        np.clip(err_test, None, x_max), bins=50, alpha=0.6,
        label="Test engines (full life)", density=True,
    )
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.3f}")
    plt.xlabel("Reconstruction error (MSE)")
    plt.ylabel("Density")
    plt.title("Reconstruction error: normal vs degraded")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "error_distribution.png", dpi=120)
    plt.close()

    # --- Plot 2: error over life for a few sample test engines ---
    test_df = data["test_df"].copy()
    test_df["error"] = err_test
    sample_engines = test_df["unit_id"].unique()[:4]

    plt.figure(figsize=(9, 5))
    for eng in sample_engines:
        sub = test_df[test_df["unit_id"] == eng].sort_values("cycle")
        plt.plot(sub["cycle"], sub["error"], label=f"Engine {eng}")
    plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
    plt.xlabel("Cycle (engine life)")
    plt.ylabel("Reconstruction error (MSE)")
    plt.title("Reconstruction error rises as engines degrade")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "error_over_life.png", dpi=120)
    plt.close()

    # Save threshold for later use (dashboard / inference)
    joblib.dump(float(threshold), ARTIFACTS_DIR / "threshold.pkl")
    print(f"\nSaved plots to {DOCS_DIR}/ and threshold to artifacts/threshold.pkl")


if __name__ == "__main__":
    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train_FD001.txt")
    main(filepath)
