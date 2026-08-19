"""
Scoring logic for anomaly detection — framework-agnostic.

This module is deliberately independent of Azure: it just loads the
trained artifacts and scores a single sensor reading. It can be tested
locally, then wrapped by an Azure Function (or any other server) later.

A reading is a dict of sensor_name -> value, e.g.:
    {"sensor_2": 641.82, "sensor_3": 1589.70, ...}
Only the sensors kept during training are used; any extra keys are
ignored, and missing required sensors raise a clear error.
"""

from pathlib import Path

import joblib
import numpy as np
import torch

# Model definition must be importable to rebuild the network architecture
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from autoencoder import Autoencoder

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


class AnomalyScorer:
    """Loads the trained model once, then scores readings on demand."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.sensor_cols = joblib.load(artifacts_dir / "sensor_cols.pkl")
        self.scaler = joblib.load(artifacts_dir / "scaler.pkl")
        self.threshold = joblib.load(artifacts_dir / "threshold.pkl")

        self.model = Autoencoder(n_features=len(self.sensor_cols))
        self.model.load_state_dict(torch.load(artifacts_dir / "autoencoder.pth"))
        self.model.eval()

    def score(self, reading: dict) -> dict:
        """Score one reading. Returns error, threshold, and anomaly flag."""
        missing = [c for c in self.sensor_cols if c not in reading]
        if missing:
            raise ValueError(f"Missing required sensors: {missing}")

        # Order values exactly as during training, then normalize.
        # Pass a DataFrame with column names so the scaler doesn't warn.
        import pandas as pd
        values = pd.DataFrame([[reading[c] for c in self.sensor_cols]], columns=self.sensor_cols)
        values_scaled = self.scaler.transform(values).astype(np.float32)

        with torch.no_grad():
            recon = self.model(torch.from_numpy(values_scaled)).numpy()
        error = float(np.mean((values_scaled - recon) ** 2))

        return {
            "reconstruction_error": error,
            "threshold": float(self.threshold),
            "is_anomaly": bool(error > self.threshold),
        }


if __name__ == "__main__":
    # Local test using a real row from the dataset
    import pandas as pd

    scorer = AnomalyScorer()
    print(f"Loaded model. {len(scorer.sensor_cols)} sensors, threshold={scorer.threshold:.4f}\n")

    COLUMN_NAMES = (
        ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )
    df = pd.read_csv(sys.argv[1], sep=r"\s+", header=None, names=COLUMN_NAMES)

    # An early-life cycle (should be normal) and a late-life cycle (likely anomaly)
    # Use a test engine known to degrade strongly. Engine 3 (held-out)
    # showed reconstruction error climbing well above threshold at end of life.
    engine = df[df["unit_id"] == 3]
    early = engine.iloc[0]
    late = engine.iloc[-1]

    for label, row in [("early-life cycle (healthy)", early), ("late-life cycle (degraded)", late)]:
        reading = {c: float(row[c]) for c in scorer.sensor_cols}
        result = scorer.score(reading)
        print(f"{label}: error={result['reconstruction_error']:.4f} "
              f"anomaly={result['is_anomaly']}")
