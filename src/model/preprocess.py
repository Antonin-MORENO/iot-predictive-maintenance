"""
C-MAPSS data preparation for autoencoder training.

Split strategy (important): the train/test split is done BY ENGINE, not
by row. Engines are divided into a train set and a test set, so the
model is evaluated on engines it has NEVER seen during training — closer
to real deployment, where new engines appear.

Steps:
1. Load the dataset and drop zero-variance sensors (uninformative)
2. Split engines into train / test groups
3. From TRAIN engines, keep only "normal" early-life cycles for training
4. Keep TEST engines' full history (normal + degraded) for evaluation
5. Normalize with a scaler fit ONLY on the training-normal data
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

COLUMN_NAMES = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

NORMAL_CYCLES_THRESHOLD = 30  # early-life cycles considered "healthy"
TEST_ENGINE_FRACTION = 0.2    # 20% of engines held out for evaluation
SEED = 42


def load_cmapss(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)


def get_useful_sensor_columns(df: pd.DataFrame, min_std: float = 1e-6) -> list[str]:
    """Drop sensors with near-zero variance (no information)."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    return [c for c in sensor_cols if df[c].std() > min_std]


def split_engines(df: pd.DataFrame, test_fraction: float = TEST_ENGINE_FRACTION):
    """Randomly assign whole engines to train or test."""
    rng = np.random.default_rng(SEED)
    engine_ids = df["unit_id"].unique()
    rng.shuffle(engine_ids)
    n_test = int(len(engine_ids) * test_fraction)
    test_ids = set(engine_ids[:n_test])
    train_ids = set(engine_ids[n_test:])
    return train_ids, test_ids


def prepare_datasets(filepath: Path, normal_cycles: int = NORMAL_CYCLES_THRESHOLD):
    df = load_cmapss(filepath)
    sensor_cols = get_useful_sensor_columns(df)

    train_ids, test_ids = split_engines(df)

    # Train = normal (early-life) cycles of TRAIN engines only
    train_normal_df = df[
        (df["unit_id"].isin(train_ids)) & (df["cycle"] <= normal_cycles)
    ].reset_index(drop=True)

    # Test = full history of TEST engines (normal + degraded)
    test_df = df[df["unit_id"].isin(test_ids)].reset_index(drop=True)

    # Scaler fit ONLY on training-normal data (no leakage)
    scaler = StandardScaler()
    scaler.fit(train_normal_df[sensor_cols])

    X_train = scaler.transform(train_normal_df[sensor_cols]).astype(np.float32)
    X_test = scaler.transform(test_df[sensor_cols]).astype(np.float32)

    return {
        "sensor_cols": sensor_cols,
        "scaler": scaler,
        "X_train": X_train,          # normal cycles, train engines
        "X_test": X_test,            # full history, test engines
        "test_df": test_df,          # keeps unit_id/cycle for evaluation
        "n_train_engines": len(train_ids),
        "n_test_engines": len(test_ids),
    }


if __name__ == "__main__":
    import sys

    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train_FD001.txt")
    data = prepare_datasets(filepath)
    print(f"{len(data['sensor_cols'])} useful sensors")
    print(f"Train engines: {data['n_train_engines']} -> X_train {data['X_train'].shape}")
    print(f"Test engines : {data['n_test_engines']} -> X_test  {data['X_test'].shape}")
