"""
Train the anomaly-detection autoencoder on normal C-MAPSS cycles.

Pipeline:
1. Load and preprocess data via preprocess.py (drops useless sensors,
   splits normal cycles, normalizes)
2. Train the autoencoder to reconstruct NORMAL cycles only
3. Save the trained weights + the scaler + the sensor list, so the model
   can later be reused for inference (locally or deployed on Azure ML)

Run:
    python src/model/train.py data/train_FD001.txt
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow importing sibling modules when run as a script
sys.path.append(str(Path(__file__).resolve().parent))
from autoencoder import Autoencoder
from preprocess import prepare_datasets

# --- Hyperparameters ---
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
BOTTLENECK = 5
SEED = 42

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def train(filepath: Path) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Data
    data = prepare_datasets(filepath)
    X_train = data["X_train"]
    n_features = X_train.shape[1]
    print(
        f"Training on {X_train.shape[0]} normal cycles "
        f"from {data['n_train_engines']} engines, {n_features} features."
    )

    X_tensor = torch.from_numpy(X_train)
    loader = DataLoader(
        TensorDataset(X_tensor), batch_size=BATCH_SIZE, shuffle=True
    )

    # 2. Model, loss, optimizer
    model = Autoencoder(n_features=n_features, bottleneck=BOTTLENECK)
    criterion = nn.MSELoss()  # reconstruction error = mean squared error
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training loop
    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)  # reconstruct the input
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(loader.dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} — loss: {epoch_loss:.6f}")

    # 4. Save artifacts (weights + scaler + sensor list)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ARTIFACTS_DIR / "autoencoder.pth")
    joblib.dump(data["scaler"], ARTIFACTS_DIR / "scaler.pkl")
    joblib.dump(data["sensor_cols"], ARTIFACTS_DIR / "sensor_cols.pkl")
    print(f"\nSaved model + scaler + sensor list to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train_FD001.txt")
    train(filepath)
