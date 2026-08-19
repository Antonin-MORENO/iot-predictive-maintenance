"""
Feedforward autoencoder for anomaly detection.

The network compresses a single cycle's sensor readings (15 features)
down to a small bottleneck (5 values), then reconstructs them. Trained
only on "normal" (healthy) cycles, it learns to reconstruct normal
behaviour well. On degraded cycles it has never seen, reconstruction
error rises — and that error is the anomaly signal.
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, n_features: int, bottleneck: int = 5):
        super().__init__()
        # Encoder: n_features -> 10 -> bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 10),
            nn.ReLU(),
            nn.Linear(10, bottleneck),
            nn.ReLU(),
        )
        # Decoder: bottleneck -> 10 -> n_features (mirror of the encoder)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 10),
            nn.ReLU(),
            nn.Linear(10, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    # Quick sanity check: a random batch should pass through and come out
    # with the same shape as it went in.
    model = Autoencoder(n_features=15)
    dummy = torch.randn(8, 15)  # batch of 8 cycles, 15 sensors each
    out = model(dummy)
    print("Input shape :", dummy.shape)
    print("Output shape:", out.shape)
    print("Model:\n", model)
