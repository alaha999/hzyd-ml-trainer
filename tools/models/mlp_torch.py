"""Simple PyTorch MLP. Edit layers here to change the architecture."""
import numpy as np
import torch
from torch import nn


def build_model(n_features):
    return nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


@torch.inference_mode()
def predict(model, x, batch_size=16384):
    model.eval()
    outputs = []
    for i in range(0, len(x), batch_size):
        batch = torch.from_numpy(np.asarray(x[i:i + batch_size], dtype=np.float32))
        outputs.append(torch.sigmoid(model(batch)).numpy().ravel())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
