"""Minimal supervised training helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _loader(iq: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(iq), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def accuracy(model: nn.Module, iq: np.ndarray, labels: np.ndarray, device: torch.device) -> float:
    """Return classification accuracy."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in _loader(iq, labels, 256, False):
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    return correct / len(labels)


def train_classifier(
    model: nn.Module,
    train_iq: np.ndarray,
    train_labels: np.ndarray,
    test_iq: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    epochs: int = 12,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
) -> dict[str, list[float]]:
    """Train an emitter classifier with cross-entropy."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    loader = _loader(train_iq, train_labels, batch_size, True)
    history = {"loss": [], "test_accuracy": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        history["loss"].append(total_loss / len(train_labels))
        history["test_accuracy"].append(accuracy(model, test_iq, test_labels, device))
        print(f"epoch {epoch + 1:2d}: loss={history['loss'][-1]:.3f}, test_acc={history['test_accuracy'][-1]:.3f}")
    return history


def embeddings(model: nn.Module, iq: np.ndarray, device: torch.device) -> np.ndarray:
    """Extract encoder embeddings."""
    model.eval()
    batches = []
    with torch.no_grad():
        for (x,) in DataLoader(TensorDataset(torch.from_numpy(iq)), batch_size=256):
            batches.append(model.encoder(x.to(device)).cpu().numpy())
    return np.concatenate(batches)
