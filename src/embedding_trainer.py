"""Training helpers for TCAN triplet metric learning."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.clustering_baselines import normalize_window_features
from src.metric_losses import build_triplet_margin_loss, l2_normalize_embeddings
from src.triplet_sampler import sample_triplets_from_batch


@dataclass
class TripletTrainingResult:
    epoch_losses: list
    skipped_batches: int
    total_triplets: int


def windows_to_tensors(windows):
    """Normalize each window and stack features/labels into training tensors."""
    features = []
    labels = []
    for window in windows:
        features.append(normalize_window_features(window.X_window))
        labels.append(window.y_window.astype(np.int64, copy=False))
    return (
        torch.from_numpy(np.stack(features).astype(np.float32)),
        torch.from_numpy(np.stack(labels).astype(np.int64)),
    )


def build_window_dataloader(windows, batch_size, shuffle=True):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    feature_tensor, label_tensor = windows_to_tensors(windows)
    dataset = TensorDataset(feature_tensor, label_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _gather_triplet_embeddings(embeddings, labels, num_triplets_per_window, rng, device):
    batch_idx, anchor_idx, positive_idx, negative_idx = sample_triplets_from_batch(
        labels.detach().cpu().numpy(),
        num_triplets_per_window=num_triplets_per_window,
        rng=rng,
    )
    if len(batch_idx) == 0:
        return None

    batch_idx = torch.from_numpy(batch_idx).long().to(device)
    anchor_idx = torch.from_numpy(anchor_idx).long().to(device)
    positive_idx = torch.from_numpy(positive_idx).long().to(device)
    negative_idx = torch.from_numpy(negative_idx).long().to(device)
    return (
        embeddings[batch_idx, anchor_idx],
        embeddings[batch_idx, positive_idx],
        embeddings[batch_idx, negative_idx],
        len(batch_idx),
    )


def train_triplet_encoder(
    model,
    dataloader,
    device,
    epochs=2,
    learning_rate=1e-3,
    margin=0.5,
    num_triplets_per_window=256,
    seed=42,
):
    """Train a TCAN encoder with random in-window triplet sampling."""
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if num_triplets_per_window <= 0:
        raise ValueError("num_triplets_per_window must be positive.")

    model.to(device)
    criterion = build_triplet_margin_loss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)

    epoch_losses = []
    skipped_batches = 0
    total_triplets = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_triplets = 0
        epoch_skipped_batches = 0

        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = model.encode(features)
            embeddings = l2_normalize_embeddings(embeddings, dim=-1)
            gathered = _gather_triplet_embeddings(
                embeddings=embeddings,
                labels=labels,
                num_triplets_per_window=num_triplets_per_window,
                rng=rng,
                device=device,
            )
            if gathered is None:
                epoch_skipped_batches += 1
                continue

            anchors, positives, negatives, triplet_count = gathered
            loss = criterion(anchors, positives, negatives)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * triplet_count
            epoch_triplets += triplet_count

        if epoch_triplets == 0:
            raise ValueError(
                "No valid triplets were sampled. Use windows with at least two "
                "labels and at least one label containing two or more pulses."
            )

        mean_loss = running_loss / float(epoch_triplets)
        epoch_losses.append(mean_loss)
        skipped_batches += epoch_skipped_batches
        total_triplets += epoch_triplets
        print(
            f"Epoch {epoch:02d}/{epochs} | triplet_loss: {mean_loss:.6f} "
            f"| triplets: {epoch_triplets} | skipped_batches: {epoch_skipped_batches}"
        )

    return TripletTrainingResult(
        epoch_losses=epoch_losses,
        skipped_batches=skipped_batches,
        total_triplets=total_triplets,
    )
