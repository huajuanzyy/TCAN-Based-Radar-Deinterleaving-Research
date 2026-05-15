"""Random triplet sampling within one TSRD pulse window."""

import numpy as np


def sample_triplets_from_window(labels, num_triplets, rng=None):
    """Sample anchor-positive-negative index triplets within one window.

    Anchor and positive come from the same label. Negative comes from a
    different label. Labels with fewer than two pulses cannot provide an
    anchor-positive pair. Windows with fewer than two unique labels are skipped.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array.")
    if num_triplets <= 0:
        raise ValueError("num_triplets must be positive.")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.empty((0, 3), dtype=np.int64)

    rng = np.random.default_rng() if rng is None else rng
    indices_by_label = {
        label: np.flatnonzero(labels == label)
        for label in unique_labels
    }
    positive_labels = [
        label
        for label, indices in indices_by_label.items()
        if len(indices) >= 2
    ]
    if not positive_labels:
        return np.empty((0, 3), dtype=np.int64)

    triplets = np.empty((num_triplets, 3), dtype=np.int64)
    for triplet_index in range(num_triplets):
        anchor_label = rng.choice(positive_labels)
        anchor_positive_indices = indices_by_label[anchor_label]
        anchor_index, positive_index = rng.choice(
            anchor_positive_indices,
            size=2,
            replace=False,
        )

        negative_labels = unique_labels[unique_labels != anchor_label]
        negative_label = rng.choice(negative_labels)
        negative_index = rng.choice(indices_by_label[negative_label])
        triplets[triplet_index] = (anchor_index, positive_index, negative_index)

    return triplets


def sample_triplets_from_batch(labels_batch, num_triplets_per_window, rng=None):
    """Sample triplets for each window in a batch.

    Returns four arrays: batch indices, anchor indices, positive indices, and
    negative indices. Each triplet is constrained to one window.
    """
    labels_batch = np.asarray(labels_batch, dtype=np.int64)
    if labels_batch.ndim != 2:
        raise ValueError("labels_batch must have shape [B, T].")

    rng = np.random.default_rng() if rng is None else rng
    batch_indices = []
    anchor_indices = []
    positive_indices = []
    negative_indices = []

    for batch_index, labels in enumerate(labels_batch):
        triplets = sample_triplets_from_window(
            labels,
            num_triplets=num_triplets_per_window,
            rng=rng,
        )
        if len(triplets) == 0:
            continue
        batch_indices.append(np.full(len(triplets), batch_index, dtype=np.int64))
        anchor_indices.append(triplets[:, 0])
        positive_indices.append(triplets[:, 1])
        negative_indices.append(triplets[:, 2])

    if not batch_indices:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty

    return (
        np.concatenate(batch_indices),
        np.concatenate(anchor_indices),
        np.concatenate(positive_indices),
        np.concatenate(negative_indices),
    )
