"""Metric-learning losses for pulse-level embeddings."""

import torch.nn.functional as F
from torch import nn


def l2_normalize_embeddings(embeddings, dim=-1, eps=1e-12):
    """L2-normalize embeddings along the embedding dimension."""
    return F.normalize(embeddings, p=2, dim=dim, eps=eps)


def build_triplet_margin_loss(margin=0.5, p=2):
    """Return the standard triplet margin loss.

    L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    """
    if margin <= 0:
        raise ValueError("margin must be positive.")
    return nn.TripletMarginLoss(margin=margin, p=p, reduction="mean")
