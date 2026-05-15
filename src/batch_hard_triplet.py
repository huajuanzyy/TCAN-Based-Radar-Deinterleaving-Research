"""Batch-hard triplet loss for pulse-level embeddings."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class BatchHardTripletResult:
    loss: torch.Tensor
    valid_anchor_count: int
    total_anchor_count: int


def _validate_context_shapes(embeddings, labels):
    if embeddings.ndim != 2:
        raise ValueError("context embeddings must have shape [N, E].")
    if labels.ndim != 1:
        raise ValueError("context labels must have shape [N].")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must have the same number of items.")


def _batch_hard_for_context(embeddings, labels, margin):
    """Compute batch-hard loss inside one local label context."""
    _validate_context_shapes(embeddings, labels)
    total_anchor_count = int(labels.numel())
    if total_anchor_count == 0:
        return embeddings.sum() * 0.0, 0, 0

    distances = torch.cdist(embeddings, embeddings, p=2)
    same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    eye = torch.eye(total_anchor_count, dtype=torch.bool, device=labels.device)
    positive_mask = same_label & ~eye
    negative_mask = ~same_label

    valid_positive = positive_mask.any(dim=1)
    valid_negative = negative_mask.any(dim=1)
    valid_anchor_mask = valid_positive & valid_negative
    valid_anchor_count = int(valid_anchor_mask.sum().item())
    if valid_anchor_count == 0:
        return embeddings.sum() * 0.0, 0, total_anchor_count

    positive_distances = distances.masked_fill(~positive_mask, float("-inf"))
    negative_distances = distances.masked_fill(~negative_mask, float("inf"))
    hardest_positive = positive_distances.max(dim=1).values
    hardest_negative = negative_distances.min(dim=1).values
    losses = F.relu(hardest_positive - hardest_negative + margin)
    return losses[valid_anchor_mask].mean(), valid_anchor_count, total_anchor_count


def batch_hard_triplet_loss(embeddings, labels, margin=0.5):
    """Compute batch-hard triplet loss.

    Inputs can be either ``embeddings=[N, E], labels=[N]`` for one local label
    context, or ``embeddings=[B, T, E], labels=[B, T]``. For batched windows,
    mining is performed independently per window so TSRD labels are never mixed
    across different local contexts.
    """
    if margin <= 0:
        raise ValueError("margin must be positive.")
    if embeddings.ndim == 2:
        if labels.ndim != 1:
            raise ValueError("labels must have shape [N] for embeddings [N, E].")
        loss, valid_count, total_count = _batch_hard_for_context(
            embeddings,
            labels,
            margin,
        )
        return BatchHardTripletResult(loss, valid_count, total_count)

    if embeddings.ndim == 3:
        if labels.ndim != 2:
            raise ValueError("labels must have shape [B, T] for embeddings [B, T, E].")
        if embeddings.shape[:2] != labels.shape:
            raise ValueError("embeddings [B, T, E] and labels [B, T] must align.")

        weighted_loss = embeddings.sum() * 0.0
        valid_anchor_count = 0
        total_anchor_count = 0
        for batch_index in range(embeddings.shape[0]):
            loss, context_valid, context_total = _batch_hard_for_context(
                embeddings[batch_index],
                labels[batch_index],
                margin,
            )
            if context_valid > 0:
                weighted_loss = weighted_loss + loss * context_valid
            valid_anchor_count += context_valid
            total_anchor_count += context_total

        if valid_anchor_count > 0:
            weighted_loss = weighted_loss / float(valid_anchor_count)
        return BatchHardTripletResult(
            weighted_loss,
            int(valid_anchor_count),
            int(total_anchor_count),
        )

    raise ValueError("embeddings must have shape [N, E] or [B, T, E].")
