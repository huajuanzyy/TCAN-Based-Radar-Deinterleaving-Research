"""Loss functions for TCAN sequence labeling."""

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss for sequence labeling.

    Accepts logits shaped [B, T, C] or [N, C] and labels shaped [B, T] or [N].
    The loss is computed per sequence position:
      -alpha_c * (1 - p_c)^gamma * log(p_c)
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", alpha.float())

    def forward(self, logits, labels):
        if logits.dim() == 3:
            num_classes = logits.size(-1)
            logits = logits.reshape(-1, num_classes)
            labels = labels.reshape(-1)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        labels = labels.long()

        true_log_probs = log_probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
        true_probs = probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - true_probs).pow(self.gamma)
        loss = -focal_factor * true_log_probs

        if self.alpha is not None:
            loss = loss * self.alpha[labels]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_inverse_frequency_alpha(labels, num_classes):
    """Compute class weights from flattened labels using inverse frequency."""
    flat_labels = labels.reshape(-1)
    counts = torch.bincount(flat_labels.long(), minlength=num_classes).float()
    total = counts.sum()
    alpha = torch.zeros(num_classes, dtype=torch.float32)
    nonzero = counts > 0
    alpha[nonzero] = total / (num_classes * counts[nonzero])
    return alpha


def build_loss(loss_name, gamma, alpha_mode, labels, num_classes, device):
    """Build CE or focal loss and return optional alpha weights."""
    if loss_name == "ce":
        return nn.CrossEntropyLoss(), None

    alpha = None
    if alpha_mode == "inverse_freq":
        alpha = compute_inverse_frequency_alpha(
            torch.from_numpy(labels),
            num_classes,
        ).to(device)

    return FocalLoss(gamma=gamma, alpha=alpha), alpha
