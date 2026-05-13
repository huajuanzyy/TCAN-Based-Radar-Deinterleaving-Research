"""TCAN pulse-level embedding extraction helpers."""

import numpy as np
import torch


def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_window_embeddings(model, X_window, device=None):
    """Extract TCAN embeddings for one pulse window.

    Parameters
    ----------
    model:
        A TCAN model exposing ``encode`` or ``forward(..., return_embeddings=True)``.
    X_window:
        Raw or normalized window features with shape [T, D].
    device:
        Optional torch device string/object.

    Returns
    -------
    np.ndarray
        Pulse embeddings with shape [T, E].
    """
    X_window = np.asarray(X_window, dtype=np.float32)
    if X_window.ndim != 2:
        raise ValueError("X_window must have shape [T, D].")

    device = resolve_device(device)
    model = model.to(device)
    model.eval()

    features = torch.from_numpy(X_window).unsqueeze(0).to(device)
    if hasattr(model, "encode"):
        embeddings = model.encode(features)
    else:
        embeddings = model(features, return_embeddings=True)
    return embeddings.squeeze(0).detach().cpu().numpy().astype(np.float32)
