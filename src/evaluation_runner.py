"""Unified evaluation runner for raw and TCAN-embedding clustering methods."""

from dataclasses import dataclass

import numpy as np
import torch

from src.clustering_baselines import normalize_window_features, run_clustering_method
from src.clustering_metrics import average_metric_dicts, compute_clustering_metrics
from src.embedding_extractor import extract_window_embeddings, resolve_device
from src.model_tcan import TCAN


EVALUATION_METHODS = ("raw", "random_embedding", "triplet_embedding")
METRIC_ORDER = (
    "homogeneity",
    "completeness",
    "v_measure",
    "adjusted_rand_index",
    "adjusted_mutual_info",
    "true_source_count",
    "estimated_source_count",
    "source_count_error",
    "abs_source_count_error",
    "noise_ratio",
)
SUMMARY_METRICS = (
    "homogeneity",
    "completeness",
    "v_measure",
    "adjusted_rand_index",
    "adjusted_mutual_info",
    "abs_source_count_error",
    "noise_ratio",
)


@dataclass
class EvaluationConfig:
    cluster_method: str
    embedding_dim: int = 64
    checkpoint: str = None
    eps: float = 0.5
    min_samples: int = 5
    min_cluster_size: int = 20


def parse_method_list(methods):
    parsed = [method.strip() for method in methods.split(",") if method.strip()]
    if not parsed:
        raise ValueError("At least one evaluation method must be provided.")
    invalid = [method for method in parsed if method not in EVALUATION_METHODS]
    if invalid:
        raise ValueError(
            f"Unsupported evaluation method(s): {invalid}. "
            f"Supported methods: {EVALUATION_METHODS}."
        )
    return parsed


def l2_normalize_numpy(features, axis=-1, eps=1e-12):
    norms = np.linalg.norm(features, ord=2, axis=axis, keepdims=True)
    return features / np.maximum(norms, eps)


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_compatible_checkpoint(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = _state_dict_from_checkpoint(checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint '{checkpoint_path}' does not contain a state dict.")

    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        clean_key = key.removeprefix("module.")
        if (
            torch.is_tensor(value)
            and clean_key in model_state
            and model_state[clean_key].shape == value.shape
        ):
            compatible[clean_key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded {len(compatible)} compatible checkpoint tensors from {checkpoint_path}. "
        f"Skipped {len(skipped)} incompatible tensors."
    )
    if missing:
        print(f"Missing model tensors after partial load: {len(missing)}")
    if unexpected:
        print(f"Unexpected checkpoint tensors after partial load: {len(unexpected)}")


def build_tcan_encoder(input_dim, embedding_dim, checkpoint_path=None, device=None):
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")
    device = resolve_device(device)
    model = TCAN(
        input_dim=input_dim,
        num_classes=1,
        embedding_dim=embedding_dim,
    ).to(device)
    if checkpoint_path:
        load_compatible_checkpoint(model, checkpoint_path, device)
    return model, device


def _cluster_features(features, true_source_count, config):
    return run_clustering_method(
        features,
        method=config.cluster_method,
        true_source_count=true_source_count,
        eps=config.eps,
        min_samples=config.min_samples,
        min_cluster_size=config.min_cluster_size,
    )


def _window_result(method, window_index, window, y_pred):
    metrics = compute_clustering_metrics(window.y_window, y_pred)
    return {
        "row_type": "window",
        "method": method,
        "window_index": window_index,
        "start_index": window.metadata["start_index"],
        "end_index": window.metadata["end_index"],
        **metrics,
    }


def _summary_result(method, method_rows):
    metrics = average_metric_dicts(
        [{key: row[key] for key in METRIC_ORDER} for row in method_rows]
    )
    return {
        "row_type": "mean",
        "method": method,
        "window_index": "",
        "start_index": "",
        "end_index": "",
        **metrics,
    }


def evaluate_raw(windows, config):
    rows = []
    for window_index, window in enumerate(windows):
        features = normalize_window_features(window.X_window)
        y_pred = _cluster_features(
            features,
            true_source_count=window.metadata["true_source_count"],
            config=config,
        )
        if y_pred is None:
            return rows, None
        rows.append(_window_result("raw", window_index, window, y_pred))
    return rows, _summary_result("raw", rows)


def evaluate_embedding_method(windows, method, config):
    if method == "random_embedding":
        print(
            "Warning: random_embedding uses an untrained TCAN encoder. "
            "Metrics are for pipeline sanity check only."
        )
        checkpoint_path = None
    elif method == "triplet_embedding":
        if not config.checkpoint:
            raise ValueError(
                "triplet_embedding requires --checkpoint pointing to a trained "
                "triplet TCAN encoder checkpoint."
            )
        checkpoint_path = config.checkpoint
    else:
        raise ValueError(f"Unsupported embedding method: {method}")

    input_dim = int(windows[0].X_window.shape[1])
    model, device = build_tcan_encoder(
        input_dim=input_dim,
        embedding_dim=config.embedding_dim,
        checkpoint_path=checkpoint_path,
    )

    rows = []
    for window_index, window in enumerate(windows):
        input_features = normalize_window_features(window.X_window)
        embeddings = extract_window_embeddings(model, input_features, device=device)
        clustering_features = l2_normalize_numpy(embeddings).astype(np.float32)
        print(f"{method} window={window_index} embeddings shape: {embeddings.shape}")
        y_pred = _cluster_features(
            clustering_features,
            true_source_count=window.metadata["true_source_count"],
            config=config,
        )
        if y_pred is None:
            return rows, None
        rows.append(_window_result(method, window_index, window, y_pred))
    return rows, _summary_result(method, rows)


def evaluate_method(windows, method, config):
    if method == "raw":
        return evaluate_raw(windows, config)
    if method in {"random_embedding", "triplet_embedding"}:
        return evaluate_embedding_method(windows, method, config)
    raise ValueError(f"Unsupported evaluation method: {method}")


def evaluate_methods(windows, methods, config):
    window_rows = []
    summary_rows = []
    for method in methods:
        method_window_rows, method_summary = evaluate_method(windows, method, config)
        window_rows.extend(method_window_rows)
        if method_summary is not None:
            summary_rows.append(method_summary)
    return window_rows, summary_rows
