"""Run TCAN-embedding clustering scaffold on fixed TSRD pulse windows."""

import argparse

import numpy as np
import torch

from src.clustering_baselines import normalize_window_features, run_clustering_method
from src.clustering_metrics import average_metric_dicts, compute_clustering_metrics
from src.embedding_extractor import extract_window_embeddings, resolve_device
from src.model_tcan import TCAN
from src.tsrd_window_dataset import load_tsrd_windows


DEFAULT_WINDOW_SIZE = 1024
DEFAULT_STRIDE = 1024
DEFAULT_MAX_WINDOWS = 10
DEFAULT_EMBEDDING_DIM = 64


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster TCAN pulse embeddings extracted from TSRD windows."
    )
    parser.add_argument(
        "--tsrd-path",
        required=True,
        help="Path to a local TSRD h5/hdf5 file containing one pulse train.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["4d", "5d"],
        default="5d",
        help="Input feature set used before TCAN embedding extraction.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Number of pulses per window.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Pulse stride between consecutive windows.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=DEFAULT_MAX_WINDOWS,
        help="Maximum number of windows to evaluate.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="TCAN pulse embedding dimension.",
    )
    parser.add_argument(
        "--method",
        choices=["dbscan", "hdbscan", "agglomerative_oracle"],
        default="dbscan",
        help="Clustering method applied to TCAN embeddings.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="DBSCAN eps on normalized embeddings.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples, also passed to HDBSCAN when method=hdbscan.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=20,
        help="HDBSCAN min_cluster_size.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional TCAN checkpoint path. Random initialization is used if omitted.",
    )
    return parser.parse_args()


def format_metric_value(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def print_metrics(prefix, metrics):
    metric_text = " | ".join(
        f"{key}={format_metric_value(metrics[key])}" for key in METRIC_ORDER
    )
    print(f"{prefix} | {metric_text}")


def l2_normalize_numpy(features, axis=-1, eps=1e-12):
    norms = np.linalg.norm(features, ord=2, axis=axis, keepdims=True)
    return features / np.maximum(norms, eps)


def state_dict_from_checkpoint(checkpoint):
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
    state_dict = state_dict_from_checkpoint(checkpoint)
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
    else:
        print(
            "Warning: using randomly initialized TCAN encoder. "
            "Metrics are for pipeline testing only."
        )
    return model, device


def main():
    args = parse_args()
    windows = load_tsrd_windows(
        tsrd_path=args.tsrd_path,
        feature_set=args.feature_set,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
    )
    input_dim = int(windows[0].X_window.shape[1])
    model, device = build_tcan_encoder(
        input_dim=input_dim,
        embedding_dim=args.embedding_dim,
        checkpoint_path=args.checkpoint,
    )

    print(f"TSRD path: {args.tsrd_path}")
    print(f"Method: {args.method}")
    print(f"Feature set: {args.feature_set}")
    print(f"Input dim: {input_dim}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Device: {device}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Loaded windows: {len(windows)}")

    metric_rows = []
    for window_index, window in enumerate(windows):
        X_scaled = normalize_window_features(window.X_window)
        embeddings = extract_window_embeddings(model, X_scaled, device=device)
        embedding_features = l2_normalize_numpy(embeddings).astype(np.float32)
        print(f"window={window_index} embeddings shape: {embeddings.shape}")

        y_pred = run_clustering_method(
            embedding_features,
            method=args.method,
            true_source_count=window.metadata["true_source_count"],
            eps=args.eps,
            min_samples=args.min_samples,
            min_cluster_size=args.min_cluster_size,
        )
        if y_pred is None:
            print("No clustering labels were produced; exiting without metrics.")
            return

        metrics = compute_clustering_metrics(window.y_window, y_pred)
        metric_rows.append(metrics)
        prefix = (
            f"window={window_index} "
            f"start={window.metadata['start_index']} "
            f"end={window.metadata['end_index']}"
        )
        print_metrics(prefix, metrics)

    mean_metrics = average_metric_dicts(metric_rows)
    print_metrics("mean", mean_metrics)


if __name__ == "__main__":
    main()
