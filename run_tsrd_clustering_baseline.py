"""Run raw-feature clustering baselines on fixed TSRD pulse windows."""

import argparse

from src.clustering_baselines import normalize_window_features, run_clustering_method
from src.clustering_metrics import average_metric_dicts, compute_clustering_metrics
from src.tsrd_window_dataset import load_tsrd_windows


DEFAULT_WINDOW_SIZE = 1024
DEFAULT_STRIDE = 1024
DEFAULT_MAX_WINDOWS = 10


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
        description="Run TSRD raw-feature clustering baselines on pulse windows."
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
        help="Feature set for clustering windows.",
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
        "--method",
        choices=["dbscan", "hdbscan", "agglomerative_oracle"],
        default="dbscan",
        help="Clustering baseline method.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="DBSCAN eps on normalized window features.",
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


def main():
    args = parse_args()
    windows = load_tsrd_windows(
        tsrd_path=args.tsrd_path,
        feature_set=args.feature_set,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
    )

    print(f"TSRD path: {args.tsrd_path}")
    print(f"Method: {args.method}")
    print(f"Feature set: {args.feature_set}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Loaded windows: {len(windows)}")

    metric_rows = []
    for window_index, window in enumerate(windows):
        X_scaled = normalize_window_features(window.X_window)
        y_pred = run_clustering_method(
            X_scaled,
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
