"""Compare raw features and TCAN embeddings under one clustering setup."""

import argparse

from src.evaluation_runner import (
    EvaluationConfig,
    METRIC_ORDER,
    SUMMARY_METRICS,
    evaluate_methods,
    parse_method_list,
)
from src.result_writer import write_evaluation_csv
from src.tsrd_window_dataset import load_tsrd_windows


DEFAULT_WINDOW_SIZE = 1024
DEFAULT_STRIDE = 1024
DEFAULT_MAX_WINDOWS = 10
DEFAULT_EMBEDDING_DIM = 64


def parse_args():
    parser = argparse.ArgumentParser(
        description="Systematically compare raw and embedding clustering methods."
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
        help="Input feature set used for raw and encoder inputs.",
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
        "--cluster-method",
        choices=["dbscan", "hdbscan", "agglomerative_oracle"],
        default="dbscan",
        help="Clustering method applied to every evaluated representation.",
    )
    parser.add_argument(
        "--methods",
        default="raw,random_embedding",
        help="Comma-separated methods: raw,random_embedding,triplet_embedding.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="TCAN pulse embedding dimension.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Triplet-trained TCAN checkpoint required by triplet_embedding.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="DBSCAN eps.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples, also passed to HDBSCAN when selected.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=20,
        help="HDBSCAN min_cluster_size.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV path for per-window and mean evaluation rows.",
    )
    args = parser.parse_args()
    args.methods = parse_method_list(args.methods)
    if "triplet_embedding" in args.methods and not args.checkpoint:
        parser.error("--checkpoint is required when --methods includes triplet_embedding.")
    return args


def format_metric_value(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def print_window_row(row):
    metric_text = " | ".join(
        f"{key}={format_metric_value(row[key])}" for key in METRIC_ORDER
    )
    print(
        f"method={row['method']} window={row['window_index']} "
        f"start={row['start_index']} end={row['end_index']} | {metric_text}"
    )


def print_summary_row(row):
    metric_text = " | ".join(
        f"mean_{key}={format_metric_value(row[key])}" for key in SUMMARY_METRICS
    )
    print(f"method={row['method']} mean | {metric_text}")


def main():
    args = parse_args()
    windows = load_tsrd_windows(
        tsrd_path=args.tsrd_path,
        feature_set=args.feature_set,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
    )
    config = EvaluationConfig(
        cluster_method=args.cluster_method,
        embedding_dim=args.embedding_dim,
        checkpoint=args.checkpoint,
        eps=args.eps,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
    )

    print(f"TSRD path: {args.tsrd_path}")
    print(f"Feature set: {args.feature_set}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Loaded windows: {len(windows)}")
    print(f"Cluster method: {args.cluster_method}")
    print(f"Evaluation methods: {args.methods}")

    window_rows, summary_rows = evaluate_methods(windows, args.methods, config)

    print("Per-window metrics:")
    for row in window_rows:
        print_window_row(row)

    print("Mean metrics by method:")
    for row in summary_rows:
        print_summary_row(row)

    if args.output_csv:
        output_path = write_evaluation_csv(args.output_csv, window_rows, summary_rows)
        print(f"Saved evaluation CSV: {output_path}")


if __name__ == "__main__":
    main()
