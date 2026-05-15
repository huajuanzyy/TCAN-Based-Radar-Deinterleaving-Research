"""Compare raw features and TCAN embeddings under one clustering setup."""

import argparse
from pathlib import Path

from src.evaluation_runner import (
    EvaluationConfig,
    METRIC_ORDER,
    SUMMARY_METRICS,
    evaluate_methods,
    parse_method_list,
    summarize_file_metric_rows,
)
from src.result_writer import write_evaluation_csv
from src.tsrd_window_dataset import load_tsrd_windows


DEFAULT_WINDOW_SIZE = 1024
DEFAULT_STRIDE = 1024
DEFAULT_MAX_WINDOWS = 10
DEFAULT_MAX_WINDOWS_PER_FILE = 10
DEFAULT_EMBEDDING_DIM = 64


def parse_args():
    parser = argparse.ArgumentParser(
        description="Systematically compare raw and embedding clustering methods."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--tsrd-path",
        help="Path to a local TSRD h5/hdf5 file containing one pulse train.",
    )
    source.add_argument(
        "--file-list",
        help="Text file containing TSRD h5/hdf5 paths, one per line.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional root used to resolve relative entries in --file-list.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to load from --file-list.",
    )
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=DEFAULT_MAX_WINDOWS_PER_FILE,
        help="Maximum number of windows loaded from each file in --file-list mode.",
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
    if args.max_files is not None and args.max_files <= 0:
        parser.error("--max-files must be positive when provided.")
    if args.max_windows_per_file <= 0:
        parser.error("--max-windows-per-file must be positive.")
    return args


def _read_file_list(file_list, data_root=None, max_files=None):
    file_list_path = Path(file_list)
    if not file_list_path.exists():
        raise FileNotFoundError(f"File list does not exist: {file_list_path}")

    root = None if data_root is None else Path(data_root)
    paths = []
    for line in file_list_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        path = Path(entry)
        if not path.is_absolute():
            path = (root / path) if root is not None else path
        paths.append(path)
        if max_files is not None and len(paths) >= max_files:
            break

    if not paths:
        raise ValueError(f"No TSRD files found in file list: {file_list_path}")
    return paths


def discover_eval_files(args):
    if args.tsrd_path:
        return [Path(args.tsrd_path)]
    return _read_file_list(
        args.file_list,
        data_root=args.data_root,
        max_files=args.max_files,
    )


def load_evaluation_windows(files, args):
    windows = []
    max_windows = args.max_windows if args.tsrd_path else args.max_windows_per_file
    for file_path in files:
        file_windows = load_tsrd_windows(
            tsrd_path=file_path,
            feature_set=args.feature_set,
            window_size=args.window_size,
            stride=args.stride,
            max_windows=max_windows,
        )
        for local_index, window in enumerate(file_windows):
            window.metadata["file_window_index"] = local_index
        windows.extend(file_windows)
        print(f"Loaded {len(file_windows)} windows from {file_path}")
    return windows


def format_metric_value(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def print_window_row(row):
    metric_text = " | ".join(
        f"{key}={format_metric_value(row[key])}" for key in METRIC_ORDER
    )
    print(
        f"method={row['method']} file={row.get('source_file', '')} "
        f"window={row['window_index']} file_window={row.get('file_window_index', '')} "
        f"start={row['start_index']} end={row['end_index']} | {metric_text}"
    )


def print_summary_row(row):
    metric_text = " | ".join(
        f"mean_{key}={format_metric_value(row[key])}" for key in SUMMARY_METRICS
    )
    print(f"method={row['method']} mean | {metric_text}")


def print_file_summary_row(row):
    metric_text = " | ".join(
        f"mean_{key}={format_metric_value(row[key])}" for key in SUMMARY_METRICS
    )
    print(f"method={row['method']} file={row['source_file']} mean | {metric_text}")


def main():
    args = parse_args()
    eval_files = discover_eval_files(args)
    windows = load_evaluation_windows(eval_files, args)
    config = EvaluationConfig(
        cluster_method=args.cluster_method,
        embedding_dim=args.embedding_dim,
        checkpoint=args.checkpoint,
        eps=args.eps,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
    )

    print(f"Evaluation files: {len(eval_files)}")
    for file_path in eval_files:
        print(f"  eval_file={file_path}")
    print(f"Feature set: {args.feature_set}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Loaded windows: {len(windows)}")
    if args.file_list:
        print(f"Max windows per file: {args.max_windows_per_file}")
    print(f"Cluster method: {args.cluster_method}")
    print(f"Evaluation methods: {args.methods}")

    window_rows, summary_rows = evaluate_methods(windows, args.methods, config)
    file_summary_rows = summarize_file_metric_rows(window_rows)

    print("Per-window metrics:")
    for row in window_rows:
        print_window_row(row)

    print("Per-file mean metrics:")
    for row in file_summary_rows:
        print_file_summary_row(row)

    print("Overall mean metrics by method:")
    for row in summary_rows:
        print_summary_row(row)

    if args.output_csv:
        output_path = write_evaluation_csv(
            args.output_csv,
            window_rows,
            summary_rows,
            file_summary_rows=file_summary_rows,
        )
        print(f"Saved evaluation CSV: {output_path}")


if __name__ == "__main__":
    main()
