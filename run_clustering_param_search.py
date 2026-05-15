"""Multi-file clustering parameter search for raw features or TCAN embeddings."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from src.clustering_baselines import normalize_window_features, run_clustering_method
from src.clustering_metrics import average_metric_dicts, compute_clustering_metrics
from src.embedding_extractor import extract_window_embeddings
from src.evaluation_runner import build_tcan_encoder, l2_normalize_numpy
from src.result_writer import write_param_search_csv
from src.tsrd_window_dataset import load_tsrd_windows


DEFAULT_WINDOW_SIZE = 1024
DEFAULT_STRIDE = 1024
DEFAULT_MAX_FILES = 3
DEFAULT_MAX_WINDOWS_PER_FILE = 3
DEFAULT_EMBEDDING_DIM = 64

METRIC_KEYS = (
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
MEAN_KEYS = tuple(f"mean_{key}" for key in METRIC_KEYS)


def parse_float_grid(value):
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("grid must contain at least one value.")
    return parsed


def parse_int_grid(value):
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("grid must contain at least one value.")
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("integer grid values must be positive.")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search DBSCAN/HDBSCAN parameters over multiple TSRD files."
    )
    parser.add_argument("--tsrd-dir", required=True, help="Directory containing TSRD h5 files.")
    parser.add_argument("--file-glob", default="*.h5", help="Glob pattern for TSRD files.")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=DEFAULT_MAX_WINDOWS_PER_FILE,
    )
    parser.add_argument("--feature-set", choices=["4d", "5d"], default="5d")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument(
        "--representation",
        choices=["raw", "triplet_embedding"],
        default="raw",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Required when --representation triplet_embedding.",
    )
    parser.add_argument("--method", choices=["dbscan", "hdbscan"], default="dbscan")
    parser.add_argument("--eps-grid", type=parse_float_grid, default=parse_float_grid("0.5"))
    parser.add_argument(
        "--min-samples-grid",
        type=parse_int_grid,
        default=parse_int_grid("5"),
    )
    parser.add_argument(
        "--min-cluster-size-grid",
        type=parse_int_grid,
        default=parse_int_grid("20"),
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="TCAN embedding dimension for triplet_embedding.",
    )
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    if args.max_files <= 0:
        parser.error("--max-files must be positive.")
    if args.max_windows_per_file <= 0:
        parser.error("--max-windows-per-file must be positive.")
    if args.window_size <= 0:
        parser.error("--window-size must be positive.")
    if args.stride <= 0:
        parser.error("--stride must be positive.")
    if args.representation == "triplet_embedding" and not args.checkpoint:
        parser.error("--checkpoint is required when --representation triplet_embedding.")
    return args


def check_hdbscan_available():
    try:
        import hdbscan  # noqa: F401
    except ImportError:
        print(
            "HDBSCAN is not installed. Install the 'hdbscan' package to run "
            "--method hdbscan parameter search."
        )
        return False
    return True


def discover_tsrd_files(tsrd_dir, file_glob, max_files):
    root = Path(tsrd_dir)
    if not root.exists():
        raise FileNotFoundError(f"TSRD directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"TSRD path is not a directory: {root}")
    files = sorted(path for path in root.glob(file_glob) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files matched {file_glob!r} under {root}")
    return files[:max_files]


def load_windows_for_files(files, feature_set, window_size, stride, max_windows_per_file):
    windows = []
    for file_path in files:
        file_windows = load_tsrd_windows(
            tsrd_path=file_path,
            feature_set=feature_set,
            window_size=window_size,
            stride=stride,
            max_windows=max_windows_per_file,
        )
        for local_index, window in enumerate(file_windows):
            window.metadata["file_window_index"] = local_index
        windows.extend(file_windows)
        print(f"Loaded {len(file_windows)} windows from {file_path}")
    if not windows:
        raise ValueError("No windows were loaded.")
    return windows


def build_parameter_grid(method, eps_grid, min_samples_grid, min_cluster_size_grid):
    if method == "dbscan":
        return [
            {"eps": eps, "min_samples": min_samples, "min_cluster_size": None}
            for eps, min_samples in product(eps_grid, min_samples_grid)
        ]
    return [
        {"eps": None, "min_samples": min_samples, "min_cluster_size": min_cluster_size}
        for min_cluster_size, min_samples in product(min_cluster_size_grid, min_samples_grid)
    ]


def prepare_representations(windows, representation, embedding_dim, checkpoint):
    prepared = []
    if representation == "raw":
        for window in windows:
            prepared.append(
                {
                    "features": normalize_window_features(window.X_window),
                    "labels": window.y_window,
                    "true_source_count": window.metadata["true_source_count"],
                    "source_file": window.metadata["source_file"],
                }
            )
        return prepared

    input_dim = int(windows[0].X_window.shape[1])
    model, device = build_tcan_encoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        checkpoint_path=checkpoint,
    )
    for window_index, window in enumerate(windows):
        input_features = normalize_window_features(window.X_window)
        embeddings = extract_window_embeddings(model, input_features, device=device)
        features = l2_normalize_numpy(embeddings).astype(np.float32)
        prepared.append(
            {
                "features": features,
                "labels": window.y_window,
                "true_source_count": window.metadata["true_source_count"],
                "source_file": window.metadata["source_file"],
            }
        )
        print(f"Prepared triplet_embedding window={window_index} shape={features.shape}")
    return prepared


def aggregate_metrics(metric_rows):
    mean_metrics = average_metric_dicts(metric_rows)
    return {f"mean_{key}": mean_metrics[key] for key in METRIC_KEYS}


def evaluate_parameter_setting(prepared_windows, representation, method, params):
    metric_rows = []
    for item in prepared_windows:
        y_pred = run_clustering_method(
            item["features"],
            method=method,
            true_source_count=item["true_source_count"],
            eps=params["eps"] if params["eps"] is not None else 0.5,
            min_samples=params["min_samples"],
            min_cluster_size=(
                params["min_cluster_size"]
                if params["min_cluster_size"] is not None
                else 20
            ),
        )
        if y_pred is None:
            return None
        metric_rows.append(compute_clustering_metrics(item["labels"], y_pred))

    row = {
        "representation": representation,
        "method": method,
        "eps": "" if params["eps"] is None else params["eps"],
        "min_samples": params["min_samples"],
        "min_cluster_size": (
            "" if params["min_cluster_size"] is None else params["min_cluster_size"]
        ),
        "num_files": len({item["source_file"] for item in prepared_windows}),
        "num_windows": len(prepared_windows),
    }
    row.update(aggregate_metrics(metric_rows))
    return row


def best_row(rows):
    return max(
        rows,
        key=lambda row: (
            row["mean_v_measure"],
            -row["mean_abs_source_count_error"],
            -row["mean_noise_ratio"],
        ),
    )


def format_value(value):
    if value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_search_row(row):
    params = (
        f"eps={row['eps']} "
        f"min_samples={row['min_samples']} "
        f"min_cluster_size={row['min_cluster_size']}"
    )
    metrics = " | ".join(f"{key}={format_value(row[key])}" for key in MEAN_KEYS)
    print(f"{row['representation']} {row['method']} {params} | {metrics}")


def main():
    args = parse_args()
    if args.method == "hdbscan" and not check_hdbscan_available():
        return

    files = discover_tsrd_files(args.tsrd_dir, args.file_glob, args.max_files)
    windows = load_windows_for_files(
        files=files,
        feature_set=args.feature_set,
        window_size=args.window_size,
        stride=args.stride,
        max_windows_per_file=args.max_windows_per_file,
    )
    prepared_windows = prepare_representations(
        windows=windows,
        representation=args.representation,
        embedding_dim=args.embedding_dim,
        checkpoint=args.checkpoint,
    )
    parameter_grid = build_parameter_grid(
        method=args.method,
        eps_grid=args.eps_grid,
        min_samples_grid=args.min_samples_grid,
        min_cluster_size_grid=args.min_cluster_size_grid,
    )

    print(f"TSRD dir: {args.tsrd_dir}")
    print(f"Matched files: {len(files)}")
    for file_path in files:
        print(f"  {file_path}")
    print(f"Total windows: {len(windows)}")
    print(f"Representation: {args.representation}")
    print(f"Method: {args.method}")
    print(f"Parameter settings: {len(parameter_grid)}")

    rows = []
    for params in parameter_grid:
        row = evaluate_parameter_setting(
            prepared_windows=prepared_windows,
            representation=args.representation,
            method=args.method,
            params=params,
        )
        if row is None:
            print("No clustering labels were produced; stopping parameter search.")
            return
        rows.append(row)
        print_search_row(row)

    selected = best_row(rows)
    print("Best parameter setting:")
    print_search_row(selected)

    if args.output_csv:
        output_path = write_param_search_csv(args.output_csv, rows)
        print(f"Saved parameter search CSV: {output_path}")


if __name__ == "__main__":
    main()
