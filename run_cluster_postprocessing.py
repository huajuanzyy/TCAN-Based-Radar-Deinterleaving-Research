"""Run conservative cluster post-processing on TCAN embedding clusters."""

import argparse
import csv
from pathlib import Path

import numpy as np

from src.cluster_refinement import (
    combine_reports,
    merge_close_clusters,
    reassign_boundary_points,
    split_dispersion_clusters,
)
from src.clustering_baselines import normalize_window_features, run_clustering_method
from src.clustering_metrics import average_metric_dicts, compute_clustering_metrics
from src.embedding_extractor import extract_window_embeddings
from src.evaluation_runner import build_tcan_encoder, l2_normalize_numpy
from src.tsrd_window_dataset import load_tsrd_windows


DEFAULT_WINDOW_SIZE = 1024
DEFAULT_STRIDE = 1024
DEFAULT_MAX_FILES = 1
DEFAULT_MAX_WINDOWS_PER_FILE = 3
DEFAULT_EMBEDDING_DIM = 64

METRIC_ORDER = (
    "homogeneity",
    "completeness",
    "v_measure",
    "adjusted_rand_index",
    "adjusted_mutual_info",
    "true_source_count",
    "estimated_source_count",
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run conservative cluster post-processing on TSRD windows."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tsrd-path", default=None)
    source.add_argument("--tsrd-dir", default=None)
    parser.add_argument("--file-glob", default="config_*.h5")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=DEFAULT_MAX_WINDOWS_PER_FILE,
    )
    parser.add_argument("--feature-set", choices=["4d", "5d"], default="5d")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--cluster-method", choices=["dbscan", "hdbscan"], default="dbscan")
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--enable-reassign", action="store_true")
    parser.add_argument("--enable-merge", action="store_true")
    parser.add_argument("--enable-split", action="store_true")
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
    return args


def discover_files(args):
    if args.tsrd_path:
        path = Path(args.tsrd_path)
        if not path.exists():
            raise FileNotFoundError(f"TSRD file does not exist: {path}")
        return [path]

    root = Path(args.tsrd_dir)
    if not root.exists():
        raise FileNotFoundError(f"TSRD directory does not exist: {root}")
    files = sorted(path for path in root.glob(args.file_glob) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files matched {args.file_glob!r} under {root}")
    return files[: args.max_files]


def load_all_windows(files, args):
    windows = []
    for file_path in files:
        file_windows = load_tsrd_windows(
            tsrd_path=file_path,
            feature_set=args.feature_set,
            window_size=args.window_size,
            stride=args.stride,
            max_windows=args.max_windows_per_file,
        )
        for local_index, window in enumerate(file_windows):
            window.metadata["file_window_index"] = local_index
        windows.extend(file_windows)
        print(f"Loaded {len(file_windows)} windows from {file_path}")
    return windows


def format_metric(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def print_metrics(prefix, metrics):
    text = " | ".join(f"{key}={format_metric(metrics[key])}" for key in METRIC_ORDER)
    print(f"{prefix} | {text}")


def metric_row(file_path, window_index, phase, metrics, report):
    return {
        "source_file": str(file_path),
        "window_index": window_index,
        "phase": phase,
        **metrics,
        "reassigned_count": len(report.reassigned_points) if report else "",
        "merged_count": len(report.merged_clusters) if report else "",
        "split_count": len(report.split_clusters) if report else "",
    }


def write_csv(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "source_file",
        "window_index",
        "phase",
        *METRIC_ORDER,
        "reassigned_count",
        "merged_count",
        "split_count",
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return output_path


def run_initial_clustering(embeddings, args, true_source_count):
    return run_clustering_method(
        embeddings,
        method=args.cluster_method,
        true_source_count=true_source_count,
        eps=args.eps,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
    )


def apply_postprocessing(labels, embeddings, pdw_features, args):
    reports = []
    refined = labels.copy()
    if args.enable_reassign:
        refined, report = reassign_boundary_points(
            refined,
            embeddings=embeddings,
            pdw_features=pdw_features,
        )
        reports.append(report)
    if args.enable_merge:
        refined, report = merge_close_clusters(
            refined,
            embeddings=embeddings,
            pdw_features=pdw_features,
        )
        reports.append(report)
    if args.enable_split:
        refined, report = split_dispersion_clusters(
            refined,
            embeddings=embeddings,
            pdw_features=pdw_features,
            enabled=True,
        )
        reports.append(report)
    return refined, combine_reports(*reports)


def print_report(report):
    print(
        "Post-processing actions: "
        f"reassigned={len(report.reassigned_points)} | "
        f"merged={len(report.merged_clusters)} | "
        f"split={len(report.split_clusters)}"
    )
    for item in report.reassigned_points[:10]:
        print(
            "  reassigned point "
            f"{item['point_index']} -> cluster {item['new_label']} "
            f"(embed_dist={item['embedding_distance']:.4f}, "
            f"pdw_dist={item['pdw_distance']:.4f})"
        )
    if len(report.reassigned_points) > 10:
        print(f"  ... {len(report.reassigned_points) - 10} more reassigned points")
    for item in report.merged_clusters:
        print(
            "  merged cluster "
            f"{item['source_label']} -> {item['target_label']} "
            f"(embed_dist={item['embedding_centroid_distance']:.4f}, "
            f"pdw_dist={item['pdw_distribution_distance']:.4f})"
        )
    for item in report.split_clusters:
        print(f"  split action: {item}")


def main():
    args = parse_args()
    files = discover_files(args)
    windows = load_all_windows(files, args)
    if not windows:
        raise ValueError("No windows loaded.")

    input_dim = int(windows[0].X_window.shape[1])
    model, device = build_tcan_encoder(
        input_dim=input_dim,
        embedding_dim=args.embedding_dim,
        checkpoint_path=args.checkpoint,
    )

    print(f"Files: {len(files)}")
    print(f"Total windows: {len(windows)}")
    print(f"Feature set: {args.feature_set}")
    print(f"Cluster method: {args.cluster_method}")
    print(f"Post-processing: reassign={args.enable_reassign}, merge={args.enable_merge}, split={args.enable_split}")

    before_metrics = []
    after_metrics = []
    csv_rows = []

    for window_index, window in enumerate(windows):
        input_features = normalize_window_features(window.X_window)
        embeddings = extract_window_embeddings(model, input_features, device=device)
        embeddings = l2_normalize_numpy(embeddings).astype(np.float32)

        initial_labels = run_initial_clustering(
            embeddings,
            args,
            true_source_count=window.metadata["true_source_count"],
        )
        if initial_labels is None:
            print("Initial clustering did not produce labels; exiting.")
            return

        before = compute_clustering_metrics(window.y_window, initial_labels)
        refined_labels, report = apply_postprocessing(
            labels=initial_labels,
            embeddings=embeddings,
            pdw_features=window.X_window,
            args=args,
        )
        after = compute_clustering_metrics(window.y_window, refined_labels)
        before_metrics.append(before)
        after_metrics.append(after)

        source_file = window.metadata["source_file"]
        print(f"window={window_index} file={source_file} start={window.metadata['start_index']} end={window.metadata['end_index']}")
        print_metrics("  before", before)
        print_metrics("  after ", after)
        print_report(report)

        csv_rows.append(metric_row(source_file, window_index, "before", before, None))
        csv_rows.append(metric_row(source_file, window_index, "after", after, report))

    mean_before = average_metric_dicts(before_metrics)
    mean_after = average_metric_dicts(after_metrics)
    print("Mean before/after metrics:")
    print_metrics("mean before", mean_before)
    print_metrics("mean after ", mean_after)

    if args.output_csv:
        output_path = write_csv(args.output_csv, csv_rows)
        print(f"Saved post-processing CSV: {output_path}")


if __name__ == "__main__":
    main()
