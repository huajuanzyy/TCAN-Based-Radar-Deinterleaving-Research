"""Source-count-aware refinement for triplet-embedding clustering results."""

import argparse
import csv
from pathlib import Path

import numpy as np

from src.cluster_error_analysis import analyze_cluster_errors
from src.clustering_baselines import normalize_window_features, run_clustering_method
from src.clustering_metrics import average_metric_dicts, compute_clustering_metrics
from src.embedding_extractor import extract_window_embeddings
from src.evaluation_runner import build_tcan_encoder, l2_normalize_numpy
from src.source_count_refinement import (
    combine_source_count_reports,
    noise_subcluster_recovery,
    split_dispersion_clusters,
)
from src.tsrd_window_dataset import load_tsrd_windows


DEFAULT_WINDOW_SIZE = 1024
DEFAULT_STRIDE = 1024
DEFAULT_MAX_FILES = 1
DEFAULT_MAX_WINDOWS_PER_FILE = 3
DEFAULT_EMBEDDING_DIM = 64

PRINT_METRICS = (
    "homogeneity",
    "completeness",
    "v_measure",
    "adjusted_rand_index",
    "adjusted_mutual_info",
    "estimated_source_count",
    "abs_source_count_error",
    "noise_ratio",
)
CSV_COLUMNS = (
    "source_file",
    "window_index",
    "phase",
    "major_error_type",
    *PRINT_METRICS,
    "recovered_cluster_count",
    "recovered_point_count",
    "split_cluster_count",
    "split_subcluster_count",
    "remaining_noise_count",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run source-count-aware clustering refinement on TSRD windows."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tsrd-path", default=None)
    source.add_argument("--tsrd-dir", default=None)
    parser.add_argument("--file-glob", default="config_*.h5")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--max-windows-per-file", type=int, default=DEFAULT_MAX_WINDOWS_PER_FILE)
    parser.add_argument("--feature-set", choices=["4d", "5d"], default="5d")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--cluster-method", choices=["dbscan", "hdbscan"], default="dbscan")
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--enable-noise-recovery", action="store_true")
    parser.add_argument("--enable-split", action="store_true")
    parser.add_argument("--noise-recovery-method", choices=["dbscan", "hdbscan"], default="dbscan")
    parser.add_argument("--noise-eps", type=float, default=0.2)
    parser.add_argument("--noise-min-samples", type=int, default=3)
    parser.add_argument("--min-recovered-cluster-size", type=int, default=8)
    parser.add_argument("--recovery-compactness-threshold", type=float, default=0.25)
    parser.add_argument("--recovery-pdw-threshold", type=float, default=3.0)
    parser.add_argument("--split-method", choices=["dbscan", "hdbscan"], default="dbscan")
    parser.add_argument("--split-eps", type=float, default=0.2)
    parser.add_argument("--split-min-samples", type=int, default=3)
    parser.add_argument("--min-split-cluster-size", type=int, default=128)
    parser.add_argument("--min-split-subcluster-size", type=int, default=32)
    parser.add_argument("--split-compactness-threshold", type=float, default=0.3)
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


def run_initial_clustering(embeddings, args, true_source_count):
    return run_clustering_method(
        embeddings,
        method=args.cluster_method,
        true_source_count=true_source_count,
        eps=args.eps,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
    )


def refine_labels(labels, embeddings, pdw_window, original_indices, args):
    refined = labels.copy()
    reports = []
    if args.enable_noise_recovery:
        refined, report = noise_subcluster_recovery(
            refined,
            embeddings=embeddings,
            pdw_window=pdw_window,
            original_indices=original_indices,
            method=args.noise_recovery_method,
            eps=args.noise_eps,
            min_samples=args.noise_min_samples,
            min_recovered_cluster_size=args.min_recovered_cluster_size,
            recovery_compactness_threshold=args.recovery_compactness_threshold,
            recovery_pdw_threshold=args.recovery_pdw_threshold,
        )
        reports.append(report)
    if args.enable_split:
        refined, report = split_dispersion_clusters(
            refined,
            embeddings=embeddings,
            pdw_window=pdw_window,
            original_indices=original_indices,
            method=args.split_method,
            eps=args.split_eps,
            min_samples=args.split_min_samples,
            min_split_cluster_size=args.min_split_cluster_size,
            min_split_subcluster_size=args.min_split_subcluster_size,
            split_compactness_threshold=args.split_compactness_threshold,
        )
        reports.append(report)
    return refined, combine_source_count_reports(*reports)


def format_metric(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def print_metrics(prefix, metrics):
    text = " | ".join(f"{key}={format_metric(metrics[key])}" for key in PRINT_METRICS)
    print(f"{prefix} | {text}")


def row_for_csv(source_file, window_index, phase, metrics, major_error_type, report, remaining_noise):
    return {
        "source_file": str(source_file),
        "window_index": window_index,
        "phase": phase,
        "major_error_type": major_error_type,
        **{key: metrics[key] for key in PRINT_METRICS},
        "recovered_cluster_count": "" if report is None else report.recovered_cluster_count,
        "recovered_point_count": "" if report is None else report.recovered_point_count,
        "split_cluster_count": "" if report is None else report.split_cluster_count,
        "split_subcluster_count": "" if report is None else report.split_subcluster_count,
        "remaining_noise_count": remaining_noise,
    }


def write_csv(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    return output_path


def print_report(report, remaining_noise_count):
    print(
        "Refinement actions: "
        f"recovered_clusters={report.recovered_cluster_count} | "
        f"recovered_points={report.recovered_point_count} | "
        f"split_clusters={report.split_cluster_count} | "
        f"split_subclusters={report.split_subcluster_count} | "
        f"remaining_noise={remaining_noise_count}"
    )
    for item in report.recovered_clusters:
        print(
            "  recovered noise cluster "
            f"{item['new_label']} size={item['size']} "
            f"compactness={item['compactness']:.4f} "
            f"pdw_consistency={item['pdw_consistency']:.4f}"
        )
    for item in report.split_clusters:
        print(
            "  split cluster "
            f"{item['source_label']} -> {item['new_labels']} "
            f"old_compactness={item['old_compactness']:.4f} "
            f"new_compactness={item['weighted_new_compactness']:.4f}"
        )


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

    before_rows = []
    after_rows = []
    csv_rows = []
    print(f"Files: {len(files)} | windows: {len(windows)}")
    print(f"Initial clustering: {args.cluster_method}")
    print(f"Noise recovery: {args.enable_noise_recovery} | Split: {args.enable_split}")

    for window_index, window in enumerate(windows):
        input_features = normalize_window_features(window.X_window)
        embeddings = extract_window_embeddings(model, input_features, device=device)
        embeddings = l2_normalize_numpy(embeddings).astype(np.float32)
        original_indices = np.arange(window.metadata["start_index"], window.metadata["end_index"])

        initial_labels = run_initial_clustering(
            embeddings,
            args,
            true_source_count=window.metadata["true_source_count"],
        )
        if initial_labels is None:
            print("Initial clustering did not produce labels; exiting.")
            return

        before = compute_clustering_metrics(window.y_window, initial_labels)
        diagnosis = analyze_cluster_errors(
            y_true=window.y_window,
            cluster_labels=initial_labels,
            embeddings=embeddings,
            pdw_window=window.X_window,
            original_indices=original_indices,
        )
        refined_labels, report = refine_labels(
            initial_labels,
            embeddings=embeddings,
            pdw_window=window.X_window,
            original_indices=original_indices,
            args=args,
        )
        after = compute_clustering_metrics(window.y_window, refined_labels)
        remaining_noise_count = int(np.sum(refined_labels == -1))

        before_rows.append(before)
        after_rows.append(after)
        source_file = window.metadata["source_file"]
        print(f"window={window_index} file={source_file} start={window.metadata['start_index']} end={window.metadata['end_index']}")
        print(f"  major_error_type={diagnosis['major_error_type']}")
        print(f"  noise_true_label_distribution={diagnosis['noise_true_label_distribution']}")
        print_metrics("  before", before)
        print_metrics("  after ", after)
        print_report(report, remaining_noise_count)

        csv_rows.append(
            row_for_csv(source_file, window_index, "before", before, diagnosis["major_error_type"], None, int(np.sum(initial_labels == -1)))
        )
        csv_rows.append(
            row_for_csv(source_file, window_index, "after", after, diagnosis["major_error_type"], report, remaining_noise_count)
        )

    mean_before = average_metric_dicts(before_rows)
    mean_after = average_metric_dicts(after_rows)
    print("Mean before/after metrics:")
    print_metrics("mean before", mean_before)
    print_metrics("mean after ", mean_after)

    if args.output_csv:
        output_path = write_csv(args.output_csv, csv_rows)
        print(f"Saved source-count refinement CSV: {output_path}")


if __name__ == "__main__":
    main()
