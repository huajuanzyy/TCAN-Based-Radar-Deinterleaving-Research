"""CSV output helpers for embedding evaluation results."""

import csv
from pathlib import Path

from src.evaluation_runner import METRIC_ORDER


CSV_COLUMNS = (
    "row_type",
    "method",
    "window_index",
    "start_index",
    "end_index",
    *METRIC_ORDER,
)

PARAM_SEARCH_COLUMNS = (
    "representation",
    "method",
    "eps",
    "min_samples",
    "min_cluster_size",
    "num_files",
    "num_windows",
    "mean_homogeneity",
    "mean_completeness",
    "mean_v_measure",
    "mean_adjusted_rand_index",
    "mean_adjusted_mutual_info",
    "mean_true_source_count",
    "mean_estimated_source_count",
    "mean_source_count_error",
    "mean_abs_source_count_error",
    "mean_noise_ratio",
)


def write_evaluation_csv(output_csv, window_rows, summary_rows):
    """Write per-window rows and per-method mean rows to one CSV file."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [*window_rows, *summary_rows]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    return output_path


def write_param_search_csv(output_csv, rows):
    """Write one aggregate row per clustering parameter setting."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PARAM_SEARCH_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in PARAM_SEARCH_COLUMNS})
    return output_path
