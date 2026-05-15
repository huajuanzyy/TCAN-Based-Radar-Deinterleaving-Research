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
