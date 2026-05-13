"""TSRD pulse-window dataset utilities for raw-feature clustering baselines."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.tsrd_loader import load_tsrd_pulse_train


VALID_FEATURE_SETS = ("4d", "5d")


@dataclass
class TSRDWindow:
    """One fixed-length pulse window from a TSRD pulse train."""

    X_window: np.ndarray
    y_window: np.ndarray
    metadata: dict


def validate_feature_set(feature_set):
    if feature_set not in VALID_FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {VALID_FEATURE_SETS}; got {feature_set!r}."
        )


def pdw_to_window_features(pdw_array, feature_set="5d"):
    """Build raw clustering features from internal PDWs.

    Input PDWs must be [TOA, PW, RF, AOA, PA], sorted by ascending TOA.
    feature_set='4d' returns [DTOA, PW, RF, AOA].
    feature_set='5d' returns [DTOA, PW, RF, AOA, PA].
    """
    validate_feature_set(feature_set)
    pdw_array = np.asarray(pdw_array, dtype=np.float64)
    if pdw_array.ndim != 2 or pdw_array.shape[1] != 5:
        raise ValueError("pdw_array must have shape [N, 5]: [TOA, PW, RF, AOA, PA].")
    if len(pdw_array) == 0:
        raise ValueError("pdw_array must contain at least one pulse.")

    toas = pdw_array[:, 0]
    dtoa = np.empty_like(toas, dtype=np.float64)
    dtoa[0] = 0.0
    dtoa[1:] = np.diff(toas)

    if feature_set == "4d":
        features = np.column_stack([dtoa, pdw_array[:, 1], pdw_array[:, 2], pdw_array[:, 3]])
    else:
        features = np.column_stack(
            [dtoa, pdw_array[:, 1], pdw_array[:, 2], pdw_array[:, 3], pdw_array[:, 4]]
        )
    return features.astype(np.float32)


def iter_fixed_pulse_windows(
    features,
    labels,
    source_file,
    window_size=1024,
    stride=1024,
    max_windows=10,
):
    """Yield fixed-length pulse windows with aligned labels and metadata."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError("features must have shape [N, D].")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same first dimension.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be positive when provided.")

    emitted = 0
    for start_index in range(0, len(features) - window_size + 1, stride):
        end_index = start_index + window_size
        X_window = features[start_index:end_index]
        y_window = labels[start_index:end_index]
        metadata = {
            "source_file": str(source_file),
            "start_index": start_index,
            "end_index": end_index,
            "window_size": window_size,
            "true_source_count": int(len(np.unique(y_window))),
        }
        yield TSRDWindow(
            X_window=X_window.astype(np.float32, copy=False),
            y_window=y_window.astype(np.int64, copy=False),
            metadata=metadata,
        )
        emitted += 1
        if max_windows is not None and emitted >= max_windows:
            break


def load_tsrd_windows(
    tsrd_path,
    feature_set="5d",
    window_size=1024,
    stride=1024,
    max_windows=10,
):
    """Load one TSRD file and split it into fixed-length pulse windows."""
    validate_feature_set(feature_set)
    tsrd_path = Path(tsrd_path)
    pdw_array, labels = load_tsrd_pulse_train(tsrd_path)
    sort_order = np.argsort(pdw_array[:, 0], kind="mergesort")
    pdw_array = pdw_array[sort_order]
    labels = labels[sort_order]

    features = pdw_to_window_features(pdw_array, feature_set=feature_set)
    windows = list(
        iter_fixed_pulse_windows(
            features=features,
            labels=labels,
            source_file=tsrd_path,
            window_size=window_size,
            stride=stride,
            max_windows=max_windows,
        )
    )
    if not windows:
        raise ValueError(
            f"Not enough pulses ({len(features)}) to build one window of size {window_size}."
        )

    for window in windows:
        window.metadata["feature_set"] = feature_set
        window.metadata["feature_dim"] = int(window.X_window.shape[1])
    return windows
