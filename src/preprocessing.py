"""DTOA conversion, normalization, and fixed-length window creation."""

import numpy as np


class MinMaxStats:
    def __init__(self, min_values, max_values):
        self.min_values = min_values
        self.max_values = max_values


def pdw_to_dtoa_features(pdw_stream):
    """Convert sorted PDWs to feature matrix [DTOA, PW, RF, DOA] and labels."""
    if pdw_stream.ndim != 2 or pdw_stream.shape[1] != 5:
        raise ValueError("pdw_stream must have shape [N, 5].")

    toas = pdw_stream[:, 0]
    dtoa = np.empty_like(toas, dtype=np.float64)
    dtoa[0] = 0.0
    dtoa[1:] = np.diff(toas)

    features = np.column_stack([dtoa, pdw_stream[:, 1], pdw_stream[:, 2], pdw_stream[:, 3]])
    labels = pdw_stream[:, 4].astype(np.int64)
    return features.astype(np.float32), labels


def fit_minmax(features):
    return MinMaxStats(
        min_values=features.min(axis=0),
        max_values=features.max(axis=0),
    )


def apply_minmax(features, stats):
    return ((features - stats.min_values) / (stats.max_values - stats.min_values + 1e-8)).astype(
        np.float32
    )


def create_binary_features(
    pdw_stream,
    ts,
    stats=None,
    start_toa=None,
    use_multiparameter=True,
):
    """Discretize TOA into binary time bins for sequence labeling.

    Feature order is [binary_presence, PW, RF, DOA]. Empty bins are all-zero
    except for the background label. If multiple pulses fall into one bin, this
    Phase 2 implementation keeps the earliest TOA pulse and ignores later ones.
    Labels are continuous for CrossEntropyLoss:
      background = 0
      radar labels 0,1,2,3 -> class labels 1,2,3,4
    """
    if pdw_stream.ndim != 2 or pdw_stream.shape[1] != 5:
        raise ValueError("pdw_stream must have shape [N, 5].")
    if ts <= 0:
        raise ValueError("ts must be positive.")
    if not use_multiparameter:
        raise ValueError("Only multiparameter binary input is supported in this branch.")

    sorted_stream = pdw_stream[np.argsort(pdw_stream[:, 0], kind="mergesort")]
    toas = sorted_stream[:, 0]
    start_toa = float(toas.min()) if start_toa is None else float(start_toa)
    bin_indices = np.floor((toas - start_toa) / ts).astype(np.int64)
    if np.any(bin_indices < 0):
        raise ValueError("pdw_stream contains TOA values earlier than start_toa.")

    num_bins = int(bin_indices.max()) + 1
    features = np.zeros((num_bins, 4), dtype=np.float32)
    labels = np.zeros(num_bins, dtype=np.int64)
    occupied = np.zeros(num_bins, dtype=bool)

    pdw_features = sorted_stream[:, 1:4].astype(np.float32)
    normalized_pdw = apply_minmax(pdw_features, stats) if stats is not None else pdw_features

    for pulse_index, bin_index in enumerate(bin_indices):
        if occupied[bin_index]:
            # Keep the earliest TOA pulse when collisions occur in the same bin.
            continue
        occupied[bin_index] = True
        features[bin_index, 0] = 1.0
        features[bin_index, 1:] = normalized_pdw[pulse_index]
        labels[bin_index] = int(sorted_stream[pulse_index, 4]) + 1

    return features, labels


def create_fixed_length_windows(
    features,
    labels,
    sequence_length,
    stride=None,
):
    """Create aligned sequence windows.

    Returns X with shape [B, T, D] and y with shape [B, T]. Incomplete tail
    windows are dropped to keep every training position supervised.
    """
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same first dimension.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    stride = sequence_length if stride is None else stride
    if stride <= 0:
        raise ValueError("stride must be positive.")

    feature_windows = []
    label_windows = []
    for start in range(0, len(features) - sequence_length + 1, stride):
        end = start + sequence_length
        feature_windows.append(features[start:end])
        label_windows.append(labels[start:end])

    if not feature_windows:
        raise ValueError("Not enough pulses to build one full sequence window.")

    return np.stack(feature_windows).astype(np.float32), np.stack(label_windows).astype(np.int64)
