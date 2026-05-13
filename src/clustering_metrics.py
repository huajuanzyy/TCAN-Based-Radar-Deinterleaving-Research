"""Clustering metrics for unknown-emitter pulse deinterleaving baselines."""

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
)


def estimate_source_count(cluster_labels):
    """Estimate emitter count from cluster IDs, excluding DBSCAN/HDBSCAN noise -1."""
    cluster_labels = np.asarray(cluster_labels)
    unique_clusters = set(cluster_labels.tolist())
    unique_clusters.discard(-1)
    return len(unique_clusters)


def compute_noise_ratio(cluster_labels):
    cluster_labels = np.asarray(cluster_labels)
    if len(cluster_labels) == 0:
        return 0.0
    return float(np.mean(cluster_labels == -1))


def compute_clustering_metrics(y_true, y_pred):
    """Compute clustering quality and source-count metrics for one window."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot compute clustering metrics for an empty window.")

    true_source_count = int(len(np.unique(y_true)))
    estimated_source_count = int(estimate_source_count(y_pred))
    source_count_error = estimated_source_count - true_source_count

    return {
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "completeness": float(completeness_score(y_true, y_pred)),
        "v_measure": float(v_measure_score(y_true, y_pred)),
        "adjusted_rand_index": float(adjusted_rand_score(y_true, y_pred)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(y_true, y_pred)),
        "true_source_count": true_source_count,
        "estimated_source_count": estimated_source_count,
        "source_count_error": int(source_count_error),
        "abs_source_count_error": int(abs(source_count_error)),
        "noise_ratio": compute_noise_ratio(y_pred),
    }


def average_metric_dicts(metric_dicts):
    if not metric_dicts:
        raise ValueError("metric_dicts must contain at least one item.")
    keys = metric_dicts[0].keys()
    return {
        key: float(np.mean([metrics[key] for metrics in metric_dicts]))
        for key in keys
    }
