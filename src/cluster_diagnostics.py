"""Cluster diagnostics for embedding-based pulse deinterleaving."""

import numpy as np


PDW_FEATURE_NAMES = ("DTOA", "PW", "RF", "AOA", "PA")


def _safe_std(values):
    if len(values) <= 1:
        return np.zeros(values.shape[1], dtype=np.float64)
    return values.std(axis=0)


def _compactness(points, centroid):
    if len(points) == 0:
        return 0.0
    distances = np.linalg.norm(points - centroid, axis=1)
    return float(distances.mean())


def compute_cluster_diagnostics(embeddings, pdw_features, cluster_labels):
    """Compute per-cluster diagnostics without using ground-truth labels."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    pdw_features = np.asarray(pdw_features, dtype=np.float64)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N, E].")
    if pdw_features.ndim != 2:
        raise ValueError("pdw_features must have shape [N, D].")
    if len(embeddings) != len(pdw_features) or len(embeddings) != len(cluster_labels):
        raise ValueError("embeddings, pdw_features, and cluster_labels must align.")

    non_noise_labels = sorted(label for label in np.unique(cluster_labels) if label != -1)
    diagnostics = {}
    centroids = {}

    for label in non_noise_labels:
        indices = np.flatnonzero(cluster_labels == label)
        cluster_embeddings = embeddings[indices]
        cluster_pdw = pdw_features[indices]
        centroid = cluster_embeddings.mean(axis=0)
        centroids[label] = centroid
        pdw_mean = cluster_pdw.mean(axis=0)
        pdw_std = _safe_std(cluster_pdw)
        dtoa_values = cluster_pdw[:, 0]

        diagnostics[label] = {
            "label": int(label),
            "indices": indices,
            "size": int(len(indices)),
            "embedding_centroid": centroid,
            "embedding_compactness": _compactness(cluster_embeddings, centroid),
            "nearest_cluster_distance": np.inf,
            "nearest_cluster_label": None,
            "pdw_mean": pdw_mean,
            "pdw_std": pdw_std,
            "dtoa_mean": float(dtoa_values.mean()),
            "dtoa_std": float(dtoa_values.std() if len(dtoa_values) > 1 else 0.0),
            "dtoa_min": float(dtoa_values.min()),
            "dtoa_max": float(dtoa_values.max()),
        }

        for feature_index, feature_name in enumerate(PDW_FEATURE_NAMES[: pdw_features.shape[1]]):
            diagnostics[label][f"{feature_name}_mean"] = float(pdw_mean[feature_index])
            diagnostics[label][f"{feature_name}_std"] = float(pdw_std[feature_index])

    for label, centroid in centroids.items():
        nearest_label = None
        nearest_distance = np.inf
        for other_label, other_centroid in centroids.items():
            if other_label == label:
                continue
            distance = float(np.linalg.norm(centroid - other_centroid))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_label = other_label
        diagnostics[label]["nearest_cluster_distance"] = nearest_distance
        diagnostics[label]["nearest_cluster_label"] = (
            None if nearest_label is None else int(nearest_label)
        )

    return {
        "clusters": diagnostics,
        "noise_ratio": float(np.mean(cluster_labels == -1)) if len(cluster_labels) else 0.0,
        "num_clusters": int(len(non_noise_labels)),
        "num_noise_points": int(np.sum(cluster_labels == -1)),
    }


def cluster_compactness(points):
    """Return mean distance to centroid for a candidate cluster."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return 0.0
    centroid = points.mean(axis=0)
    return _compactness(points, centroid)
