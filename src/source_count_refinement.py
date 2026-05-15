"""Source-count-aware cluster refinement without ground-truth decisions."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.cluster_diagnostics import cluster_compactness, compute_cluster_diagnostics


@dataclass
class SourceCountRefinementReport:
    recovered_clusters: list = field(default_factory=list)
    split_clusters: list = field(default_factory=list)

    @property
    def recovered_cluster_count(self):
        return len(self.recovered_clusters)

    @property
    def recovered_point_count(self):
        return int(sum(item["size"] for item in self.recovered_clusters))

    @property
    def split_cluster_count(self):
        return len(self.split_clusters)

    @property
    def split_subcluster_count(self):
        return int(sum(item["new_subclusters"] for item in self.split_clusters))


def _standardize_pdw(pdw_window):
    pdw_window = np.asarray(pdw_window, dtype=np.float64)
    if len(pdw_window) == 0:
        return pdw_window
    return StandardScaler().fit_transform(pdw_window)


def _secondary_cluster(features, method, eps, min_samples, min_cluster_size):
    if method == "dbscan":
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features).astype(np.int64)
    if method == "hdbscan":
        try:
            import hdbscan
        except ImportError:
            print("HDBSCAN is not installed; skipping secondary HDBSCAN refinement.")
            return None
        return hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        ).fit_predict(features).astype(np.int64)
    raise ValueError(f"Unsupported secondary clustering method: {method}")


def _next_cluster_id(labels):
    non_noise = [label for label in np.unique(labels) if label != -1]
    return int(max(non_noise) + 1) if non_noise else 0


def _pdw_consistency(pdw_scaled):
    if len(pdw_scaled) <= 1:
        return 0.0
    return float(np.linalg.norm(pdw_scaled.std(axis=0)))


def _nearest_existing_distance(centroid, pdw_mean, diagnostics):
    clusters = diagnostics["clusters"]
    if not clusters:
        return np.inf, np.inf
    embedding_distances = []
    pdw_distances = []
    for cluster in clusters.values():
        embedding_distances.append(float(np.linalg.norm(centroid - cluster["embedding_centroid"])))
        pdw_distances.append(float(np.linalg.norm(pdw_mean - cluster["pdw_mean"])))
    return min(embedding_distances), min(pdw_distances)


def noise_subcluster_recovery(
    cluster_labels,
    embeddings,
    pdw_window,
    original_indices=None,
    method="dbscan",
    eps=0.2,
    min_samples=5,
    min_recovered_cluster_size=8,
    recovery_compactness_threshold=0.25,
    recovery_pdw_threshold=3.0,
):
    """Recover stable subclusters from points currently labeled as noise."""
    labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    embeddings = np.asarray(embeddings, dtype=np.float64)
    pdw_scaled = _standardize_pdw(pdw_window)
    report = SourceCountRefinementReport()

    noise_indices = np.flatnonzero(labels == -1)
    if len(noise_indices) < min_recovered_cluster_size:
        return labels, report

    secondary_labels = _secondary_cluster(
        embeddings[noise_indices],
        method=method,
        eps=eps,
        min_samples=min_samples,
        min_cluster_size=min_recovered_cluster_size,
    )
    if secondary_labels is None:
        return labels, report

    diagnostics = compute_cluster_diagnostics(embeddings, pdw_scaled, labels)
    next_id = _next_cluster_id(labels)
    for sub_label in sorted(label for label in np.unique(secondary_labels) if label != -1):
        local_indices = np.flatnonzero(secondary_labels == sub_label)
        if len(local_indices) < min_recovered_cluster_size:
            continue
        global_indices = noise_indices[local_indices]
        sub_embeddings = embeddings[global_indices]
        sub_pdw = pdw_scaled[global_indices]
        compactness = cluster_compactness(sub_embeddings)
        pdw_consistency = _pdw_consistency(sub_pdw)
        if compactness > recovery_compactness_threshold:
            continue
        if pdw_consistency > recovery_pdw_threshold:
            continue

        centroid = sub_embeddings.mean(axis=0)
        pdw_mean = sub_pdw.mean(axis=0)
        nearest_embedding, nearest_pdw = _nearest_existing_distance(
            centroid,
            pdw_mean,
            diagnostics,
        )
        # Reject obvious boundary extensions of an existing cluster.
        if nearest_embedding <= recovery_compactness_threshold and nearest_pdw <= recovery_pdw_threshold:
            continue

        labels[global_indices] = next_id
        report.recovered_clusters.append(
            {
                "new_label": int(next_id),
                "size": int(len(global_indices)),
                "compactness": float(compactness),
                "pdw_consistency": float(pdw_consistency),
                "nearest_existing_embedding_distance": float(nearest_embedding),
                "nearest_existing_pdw_distance": float(nearest_pdw),
                "original_indices": (
                    []
                    if original_indices is None
                    else [int(original_indices[index]) for index in global_indices]
                ),
            }
        )
        next_id += 1

    return labels, report


def split_dispersion_clusters(
    cluster_labels,
    embeddings,
    pdw_window,
    original_indices=None,
    method="dbscan",
    eps=0.2,
    min_samples=5,
    min_split_cluster_size=128,
    min_split_subcluster_size=32,
    split_compactness_threshold=0.3,
):
    """Conservatively split large high-dispersion clusters with secondary clustering."""
    labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    embeddings = np.asarray(embeddings, dtype=np.float64)
    report = SourceCountRefinementReport()
    next_id = _next_cluster_id(labels)

    for cluster_label in sorted(label for label in np.unique(labels) if label != -1):
        cluster_indices = np.flatnonzero(labels == cluster_label)
        if len(cluster_indices) < min_split_cluster_size:
            continue
        compactness = cluster_compactness(embeddings[cluster_indices])
        if compactness <= split_compactness_threshold:
            continue

        secondary_labels = _secondary_cluster(
            embeddings[cluster_indices],
            method=method,
            eps=eps,
            min_samples=min_samples,
            min_cluster_size=min_split_subcluster_size,
        )
        if secondary_labels is None:
            continue

        valid_subclusters = []
        for sub_label in sorted(label for label in np.unique(secondary_labels) if label != -1):
            local_indices = np.flatnonzero(secondary_labels == sub_label)
            if len(local_indices) >= min_split_subcluster_size:
                valid_subclusters.append(local_indices)
        if len(valid_subclusters) < 2:
            continue

        valid_point_count = sum(len(indices) for indices in valid_subclusters)
        if valid_point_count < 0.8 * len(cluster_indices):
            continue

        weighted_compactness = 0.0
        for local_indices in valid_subclusters:
            sub_embeddings = embeddings[cluster_indices[local_indices]]
            weighted_compactness += cluster_compactness(sub_embeddings) * len(local_indices)
        weighted_compactness /= float(valid_point_count)
        if weighted_compactness >= compactness:
            continue

        assigned_labels = []
        for subcluster_index, local_indices in enumerate(valid_subclusters):
            global_indices = cluster_indices[local_indices]
            if subcluster_index == 0:
                assigned_label = int(cluster_label)
            else:
                assigned_label = next_id
                next_id += 1
            labels[global_indices] = assigned_label
            assigned_labels.append(assigned_label)

        report.split_clusters.append(
            {
                "source_label": int(cluster_label),
                "new_labels": [int(label) for label in assigned_labels],
                "new_subclusters": int(len(valid_subclusters)),
                "original_size": int(len(cluster_indices)),
                "valid_point_count": int(valid_point_count),
                "old_compactness": float(compactness),
                "weighted_new_compactness": float(weighted_compactness),
            }
        )

    return labels, report


def combine_source_count_reports(*reports):
    combined = SourceCountRefinementReport()
    for report in reports:
        combined.recovered_clusters.extend(report.recovered_clusters)
        combined.split_clusters.extend(report.split_clusters)
    return combined
