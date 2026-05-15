"""Conservative cluster post-processing for embedding deinterleaving."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.cluster_diagnostics import cluster_compactness, compute_cluster_diagnostics


@dataclass
class RefinementReport:
    reassigned_points: list = field(default_factory=list)
    merged_clusters: list = field(default_factory=list)
    split_clusters: list = field(default_factory=list)


def _standardize_pdw(pdw_features):
    pdw_features = np.asarray(pdw_features, dtype=np.float64)
    if len(pdw_features) == 0:
        return pdw_features
    return StandardScaler().fit_transform(pdw_features)


def _cluster_labels(labels):
    return sorted(label for label in np.unique(labels) if label != -1)


def _candidate_compactness_after_add(embeddings, labels, cluster_label, point_index):
    cluster_indices = np.flatnonzero(labels == cluster_label)
    candidate_indices = np.concatenate([cluster_indices, np.asarray([point_index])])
    return cluster_compactness(embeddings[candidate_indices])


def reassign_boundary_points(
    cluster_labels,
    embeddings,
    pdw_features,
    max_embedding_distance=0.35,
    min_distance_margin=0.08,
    max_pdw_distance=2.5,
    max_compactness_degradation=0.2,
):
    """Conservatively reassign noise points to nearby confident clusters.

    Ground-truth labels are not used. This first implementation only considers
    points currently labeled -1.
    """
    labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    embeddings = np.asarray(embeddings, dtype=np.float64)
    pdw_scaled = _standardize_pdw(pdw_features)
    report = RefinementReport()

    diagnostics = compute_cluster_diagnostics(embeddings, pdw_scaled, labels)
    clusters = diagnostics["clusters"]
    if len(clusters) < 1:
        return labels, report

    cluster_ids = _cluster_labels(labels)
    centroids = np.stack([clusters[label]["embedding_centroid"] for label in cluster_ids])
    pdw_means = np.stack([clusters[label]["pdw_mean"] for label in cluster_ids])
    noise_indices = np.flatnonzero(labels == -1)

    for point_index in noise_indices:
        embedding_distances = np.linalg.norm(centroids - embeddings[point_index], axis=1)
        order = np.argsort(embedding_distances)
        nearest_position = int(order[0])
        nearest_label = cluster_ids[nearest_position]
        nearest_distance = float(embedding_distances[nearest_position])
        second_distance = (
            float(embedding_distances[int(order[1])]) if len(order) > 1 else np.inf
        )
        if nearest_distance > max_embedding_distance:
            continue
        if second_distance - nearest_distance < min_distance_margin:
            continue

        pdw_distance = float(
            np.linalg.norm(pdw_means[nearest_position] - pdw_scaled[point_index])
        )
        if pdw_distance > max_pdw_distance:
            continue

        old_compactness = clusters[nearest_label]["embedding_compactness"]
        new_compactness = _candidate_compactness_after_add(
            embeddings,
            labels,
            nearest_label,
            point_index,
        )
        allowed_compactness = old_compactness * (1.0 + max_compactness_degradation) + 1e-8
        if new_compactness > allowed_compactness:
            continue

        labels[point_index] = nearest_label
        report.reassigned_points.append(
            {
                "point_index": int(point_index),
                "new_label": int(nearest_label),
                "embedding_distance": nearest_distance,
                "second_distance": second_distance,
                "pdw_distance": pdw_distance,
                "old_compactness": float(old_compactness),
                "new_compactness": float(new_compactness),
            }
        )

    return labels, report


def _merged_compactness(embeddings, labels, label_a, label_b):
    indices = np.flatnonzero((labels == label_a) | (labels == label_b))
    return cluster_compactness(embeddings[indices])


def merge_close_clusters(
    cluster_labels,
    embeddings,
    pdw_features,
    max_centroid_distance=0.25,
    max_pdw_distribution_distance=1.2,
    max_compactness_degradation=0.15,
):
    """Conservatively merge clusters with similar embedding and PDW distributions."""
    labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    embeddings = np.asarray(embeddings, dtype=np.float64)
    pdw_scaled = _standardize_pdw(pdw_features)
    report = RefinementReport()

    changed = True
    while changed:
        changed = False
        diagnostics = compute_cluster_diagnostics(embeddings, pdw_scaled, labels)
        clusters = diagnostics["clusters"]
        cluster_ids = _cluster_labels(labels)
        best_pair = None
        best_distance = np.inf

        for index_a, label_a in enumerate(cluster_ids):
            for label_b in cluster_ids[index_a + 1 :]:
                centroid_distance = float(
                    np.linalg.norm(
                        clusters[label_a]["embedding_centroid"]
                        - clusters[label_b]["embedding_centroid"]
                    )
                )
                if centroid_distance > max_centroid_distance:
                    continue

                pdw_distance = float(
                    np.linalg.norm(clusters[label_a]["pdw_mean"] - clusters[label_b]["pdw_mean"])
                )
                if pdw_distance > max_pdw_distribution_distance:
                    continue

                merged_compactness = _merged_compactness(
                    embeddings,
                    labels,
                    label_a,
                    label_b,
                )
                reference_compactness = max(
                    clusters[label_a]["embedding_compactness"],
                    clusters[label_b]["embedding_compactness"],
                )
                allowed = reference_compactness * (1.0 + max_compactness_degradation) + 1e-8
                if merged_compactness > allowed:
                    continue

                if centroid_distance < best_distance:
                    best_distance = centroid_distance
                    best_pair = (
                        label_a,
                        label_b,
                        centroid_distance,
                        pdw_distance,
                        merged_compactness,
                        reference_compactness,
                    )

        if best_pair is None:
            continue

        label_a, label_b, centroid_distance, pdw_distance, merged_compactness, old_compactness = (
            best_pair
        )
        target_label = min(label_a, label_b)
        source_label = max(label_a, label_b)
        labels[labels == source_label] = target_label
        report.merged_clusters.append(
            {
                "target_label": int(target_label),
                "source_label": int(source_label),
                "embedding_centroid_distance": float(centroid_distance),
                "pdw_distribution_distance": float(pdw_distance),
                "old_compactness": float(old_compactness),
                "merged_compactness": float(merged_compactness),
            }
        )
        changed = True

    return labels, report


def split_dispersion_clusters(
    cluster_labels,
    embeddings,
    pdw_features,
    enabled=False,
):
    """Optional split scaffold.

    The conservative default is a no-op. A future branch can add internal
    DBSCAN/HDBSCAN checks and stability criteria here.
    """
    labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    report = RefinementReport()
    if enabled:
        report.split_clusters.append(
            {
                "status": "skipped",
                "reason": "split scaffold only; conservative no-op in Phase 3B",
            }
        )
    return labels, report


def combine_reports(*reports):
    combined = RefinementReport()
    for report in reports:
        combined.reassigned_points.extend(report.reassigned_points)
        combined.merged_clusters.extend(report.merged_clusters)
        combined.split_clusters.extend(report.split_clusters)
    return combined
