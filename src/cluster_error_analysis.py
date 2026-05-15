"""Ground-truth-based clustering error diagnostics for analysis only."""

from collections import Counter, defaultdict

import numpy as np

from src.clustering_metrics import estimate_source_count, compute_noise_ratio


def _counter_to_plain_dict(counter):
    return {int(key): int(value) for key, value in sorted(counter.items())}


def build_true_to_cluster_table(y_true, cluster_labels):
    table = defaultdict(Counter)
    for true_label, cluster_label in zip(y_true, cluster_labels):
        table[int(true_label)][int(cluster_label)] += 1
    return {int(key): _counter_to_plain_dict(value) for key, value in sorted(table.items())}


def build_cluster_to_true_table(y_true, cluster_labels):
    table = defaultdict(Counter)
    for true_label, cluster_label in zip(y_true, cluster_labels):
        table[int(cluster_label)][int(true_label)] += 1
    return {int(key): _counter_to_plain_dict(value) for key, value in sorted(table.items())}


def summarize_major_error_type(y_true, cluster_labels):
    y_true = np.asarray(y_true, dtype=np.int64)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    true_count = int(len(np.unique(y_true)))
    estimated_count = int(estimate_source_count(cluster_labels))
    noise_ratio = compute_noise_ratio(cluster_labels)
    cluster_to_true = build_cluster_to_true_table(y_true, cluster_labels)
    true_to_cluster = build_true_to_cluster_table(y_true, cluster_labels)

    non_noise_compositions = {
        cluster: composition
        for cluster, composition in cluster_to_true.items()
        if cluster != -1
    }
    mixed_clusters = sum(
        1 for composition in non_noise_compositions.values() if len(composition) > 1
    )
    split_true_labels = 0
    for composition in true_to_cluster.values():
        non_noise_hits = [cluster for cluster, count in composition.items() if cluster != -1 and count > 0]
        if len(non_noise_hits) > 1:
            split_true_labels += 1

    missing_as_noise = noise_ratio >= 0.2 and estimated_count < true_count
    over_merged = estimated_count < true_count and mixed_clusters > 0
    over_split = estimated_count > true_count or split_true_labels > 0

    if not missing_as_noise and not over_merged and not over_split:
        return "clean"
    if sum([missing_as_noise, over_merged, over_split]) > 1:
        return "mixed"
    if missing_as_noise:
        return "missing_as_noise"
    if over_merged:
        return "over_merged"
    return "over_split"


def analyze_cluster_errors(y_true, cluster_labels, embeddings=None, pdw_window=None, original_indices=None):
    """Analyze errors with true labels.

    This function is for diagnosis and reporting only. Refinement decisions must
    not use its output.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    if y_true.ndim != 1 or cluster_labels.ndim != 1:
        raise ValueError("y_true and cluster_labels must be one-dimensional.")
    if len(y_true) != len(cluster_labels):
        raise ValueError("y_true and cluster_labels must have the same length.")

    noise_mask = cluster_labels == -1
    noise_distribution = Counter(y_true[noise_mask].tolist())
    return {
        "true_source_count": int(len(np.unique(y_true))),
        "estimated_source_count": int(estimate_source_count(cluster_labels)),
        "noise_ratio": compute_noise_ratio(cluster_labels),
        "true_to_cluster": build_true_to_cluster_table(y_true, cluster_labels),
        "cluster_to_true": build_cluster_to_true_table(y_true, cluster_labels),
        "noise_true_label_distribution": _counter_to_plain_dict(noise_distribution),
        "major_error_type": summarize_major_error_type(y_true, cluster_labels),
        "original_index_range": (
            None
            if original_indices is None or len(original_indices) == 0
            else (int(np.min(original_indices)), int(np.max(original_indices)))
        ),
    }
