"""Raw-feature clustering baselines for TSRD pulse windows."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler


def normalize_window_features(X_window):
    """Normalize one window independently before clustering."""
    X_window = np.asarray(X_window, dtype=np.float32)
    if X_window.ndim != 2:
        raise ValueError("X_window must have shape [window_size, feature_dim].")
    return StandardScaler().fit_transform(X_window).astype(np.float32)


def cluster_dbscan(X_window, eps=0.5, min_samples=5):
    if eps <= 0:
        raise ValueError("eps must be positive for DBSCAN.")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive for DBSCAN.")
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_window).astype(np.int64)


def cluster_agglomerative_oracle(X_window, true_source_count):
    if true_source_count <= 0:
        raise ValueError("true_source_count must be positive for agglomerative clustering.")
    if true_source_count > len(X_window):
        raise ValueError("true_source_count cannot exceed the number of samples.")
    return AgglomerativeClustering(n_clusters=int(true_source_count)).fit_predict(
        X_window
    ).astype(np.int64)


def cluster_hdbscan(X_window, min_cluster_size=20, min_samples=None):
    if min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be positive for HDBSCAN.")
    if min_samples is not None and min_samples <= 0:
        raise ValueError("min_samples must be positive for HDBSCAN when provided.")

    try:
        import hdbscan
    except ImportError:
        print(
            "HDBSCAN is not installed. Install the 'hdbscan' package to use "
            "--method hdbscan; skipping this run."
        )
        return None

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    return clusterer.fit_predict(X_window).astype(np.int64)


def run_clustering_method(
    X_window,
    method,
    true_source_count=None,
    eps=0.5,
    min_samples=5,
    min_cluster_size=20,
):
    """Dispatch one normalized window to the selected clustering baseline."""
    if method == "dbscan":
        return cluster_dbscan(X_window, eps=eps, min_samples=min_samples)
    if method == "agglomerative_oracle":
        if true_source_count is None:
            raise ValueError("true_source_count is required for agglomerative_oracle.")
        return cluster_agglomerative_oracle(X_window, true_source_count=true_source_count)
    if method == "hdbscan":
        return cluster_hdbscan(
            X_window,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
    raise ValueError(f"Unsupported clustering method: {method}")
