"""Nonideal receiving condition simulation at the PDW level."""

import numpy as np


def _labels_from_pdw(pdw_array, labels):
    if labels is None:
        return pdw_array[:, 4].astype(np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != len(pdw_array):
        raise ValueError("labels must have the same length as pdw_array.")
    return labels


def _with_synced_labels(pdw_array, labels):
    synced = pdw_array.copy()
    synced[:, 4] = labels.astype(np.float64)
    return synced


def apply_measurement_error(
    pdw_array,
    labels=None,
    toa_std=0.0,
    pw_std=0.0,
    rf_std=0.0,
    doa_std=0.0,
    seed=None,
):
    """Add Gaussian measurement error to TOA, PW, RF, and DOA.

    PW is clipped to stay positive. If TOA is perturbed, pulses are sorted by
    TOA again and labels are reordered with the same permutation.
    """
    if pdw_array.ndim != 2 or pdw_array.shape[1] != 5:
        raise ValueError("pdw_array must have shape [N, 5].")
    labels = _labels_from_pdw(pdw_array, labels)
    rng = np.random.RandomState(seed)

    noisy = _with_synced_labels(pdw_array, labels)
    if toa_std > 0:
        noisy[:, 0] += rng.normal(0.0, toa_std, len(noisy))
    if pw_std > 0:
        noisy[:, 1] += rng.normal(0.0, pw_std, len(noisy))
        noisy[:, 1] = np.clip(noisy[:, 1], 1e-6, None)
    if rf_std > 0:
        noisy[:, 2] += rng.normal(0.0, rf_std, len(noisy))
    if doa_std > 0:
        noisy[:, 3] += rng.normal(0.0, doa_std, len(noisy))

    sort_order = np.argsort(noisy[:, 0], kind="mergesort")
    noisy = noisy[sort_order]
    sorted_labels = labels[sort_order]
    noisy[:, 4] = sorted_labels.astype(np.float64)
    return noisy, sorted_labels


def apply_random_pulse_loss(
    pdw_array,
    labels=None,
    loss_rate=0.0,
    seed=None,
):
    """Randomly delete individual pulses with probability loss_rate."""
    if pdw_array.ndim != 2 or pdw_array.shape[1] != 5:
        raise ValueError("pdw_array must have shape [N, 5].")
    if loss_rate < 0.0 or loss_rate >= 1.0:
        raise ValueError("loss_rate must be in [0, 1).")
    labels = _labels_from_pdw(pdw_array, labels)
    if loss_rate == 0.0:
        return _with_synced_labels(pdw_array, labels), labels

    rng = np.random.RandomState(seed)
    keep_mask = rng.rand(len(pdw_array)) >= loss_rate
    if not np.any(keep_mask):
        raise ValueError("pulse loss removed all pulses; lower loss_rate.")
    kept_labels = labels[keep_mask]
    kept_pdw = _with_synced_labels(pdw_array[keep_mask], kept_labels)
    return kept_pdw, kept_labels


def apply_spurious_pulses(
    pdw_array,
    labels=None,
    spurious_rate=0.0,
    spurious_label=4,
    seed=None,
):
    """Insert false pulses sampled from the current observation ranges."""
    if pdw_array.ndim != 2 or pdw_array.shape[1] != 5:
        raise ValueError("pdw_array must have shape [N, 5].")
    if spurious_rate < 0.0:
        raise ValueError("spurious_rate must be non-negative.")
    labels = _labels_from_pdw(pdw_array, labels)
    base_pdw = _with_synced_labels(pdw_array, labels)
    if spurious_rate == 0.0:
        return base_pdw, labels

    count = int(round(len(base_pdw) * spurious_rate))
    if count == 0:
        return base_pdw, labels

    rng = np.random.RandomState(seed)
    mins = base_pdw[:, :4].min(axis=0)
    maxs = base_pdw[:, :4].max(axis=0)
    spurious_toa = rng.uniform(mins[0], maxs[0], count)
    spurious_pw = rng.uniform(mins[1], maxs[1], count)
    spurious_rf = rng.uniform(mins[2], maxs[2], count)
    spurious_doa = rng.uniform(mins[3], maxs[3], count)
    spurious_labels = np.full(count, spurious_label, dtype=np.int64)
    spurious = np.column_stack(
        [spurious_toa, spurious_pw, spurious_rf, spurious_doa, spurious_labels]
    ).astype(np.float64)

    combined = np.vstack([base_pdw, spurious])
    combined_labels = np.concatenate([labels, spurious_labels])
    sort_order = np.argsort(combined[:, 0], kind="mergesort")
    combined = combined[sort_order]
    sorted_labels = combined_labels[sort_order]
    combined[:, 4] = sorted_labels.astype(np.float64)
    return combined, sorted_labels
