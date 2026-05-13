"""Signal sparsity simulation for intercepted radar pulse streams."""

import numpy as np


SPARSITY_GAP_RATIOS = {
    "1:3": 3.0,
    "1:5": 5.0,
    "1:8": 8.0,
}


def sparsity_ratio_to_gap_ratio(sparsity_ratio):
    if sparsity_ratio == "none":
        return None
    if sparsity_ratio not in SPARSITY_GAP_RATIOS:
        raise ValueError("Unsupported sparsity_ratio: {}".format(sparsity_ratio))
    return SPARSITY_GAP_RATIOS[sparsity_ratio]


def apply_signal_sparsity(
    pdw_array,
    labels=None,
    visible_duration=5000.0,
    gap_ratio=3.0,
    phase_offset=0.0,
):
    """Keep pulses only inside periodic visible scan-gate intervals.

    This models signal sparsity from main-lobe scanning:
      visible segment -> missing interval -> visible segment -> missing interval

    It is not random pulse loss because whole continuous TOA intervals are
    removed. If labels is provided, the same keep mask is applied to labels.
    """
    if pdw_array.ndim != 2 or pdw_array.shape[1] != 5:
        raise ValueError("pdw_array must have shape [N, 5].")
    if visible_duration <= 0:
        raise ValueError("visible_duration must be positive.")
    if gap_ratio < 0:
        raise ValueError("gap_ratio must be non-negative.")
    if labels is not None and len(labels) != len(pdw_array):
        raise ValueError("labels must have the same length as pdw_array.")

    scan_period = visible_duration * (1.0 + gap_ratio)
    toas = pdw_array[:, 0]
    phase = (toas + phase_offset) % scan_period
    keep_mask = phase < visible_duration
    sparse_pdw = pdw_array[keep_mask]

    if labels is None:
        return sparse_pdw, keep_mask
    return sparse_pdw, labels[keep_mask], keep_mask
