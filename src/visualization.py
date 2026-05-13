"""Basic visual checks for Phase 1 radar pulse simulation."""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.data_simulator import generate_emitter_streams, merge_streams_by_toa
from src.preprocessing import pdw_to_dtoa_features


DEFAULT_OUTPUT_DIR = os.path.join("outputs", "figures")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def align_streams_to_common_observation(emitter_streams):
    """Trim visualization streams to a shared observation window."""
    common_max_toa = min(stream[:, 0].max() for _, _, stream in emitter_streams)
    aligned_streams = []
    for name, title, stream in emitter_streams:
        aligned = stream[stream[:, 0] <= common_max_toa]
        if len(aligned) < 2:
            raise ValueError("Each emitter needs at least two pulses after TOA alignment.")
        aligned_streams.append((name, title, aligned))
    return aligned_streams, common_max_toa


def summarize_by_label(stream):
    """Compute per-emitter observation statistics from the merged stream."""
    summaries = []
    labels = np.unique(stream[:, 4].astype(int))
    for label in labels:
        emitter_stream = stream[stream[:, 4].astype(int) == label]
        toas = np.sort(emitter_stream[:, 0])
        intervals = np.diff(toas)
        mean_interval = float(np.mean(intervals)) if len(intervals) > 0 else float("nan")
        summaries.append(
            {
                "label": int(label),
                "pulse_count": int(len(toas)),
                "min_toa": float(toas.min()),
                "max_toa": float(toas.max()),
                "mean_same_emitter_interval": mean_interval,
            }
        )
    return summaries


def print_label_summaries(summaries):
    print("Per-label simulation summary:")
    for item in summaries:
        print(
            "  label {label}: pulse_count={pulse_count}, "
            "min TOA={min_toa:.2f}, max TOA={max_toa:.2f}, "
            "mean same-emitter interval={mean_same_emitter_interval:.2f}".format(**item)
        )


def save_single_emitter_plot(name, title, stream, output_dir):
    """Plot one emitter's TOA-DTOA relation to inspect its PRI pattern."""
    toas = stream[:, 0]
    label = int(stream[0, 4])
    dtoa = np.diff(toas)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.scatter(toas[1:], dtoa, s=24, alpha=0.85)
    ax.plot(toas[1:], dtoa, linewidth=0.8, alpha=0.55)
    ax.set_title(title + " TOA vs DTOA")
    ax.set_xlabel("TOA")
    ax.set_ylabel("DTOA / PRI")
    ax.text(
        0.01,
        0.92,
        "label " + str(label),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = os.path.join(output_dir, "emitter_" + name + ".png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_true_pri_by_emitter_plot(stream, output_dir):
    """Group by true label and plot each emitter's own adjacent interval."""
    labels = np.unique(stream[:, 4].astype(int))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    cmap = plt.get_cmap("tab10")
    for label in labels:
        emitter_stream = stream[stream[:, 4].astype(int) == label]
        toas = np.sort(emitter_stream[:, 0])
        intervals = np.diff(toas)
        ax.scatter(
            toas[1:],
            intervals,
            s=22,
            alpha=0.85,
            color=cmap(label),
            label="label " + str(label),
        )
        ax.plot(toas[1:], intervals, linewidth=0.8, alpha=0.45, color=cmap(label))

    ax.set_title("True PRI by emitter grouped by label")
    ax.set_xlabel("TOA")
    ax.set_ylabel("Same-emitter interval / true PRI")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    path = os.path.join(output_dir, "true_pri_by_emitter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_interleaved_label_vs_toa_plot(stream, output_dir):
    """Plot true labels over TOA to inspect the interleaved pulse stream."""
    toas = stream[:, 0]
    labels = stream[:, 4].astype(int)

    fig, ax = plt.subplots(figsize=(10, 4))
    scatter = ax.scatter(toas, labels, c=labels, cmap="tab10", s=20, alpha=0.85)
    ax.set_title("Interleaved pulse stream label vs TOA")
    ax.set_xlabel("TOA")
    ax.set_ylabel("True label")
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(True, axis="x", alpha=0.25)
    fig.colorbar(scatter, ax=ax, ticks=[0, 1, 2, 3], label="label")
    fig.tight_layout()

    path = os.path.join(output_dir, "interleaved_label_vs_toa.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_mixed_stream_adjacent_dtoa_plot(stream, output_dir):
    """Plot mixed-stream TOA against adjacent pulse interval after sorting."""
    toas = stream[:, 0]
    labels = stream[:, 4].astype(int)
    dtoa = np.diff(toas)

    fig, ax = plt.subplots(figsize=(10, 4))
    scatter = ax.scatter(toas[1:], dtoa, c=labels[1:], cmap="tab10", s=22, alpha=0.85)
    ax.set_title("Mixed-stream adjacent DTOA after TOA sorting")
    ax.set_xlabel("TOA")
    ax.set_ylabel("Adjacent interval / mixed-stream DTOA")
    ax.text(
        0.01,
        0.94,
        "Model DTOA input, not same-emitter true PRI",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
    )
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, ticks=[0, 1, 2, 3], label="current pulse label")
    fig.tight_layout()

    path = os.path.join(output_dir, "mixed_stream_adjacent_dtoa.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_dtoa_sequence_plot(stream, output_dir):
    """Plot DTOA by pulse index to inspect adjacent pulse spacing after mixing."""
    features, _ = pdw_to_dtoa_features(stream)
    dtoa = features[:, 0]
    pulse_index = np.arange(len(dtoa))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pulse_index, dtoa, linewidth=1.0)
    ax.set_title("DTOA sequence of interleaved pulse stream")
    ax.set_xlabel("Pulse index")
    ax.set_ylabel("DTOA")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = os.path.join(output_dir, "dtoa_sequence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_dtoa_histogram(stream, output_dir):
    """Plot DTOA distribution to inspect PRI-derived spacing statistics."""
    features, _ = pdw_to_dtoa_features(stream)
    dtoa = features[1:, 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dtoa, bins=35, color="#4C78A8", edgecolor="white")
    ax.set_title("DTOA histogram of interleaved pulse stream")
    ax.set_xlabel("DTOA")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    path = os.path.join(output_dir, "dtoa_histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def visualize_simulation(
    pulses_per_emitter=80,
    seed=42,
    output_dir=DEFAULT_OUTPUT_DIR,
    align_common_observation=True,
):
    """Generate short simulated streams and save basic diagnostic figures."""
    ensure_dir(output_dir)

    emitter_streams = generate_emitter_streams(
        pulses_per_emitter=pulses_per_emitter,
        seed=seed,
    )
    if align_common_observation:
        emitter_streams, common_max_toa = align_streams_to_common_observation(emitter_streams)
        print("Common visualization max TOA: {:.2f}".format(common_max_toa))

    merged_stream = merge_streams_by_toa([stream for _, _, stream in emitter_streams])
    print_label_summaries(summarize_by_label(merged_stream))

    saved_paths = []
    for name, title, stream in emitter_streams:
        saved_paths.append(save_single_emitter_plot(name, title, stream, output_dir))

    saved_paths.append(save_true_pri_by_emitter_plot(merged_stream, output_dir))
    saved_paths.append(save_interleaved_label_vs_toa_plot(merged_stream, output_dir))
    saved_paths.append(save_mixed_stream_adjacent_dtoa_plot(merged_stream, output_dir))
    saved_paths.append(save_dtoa_sequence_plot(merged_stream, output_dir))
    saved_paths.append(save_dtoa_histogram(merged_stream, output_dir))
    return [os.path.abspath(path) for path in saved_paths]


def main():
    saved_paths = visualize_simulation()
    print("Saved simulation figures:")
    for path in saved_paths:
        print("  " + path)


if __name__ == "__main__":
    main()
