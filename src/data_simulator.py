"""Synthetic PDW stream generation for the Phase 1 DTOA pipeline."""

import numpy as np


class EmitterConfig:
    def __init__(self, label, pw, rf, doa, start_toa):
        self.label = label
        self.pw = pw
        self.rf = rf
        self.doa = doa
        self.start_toa = start_toa


PDW_COLUMNS = ("TOA", "PW", "RF", "DOA", "label")


def get_default_emitter_configs():
    """Return simplified conventional emitter settings for Phase 1."""
    return [
        EmitterConfig(label=0, pw=0.50, rf=9200.0, doa=-32.0, start_toa=0.0),
        EmitterConfig(label=1, pw=0.70, rf=9400.0, doa=-8.0, start_toa=120.0),
        EmitterConfig(label=2, pw=0.90, rf=9650.0, doa=18.0, start_toa=240.0),
        EmitterConfig(label=3, pw=1.10, rf=9900.0, doa=43.0, start_toa=360.0),
    ]


def _make_pdws(
    toas,
    config,
    rng,
    pw_std=0.04,
    rf_std=3.0,
    doa_std=0.4,
):
    """Build a PDW array with light measurement noise."""
    count = len(toas)
    pw = np.clip(rng.normal(config.pw, pw_std, count), 0.05, None)
    rf = rng.normal(config.rf, rf_std, count)
    doa = rng.normal(config.doa, doa_std, count)
    label = np.full(count, config.label, dtype=np.float64)
    return np.column_stack([toas, pw, rf, doa, label]).astype(np.float64)


def _toas_from_pris(start_toa, pris):
    """Convert PRI intervals to absolute TOA values."""
    return start_toa + np.cumsum(pris)


def generate_fixed_pri(
    count, config, rng, pri
):
    pris = np.full(count, pri, dtype=np.float64)
    return _make_pdws(_toas_from_pris(config.start_toa, pris), config, rng)


def generate_jitter_pri(
    count,
    config,
    rng,
    base_pri,
    jitter_fraction,
):
    jitter = rng.uniform(-jitter_fraction, jitter_fraction, count)
    pris = base_pri * (1.0 + jitter)
    return _make_pdws(_toas_from_pris(config.start_toa, pris), config, rng)


def generate_stagger_pri(
    count,
    config,
    rng,
    pri_pattern,
):
    pattern = np.asarray(pri_pattern, dtype=np.float64)
    pris = np.resize(pattern, count)
    return _make_pdws(_toas_from_pris(config.start_toa, pris), config, rng)


def generate_dwell_switch_pri(
    count,
    config,
    rng,
    pri_values,
    dwell_size,
):
    values = np.asarray(pri_values, dtype=np.float64)
    dwell_indices = (np.arange(count) // dwell_size) % len(values)
    pris = values[dwell_indices]
    return _make_pdws(_toas_from_pris(config.start_toa, pris), config, rng)


def generate_emitter_streams(
    pulses_per_emitter=900,
    seed=None,
):
    """Generate individual streams for the four conventional emitter types."""
    rng = np.random.RandomState(seed)
    configs = get_default_emitter_configs()

    return [
        (
            "fixed_pri",
            "Fixed PRI",
            generate_fixed_pri(pulses_per_emitter, configs[0], rng, pri=1000.0),
        ),
        (
            "jitter_pri",
            "Jitter PRI",
            generate_jitter_pri(
                pulses_per_emitter,
                configs[1],
                rng,
                base_pri=850.0,
                jitter_fraction=0.12,
            ),
        ),
        (
            "stagger_pri",
            "Stagger PRI",
            generate_stagger_pri(
                pulses_per_emitter,
                configs[2],
                rng,
                pri_pattern=(720.0, 930.0, 1110.0, 880.0),
            ),
        ),
        (
            "dwell_switch_pri",
            "Dwell-and-Switch PRI",
            generate_dwell_switch_pri(
                pulses_per_emitter,
                configs[3],
                rng,
                pri_values=(650.0, 780.0, 980.0),
                dwell_size=12,
            ),
        ),
    ]


def merge_streams_by_toa(streams):
    """Merge individual PDW streams and sort them by ascending TOA."""
    merged = np.vstack(streams)
    sort_order = np.argsort(merged[:, 0], kind="mergesort")
    return merged[sort_order]


def generate_interleaved_stream(
    pulses_per_emitter=900,
    seed=None,
):
    """Generate four conventional emitters and merge them by ascending TOA.

    Columns are TOA, PW, RF, DOA, label. Parameters are simplified for the
    minimal runnable reproduction and should not be treated as exact paper
    settings.
    """
    emitter_streams = generate_emitter_streams(pulses_per_emitter, seed)
    streams = [stream for _, _, stream in emitter_streams]
    return merge_streams_by_toa(streams)
