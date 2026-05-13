"""Loader and field adapter for the Turing Synthetic Radar Dataset.

The adapter returns one pulse train in the project-internal representation:

    pdw_array: [TOA, PW, RF, AOA, PA]
    labels: integer emitter labels local to this pulse train
"""

from pathlib import Path

import numpy as np


INTERNAL_PDW_COLUMNS = ("TOA", "PW", "RF", "AOA", "PA")
TSRD_FIELD_MAPPING = {
    "TOA": "TOA",
    "Pulse Width": "PW",
    "Centre Frequency": "RF",
    "Angle of Arrival": "AOA",
    "Amplitude": "PA",
}

_FIELD_ALIASES = {
    "TOA": (
        "toa",
        "timeofarrival",
        "timearrival",
        "arrivaltime",
    ),
    "PW": (
        "pulsewidth",
        "pw",
        "pulse_width",
    ),
    "RF": (
        "centrefrequency",
        "centerfrequency",
        "carrierfrequency",
        "rf",
        "frequency",
        "freq",
        "cf",
    ),
    "AOA": (
        "angleofarrival",
        "arrivalangle",
        "aoa",
        "doa",
        "directionofarrival",
    ),
    "PA": (
        "amplitude",
        "pa",
        "pulseamplitude",
        "amp",
        "power",
    ),
    "label": (
        "label",
        "labels",
        "emitter",
        "emitterid",
        "emitterlabel",
        "source",
        "sourceid",
        "class",
        "target",
    ),
}


class TSRDLoadError(ValueError):
    """Raised when a TSRD file cannot be mapped to the internal format."""


def _normalize_name(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _alias_lookup(names):
    lookup = {}
    for name in names:
        normalized = _normalize_name(name)
        if normalized:
            lookup[normalized] = name
        basename = str(name).split("/")[-1]
        normalized_basename = _normalize_name(basename)
        if normalized_basename:
            lookup[normalized_basename] = name
    return lookup


def _find_field(names, canonical_name):
    lookup = _alias_lookup(names)
    for alias in _FIELD_ALIASES[canonical_name]:
        normalized = _normalize_name(alias)
        if normalized in lookup:
            return lookup[normalized]
    return None


def _require_field(names, canonical_name):
    field = _find_field(names, canonical_name)
    if field is None:
        aliases = ", ".join(_FIELD_ALIASES[canonical_name])
        available = ", ".join(map(str, names)) if names else "<none>"
        raise TSRDLoadError(
            f"TSRD field mapping failed: missing required field '{canonical_name}'. "
            f"Accepted aliases: {aliases}. Available fields: {available}."
        )
    return field


def _as_1d_column(values, field_name):
    column = np.asarray(values)
    if column.ndim == 2 and 1 in column.shape:
        column = column.reshape(-1)
    if column.ndim != 1:
        raise TSRDLoadError(
            f"TSRD field '{field_name}' must be one-dimensional; got shape {column.shape}."
        )
    return column


def _records_to_columns(records):
    if records.dtype.names is None:
        raise TSRDLoadError(
            "TSRD array does not contain field names. Provide a CSV/HDF5/NPZ file "
            "with named columns such as TOA, Pulse Width, Centre Frequency, "
            "Angle of Arrival, Amplitude, and label."
        )
    return {name: records[name] for name in records.dtype.names}


def _load_text_table(path):
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    if path.suffix.lower() in {".txt", ".dat"}:
        delimiter = None
    try:
        records = np.genfromtxt(
            path,
            names=True,
            delimiter=delimiter,
            dtype=None,
            encoding=None,
            autostrip=True,
        )
    except Exception as exc:
        raise TSRDLoadError(f"Failed to read TSRD text table '{path}': {exc}") from exc

    if records.size == 0:
        raise TSRDLoadError(f"TSRD file '{path}' contains no pulse rows.")
    return _records_to_columns(np.atleast_1d(records))


def _load_numpy_file(path):
    try:
        loaded = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise TSRDLoadError(f"Failed to read TSRD NumPy file '{path}': {exc}") from exc

    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            keys = list(loaded.files)
            if not keys:
                raise TSRDLoadError(f"TSRD NPZ file '{path}' contains no arrays.")
            if len(keys) == 1 and loaded[keys[0]].dtype.names is not None:
                return _records_to_columns(np.atleast_1d(loaded[keys[0]]))
            return {key: loaded[key] for key in keys}
        finally:
            loaded.close()

    return _records_to_columns(np.atleast_1d(loaded))


def _collect_hdf5_datasets(group, prefix=""):
    datasets = {}
    for name, item in group.items():
        key = f"{prefix}/{name}" if prefix else name
        if hasattr(item, "items"):
            datasets.update(_collect_hdf5_datasets(item, key))
        else:
            datasets[key] = item[()]
    return datasets


def _decode_hdf5_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _columns_from_feature_matrix(datasets):
    if "data" not in datasets or "metadata/feature_names" not in datasets:
        return None

    data = np.asarray(datasets["data"])
    feature_names = np.asarray(datasets["metadata/feature_names"]).reshape(-1)
    if data.ndim != 2:
        raise TSRDLoadError(
            f"TSRD dataset 'data' must be a two-dimensional feature matrix; got {data.shape}."
        )
    if data.shape[1] != len(feature_names):
        raise TSRDLoadError(
            "TSRD dataset 'data' column count does not match "
            f"metadata/feature_names length: {data.shape[1]} != {len(feature_names)}."
        )

    columns = {
        _decode_hdf5_name(feature_name): data[:, index]
        for index, feature_name in enumerate(feature_names)
    }
    if "labels" in datasets:
        columns["labels"] = datasets["labels"]
    return columns


def _load_hdf5_file(path):
    try:
        import h5py
    except ImportError as exc:
        raise TSRDLoadError(
            "Reading .h5/.hdf5 TSRD files requires h5py. Install h5py or export the "
            "pulse train as CSV/NPZ with named fields."
        ) from exc

    try:
        with h5py.File(path, "r") as handle:
            datasets = _collect_hdf5_datasets(handle)
    except Exception as exc:
        raise TSRDLoadError(f"Failed to read TSRD HDF5 file '{path}': {exc}") from exc

    if not datasets:
        raise TSRDLoadError(f"TSRD HDF5 file '{path}' contains no datasets.")

    matrix_columns = _columns_from_feature_matrix(datasets)
    if matrix_columns is not None:
        return matrix_columns

    compound = {
        name: values
        for name, values in datasets.items()
        if getattr(np.asarray(values).dtype, "names", None) is not None
    }
    if len(compound) == 1:
        return _records_to_columns(np.atleast_1d(next(iter(compound.values()))))

    return datasets


def _load_columns(path):
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt", ".dat"}:
        return _load_text_table(path)
    if suffix in {".npy", ".npz"}:
        return _load_numpy_file(path)
    if suffix in {".h5", ".hdf5"}:
        return _load_hdf5_file(path)
    raise TSRDLoadError(
        f"Unsupported TSRD file extension '{suffix}'. Supported formats are "
        ".csv, .tsv, .txt, .dat, .npy, .npz, .h5, and .hdf5."
    )


def _extract_mapped_columns(columns, label_field=None):
    names = list(columns.keys())
    mapped = {}
    for canonical_name in INTERNAL_PDW_COLUMNS:
        source_name = _require_field(names, canonical_name)
        mapped[canonical_name] = _as_1d_column(columns[source_name], source_name)

    if label_field is not None:
        if label_field not in columns:
            available = ", ".join(map(str, names)) if names else "<none>"
            raise TSRDLoadError(
                f"Requested label field '{label_field}' is not present. "
                f"Available fields: {available}."
            )
        label_source = label_field
    else:
        label_source = _require_field(names, "label")

    labels = _as_1d_column(columns[label_source], label_source)
    lengths = {name: len(values) for name, values in mapped.items()}
    lengths["label"] = len(labels)
    if len(set(lengths.values())) != 1:
        raise TSRDLoadError(f"TSRD fields have inconsistent lengths: {lengths}.")

    try:
        pdw_array = np.column_stack(
            [mapped[name].astype(np.float64) for name in INTERNAL_PDW_COLUMNS]
        )
    except ValueError as exc:
        raise TSRDLoadError(f"TSRD PDW fields must be numeric: {exc}") from exc

    if not np.all(np.isfinite(pdw_array)):
        raise TSRDLoadError("TSRD PDW fields contain NaN or infinite values.")

    if np.asarray(labels).dtype.kind in {"f", "i", "u"} and not np.all(np.isfinite(labels)):
        raise TSRDLoadError("TSRD labels contain NaN or infinite values.")

    return pdw_array, labels


def _remap_labels(labels):
    labels = np.asarray(labels)
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    if len(unique_labels) == 0:
        raise TSRDLoadError("TSRD pulse train contains no labels.")
    return inverse.astype(np.int64), unique_labels


def load_tsrd_pulse_train(path, label_field=None):
    """Load one TSRD pulse train and map it to internal PDW fields.

    Labels are remapped to contiguous integers local to the current pulse train,
    then PDWs and labels are stably sorted by ascending TOA.
    """
    path = Path(path)
    if not path.exists():
        raise TSRDLoadError(f"TSRD file does not exist: {path}")
    if not path.is_file():
        raise TSRDLoadError(f"TSRD path is not a file: {path}")

    columns = _load_columns(path)
    pdw_array, raw_labels = _extract_mapped_columns(columns, label_field=label_field)
    labels, _ = _remap_labels(raw_labels)

    sort_order = np.argsort(pdw_array[:, 0], kind="mergesort")
    pdw_array = pdw_array[sort_order].astype(np.float64)
    labels = labels[sort_order]

    if len(pdw_array) == 0:
        raise TSRDLoadError("TSRD pulse train contains no pulses.")

    return pdw_array, labels
