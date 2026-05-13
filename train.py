"""Run TCAN sequence-labeling experiments with DTOA or binary input."""

import argparse
import numpy as np

from src.data_simulator import PDW_COLUMNS, generate_interleaved_stream
from src.metrics import average_recall, class_counts, confusion_matrix, recall_per_class
from src.nonideal import (
    apply_measurement_error,
    apply_random_pulse_loss,
    apply_spurious_pulses,
)
from src.preprocessing import (
    apply_minmax,
    create_binary_features,
    create_fixed_length_windows,
    fit_minmax,
    pdw_to_dtoa_features,
)
from src.sparsity import apply_signal_sparsity, sparsity_ratio_to_gap_ratio
from src.tsrd_loader import INTERNAL_PDW_COLUMNS, TSRDLoadError, load_tsrd_pulse_train


SEED = 42
DEFAULT_WINDOW_SIZE = 64
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 12
LEARNING_RATE = 1e-3
SYNTHETIC_PA_VALUE = 1.0

torch = None
DataLoader = None
TensorDataset = None


def split_pdw_and_labels(pdw_array, labels, train_fraction=0.8):
    split_index = int(len(pdw_array) * train_fraction)
    return (
        pdw_array[:split_index],
        labels[:split_index],
        pdw_array[split_index:],
        labels[split_index:],
    )


def build_dataloader(
    x_windows,
    y_windows,
    batch_size,
    shuffle,
):
    dataset = TensorDataset(
        torch.from_numpy(x_windows).float(),
        torch.from_numpy(y_windows).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    num_classes,
):
    model.train()
    total_loss = 0.0
    total_positions = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits.reshape(-1, num_classes), labels.reshape(-1))
        loss.backward()
        optimizer.step()

        positions = labels.numel()
        total_loss += loss.item() * positions
        total_positions += positions

    return total_loss / total_positions


def evaluate(
    model,
    dataloader,
    device,
):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            logits = model(features)
            predictions.append(torch.argmax(logits, dim=-1).cpu().numpy())
            targets.append(labels.numpy())

    return np.concatenate(targets, axis=0), np.concatenate(predictions, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TCAN with DTOA or binary input for radar pulse labeling."
    )
    parser.add_argument(
        "--data-source",
        choices=["synthetic", "tsrd"],
        default="synthetic",
        help="Dataset source. Defaults to the inherited synthetic pipeline.",
    )
    parser.add_argument(
        "--tsrd-path",
        default=None,
        help="Path to one local TSRD pulse-train file when --data-source tsrd.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["4d", "5d"],
        default="4d",
        help="Feature set: 4d excludes PA, 5d includes PA.",
    )
    parser.add_argument(
        "--input-format",
        choices=["dtoa", "binary"],
        default="dtoa",
        help="Input preprocessing format.",
    )
    parser.add_argument(
        "--ts",
        type=float,
        default=10.0,
        help="Sampling interval for binary time bins.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Fixed sequence window length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--loss",
        choices=["ce", "focal"],
        default="ce",
        help="Loss function.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Focal loss focusing strength.",
    )
    parser.add_argument(
        "--alpha-mode",
        choices=["none", "inverse_freq"],
        default="inverse_freq",
        help="Focal loss alpha weighting mode.",
    )
    parser.add_argument(
        "--sparsity-ratio",
        choices=["none", "1:3", "1:5", "1:8"],
        default="none",
        help="Periodic signal sparsity ratio. 'none' keeps the full stream.",
    )
    parser.add_argument(
        "--visible-duration",
        type=float,
        default=5000.0,
        help="Duration of each visible scan-gate segment.",
    )
    parser.add_argument(
        "--sparsity-phase-offset",
        type=float,
        default=0.0,
        help="Phase offset applied before the periodic sparsity gate.",
    )
    parser.add_argument(
        "--toa-error-std",
        type=float,
        default=0.0,
        help="Gaussian TOA measurement error standard deviation.",
    )
    parser.add_argument(
        "--pw-error-std",
        type=float,
        default=0.0,
        help="Gaussian PW measurement error standard deviation.",
    )
    parser.add_argument(
        "--rf-error-std",
        type=float,
        default=0.0,
        help="Gaussian RF measurement error standard deviation.",
    )
    parser.add_argument(
        "--doa-error-std",
        type=float,
        default=0.0,
        help="Gaussian DOA measurement error standard deviation.",
    )
    parser.add_argument(
        "--pulse-loss-rate",
        type=float,
        default=0.0,
        help="Probability of randomly deleting each pulse.",
    )
    parser.add_argument(
        "--spurious-rate",
        type=float,
        default=0.0,
        help="Spurious pulse insertion rate relative to current pulse count.",
    )
    args = parser.parse_args()
    if args.data_source == "tsrd" and not args.tsrd_path:
        parser.error("--tsrd-path is required when --data-source tsrd.")
    return args


def maybe_apply_sparsity(pdw_stream, args):
    before_count = len(pdw_stream)
    gap_ratio = sparsity_ratio_to_gap_ratio(args.sparsity_ratio)
    if gap_ratio is None:
        retained_ratio = 1.0
        return pdw_stream, before_count, before_count, retained_ratio

    sparse_stream, _ = apply_signal_sparsity(
        pdw_stream,
        visible_duration=args.visible_duration,
        gap_ratio=gap_ratio,
        phase_offset=args.sparsity_phase_offset,
    )
    after_count = len(sparse_stream)
    retained_ratio = after_count / float(before_count) if before_count > 0 else 0.0
    return sparse_stream, before_count, after_count, retained_ratio


def apply_nonideal_conditions(pdw_stream, args):
    labels = pdw_stream[:, 4].astype(np.int64)
    before_count = len(pdw_stream)

    pdw_stream, labels = apply_measurement_error(
        pdw_stream,
        labels=labels,
        toa_std=args.toa_error_std,
        pw_std=args.pw_error_std,
        rf_std=args.rf_error_std,
        doa_std=args.doa_error_std,
        seed=SEED + 1,
    )
    after_measurement_error = len(pdw_stream)

    pdw_stream, labels = apply_random_pulse_loss(
        pdw_stream,
        labels=labels,
        loss_rate=args.pulse_loss_rate,
        seed=SEED + 2,
    )
    after_pulse_loss = len(pdw_stream)

    pdw_stream, labels = apply_spurious_pulses(
        pdw_stream,
        labels=labels,
        spurious_rate=args.spurious_rate,
        spurious_label=4,
        seed=SEED + 3,
    )
    after_spurious = len(pdw_stream)

    return pdw_stream, {
        "before": before_count,
        "after_measurement_error": after_measurement_error,
        "after_pulse_loss": after_pulse_loss,
        "after_spurious": after_spurious,
    }


def synthetic_stream_to_internal(pdw_stream):
    labels = pdw_stream[:, 4].astype(np.int64)
    pa = np.full((len(pdw_stream), 1), SYNTHETIC_PA_VALUE, dtype=np.float64)
    pdw_array = np.column_stack([pdw_stream[:, :4], pa]).astype(np.float64)
    return pdw_array, labels


def load_source_data(args):
    if args.data_source == "synthetic":
        pdw_stream = generate_interleaved_stream(pulses_per_emitter=900, seed=SEED)
        pdw_stream, before_count, after_count, retained_ratio = maybe_apply_sparsity(
            pdw_stream,
            args,
        )
        pdw_stream, nonideal_counts = apply_nonideal_conditions(pdw_stream, args)
        pdw_array, labels = synthetic_stream_to_internal(pdw_stream)
        return pdw_array, labels, {
            "before_sparsity": before_count,
            "after_sparsity": after_count,
            "retained_ratio": retained_ratio,
            "nonideal_counts": nonideal_counts,
        }

    try:
        pdw_array, labels = load_tsrd_pulse_train(args.tsrd_path)
    except TSRDLoadError as exc:
        raise SystemExit(f"Failed to load TSRD pulse train: {exc}") from exc

    return pdw_array, labels, {
        "path": args.tsrd_path,
    }


def build_dtoa_windows(
    train_pdw,
    train_labels,
    test_pdw,
    test_labels,
    window_size,
    num_classes,
    feature_set,
):
    train_features, train_labels = pdw_to_dtoa_features(
        train_pdw,
        labels=train_labels,
        feature_set=feature_set,
    )
    test_features, test_labels = pdw_to_dtoa_features(
        test_pdw,
        labels=test_labels,
        feature_set=feature_set,
    )

    norm_stats = fit_minmax(train_features)
    train_features = apply_minmax(train_features, norm_stats)
    test_features = apply_minmax(test_features, norm_stats)

    x_train, y_train = create_fixed_length_windows(
        train_features,
        train_labels,
        sequence_length=window_size,
    )
    x_test, y_test = create_fixed_length_windows(
        test_features,
        test_labels,
        sequence_length=window_size,
    )
    return x_train, y_train, x_test, y_test, num_classes


def build_binary_windows(
    train_pdw,
    train_labels,
    test_pdw,
    test_labels,
    window_size,
    ts,
    num_classes,
    feature_set,
):
    parameter_end = 5 if feature_set == "5d" else 4
    pdw_norm_stats = fit_minmax(train_pdw[:, 1:parameter_end].astype(np.float32))
    train_features, train_labels = create_binary_features(
        train_pdw,
        ts=ts,
        labels=train_labels,
        stats=pdw_norm_stats,
        feature_set=feature_set,
    )
    test_features, test_labels = create_binary_features(
        test_pdw,
        ts=ts,
        labels=test_labels,
        stats=pdw_norm_stats,
        feature_set=feature_set,
    )

    x_train, y_train = create_fixed_length_windows(
        train_features,
        train_labels,
        sequence_length=window_size,
    )
    x_test, y_test = create_fixed_length_windows(
        test_features,
        test_labels,
        sequence_length=window_size,
    )
    return x_train, y_train, x_test, y_test, num_classes


def determine_num_classes(input_format, labels):
    pulse_classes = int(labels.max()) + 1
    if input_format == "dtoa":
        return pulse_classes
    return pulse_classes + 1


def import_training_dependencies():
    global torch, DataLoader, TensorDataset
    try:
        import torch as torch_module
        from torch.utils.data import DataLoader as data_loader_class
        from torch.utils.data import TensorDataset as tensor_dataset_class
        from src.losses import build_loss
        from src.model_tcan import TCAN
        from src.utils import get_device, set_random_seed
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "a required module"
        raise SystemExit(
            f"Training requires PyTorch and the TCAN dependencies, but '{missing_name}' "
            "is not available in this Python environment."
        ) from exc

    torch = torch_module
    DataLoader = data_loader_class
    TensorDataset = tensor_dataset_class
    return build_loss, TCAN, get_device, set_random_seed


def main():
    args = parse_args()
    pdw_array, labels, source_info = load_source_data(args)
    train_pdw, train_labels, test_pdw, test_labels = split_pdw_and_labels(pdw_array, labels)

    build_loss, TCAN, get_device, set_random_seed = import_training_dependencies()
    set_random_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Selected data source: {args.data_source}")
    print(f"Selected input format: {args.input_format}")
    print(f"Selected feature set: {args.feature_set}")
    print(f"Selected loss function: {args.loss}")
    print(f"Selected sparsity ratio: {args.sparsity_ratio}")
    print(f"Visible duration: {args.visible_duration}")
    print(f"Sparsity phase offset: {args.sparsity_phase_offset}")
    print(
        "Measurement error stds: "
        f"TOA={args.toa_error_std}, PW={args.pw_error_std}, "
        f"RF={args.rf_error_std}, DOA={args.doa_error_std}"
    )
    print(f"Pulse loss rate: {args.pulse_loss_rate}")
    print(f"Spurious pulse rate: {args.spurious_rate}")

    print(f"Internal PDW columns: {INTERNAL_PDW_COLUMNS}")
    if args.data_source == "synthetic":
        nonideal_counts = source_info["nonideal_counts"]
        print(f"Synthetic source columns: {PDW_COLUMNS}")
        print(f"Synthetic PA placeholder value: {SYNTHETIC_PA_VALUE}")
        print(f"Pulse count before sparsity: {source_info['before_sparsity']}")
        print(f"Pulse count after sparsity: {source_info['after_sparsity']}")
        print(f"Retained pulse ratio: {source_info['retained_ratio']:.4f}")
        print(f"Pulse count before nonideal conditions: {nonideal_counts['before']}")
        print(
            "Pulse count after measurement error: "
            f"{nonideal_counts['after_measurement_error']}"
        )
        print(f"Pulse count after pulse loss: {nonideal_counts['after_pulse_loss']}")
        print(f"Pulse count after spurious insertion: {nonideal_counts['after_spurious']}")
    else:
        print(f"TSRD path: {source_info['path']}")
        print("TSRD mode loads one pulse train and remaps labels locally.")
    print(
        f"Total pulses: {len(pdw_array)} | train: {len(train_pdw)} | test: {len(test_pdw)}"
    )

    num_classes = determine_num_classes(args.input_format, labels)
    if args.input_format == "dtoa":
        x_train, y_train, x_test, y_test, num_classes = build_dtoa_windows(
            train_pdw,
            train_labels,
            test_pdw,
            test_labels,
            args.window_size,
            num_classes,
            args.feature_set,
        )
    else:
        print(f"Binary sampling interval Ts: {args.ts}")
        x_train, y_train, x_test, y_test, num_classes = build_binary_windows(
            train_pdw,
            train_labels,
            test_pdw,
            test_labels,
            args.window_size,
            args.ts,
            num_classes,
            args.feature_set,
        )

    print(f"Train input tensor shape [B, T, D]: {x_train.shape}")
    print(f"Train label tensor shape [B, T]: {y_train.shape}")
    print(f"Test input tensor shape [B, T, D]: {x_test.shape}")
    print(f"Test label tensor shape [B, T]: {y_test.shape}")
    print(f"Number of classes: {num_classes}")
    train_counts = class_counts(y_train, num_classes)
    print(f"Train class counts: {train_counts}")

    train_loader = build_dataloader(x_train, y_train, args.batch_size, shuffle=True)
    test_loader = build_dataloader(x_test, y_test, args.batch_size, shuffle=False)

    model = TCAN(input_dim=x_train.shape[-1], num_classes=num_classes).to(device)
    criterion, alpha = build_loss(
        loss_name=args.loss,
        gamma=args.gamma,
        alpha_mode=args.alpha_mode,
        labels=y_train,
        num_classes=num_classes,
        device=device,
    )
    if args.loss == "focal":
        print(f"Focal gamma: {args.gamma}")
        print(f"Focal alpha mode: {args.alpha_mode}")
        if alpha is not None:
            print(f"Focal alpha weights: {alpha.detach().cpu().numpy()}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    sample_features, _ = next(iter(train_loader))
    sample_logits = model(sample_features.to(device))
    print(f"Model output tensor shape [B, T, C]: {tuple(sample_logits.shape)}")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes,
        )
        print(f"Epoch {epoch:02d}/{args.epochs} | loss: {loss:.4f}")

    y_true, y_pred = evaluate(model, test_loader, device)
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    eval_counts = class_counts(y_true, num_classes)
    recalls = recall_per_class(matrix)

    print(f"Evaluation class counts: {eval_counts}")
    print("Recall per class:")
    for class_index, recall in enumerate(recalls):
        print(f"  class {class_index}: {recall:.4f}")
    print(f"Average recall: {average_recall(recalls):.4f}")
    if args.input_format == "binary":
        pulse_only_average = average_recall(recalls[1:])
        print(
            "Pulse-only average recall "
            f"(classes 1-{num_classes - 1}): {pulse_only_average:.4f}"
        )
    print("Confusion matrix (rows=true labels, cols=predicted labels):")
    print(matrix)


if __name__ == "__main__":
    main()
