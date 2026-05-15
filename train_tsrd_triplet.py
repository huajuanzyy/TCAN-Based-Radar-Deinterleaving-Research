"""Train TCAN pulse embeddings with in-window triplet metric learning."""

import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.embedding_extractor import resolve_device
from src.embedding_trainer import build_window_dataloader, train_triplet_encoder
from src.model_tcan import TCAN
from src.tsrd_window_dataset import load_tsrd_windows


DEFAULT_WINDOW_SIZE = 1024
DEFAULT_STRIDE = 1024
DEFAULT_MAX_WINDOWS = 10
DEFAULT_MAX_WINDOWS_PER_FILE = 10
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MARGIN = 0.5
DEFAULT_TRIPLETS_PER_WINDOW = 256
DEFAULT_CHECKPOINT_DIR = "checkpoints"
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TCAN encoder embeddings with triplet margin loss."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--tsrd-path",
        help="Path to a local TSRD h5/hdf5 file containing one pulse train.",
    )
    source.add_argument(
        "--file-list",
        help="Text file containing TSRD h5/hdf5 paths, one per line.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional root used to resolve relative entries in --file-list.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to load from --file-list.",
    )
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=DEFAULT_MAX_WINDOWS_PER_FILE,
        help="Maximum number of windows loaded from each file in --file-list mode.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["4d", "5d"],
        default="5d",
        help="Input feature set used before TCAN embedding extraction.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Number of pulses per window.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Pulse stride between consecutive windows.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=DEFAULT_MAX_WINDOWS,
        help="Maximum number of windows to train on.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="TCAN pulse embedding dimension.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of windows per batch.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="Triplet margin.",
    )
    parser.add_argument(
        "--num-triplets-per-window",
        type=int,
        default=DEFAULT_TRIPLETS_PER_WINDOW,
        help="Number of random triplets sampled from each window per epoch.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory for saving checkpoints.",
    )
    parser.add_argument(
        "--triplet-mining",
        choices=["random"],
        default="random",
        help="Triplet mining strategy. Only random triplet mining is supported in this branch.",
    )
    args = parser.parse_args()
    if args.max_files is not None and args.max_files <= 0:
        parser.error("--max-files must be positive when provided.")
    if args.max_windows_per_file <= 0:
        parser.error("--max-windows-per-file must be positive.")
    return args


def _read_file_list(file_list, data_root=None, max_files=None):
    file_list_path = Path(file_list)
    if not file_list_path.exists():
        raise FileNotFoundError(f"File list does not exist: {file_list_path}")

    root = None if data_root is None else Path(data_root)
    paths = []
    for line in file_list_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        path = Path(entry)
        if not path.is_absolute():
            path = (root / path) if root is not None else path
        paths.append(path)
        if max_files is not None and len(paths) >= max_files:
            break

    if not paths:
        raise ValueError(f"No TSRD files found in file list: {file_list_path}")
    return paths


def discover_train_files(args):
    if args.tsrd_path:
        return [Path(args.tsrd_path)]
    return _read_file_list(
        args.file_list,
        data_root=args.data_root,
        max_files=args.max_files,
    )


def load_training_windows(files, args):
    windows = []
    max_windows = args.max_windows if args.tsrd_path else args.max_windows_per_file
    for file_path in files:
        file_windows = load_tsrd_windows(
            tsrd_path=file_path,
            feature_set=args.feature_set,
            window_size=args.window_size,
            stride=args.stride,
            max_windows=max_windows,
        )
        for local_index, window in enumerate(file_windows):
            window.metadata["file_window_index"] = local_index
        windows.extend(file_windows)
        print(f"Loaded {len(file_windows)} windows from {file_path}")
    return windows


def save_checkpoint(model, args, input_dim, result):
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / (
        f"tcan_triplet_{args.feature_set}_edim{args.embedding_dim}_{timestamp}.pt"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "embedding_dim": args.embedding_dim,
            "feature_set": args.feature_set,
            "window_size": args.window_size,
            "stride": args.stride,
            "max_windows": args.max_windows,
            "max_windows_per_file": args.max_windows_per_file,
            "train_files": [str(path) for path in getattr(args, "train_files", [])],
            "margin": args.margin,
            "epoch_losses": result.epoch_losses,
            "total_triplets": result.total_triplets,
            "args": vars(args),
        },
        checkpoint_path,
    )
    return checkpoint_path


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = resolve_device()
    train_files = discover_train_files(args)
    args.train_files = train_files
    windows = load_training_windows(train_files, args)
    input_dim = int(windows[0].X_window.shape[1])
    dataloader = build_window_dataloader(
        windows,
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = TCAN(
        input_dim=input_dim,
        num_classes=1,
        embedding_dim=args.embedding_dim,
    )

    print(f"Using device: {device}")
    print(f"Training files: {len(train_files)}")
    for file_path in train_files:
        print(f"  train_file={file_path}")
    print(f"Feature set: {args.feature_set}")
    print(f"Input dim: {input_dim}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Loaded windows: {len(windows)}")
    if args.file_list:
        print(f"Max windows per file: {args.max_windows_per_file}")
    print(f"Batch size: {args.batch_size}")
    print(f"Triplet mining: {args.triplet_mining}")
    print(f"Triplets per window: {args.num_triplets_per_window}")
    print(f"Margin: {args.margin}")

    result = train_triplet_encoder(
        model=model,
        dataloader=dataloader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        margin=args.margin,
        num_triplets_per_window=args.num_triplets_per_window,
        seed=SEED,
    )
    checkpoint_path = save_checkpoint(
        model=model,
        args=args,
        input_dim=input_dim,
        result=result,
    )
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
