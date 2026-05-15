# AGENTS.md

## Project

This repository targets radar pulse deinterleaving under an unknown number of emitters.

The long-term target pipeline is:

TSRD pulse train
-> pulse windows
-> TCAN/Transformer pulse-level embeddings
-> metric learning
-> non-parametric clustering
-> source-count estimation
-> clustering metrics

## Current branch

feat/triplet-metric-learning

## Current objective

Implement Phase 2B: triplet metric learning for TCAN pulse-level embeddings.

Phase 1A TSRD loader has been implemented.
Phase 1B TSRD windowing and raw-feature clustering baseline has been implemented.
Phase 2A TCAN encoder embedding extraction has been implemented.

This branch should train the TCAN encoder so that same-emitter pulses are close in embedding space and different-emitter pulses are far apart.

## Main task

Add triplet-loss training for TCAN embeddings.

The pipeline should be:

TSRD windows
-> TCAN encoder
-> embeddings [B, T, E]
-> triplet sampling within each window
-> triplet loss
-> checkpoint saving
-> embedding clustering evaluation

## Important label rule

TSRD labels are only meaningful within the current pulse train.

Do not assume that label 0 in one file is the same physical emitter as label 0 in another file.

For the first implementation, construct triplets within the same window only.

## Required files

Create or update:

- src/metric_losses.py
- src/triplet_sampler.py
- src/embedding_trainer.py
- train_tsrd_triplet.py
- run_tsrd_embedding_clustering.py
- README.md

## Triplet loss

Use standard triplet margin loss:

L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

Default:

--margin 0.5

Support:

--embedding-dim
--margin
--epochs
--batch-size
--learning-rate
--num-triplets-per-window
--checkpoint-dir

## Triplet sampling

For each window:

- anchor and positive must have the same label.
- anchor and negative must have different labels.
- Skip labels that appear fewer than 2 times in the window.
- Skip windows that contain fewer than 2 unique labels.
- Do not sample positives or negatives across different files in this branch.

## Embedding normalization

L2-normalize embeddings before triplet loss and before clustering.

## Checkpoint

Save trained encoder checkpoint to:

checkpoints/

Do not commit checkpoint files to GitHub.

## Evaluation

After training, allow evaluation with:

run_tsrd_embedding_clustering.py --checkpoint <path>

The evaluation should compute:

- homogeneity
- completeness
- v_measure
- adjusted_rand_index
- adjusted_mutual_info
- true_source_count
- estimated_source_count
- source_count_error
- abs_source_count_error
- noise_ratio

## Constraints

Do not implement supervised contrastive loss in this branch.
Do not implement batch-hard mining in this branch unless the simple random triplet version is already complete.
Do not implement post-processing.
Do not implement cluster merging or splitting.
Do not upload datasets, checkpoints, or outputs.

## Verification

The following should run:

python train_tsrd_triplet.py --tsrd-path <local_h5_file> --feature-set 5d --window-size 1024 --max-windows 10 --embedding-dim 64 --epochs 2 --num-triplets-per-window 256

python run_tsrd_embedding_clustering.py --tsrd-path <local_h5_file> --feature-set 5d --window-size 1024 --max-windows 3 --embedding-dim 64 --checkpoint <checkpoint_path> --method dbscan

After implementation, report:

1. Files modified
2. Triplet sampling strategy
3. Training loss behavior
4. Checkpoint path
5. Embedding clustering results before and after training if available
6. Current limitations