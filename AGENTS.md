# AGENTS.md

## Current branch

feat/batch-hard-triplet

## Current objective

Implement Phase 4A: batch-hard triplet metric learning for TCAN pulse-level embeddings.

Previous phases have implemented:

- TSRD loader
- TSRD windowing
- raw-feature clustering baseline
- TCAN embedding extraction
- random triplet metric learning
- unified embedding evaluation
- clustering parameter search
- cluster-to-original-pulse mapping
- source-count-aware refinement

Current observation:

Source-count-aware refinement improves metrics slightly:

V-measure: 0.6591 -> 0.6837
ARI:       0.5922 -> 0.6460
AMI:       0.6569 -> 0.6819
Abs error: 4.0000 -> 3.6667
Noise:     0.0228 -> 0.0228

Since noise ratio is already low and source-count error remains high, the remaining issue is likely over-merged clusters caused by insufficient embedding separation.

## Main task

Improve TCAN embedding separation using batch-hard triplet loss.

## Required changes

Create or update:

- src/batch_hard_triplet.py
- src/metric_losses.py
- train_tsrd_triplet.py
- run_embedding_evaluation.py
- README.md

## Batch-hard triplet loss

For each anchor in a batch:

- hardest positive: same-label sample with the largest distance
- hardest negative: different-label sample with the smallest distance

Loss:

L = max(0, d(anchor, hardest_positive) - d(anchor, hardest_negative) + margin)

Support:

--triplet-mining random/batch_hard
--margin
--embedding-dim
--epochs
--learning-rate
--batch-size

Default should preserve the existing random triplet training behavior.

## Important label rule

TSRD labels are local to each pulse train/window.

Do not assume the same label ID across different files represents the same physical emitter.

Batch-hard mining may operate within a window or within a batch only when labels are valid in the same local context.

## Evaluation

Compare:

- raw
- random_triplet_embedding
- batch_hard_triplet_embedding

using the existing evaluation scripts.

Metrics:

- V-measure
- ARI
- AMI
- Homogeneity
- Completeness
- Abs source-count error
- Noise ratio

## Constraints

Do not implement supervised contrastive loss in this branch.
Do not implement new post-processing rules in this branch.
Do not use ground-truth labels during clustering or refinement decisions.
Do not commit checkpoints, outputs, or h5 files.