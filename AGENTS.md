# AGENTS.md

## Project

This repository is a radar pulse deinterleaving research project.

It is based on a previous TCAN paper reproduction repository, but the goal of this repository is different.

The previous repository reproduced a closed-set supervised TCAN sequence labeling pipeline.

This repository targets the task described in the project proposal:

- unknown number of emitters
- mixed radar pulse streams
- pulse-level embedding learning
- non-parametric clustering
- source-count estimation
- robust deinterleaving under complex electromagnetic conditions

## Current objective

Phase 1 of this repository is to prepare the project for the proposal-oriented deinterleaving task.

The first technical step is:

Phase 1A: Turing Synthetic Radar Dataset loader and data adapter.

Do not implement embedding clustering yet.
Do not implement triplet loss yet.
Do not implement HDBSCAN yet.
Do not rewrite TCAN yet.

First, only make the dataset loading and field mapping clean.

## Existing inherited components

The repository currently inherits code from the TCAN reproduction project:

- synthetic PDW simulator
- DTOA preprocessing
- binary input preprocessing
- focal loss
- signal sparsity simulation
- nonideal condition simulation
- TCAN sequence labeling model

These components can be reused, but the long-term goal is not fixed-class classification.

## Long-term target pipeline

The target proposal-oriented pipeline is:

1. Load PDW pulse trains.
2. Normalize and preprocess PDW features.
3. Generate pulse-level contextual embeddings using TCAN or Transformer encoder.
4. Train embeddings using metric learning, such as triplet loss or supervised contrastive loss.
5. Cluster embeddings using HDBSCAN, DBSCAN, or hierarchical clustering.
6. Estimate the number of emitters from clustering results.
7. Evaluate clustering quality using V-measure, ARI, AMI, Homogeneity, Completeness, and source-count error.
8. Apply post-processing such as cluster merging, cluster splitting, and boundary pulse reassignment.

## Current branch recommendation

Use a new branch:

feat/tsrd-loader

## Current task

Implement a dataset adapter for the Turing Synthetic Radar Dataset.

The adapter should convert one pulse train into the internal format used by this project.

Expected internal PDW format:

[TOA, PW, RF, AOA, PA]

Expected label format:

labels: integer emitter labels within the current pulse train.

Important:

- Sort pulses by TOA.
- Reorder labels consistently after sorting.
- Do not assume that the same label ID across different pulse trains represents the same physical emitter.
- Do not upload dataset files to GitHub.

## Files to create or update

Create:

- src/tsrd_loader.py

Update if needed:

- src/preprocessing.py
- train.py
- README.md
- .gitignore

## Command-line support

Add optional arguments to train.py:

--data-source synthetic/tsrd
--tsrd-path
--feature-set 4d/5d

Default must remain:

--data-source synthetic

so that existing synthetic experiments still run.

## Feature mapping

TSRD PDW fields should be mapped as:

TOA -> TOA
Pulse Width -> PW
Centre Frequency -> RF
Angle of Arrival -> AOA
Amplitude -> PA

For feature-set 4d:

DTOA input:
[DTOA, PW, RF, AOA]

Binary input:
[binary_presence, PW, RF, AOA]

For feature-set 5d:

DTOA input:
[DTOA, PW, RF, AOA, PA]

Binary input:
[binary_presence, PW, RF, AOA, PA]

## Constraints

Do not train on the full TSRD dataset in this branch.

Do not implement clustering in this branch.

Do not implement triplet loss in this branch.

Do not implement HDBSCAN in this branch.

Do not implement source-count estimation in this branch.

Do not commit large data files.

## Verification

The following commands should still run:

python train.py --data-source synthetic --input-format dtoa --epochs 2
python train.py --data-source synthetic --input-format binary --epochs 2

If a local TSRD file is provided, this command should run or fail gracefully with a clear error message:

python train.py --data-source tsrd --tsrd-path <path_to_file> --input-format dtoa --feature-set 5d --epochs 2

## Git

Do not commit automatically unless asked.

After implementation, report:

1. files modified
2. TSRD field mapping
3. supported feature sets
4. how to run synthetic mode
5. how to run TSRD mode
6. current limitations
7. next step toward embedding clustering