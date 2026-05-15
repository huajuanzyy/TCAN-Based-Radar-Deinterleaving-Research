# AGENTS.md

## Project

This repository targets radar pulse deinterleaving under an unknown number of emitters.

The long-term target pipeline is:

TSRD pulse train
-> pulse windows
-> pulse-level embeddings
-> non-parametric clustering
-> source-count estimation
-> clustering metrics
-> post-processing and sequence reconstruction

## Current branch

feat/embedding-evaluation

## Current objective

Implement Phase 2C: systematic embedding evaluation and clustering comparison.

Previous phases have implemented:

- TSRD loader
- TSRD windowing
- raw-feature clustering baseline
- TCAN encoder embedding extraction
- triplet metric learning for TCAN embeddings

This branch should compare raw features, randomly initialized TCAN embeddings, and triplet-trained TCAN embeddings under the same windows and clustering metrics.

## Main task

Create a unified evaluation script that can compare:

1. raw-feature clustering
2. random TCAN embedding clustering
3. triplet-trained TCAN embedding clustering

using the same TSRD file, same windows, same clustering method, and same metrics.

## Required files

Create or update:

- run_embedding_evaluation.py
- src/evaluation_runner.py
- src/result_writer.py
- README.md

Reuse existing:

- src/tsrd_loader.py
- src/tsrd_window_dataset.py
- src/clustering_baselines.py
- src/clustering_metrics.py
- src/embedding_extractor.py
- src/model_tcan.py

## Evaluation methods

Support method names:

- raw
- random_embedding
- triplet_embedding

For raw:

Use 4d or 5d raw features directly.

For random_embedding:

Use randomly initialized TCAN encoder and print a warning.

For triplet_embedding:

Load a checkpoint from --checkpoint.

## Command-line arguments

run_embedding_evaluation.py should support:

--tsrd-path
--feature-set 4d/5d
--window-size
--stride
--max-windows
--cluster-method dbscan/hdbscan/agglomerative_oracle
--methods raw,random_embedding,triplet_embedding
--embedding-dim
--checkpoint
--eps
--min-samples
--min-cluster-size
--output-csv optional

## Metrics

For each method and each window, compute:

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

Also print mean metrics grouped by method.

## Constraints

Do not implement post-processing in this branch.

Do not implement cluster merging or splitting.

Do not implement source-count correction.

Do not implement batch-hard triplet mining.

Do not commit outputs, checkpoints, or data files.

## Verification

The following should run:

python run_embedding_evaluation.py --tsrd-path <local_h5_file> --feature-set 5d --window-size 1024 --max-windows 3 --cluster-method dbscan --methods raw,random_embedding

If a checkpoint is available:

python run_embedding_evaluation.py --tsrd-path <local_h5_file> --feature-set 5d --window-size 1024 --max-windows 3 --cluster-method dbscan --methods raw,triplet_embedding --checkpoint <checkpoint_path>

After implementation, report:

1. files modified
2. compared methods
3. metric table
4. output CSV path if used
5. current limitations
6. next step toward HDBSCAN tuning or post-processing