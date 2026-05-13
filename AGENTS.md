# AGENTS.md

## Project

This repository targets radar pulse deinterleaving under an unknown number of emitters.

The long-term goal is not fixed-class sequence labeling. The target pipeline is:

1. Load PDW pulse trains.
2. Split pulse trains into pulse-level windows.
3. Generate pulse-level contextual embeddings.
4. Cluster embeddings using non-parametric clustering.
5. Estimate emitter count.
6. Evaluate with clustering metrics.

## Current branch

feat/tsrd-window-clustering-baseline

## Current objective

Implement Phase 1B: TSRD windowing and raw-feature clustering baseline.

Phase 1A TSRD loader has already been implemented.

Do not implement TCAN embedding yet.
Do not implement triplet loss yet.
Do not implement supervised contrastive loss yet.
Do not modify the TCAN model.
Do not train deep models in this branch.

## Main task

Add a basic unknown-source-count clustering evaluation pipeline using raw PDW-derived features.

The pipeline should be:

TSRD pulse train
-> sort by TOA
-> compute DTOA
-> build 4d or 5d features
-> split into fixed-length pulse windows
-> normalize features
-> cluster each window
-> evaluate clustering against true labels

## Required files

Create or update:

- src/tsrd_window_dataset.py
- src/clustering_baselines.py
- src/clustering_metrics.py
- run_tsrd_clustering_baseline.py
- README.md

## Windowing

Support pulse-count windows:

- window_size
- stride
- max_windows

Default:

- window_size = 1024
- stride = 1024
- max_windows = 10

Each window should return:

X_window: [window_size, feature_dim]
y_window: [window_size]
metadata:
  source file
  start index
  end index
  true source count

## Feature sets

Support:

4d:
[DTOA, PW, RF, AOA]

5d:
[DTOA, PW, RF, AOA, PA]

Normalize features per window using StandardScaler or RobustScaler.

## Clustering methods

Implement:

1. DBSCAN
2. AgglomerativeClustering with oracle true source count for sanity check
3. Optional HDBSCAN if the package is installed

If hdbscan is not installed, print a clear message and continue.

## Metrics

For each window, compute:

- homogeneity
- completeness
- v_measure
- adjusted_rand_index
- adjusted_mutual_info
- true_source_count
- estimated_source_count
- source_count_error
- abs_source_count_error
- noise_ratio if cluster label -1 exists

Aggregate metrics across windows and print mean values.

## Command-line script

Create:

run_tsrd_clustering_baseline.py

Arguments:

--tsrd-path
--feature-set 4d/5d
--window-size
--stride
--max-windows
--method dbscan/hdbscan/agglomerative_oracle
--eps
--min-samples
--min-cluster-size

Example:

python run_tsrd_clustering_baseline.py --tsrd-path E:\Datasets\TSRD\scan\train_scan\config_0.h5 --feature-set 5d --window-size 1024 --max-windows 5 --method dbscan

## Constraints

Do not upload data files.
Do not commit outputs.
Do not implement neural embedding in this branch.
Do not implement triplet loss in this branch.
Do not implement source-count correction or post-processing yet.

## Verification

The following should run:

python run_tsrd_clustering_baseline.py --tsrd-path <local_h5_file> --feature-set 5d --window-size 1024 --max-windows 3 --method dbscan

python run_tsrd_clustering_baseline.py --tsrd-path <local_h5_file> --feature-set 5d --window-size 1024 --max-windows 3 --method agglomerative_oracle

After implementation, report:

1. Files modified
2. How windows are constructed
3. How source count is estimated
4. Which clustering metrics are computed
5. Baseline results on the test file
6. Limitations