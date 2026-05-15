# AGENTS.md

## Project

This repository targets radar pulse deinterleaving under an unknown number of emitters.

The current technical route is:

TSRD pulse trains
-> pulse windows
-> TCAN triplet embeddings
-> non-parametric clustering
-> cluster post-processing
-> source-count estimation
-> clustering metrics

## Current branch

feat/cluster-postprocessing

## Current objective

Implement Phase 3B: cluster post-processing and sequence reconstruction.

Previous phases have implemented:

- TSRD loader
- TSRD windowing
- raw-feature clustering baseline
- TCAN embedding extraction
- triplet metric learning
- unified embedding evaluation
- multi-file clustering parameter search

This branch should refine initial clustering results by mapping clusters back to the original pulse sequence and applying conservative post-processing.

## Conceptual inspiration

This branch borrows the idea of mapping clustering results back to original pulse indices and iteratively refining deinterleaving results.

Do not implement multi-receiver TDOA mapping.

Do not implement SSC-DBSCAN.

Do not implement multi-receiver TDOA generation.

Instead, adapt the idea to embedding clustering:

embedding cluster
-> original TOA/PDW pulse indices
-> cluster diagnostics
-> boundary reassignment
-> conservative cluster merging
-> optional cluster splitting

## Main task

Create a post-processing pipeline for clustering labels.

The pipeline should compare metrics before and after post-processing.

## Required files

Create or update:

- src/cluster_diagnostics.py
- src/cluster_refinement.py
- run_cluster_postprocessing.py
- README.md

Reuse existing:

- src/tsrd_loader.py
- src/tsrd_window_dataset.py
- src/model_tcan.py
- src/embedding_extractor.py
- src/clustering_baselines.py
- src/clustering_metrics.py

## Cluster diagnostics

Implement diagnostic metrics for each cluster:

- cluster size
- embedding centroid
- embedding compactness
- nearest cluster distance
- PW/RF/AOA/PA mean and std
- DTOA statistics
- noise ratio if applicable

## Boundary pulse reassignment

Implement conservative reassignment for noise points or boundary points.

Only reassign a pulse if:

- nearest cluster distance is below a threshold
- nearest-vs-second-nearest margin is large enough
- PDW distance is within threshold
- cluster compactness does not degrade significantly

Do not force all noise points to be assigned.

## Cluster merging

Implement conservative merging of clusters.

Merge two clusters only if:

- embedding centroid distance is below threshold
- PDW distribution distance is below threshold
- merged compactness remains acceptable

## Cluster splitting

Implement optional conservative splitting.

Only attempt splitting for clusters with high internal dispersion.

Use internal DBSCAN/HDBSCAN if available.

If splitting is unstable, leave the cluster unchanged.

## Important rule

Do not use ground-truth labels to make post-processing decisions.

Ground-truth labels may only be used for evaluation metrics.

## Command-line script

Create:

run_cluster_postprocessing.py

Arguments:

--tsrd-path or --tsrd-dir
--file-glob
--max-files
--max-windows-per-file
--feature-set 4d/5d
--window-size
--stride
--checkpoint
--cluster-method dbscan/hdbscan
--eps
--min-samples
--min-cluster-size
--enable-reassign
--enable-merge
--enable-split
--output-csv optional

## Output

For each window, print metrics before and after post-processing:

- homogeneity
- completeness
- v_measure
- adjusted_rand_index
- adjusted_mutual_info
- true_source_count
- estimated_source_count
- abs_source_count_error
- noise_ratio

Also print mean before/after metrics.

## Constraints

Do not implement multi-receiver TDOA processing.
Do not implement SSC-DBSCAN.
Do not implement new metric-learning loss in this branch.
Do not retrain the encoder in this branch.
Do not commit outputs, checkpoints, or h5 files.

## Verification

The following should run:

python run_cluster_postprocessing.py --tsrd-path <local_h5_file> --feature-set 5d --window-size 1024 --max-windows-per-file 3 --checkpoint <checkpoint_path> --cluster-method dbscan --eps 0.5 --min-samples 5 --enable-reassign --enable-merge

After implementation, report:

1. files modified
2. diagnostics computed
3. reassignment rule
4. merge rule
5. split rule if implemented
6. before/after metrics
7. current limitations