# AGENTS.md

## Project

This repository targets radar pulse deinterleaving under an unknown number of emitters.

The current technical route is:

TSRD pulse trains
-> pulse windows
-> TCAN triplet embeddings
-> non-parametric clustering
-> cluster-to-original-pulse mapping
-> source-count-aware refinement
-> clustering metrics

## Current branch

feat/source-count-refinement

## Current objective

Implement Phase 3C: source-count-aware clustering refinement.

Previous phases have implemented:

- TSRD loader
- TSRD windowing
- raw-feature clustering baseline
- TCAN encoder embedding extraction
- triplet metric learning
- unified embedding evaluation
- multi-file clustering parameter search
- cluster-to-original-pulse mapping
- conservative cluster post-processing

The previous conservative post-processing only slightly improved metrics.

Current observed before/after post-processing metrics:

mean before:
homogeneity=0.8781
completeness=0.8915
v_measure=0.8707
adjusted_rand_index=0.8421
adjusted_mutual_info=0.8689
true_source_count=8.0000
estimated_source_count=4.3333
abs_source_count_error=3.6667
noise_ratio=0.2520

mean after:
homogeneity=0.8785
completeness=0.8922
v_measure=0.8712
adjusted_rand_index=0.8423
adjusted_mutual_info=0.8694
true_source_count=8.0000
estimated_source_count=4.3333
abs_source_count_error=3.6667
noise_ratio=0.2514

The main remaining problem is source-count underestimation and high noise ratio.

## Conceptual reference

This branch continues the cluster-to-original-pulse mapping idea.

The project borrows the high-level idea of mapping clustering results back to original pulse indices and iteratively refining the deinterleaving result.

Do not implement multi-receiver TDOA processing.

Do not implement SSC-DBSCAN.

Do not implement TDOA generation.

Do not implement direct TDOA mapping.

Instead, adapt the idea to embedding clustering:

embedding cluster
-> original pulse index / original PDW record
-> error diagnosis
-> noise subcluster recovery
-> conservative cluster splitting
-> before/after evaluation

## Main task

Improve source-count estimation by recovering possible emitter clusters from noise points and splitting over-merged clusters.

This branch should focus on:

1. Error diagnosis
2. Noise subcluster recovery
3. Conservative split of high-dispersion clusters
4. Before/after metric comparison

## Required files

Create or update:

- src/cluster_error_analysis.py
- src/source_count_refinement.py
- src/cluster_refinement.py
- run_source_count_refinement.py
- README.md

Reuse existing:

- src/tsrd_loader.py
- src/tsrd_window_dataset.py
- src/model_tcan.py
- src/embedding_extractor.py
- src/clustering_baselines.py
- src/clustering_metrics.py
- src/cluster_diagnostics.py

## Error diagnosis

Implement cluster error diagnosis for analysis only.

Input:

- y_true
- cluster_labels
- embeddings
- pdw_window
- original_indices

Output:

- true_source_count
- estimated_source_count
- noise_ratio
- true-label to cluster contingency table
- cluster to true-label composition table
- noise true-label distribution
- major error type summary

Possible error types:

- missing_as_noise
- over_merged
- over_split
- mixed
- clean

Important:

Ground-truth labels may be used only for diagnosis and metric computation.

Ground-truth labels must not be used for refinement decisions.

## Noise subcluster recovery

Implement:

recover_noise_subclusters(
    cluster_labels,
    embeddings,
    pdw_window,
    original_indices,
    ...
)

Purpose:

Some real emitters may be missed because their pulses are labeled as noise by DBSCAN/HDBSCAN.

This function should:

1. Extract points with cluster label = -1.
2. Run secondary clustering only on the noise points.
3. Accept stable subclusters as new emitter clusters.
4. Assign new cluster IDs to accepted recovered clusters.
5. Leave unstable noise points as -1.

Decision rules must not use ground-truth labels.

Acceptance conditions should include:

- recovered subcluster size >= min_recovered_cluster_size
- embedding compactness below threshold
- PDW distribution consistency within threshold
- recovered subcluster should not simply be a boundary extension of an existing cluster

## Cluster splitting

Implement or improve:

split_dispersion_clusters(
    cluster_labels,
    embeddings,
    pdw_window,
    original_indices,
    ...
)

Purpose:

Some predicted clusters may merge multiple real emitters.

This function should:

1. Identify large and high-dispersion clusters.
2. Run secondary clustering inside those clusters.
3. Accept split only if subclusters are stable and sufficiently large.
4. Leave the original cluster unchanged if splitting is not reliable.

Decision rules must not use ground-truth labels.

Trigger conditions:

- cluster size >= min_split_cluster_size
- cluster compactness > split_compactness_threshold
- secondary clustering produces at least two valid subclusters
- each accepted subcluster size >= min_split_subcluster_size

## Metrics

For each window, compute before and after:

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

Also report:

- recovered_cluster_count
- recovered_point_count
- split_cluster_count
- split_subcluster_count
- remaining_noise_count
- major error type summary

## Command-line script

Create or update:

run_source_count_refinement.py

Arguments:

--tsrd-path
--tsrd-dir
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

--enable-noise-recovery
--enable-split

--noise-recovery-method dbscan/hdbscan
--noise-eps
--noise-min-samples
--min-recovered-cluster-size
--recovery-compactness-threshold
--recovery-pdw-threshold

--split-method dbscan/hdbscan
--split-eps
--split-min-samples
--min-split-cluster-size
--min-split-subcluster-size
--split-compactness-threshold

--output-csv optional

## Output

Print per-window metrics:

Before refinement:
- homogeneity
- completeness
- v_measure
- adjusted_rand_index
- adjusted_mutual_info
- estimated_source_count
- abs_source_count_error
- noise_ratio

After refinement:
- homogeneity
- completeness
- v_measure
- adjusted_rand_index
- adjusted_mutual_info
- estimated_source_count
- abs_source_count_error
- noise_ratio

Also print:

- recovered clusters
- recovered points
- split clusters
- split subclusters
- error type summary

Finally print mean before/after metrics.

## Decision rule

The refinement objective is not to force every noise point into a cluster.

Priority order:

1. Do not significantly reduce V-measure or ARI.
2. Reduce abs_source_count_error.
3. Reduce noise_ratio.
4. Keep refinement conservative.

## Constraints

Do not use true labels to make refinement decisions.

Do not retrain the TCAN encoder in this branch.

Do not implement new metric-learning losses.

Do not implement batch-hard triplet mining.

Do not implement supervised contrastive loss.

Do not implement multi-receiver TDOA processing.

Do not implement SSC-DBSCAN.

Do not commit outputs, checkpoints, or h5 data files.

## Verification

The following should run:

python run_source_count_refinement.py --tsrd-path <local_h5_file> --feature-set 5d --window-size 1024 --max-windows-per-file 3 --checkpoint <checkpoint_path> --cluster-method dbscan --eps <best_eps> --min-samples <best_min_samples> --enable-noise-recovery --enable-split

A multi-file test should also run:

python run_source_count_refinement.py --tsrd-dir E:\Datasets\TSRD\scan\train_scan --file-glob "config_*.h5" --max-files 3 --max-windows-per-file 3 --feature-set 5d --window-size 1024 --checkpoint <checkpoint_path> --cluster-method dbscan --eps <best_eps> --min-samples <best_min_samples> --enable-noise-recovery --enable-split

After implementation, report:

1. files modified
2. error diagnosis output
3. number of recovered clusters
4. number of recovered noise points
5. number of split clusters
6. before/after metrics
7. whether abs_source_count_error decreased
8. whether noise_ratio decreased
9. whether V-measure / ARI remained stable
10. current limitations