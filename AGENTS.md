# AGENTS.md

## Project

This repository targets radar pulse deinterleaving under an unknown number of emitters.

The long-term target pipeline is:

1. Load TSRD pulse trains.
2. Split pulse trains into fixed-length pulse windows.
3. Extract pulse-level contextual embeddings using TCAN or Transformer encoder.
4. Cluster embeddings using DBSCAN, HDBSCAN, or hierarchical clustering.
5. Estimate emitter count.
6. Evaluate clustering with V-measure, ARI, AMI, Homogeneity, Completeness, and source-count error.

## Current branch

feat/tcan-embedding-extraction

## Current objective

Implement Phase 2A: TCAN encoder embedding extraction and embedding clustering scaffold.

Phase 1A TSRD loader has already been implemented.

Phase 1B TSRD windowing and raw-feature clustering baseline has already been implemented.

This branch should not implement triplet loss yet.

This branch should not implement supervised contrastive loss yet.

This branch should not implement post-processing such as cluster merging or splitting.

## Main task

Add a TCAN encoder mode that outputs pulse-level embeddings instead of fixed-class logits.

The goal is to connect:

TSRD window
-> feature construction
-> TCAN encoder
-> pulse embeddings
-> clustering baseline
-> clustering metrics

## Required implementation

Create or update:

- src/model_tcan.py
- src/embedding_extractor.py
- run_tsrd_embedding_clustering.py
- README.md

Reuse existing:

- src/tsrd_loader.py
- src/tsrd_window_dataset.py
- src/clustering_baselines.py
- src/clustering_metrics.py

## Embedding output

The TCAN encoder should accept:

X: [B, T, D]

and return:

embeddings: [B, T, E]

where:

- B is batch size
- T is pulse-window length
- D is input feature dimension
- E is embedding dimension

The embedding dimension should be configurable, default:

--embedding-dim 64

## Important design

Do not remove the existing TCAN classification pipeline.

Add encoder functionality without breaking existing code.

Possible implementation options:

1. Add a `return_embeddings` flag to TCAN forward.
2. Split TCAN into encoder and classifier head.
3. Add a wrapper class TCANEncoder.

Choose the cleanest option while preserving backward compatibility.

## Clustering script

Create:

run_tsrd_embedding_clustering.py

Arguments:

--tsrd-path
--feature-set 4d/5d
--window-size
--stride
--max-windows
--embedding-dim
--method dbscan/hdbscan/agglomerative_oracle
--eps
--min-samples
--min-cluster-size
--checkpoint optional

For this branch, if no checkpoint is provided, allow random initialized embeddings for pipeline testing, but clearly print a warning.

If a checkpoint is provided, load model weights if compatible.

## Metrics

Use existing clustering metrics:

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

Do not implement triplet loss in this branch.

Do not implement supervised contrastive loss in this branch.

Do not implement HDBSCAN-specific tuning in this branch.

Do not implement cluster merging/splitting in this branch.

Do not train a deep embedding model in this branch.

Do not commit datasets or outputs.

## Verification

The following should run:

python run_tsrd_embedding_clustering.py --tsrd-path <local_h5_file> --feature-set 5d --window-size 1024 --max-windows 3 --embedding-dim 64 --method dbscan

python run_tsrd_embedding_clustering.py --tsrd-path <local_h5_file> --feature-set 5d --window-size 1024 --max-windows 3 --embedding-dim 64 --method agglomerative_oracle

Expected result:

- The script loads TSRD windows.
- The TCAN encoder outputs embeddings with shape [T, E].
- Clustering runs on embeddings.
- Clustering metrics are printed.

It is acceptable if metrics are poor when the encoder is randomly initialized. The goal of this branch is pipeline construction, not final performance.

## Git

Do not commit automatically unless asked.

After implementation, report:

1. files modified
2. how TCAN encoder embeddings are produced
3. embedding shape
4. how clustering uses embeddings
5. smoke test results
6. next step toward triplet loss or supervised contrastive loss