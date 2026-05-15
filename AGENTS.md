# AGENTS.md

## Current branch

feat/multifile-validation

## Current objective

Implement Phase 4A: multi-file train/validation evaluation for the random-triplet TCAN embedding pipeline.

Previous experiments showed that batch-hard triplet mining performed much worse than random triplet mining. Therefore, the current mainline should keep random triplet as the default metric-learning strategy.

Current goal:

Train the random-triplet TCAN encoder on multiple TSRD files and evaluate on different held-out TSRD files.

## Data split

Use local TSRD scan-mode files:

Training files:
- config_0.h5
- config_1.h5
- config_2.h5
- config_3.h5
- config_4.h5

Evaluation files:
- config_5.h5
- config_6.h5
- config_7.h5
- config_8.h5
- config_9.h5

Do not assume that the same integer label across different files represents the same physical emitter.

Triplet sampling must remain within a valid local context, such as the same window or same pulse train.

## Main task

Add multi-file support for training and evaluation.

The pipeline should be:

Training:
TSRD train files
-> fixed-length windows
-> TCAN encoder
-> random triplet loss
-> save checkpoint

Evaluation:
TSRD eval files
-> fixed-length windows
-> raw feature clustering
-> triplet embedding clustering
-> clustering metrics
-> mean metrics by method

## Required changes

Create or update:

- train_tsrd_triplet.py
- run_embedding_evaluation.py
- src/evaluation_runner.py
- src/result_writer.py
- README.md

Optional:

- configs/train_files_scan_small.txt
- configs/eval_files_scan_small.txt

## Required command-line arguments

Support:

--file-list
--data-root
--max-files
--max-windows-per-file

Existing single-file arguments must continue to work.

## Evaluation methods

Compare:

- raw
- triplet_embedding

Optional:

- random_embedding

## Metrics

Report:

- homogeneity
- completeness
- v_measure
- adjusted_rand_index
- adjusted_mutual_info
- true_source_count
- estimated_source_count
- abs_source_count_error
- noise_ratio

Print:

- per-window metrics
- per-file mean metrics
- overall mean metrics by method

## Constraints

Do not implement batch-hard triplet in this branch.

Do not implement supervised contrastive loss.

Do not implement new post-processing rules.

Do not use true labels for clustering or refinement decisions.

Do not commit checkpoints, outputs, or h5 files.

## Verification

The following should run:

python train_tsrd_triplet.py --data-root E:\Datasets\TSRD --file-list configs/train_files_scan_small.txt --feature-set 5d --window-size 1024 --max-windows-per-file 10 --embedding-dim 64 --epochs 5 --triplet-mining random

python run_embedding_evaluation.py --data-root E:\Datasets\TSRD --file-list configs/eval_files_scan_small.txt --feature-set 5d --window-size 1024 --max-windows-per-file 5 --cluster-method dbscan --methods raw,triplet_embedding --checkpoint <checkpoint_path>

After implementation, report:

1. files modified
2. train files used
3. evaluation files used
4. checkpoint path
5. raw vs triplet_embedding mean metrics
6. whether triplet_embedding generalizes to held-out files
7. current limitations