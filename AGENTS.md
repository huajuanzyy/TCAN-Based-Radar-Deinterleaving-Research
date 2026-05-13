# AGENTS.md

## Project

This repository reproduces the radar pulse deinterleaving method from:

"Deinterleaving of Intercepted Radar Pulse Streams via Temporal Convolutional Attention Network"

The implementation uses Python and PyTorch.

## Current branch

feat/nonideal-conditions

## Current objective

This branch implements Phase 5: nonideal receiving conditions for the TCAN radar pulse deinterleaving pipeline.

Previous phases have already been implemented and merged into main:

- Phase 1: DTOA input
- Phase 2: Binary input
- Phase 3: Focal loss
- Phase 4: Signal sparsity

Do not rewrite the existing DTOA pipeline, binary pipeline, TCAN model, focal loss, or signal sparsity unless necessary.

## Main task for this branch

Add three nonideal conditions:

1. Measurement error
2. Random pulse loss
3. Spurious pulses

These conditions should be applied at the PDW level before DTOA or binary preprocessing.

The pipeline should be:

1. Generate full PDW pulse stream.
2. Optionally apply signal sparsity.
3. Apply nonideal conditions.
4. Apply DTOA or binary preprocessing.
5. Train and evaluate TCAN.

## Important distinction

Signal sparsity is not random pulse loss.

Signal sparsity removes continuous time intervals due to periodic visibility caused by radar scanning.

Random pulse loss randomly removes individual pulses.

Do not mix these two mechanisms.

## Required implementation

Create or update:

- src/nonideal.py
- train.py
- README.md

If needed, small changes are allowed in:

- src/data_simulator.py
- src/preprocessing.py
- src/metrics.py
- src/utils.py

## Measurement error

Implement:

apply_measurement_error(
    pdw_array,
    toa_std=0.0,
    pw_std=0.0,
    rf_std=0.0,
    doa_std=0.0,
    seed=None
)

The function should add Gaussian noise to:

- TOA
- PW
- RF
- DOA

PW must be clipped to remain positive.

After adding TOA noise, sort the pulses by TOA again and reorder labels consistently.

## Random pulse loss

Implement:

apply_random_pulse_loss(
    pdw_array,
    labels,
    loss_rate,
    seed=None
)

The function should randomly delete individual pulses.

This is different from signal sparsity.

## Spurious pulses

Implement:

apply_spurious_pulses(
    pdw_array,
    labels,
    spurious_rate,
    spurious_label,
    seed=None
)

The function should insert false pulses into the PDW stream.

Spurious pulse TOA should be sampled within the observation interval.

Spurious PW/RF/DOA can be sampled from the global PDW ranges or from configurable ranges.

After insertion, sort by TOA and reorder labels consistently.

## Label design for spurious pulses

Use an extra spurious class.

For DTOA input:

- radar classes: 0, 1, 2, 3
- spurious class: 4

For binary input:

- background class: 0
- radar classes: 1, 2, 3, 4
- spurious class: 5

Make sure class labels are continuous and compatible with CrossEntropyLoss and FocalLoss.

## Command-line arguments

Update train.py to support:

--toa-error-std
--pw-error-std
--rf-error-std
--doa-error-std

--pulse-loss-rate

--spurious-rate

Default values should disable these conditions:

--toa-error-std 0.0
--pw-error-std 0.0
--rf-error-std 0.0
--doa-error-std 0.0
--pulse-loss-rate 0.0
--spurious-rate 0.0

Existing commands without nonideal options must still run.

## Training output requirements

Print:

1. selected input format
2. selected loss function
3. selected sparsity ratio
4. measurement error standard deviations
5. pulse loss rate
6. spurious pulse rate
7. pulse count before nonideal conditions
8. pulse count after measurement error
9. pulse count after pulse loss
10. pulse count after spurious insertion
11. input tensor shape [B, T, D]
12. label tensor shape [B, T]
13. model output tensor shape [B, T, C]
14. number of classes
15. class counts
16. recall per class
17. average recall
18. pulse-only average recall for binary input
19. confusion matrix

## Verification criteria

This branch is complete only when the following smoke tests run successfully:

```bash
python train.py --input-format dtoa --loss focal --epochs 2 --toa-error-std 2.0
python train.py --input-format dtoa --loss focal --epochs 2 --pulse-loss-rate 0.10
python train.py --input-format dtoa --loss focal --epochs 2 --spurious-rate 0.10

python train.py --input-format binary --loss focal --epochs 2 --toa-error-std 2.0
python train.py --input-format binary --loss focal --epochs 2 --pulse-loss-rate 0.10
python train.py --input-format binary --loss focal --epochs 2 --spurious-rate 0.10

Also verify that the original non-noisy commands still work:

python train.py --input-format dtoa --loss focal --epochs 2
python train.py --input-format binary --loss focal --epochs 2
Constraints

Do not implement MFR simulation in this branch.

Do not implement CDIF, SDIF, PRI Transform, GRU, LSTM, or TCN baselines in this branch.

Do not generate complex comparison plots in this branch.

Do not rewrite TCAN architecture.

The goal is to add nonideal condition simulation cleanly and verify that existing pipelines still work.

Git

Do not commit automatically unless asked.

After implementation, report:

files modified
how each nonideal condition is implemented
where in the pipeline nonideal conditions are applied
how spurious pulse labels are handled
smoke test commands used
current limitations