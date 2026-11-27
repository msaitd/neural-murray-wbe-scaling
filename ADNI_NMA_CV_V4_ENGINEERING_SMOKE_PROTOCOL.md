# ADNI NMA CV v4 Engineering Smoke Protocol

## Status

This engineering protocol is frozen after the successful v4 Phase-0
preflight and before any v4 network is trained. It exercises one
validation-only fold-seed cell and cannot be used for predictive inference,
model selection, or modification of the locked full protocol.

## Cell and data firewall

- fold: `1`;
- seed: `20260820`;
- training rows: `1,146` from `382` participants;
- validation rows: `128` baseline scans from `128` participants;
- outer-test rows: `129` baseline scans from `129` participants;
- outer-test prediction: prohibited;
- global ADNI-trained teacher reuse: prohibited.

The raw ImageNet-inflated 3-D ResNet-18 initialization is loaded, its
three-class head is reset with seed `20260820`, and only Fold 1 training data
may update the network.

## Reduced engineering schedule

The smoke deliberately reduces computation while retaining every code path:

1. dense head-only training: one epoch, seed `20260820`;
2. dense all-tensor supervised training: one epoch, seed `20260821`;
3. multiplicative activation-gradient calibration: twelve Fold 1 training
   baseline scans, four per class, selected deterministically after sorting
   by participant and image identifier;
4. proportional student training: one epoch, seed `20260820`;
5. multi-resource NMA student training: one epoch, seed `20260820`.

The dense stage otherwise uses the locked learning rate, hard-label loss,
regularization, clipping, augmentation, and all-visit sampling settings. Both
students otherwise use the locked low-KD recipe and one-visit-per-participant
sampling. The full study remains fixed at eight dense epochs, six student
epochs, and calibration over every Fold 1 training visit.

## Allocation contract

The proportional control is fixed at `[52,120,224,440]`. The engineering NMA
allocation is solved from the twelve-scan training-only calibration with:

- resources: parameters and MACs;
- ceilings: `24,923,027` parameters and `9,865,390,056` MACs;
- estimator: multiplicative;
- exponent: `p=2`;
- granularity: `[4,8,8,8]`;
- minimum widths: `[32,64,128,256]`;
- maximum widths: `[64,128,256,512]`.

The resulting smoke NMA widths are engineering output, not frozen full-study
widths. Every full fold-seed cell must recalibrate on all of its own training
visits.

## Required pass conditions

- all losses, gradients, validation scores, and resource values are finite;
- both compressed allocations satisfy both exact ceilings;
- analytical and actual parameter counts agree;
- saved networks reproduce their saved validation predictions;
- every checkpoint is atomically written and can be loaded without retraining;
- a second identical invocation reuses all five stage artifacts unchanged;
- no training, calibration, allocation, or evaluation function reads the
  outer-test split;
- the independent smoke validator passes.

Smoke accuracy, balanced accuracy, macro-F1, and class recall must be reported
only as execution diagnostics. They cannot alter the full v4 protocol.
