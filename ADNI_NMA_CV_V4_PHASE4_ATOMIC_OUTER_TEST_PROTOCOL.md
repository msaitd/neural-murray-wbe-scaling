# ADNI NMA CV v4 Phase-4 Atomic Outer-Test Execution Protocol

## Relationship to the locked protocol

This document fixes implementation details for Phase 4 of
`ADNI_NMA_CV_V4_LOCKED_INTERNAL_VALIDATION_PROTOCOL.md`. It is created after
the complete Phase-3 recovery authorization passed and before any v4 outer-test
network prediction is computed. It does not change any model, seed, fold,
allocation, checkpoint, training hyperparameter, or endpoint.

## Entry gate

Phase 4 may start only if both of the following atomic artifacts pass and their
complete input metadata remain unchanged:

- `pretest_audit/v4_validation_training_pretest_audit.mat`;
- `pretest_audit/v4_phase3_recovery_authorization.mat`.

The gate must confirm 25 audited fold-seed cells, 75 validated training
artifacts, 75 reproduced validation prediction sets, no post-hoc selection,
and zero prior outer-test evaluations.

## Atomic evaluation unit

The release unit is the complete `5 folds x 5 seeds x 3 methods` grid. The
execution order remains fold-major, seed-major, then method order:

1. dense;
2. proportional;
3. multi-resource NMA.

All 75 test evaluations are retained in memory. No outer prediction, metric,
or partial result is written during the loop. Only after the following
conditions pass is the complete batch written atomically:

- exactly 75 evaluations completed;
- exactly 9,585 prediction rows exist;
- for each seed and method, the five folds concatenate to exactly 639 unique
  participants;
- participant truth labels agree across methods and seeds;
- every method uses its corresponding frozen fold-seed checkpoint;
- all checkpoint, audit, manifest, and source stamps remain unchanged.

A process failure before atomic save requires a complete rerun under unchanged
code. It may not produce a selectable partial result.

## Endpoints stored at release

For every fold-seed-method cell the release stores:

- participant predictions and three class scores;
- accuracy, balanced accuracy, macro-F1, and AD/CN/MCI recall;
- macro-AUROC, negative log likelihood, Brier score, and 10-bin ECE;
- confusion matrix;
- widths, exact learnable parameters, analytical MACs, activation proxy, and
  exact learnable-plus-state tensor bytes;
- selected checkpoint epoch and number of executed epochs;
- immutable checkpoint path, size, and modification stamp.

Per-cell bootstrap intervals are not used. Inference is performed on the
participant-pooled repeated-seed design below.

## Statistical analysis frozen before release

For each seed, predictions from all five disjoint test folds are concatenated,
giving one prediction for each of 639 participants per method. Cells are not
treated as 25 independent observations.

The primary confirmatory contrast is:

```text
balanced accuracy: multi-resource NMA minus proportional control
```

The dense-reference secondary non-inferiority contrast is:

```text
balanced accuracy: multi-resource NMA minus dense teacher
margin = -0.03
```

The analysis reports per-cell results, per-seed pooled metrics, seed mean,
standard deviation, median, range, exact df=4 t intervals, paired seed-level
differences, and the proportion of seeds favoring NMA.

The primary uncertainty analysis uses 5,000 deterministic hierarchical
bootstrap replicates with seed `20260825`. Each replicate:

1. resamples participants with replacement within each true class;
2. keeps every selected participant's predictions paired across methods and
   repeated seeds;
3. resamples the five seeds with replacement;
4. computes each metric within every selected seed and averages the five
   paired seed contrasts.

The two-sided percentile 95% interval is primary. NMA allocation superiority
requires the lower bound of the primary interval to exceed zero. Dense
non-inferiority requires the lower bound of the NMA-minus-dense balanced
accuracy interval to exceed `-0.03`.

The confirmatory secondary NMA-minus-proportional family contains accuracy,
macro-F1, three class recalls, macro-AUROC, NLL, Brier score, and ECE. For NLL,
Brier, and ECE, signs are oriented so positive means improvement. Two-sided
bootstrap p-values receive Holm step-down adjustment; rank-specific percentile
intervals use the corresponding Holm alpha threshold and are reported
explicitly as step-down intervals.

Two-way method-of-moments fold, seed, and residual fold-by-seed variance
components are reported for balanced accuracy and the two prespecified paired
balanced-accuracy contrasts. Negative component estimates are truncated at
zero and labeled as such.

## Decision rules

- Allocation advantage: the primary hierarchical-bootstrap lower bound is
  greater than zero and every NMA checkpoint is no worse than the proportional
  ceiling in both exact parameters and MACs.
- Dense non-inferior compression: the hierarchical-bootstrap lower bound for
  NMA minus dense balanced accuracy is greater than `-0.03`, and every NMA
  checkpoint has fewer parameters and MACs than dense.
- Measured Pareto improvement requires the dense non-inferior-compression tier
  plus a separately preflighted paired device benchmark whose upper confidence
  bound demonstrates lower latency or another directly measured resource.
- Otherwise, exact parameter/MAC reduction alone is reported as an
  architecture-only result, without an accuracy-retention claim.

## Resource benchmark boundary

Paired RTX 5070 median latency, p95 latency, and throughput will be measured in
a separately stamped, fold-seed-paired benchmark after the atomic predictive
release. Windows WDDM does not expose reliable per-process peak VRAM through
`nvidia-smi`; therefore retained GPU allocation from MATLAB may be reported as
a proxy but must not be mislabeled as exact peak memory. Energy is not claimed.

## Prohibitions after entry

- no training, recalibration, allocation, reconstruction, soup, ensembling,
  threshold tuning, checkpoint selection, or architecture repair;
- no exclusion of a fold, seed, participant, method, or failed metric;
- no editing of stamped Phase-4 source or protocol files;
- no external-validation or clinical-utility claim from this internal ADNI
  experiment.

