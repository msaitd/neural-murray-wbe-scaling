# ADNI NMA CV v4 Locked Pretrained Multi-Resource Internal Validation Protocol

## Status and freeze point

This protocol is frozen after:

- independent validation of the five-seed Run-3 multi-resource NMA result;
- completion of the v3.9-v3.12 random-initialized compact-dense diagnostics;
- the v3.13 audit that closed the failed compact-dense optimization branch;

and before any v4 model is trained or any v4 outer-fold prediction is
obtained.

The study is a prospective, leakage-controlled internal repeated
cross-validation of the already developed pretrained-teacher,
multi-resource NMA procedure. It is not described as a pristine external
confirmation because ADNI participants and earlier fixed-split outcomes have
already informed development. A separately frozen external cohort remains
necessary for an external-validity claim.

## Primary scientific question

Under fixed parameter and MAC ceilings, does multiplicative forward-backward
multi-resource NMA allocate channels more effectively than a matched
proportional-width control when both students are inherited from the same
fold-specific dense teacher and trained under the same recipe?

The primary predictive contrast is participant-level balanced accuracy:

```text
multi-resource NMA minus proportional control
```

The dense teacher is a reference for predictive retention. Dense-teacher
non-inferiority is secondary and uses the previously documented exploratory
absolute margin of 0.03. It is achieved only if the lower confidence bound
for `BA_NMA - BA_dense` is greater than `-0.03`. Failure of this secondary
criterion does not erase a valid NMA-versus-proportional allocation result.

## Locked data design

The source is the five-fold longitudinal, participant-disjoint ADNI1 1.5 T
manifest collection under `adni_work/adni_3d_cv_v2`.

For each outer fold:

- three folds are training;
- one cyclic fold is validation;
- one fold is outer test;
- all available longitudinal visits are eligible only for training
  participants;
- validation and outer-test participants contribute one canonical baseline
  scan;
- no participant crosses a split.

Expected fold sizes are frozen:

| Outer fold | Training rows | Training participants | Validation participants | Test participants |
|---:|---:|---:|---:|---:|
| 1 | 1,146 | 382 | 128 | 129 |
| 2 | 1,149 | 383 | 128 | 128 |
| 3 | 1,152 | 384 | 127 | 128 |
| 4 | 1,155 | 385 | 127 | 127 |
| 5 | 1,149 | 383 | 129 | 127 |

Every ADNI participant must occur in exactly one outer-test fold per seed.

## Leakage-safe initialization

No ADNI-trained Run-3 teacher checkpoint is reused in v4. Reusing it would
leak participants across outer folds.

Each fold-seed cell starts from the immutable raw ImageNet-inflated 3-D
ResNet-18 initialization. The development-stage choice of ImageNet rather
than MedicalNet is frozen, but all ADNI-trained weights are discarded. The
three-class classifier is reset independently under the cell seed before any
fold data are used.

The stage-level seed mapping is fixed before training:

- classifier reset and head-only training: the cell seed;
- dense all-tensor training: the cell seed plus one;
- deterministic calibration GPU generator: the cell seed plus two;
- both compressed-student training runs: the cell seed.

Using the same student seed resets ordering, visit choice, augmentation, and
GPU random state for the paired proportional and NMA students.

## Frozen dense-teacher recipe

The fold-specific dense teacher uses widths `[64,128,256,512]`.

1. one head-only epoch, learning rate `1e-3`;
2. all-tensor training for at most eight epochs;
3. learning rate `2e-5`;
4. cosine minimum learning-rate ratio `0`;
5. hard-label weight `alpha=1`;
6. temperature `4` (inactive when `alpha=1`);
7. L2 coefficient `1e-4`;
8. trainable-gradient norm threshold `5`;
9. validation patience `3`;
10. augmentation enabled;
11. all training visits used;
12. deterministic GPU execution;
13. checkpoint selection by validation balanced accuracy, with macro-F1 as
    an exact-tie breaker.

The dense teacher is trained once per fold and seed and shared by both
compressed candidates in that cell.

## Frozen allocation and controls

### Dense reference

```text
widths = [64,128,256,512]
parameters = 33,166,339
MACs = 13,249,807,872
```

### Proportional control

```text
widths = [52,120,224,440]
parameter ceiling = 24,923,027
MAC ceiling = 9,865,390,056
```

### Multi-resource NMA

For every fold-seed teacher, estimate full-training multiplicative demand:

```text
mean(abs(activation .* gradient))
```

Calibration uses every training-manifest row in its frozen manifest order,
the class weights computed from all training visits, minibatch size one, and
no augmentation. Validation and outer-test rows are excluded before file
paths are passed to the calibrator.

Then solve the frozen parameter-plus-MAC problem with:

- `p=2`;
- resources `parameters` and `macs`;
- exact ceilings `24,923,027` and `9,865,390,056`;
- granularity `[4,8,8,8]`;
- minimum widths equal to one half of each dense stage, tiled upward;
- maximum widths equal to the dense widths;
- deterministic feasible rounding and slack repair;
- multiplicative per-channel salience for inherited channel identities.

Validation or outer-test performance is not an input to allocation. Different
fold-seed cells may produce different feasible NMA widths; the frozen object
of inference is the allocation procedure, not one globally reused ADNI width
vector.

## Frozen compressed-student recipe

Both proportional and multi-resource NMA students are rebuilt directly from
the same cell-specific dense teacher and independently trained with:

- maximum epochs `6`;
- learning rate `2e-5`;
- cosine minimum learning-rate ratio `0.1`;
- hard-loss weight `alpha=0.9`;
- distillation temperature `2`;
- L2 coefficient `1e-4`;
- gradient threshold `5`;
- validation patience `3`;
- augmentation enabled;
- one randomly selected visit per training participant per epoch;
- deterministic GPU execution;
- the same cell seed and stochastic-data protocol;
- validation balanced accuracy as the checkpoint criterion, with macro-F1
  as the exact-tie breaker.

No v3.9-v3.12 compact-dense optimizer intervention is imported into v4.

## Frozen seeds

Five new, no-selection seeds are used:

```text
20260820, 20260821, 20260822, 20260823, 20260824
```

All five folds and five seeds are retained, producing 25 fold-seed cells per
method. A failed process may be repaired and rerun only under the same seed
and unchanged protocol. No seed, fold, or architecture is discarded because
of its metric.

## Staged execution and outer-test firewall

### Phase 0: read-only preflight

Validate manifests, participant disjointness, raw initialization, resource
polynomials, code, storage, and deterministic reconstruction. No model is
trained and no outer-test prediction is computed.

### Phase 1: engineering smoke

Run one fold-seed with reduced epochs solely to verify finite execution,
artifact restartability, allocation feasibility, and identical test sealing.
The exact reductions and pass conditions are frozen separately in
`ADNI_NMA_CV_V4_ENGINEERING_SMOKE_PROTOCOL.md`. Smoke predictive metrics
cannot change the protocol.

### Phase 2: complete validation-only training

Train all 25 dense teachers and all 50 compressed students. Save validation
predictions and complete histories. Execution order is fold-major (`1` to
`5`) and then seed-major (`20260820` to `20260824`). Each head, teacher,
calibration/allocation, and student artifact is written atomically. A stopped
run resumes only by validating and reusing an unchanged artifact; a stale or
conflicting artifact causes a hard stop. No function in this phase may call
`Split="test"`.

### Phase 3: independent pre-test audit

Independently reconstruct manifests, initializations, resources, validation
predictions, and all 75 training artifacts. The audit must confirm that every
fold-seed cell is complete and that no method, seed, fold, or checkpoint was
selected post hoc.

### Phase 4: atomic outer-fold evaluation

Only after Phase 3 passes, evaluate all frozen dense, proportional, and NMA
networks on their corresponding outer folds. Outer predictions are written
as a complete batch; no training or allocation code is callable afterward
under this protocol.

## Primary and secondary endpoints

Primary task endpoint:

- participant-level three-class balanced accuracy.

Primary resource endpoints:

- exact parameter count;
- exact analytical MAC count.

Secondary endpoints:

- macro-F1 and ordinary accuracy;
- AD, CN, and MCI recall;
- macro-AUROC;
- negative log likelihood, Brier score, and expected calibration error;
- confusion matrix;
- checkpoint epoch and convergence history;
- model bytes, activation proxy, peak GPU memory;
- paired median and p95 RTX 5070 latency and throughput.

Energy is not claimed unless a separate synchronized telemetry or wall-power
protocol is completed.

## Statistical analysis

For each seed, concatenate predictions from the five disjoint outer folds so
that every participant contributes one outer prediction per method. Do not
treat the 25 cells as 25 independent samples.

Report:

1. every fold-seed result;
2. per-seed pooled metrics over all participants;
3. mean, standard deviation, median, range, and 95% intervals;
4. paired seed-level differences;
5. a participant-clustered hierarchical bootstrap that preserves method
   pairing and repeated predictions across seeds;
6. fold and seed variance components;
7. the proportion of seeds favoring NMA;
8. Holm-adjusted intervals for confirmatory secondary comparisons.

## Decision tiers

1. **Allocation advantage:** NMA exceeds proportional control on the primary
   paired BA contrast and is not worse on both exact constrained resources.
2. **Dense non-inferior compression:** the lower interval bound for
   `BA_NMA - BA_dense` exceeds `-0.03`, with lower parameters and MACs.
3. **Measured Pareto improvement:** tier 2 plus a statistically supported
   reduction in a directly measured resource such as latency or peak memory.
4. **Architecture-only result:** parameter/MAC reduction is exact, but
   predictive retention or direct-device benefit is not established.

Point estimates alone are not called superiority, equivalence, or
non-inferiority.

## Prohibited actions

- no reuse of any ADNI-trained global teacher as a fold initialization;
- no outer-test call before the atomic Phase-4 release;
- no seed, fold, checkpoint, or class-specific threshold selection;
- no adjustment of budgets, widths, `p`, granularity, loss, learning rate,
  patience, or training length after any v4 metric is observed;
- no soup, probability ensemble, calibration refit, or post-hoc architecture
  repair based on outer predictions;
- no claim that this internal ADNI study is external clinical validation.

## Publication continuation

After the locked ADNI analysis, freeze the allocation procedure and evaluate
it without ADNI-driven retuning on an external cohort such as OASIS or AIBL.
A structured-pruning comparator, a forward-only/backward-only demand
ablation, and a second materially different inference device are required
before a broad resource-allometry claim is submitted to a highly selective
general or machine-intelligence journal.
