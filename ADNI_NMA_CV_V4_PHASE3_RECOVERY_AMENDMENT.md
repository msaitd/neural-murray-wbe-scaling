# ADNI NMA CV v4 Phase-3 Recovery Amendment

## Scope

This amendment repairs a non-statistical MATLAB string-construction defect in
the independent Phase-3 audit path. It does not alter, retrain, select, or
evaluate any dense, proportional, or multi-resource NMA network. It does not
authorize access to the outer-test images or predictions.

## Triggering failure

The complete locked Phase-2 run finished all 25 fold-by-seed cells (25 dense
teachers and 50 compressed students) and produced its atomic validation-only
aggregate. The first Phase-3 audit process then stopped before loading a
network, calibration batch, validation image, or outer-test image. MATLAB
reported `Error using vertcat: Dimensions of arrays being concatenated are not
consistent` while constructing `inputPaths`.

The defect was caused by this expression in the original auditor and aggregate
auditor:

```matlab
string([mfilename("fullpath") ".m"])
```

Square-bracket concatenation creates two string elements rather than one file
path. The corrected expression is:

```matlab
string(mfilename("fullpath"))+".m"
```

The same MATLAB string-array construction was also present in one terminal
status message in the cell auditor and one in the aggregate auditor. Those
messages execute only after their atomic artifacts are saved, but they would
still return a nonzero process status. Their format strings are therefore
joined with scalar string concatenation (`+`) instead of square brackets.

The controlled first-cell recovery run then exposed a second audit-only schema
defect before recalibration or prediction reproduction. The auditor applied the
distillation trainer's `BestValidation` contract to the one-epoch head artifact,
although the head is produced by `adni_train_resnet3d18`, whose locked output
schema contains `Info` and does not contain `BestValidation`. The original
training producer and its own artifact validator confirm this distinction. The
recovery auditor therefore replaces that impossible assertion with explicit
checks for the correct head schema: `Info` must exist and `BestValidation` must
not exist. Dense-teacher and student `BestValidation` checks remain unchanged.

## Integrity controls

1. Every file stamped by the original Phase-0 and Phase-2 preflights remains
   byte- and timestamp-identical. The original audit sources are retained and
   are not edited.
2. Recovery copies are accepted only when their complete text equals the
   original source after the explicitly enumerated substitutions above (and,
   for the function/runner pair, the necessary recovery function-name change).
3. The recovery preflight independently validates the completed Phase-2
   aggregate, its original preflight stamp, the 25 completed cells, and all
   original preflight input metadata.
4. Static analysis must pass for every recovery MATLAB source. Recovery sources
   are rejected if they contain an outer-test evaluator call.
5. Each recovered cell audit records the recovery auditor itself in its input
   metadata. The aggregate audit records the recovery aggregate source and all
   25 cell-audit artifacts.
6. A separate atomic recovery authorization artifact links the recovery
   preflight to the completed 25-cell audit. Phase 4 must require both that
   authorization and the canonical Phase-3 audit artifact.
7. No training or allocation routine is callable after Phase 4 begins. Outer
   test evaluation remains sealed until the recovery authorization passes.

## Permitted code substitutions

- Auditor: rename the top-level function to the recovery name, replace the
  malformed `auditSourceFile` expression, and scalarize its terminal status
  format string. In the head-only validator, replace only the inapplicable
  distillation `BestValidation` assertion with the correct producer-schema
  assertions described above.
- Audit-cell runner: replace only the called auditor function name.
- Aggregate auditor: replace the malformed `sourceFile` expression and
  scalarize its terminal status format string.

Any additional difference invalidates this amendment and requires a new,
explicitly documented protocol change.
