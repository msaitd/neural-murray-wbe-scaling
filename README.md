# Neural Murray-WBE scaling: forward-backward transport laws for resource-constrained deep neural architectures

[![MATLAB R2025b](https://img.shields.io/badge/MATLAB-R2025b-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![Code only](https://img.shields.io/badge/release-code%20only-blue.svg)](#-data-availability-read-first)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Reproducible, data-free source code for **Neural Murray-WBE scaling**, a framework that treats a
trainable neural computation graph as a capacity-limited transport network. Forward activation demand
and backward loss sensitivity are combined, constrained channel widths are allocated under exact
parameter and multiply-accumulate (MAC) ceilings, graph compatibility is preserved during network
surgery, and the resulting architectures are evaluated under a locked three-dimensional ResNet-18
workflow.

> **Central finding.** In the locked internal ADNI proof of concept, multi-resource Neural Murray
> Allocation (NMA) reduced mean parameter count by 25.47%, analytical MACs by 30.51%, the activation
> proxy by 18.17%, and paired RTX 5070 model-only median latency by 9.77% relative to the dense
> reference while satisfying the prespecified dense-reference balanced-accuracy non-inferiority
> margin. NMA did **not** establish accuracy superiority over the resource-matched proportional
> control, and absolute discrimination remained near chance; the results do not support clinical use.

---

## ⚠️ Data availability (read first)

**This repository contains code and protocol text only. It contains NO ADNI images, metadata rows,
participant manifests, predictions, derived participant-level tables, checkpoints, trained weights,
or raw timing matrices.**

The magnetic resonance images and metadata used in the paper are controlled-access data from the
[Alzheimer's Disease Neuroimaging Initiative](https://adni.loni.usc.edu/). Qualified researchers must
obtain their own authorization through the LONI Image and Data Archive and comply with the current
ADNI Data Use Agreement. Do not open a GitHub issue containing participant-level data, local ADNI
paths, screenshots, logs, or identifiers.

The public code can reproduce the analytic allocator, exact resource accounting, network construction,
synthetic smoke example, and the locked computation logic. Reproducing the participant-level experiment
requires separately authorized ADNI inputs. The original Phase-0 audit also depends on controlled-data-
derived predecessor audit artifacts that are not redistributed; they must not be fabricated or replaced
with post hoc substitutes.

---

## Repository structure

```text
README.md                                      Study overview and reproduction guide.
LICENSE                                        MIT license for code only.
CITATION.cff                                   GitHub citation metadata.
REPOSITORY_METADATA.md                         Proposed GitHub name, title, description, and topics.
DATA_ACCESS.md                                 ADNI access and non-redistribution boundary.
validate_release.py                            Code-only and path/identifier safety audit.
adni_inventory.py                              Match an authorized metadata export to local NIfTI files.
adni_build_subject_splits.py                   Deterministic participant-level baseline split.
adni_build_longitudinal_training_manifest.py   Longitudinal training-only visit selection.
adni_build_5fold_manifests.py                  Participant-stratified five-fold baseline manifests.
adni_build_cv_v2_manifests.py                  Frozen three-visit v4 fold manifests.
adni_validate_3d.py                            Baseline 3D materialization audit.
adni_validate_longitudinal.py                  Longitudinal materialization audit.
run_adni_resnet3d18_pipeline.ps1               V4-only preflight, smoke, training, and audit runner.
run_adni_resnet3d18_cv_v4_phase3_recovery.ps1  Frozen Phase-3 recovery route.
run_adni_resnet3d18_cv_v4_phase4.ps1           Atomic outer-test release and analysis.
run_adni_resnet3d18_cv_v4_paired_rtx5070.ps1   Fresh-process paired latency benchmark.
matlab/                                        Allocator, 3D ResNet, training, audit, and analysis code.
docs/LOCKED_RESULTS_SUMMARY.md                  Publication-safe aggregate result summary.
ADNI_NMA_CV_V4_*.md                            Immutable locked protocol documents.
```

## Requirements

- MATLAB R2025b, matching the reported run.
- Deep Learning Toolbox.
- Deep Learning Toolbox Model for ResNet-18 Network.
- Medical Imaging Toolbox.
- Image Processing Toolbox.
- Parallel Computing Toolbox for GPU execution.
- Optimization Toolbox for the coupled multi-resource allocator.
- Python 3.10 or later; the included Python scripts use only the standard library.
- PowerShell 7 or Windows PowerShell 5.1 for the supplied orchestration scripts.
- A CUDA-capable GPU for the locked training and paired benchmark. The reported device result is
  specific to the stated RTX 5070/MATLAB stack and is not a cross-device latency claim.

## Installation

```powershell
git clone https://github.com/msaitd/neural-murray-wbe-scaling.git
cd neural-murray-wbe-scaling
python .\validate_release.py
```

No Python package installation is required. Install the MATLAB products and the ResNet-18 support
package through MathWorks before running the network code.

## Quick start without ADNI data

The following example evaluates only the analytic single-resource allocator and graph-group projection.
It is a software smoke test, not empirical evidence for the paper.

```powershell
matlab -batch "addpath('matlab'); run('matlab/nma_demo_allocator.m')"
matlab -batch "run('matlab/audit_neural_murray_core_code.m')"
```

The smoke example asserts budget feasibility and equality of widths tied by residual compatibility
groups. The core audit independently checks the exact dense and proportional parameter, MAC, and
activation-proxy values reported in the study.

## Preparing authorized ADNI inputs

Only researchers who already have valid ADNI access should run this section. Supply paths on the
command line; the repository contains no personal-machine or ADNI source paths.

```powershell
python .\adni_inventory.py `
  --image-root "X:\path\to\authorized\ADNI\NIfTI" `
  --metadata-csv "X:\path\to\authorized\ADNI_export.csv"
python .\adni_build_subject_splits.py
```

Materialize the fixed 96 × 96 × 96 volumes and validate them:

```powershell
matlab -batch "addpath('matlab'); adni_prepare_3d_dataset('adni_work/adni_baseline_subject_split.csv','adni_work/adni_3d',OutputSize=[96 96 96],Verbose=true)"
python .\adni_validate_3d.py
python .\adni_build_longitudinal_training_manifest.py
matlab -batch "addpath('matlab'); adni_prepare_3d_dataset('adni_work/adni_longitudinal_training_split.csv','adni_work/adni_3d_longitudinal',OutputSize=[96 96 96],Verbose=true)"
python .\adni_validate_longitudinal.py
python .\adni_build_5fold_manifests.py
python .\adni_build_cv_v2_manifests.py --stage plan
matlab -batch "run('matlab/run_adni_prepare_cv_v2_missing_3d.m')"
python .\adni_build_cv_v2_manifests.py --stage finalize
```

All generated manifests and volumes remain under ignored `adni_work/` paths. Review the generated QA
summaries before training; do not commit any of these files.

## Creating the raw ImageNet-inflated 3D initialization

```powershell
matlab -batch "run('matlab/create_adni_resnet3d18_imagenet_initialization.m')"
```

This creates an immutable `adni_work/adni_resnet3d18_imagenet_initialization.mat` containing the
three-dimensional ResNet-18, its inflation report, and exact analytical resource counts. The helper
refuses to overwrite an existing initialization.

## Reproducing the locked v4 workflow

The exact original Phase-0 preflight requires the predecessor selection and branch-closure audit
artifacts named in the locked protocol. They contain or depend on controlled ADNI analyses and are not
public. An authorized reproducer who possesses the complete audit lineage can execute:

```powershell
.\run_adni_resnet3d18_pipeline.ps1 -Stage cv4internalpreflight
.\run_adni_resnet3d18_pipeline.ps1 -Stage cv4smokepreflight
.\run_adni_resnet3d18_pipeline.ps1 -Stage cv4smoke
.\run_adni_resnet3d18_pipeline.ps1 -Stage cv4fullpreflight
.\run_adni_resnet3d18_pipeline.ps1 -Stage cv4full
.\run_adni_resnet3d18_cv_v4_phase3_recovery.ps1
.\run_adni_resnet3d18_cv_v4_phase4.ps1 -Stage all
.\run_adni_resnet3d18_cv_v4_paired_rtx5070.ps1 -Stage all
```

Pass `-MatlabPath "C:\path\to\matlab.exe"` if MATLAB is not available on `PATH`. Training is
computationally expensive: the locked grid retains five folds, five no-selection seeds, 25 dense
teachers, and 50 compressed students.

### Which code is used where

| Analysis component | Principal source |
|---|---|
| Closed-form separable allocation and tile repair | `matlab/nma_allocate_single_resource.m` |
| Forward/backward demand probing | `matlab/nma_probe_demand.m` |
| Multi-resource parameter/MAC allocation | `matlab/nma_resnet3d18_allocate_multiresource.m` |
| Exact 3D ResNet resource polynomials | `matlab/nma_resnet3d18_resource.m` |
| Graph-compatible inherited-channel reconstruction | `matlab/nma_resnet3d18_rebuild.m` |
| Locked fold-seed training cell | `matlab/adni_run_resnet3d18_cv_v4_validation_cell.m` |
| Independent pre-test reconstruction audit | `matlab/adni_audit_resnet3d18_cv_v4_validation_cell_recovery.m` |
| Atomic participant-level outer evaluation | `matlab/run_adni_resnet3d18_cv_v4_atomic_outer_test.m` |
| Hierarchical predictive analysis | `matlab/analyze_adni_resnet3d18_cv_v4_outer_test.m` |
| Paired fresh-process latency measurement | `matlab/run_adni_resnet3d18_cv_v4_benchmark_cell.m` |
| Crossed fold-by-seed device bootstrap | `matlab/aggregate_adni_resnet3d18_cv_v4_paired_rtx5070.m` |

## Leakage and circularity controls

- **Participant-disjoint folds** — every participant appears in one outer fold and never crosses a
  training, validation, or test boundary within a fold.
- **Training-only allocation** — activation-gradient demand is estimated only from the fold's training
  manifests; validation and test inputs are not passed to the allocator.
- **Fold-local initialization** — no ADNI-trained global teacher is reused across outer folds.
- **Shared paired teacher** — proportional and NMA students inherit from the same fold-seed dense
  teacher and use matched training seeds and recipes.
- **Atomic outer release** — all 75 frozen model evaluations are released together only after the
  independent Phase-3 audit.
- **No selective exclusion** — all five folds and all five prespecified seeds are retained.
- **Device pairing** — each benchmark cell loads all three networks in a fresh MATLAB process and uses
  randomized, interleaved, GPU-synchronized timing blocks.

## Reproducibility and interpretation note

The v4 programs preserve the exact seeds, budgets, width granularity, dense and proportional controls,
training recipes, statistical decisions, and hardware-timing procedure used for the reported release.
The public package intentionally excludes the controlled inputs and intermediate/result artifacts.
Consequently, a fresh ADNI download can differ in archive version or file organization, and an exact
bitwise rerun additionally requires the original authorized snapshot and audit lineage.

Parameter reduction does not mathematically guarantee latency reduction. Latency in the paper was a
separate directly measured endpoint and remains specific to one MATLAB/GPU stack. The theoretical
fixed-demand optimum is not an unconditional convergence or accuracy guarantee for the endogenous
train-allocate-fine-tune loop.

## Citation

> Dündar, M. S. (2026). *Neural Murray-WBE scaling: forward-backward transport laws for
> resource-constrained deep neural architectures.* Manuscript in preparation.

The citation will be updated with the journal and DOI after publication. GitHub can also generate a
software citation from [`CITATION.cff`](CITATION.cff).

## License

Code is released under the [MIT License](LICENSE). This license covers **the source code only**. ADNI
images, metadata, derived participant-level material, MATLAB, MathWorks support packages, and any
third-party model weights remain governed by their respective licenses and access terms and are not
redistributed here.

## Author

**Mehmet Sait Dündar**, Medical Imaging Techniques, Halil Bayraktar Health Services Vocational School,
Erciyes University, Kayseri, Türkiye.  
ORCID: [0000-0002-0336-4825](https://orcid.org/0000-0002-0336-4825)  
Email: [msaitdundar@erciyes.edu.tr](mailto:msaitdundar@erciyes.edu.tr)

## Acknowledgement

Data used in the associated study were obtained from ADNI. ADNI investigators contributed to the
design and implementation of ADNI and/or provided data but did not participate in this analysis or
writing. The current official ADNI acknowledgement and publication-review requirements apply to any
manuscript produced with those controlled data.
