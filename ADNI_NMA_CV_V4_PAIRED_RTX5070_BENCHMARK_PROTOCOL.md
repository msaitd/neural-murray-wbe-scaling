# ADNI NMA CV v4 Paired RTX 5070 Benchmark Protocol

## Status and purpose

This device protocol is frozen after the independently validated atomic v4
outer-test release and before any confirmatory v4 device timing is measured.
The predictive result cannot change this benchmark design. The purpose is to
test whether exact parameter/MAC compression produces a directly measured
batch-one inference benefit on the configured NVIDIA GeForce RTX 5070.

## Benchmark unit and pairing

All 25 fold-seed cells are retained. Each cell loads its corresponding frozen
dense, proportional, and multi-resource NMA checkpoints in a fresh MATLAB
process. No training, calibration, allocation, reconstruction, checkpoint
selection, or test prediction is callable.

The timed input is a zero-valued single-precision tensor with the exact spatial
and channel shape of the first training volume in the cell's fold and batch
size one. Image I/O, checkpoint loading, and CPU preprocessing are excluded.

Within each cell:

- 30 untimed warm-up inferences are run per network;
- 200 timing blocks are run;
- every block executes all three networks once in a newly randomized order;
- the GPU is synchronized after every `predict` call;
- latency is recorded with `tic/toc` around only `predict` plus synchronization;
- the cell timing seed is `20260830 + cell_index`, with cell index 1 through 25;
- 5,000 paired block-bootstrap replicates estimate median and p95 latency
  ratios relative to dense.

The primary measured endpoint is the NMA-to-dense median-latency ratio.
Secondary measured endpoints are p95 latency ratio, throughput, proportional
ratios, and NMA-to-proportional ratios.

## Across-cell inference

For each latency endpoint and comparison, cell ratios are log transformed.
Five thousand deterministic replicates with seed `20260831` independently
resample the five folds and five training seeds with replacement, retain the
complete selected fold-by-seed submatrix, average log ratios, and exponentiate.
The geometric mean ratio and percentile 95% interval are reported.

A directly measured dense-latency reduction is supported only if the upper 95%
bound for the NMA-to-dense median-latency ratio is below one. Measured Pareto
improvement additionally requires the already frozen dense non-inferior
compression result. Point estimates alone do not satisfy this tier.

## GPU memory and energy boundary

Windows WDDM does not expose reliable per-process peak VRAM through
`nvidia-smi`. MATLAB `gpuDevice.AvailableMemory` before and after synchronized
warm-up may be stored as a retained-allocation proxy in a fresh reset context.
It must be labeled `RetainedGPUAllocationProxyBytes`, not peak GPU memory, and
cannot by itself establish measured Pareto improvement. Energy is not claimed.

## Required outputs

- raw 200-by-3 paired latency matrix and execution order for every cell;
- per-cell median, p95, mean, standard deviation, throughput, exact resources,
  paired bootstrap ratio intervals, and device metadata;
- complete 75-row benchmark table;
- across-cell fold-seed bootstrap comparison table;
- final decision table that links measured latency evidence to the immutable
  predictive-analysis decision table.

