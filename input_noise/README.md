# `input_noise/` — input-noise vs. pruning iso-accuracy experiment

Sweeps test-time **input noise** `σ_x` jointly with pruning **density** `s`,
extracts the iso-accuracy contour `A(s, σ_x) = A_*`, and tests the
signal-to-noise collapse `η = 1 − ξ` in the normalised coordinates of the
paper. Depends on `unstructured_pruning` (`apply_mask`, `evaluate_masked_accuracy`,
`random_masks`) and `pruning` (`FCNetwork`).

## Files

| File | Role |
|---|---|
| `core.py` | Shared helpers: `add_gaussian_noise`, `evaluate_noisy_accuracy`, `evaluate_joint` (the `(s, σ_x)` grid), iso-accuracy contour extraction. |
| `run_experiment.py` | Pilot single-machine runner: per `(dataset, H, L)` cell sweep `A(s)` and `A(σ_x)`, fit sigmoids → `assets/input_noise/pilot/results.json`. |
| `cluster_sweep.py` | Cluster-scale resumable sweep over saved checkpoints; one JSON per cell under `input_noise/results_cluster/` (raw data, **not** an asset). |
| `aggregate.py` | Pool the per-cell JSONs into `input_noise/results_cluster_all.json`. |
| `cluster_analyze.py` | Read the aggregated JSON, extract iso-`A=0.5` contours, fit the framework rational curve, build the collapse figures → `assets/input_noise/cluster/`. |
| `plots.py` | Per-cell `A(s)`/`A(σ_x)` panels, iso-contour maps, SNR-collapse and conversion-fit plots → `assets/input_noise/pilot/`. |
| `extensions/` | Self-contained follow-up probes (`iso_levels/`, `depth_cells/`, `seed_replicates/`, `falsifiability/`, `cov_wx/`) + shared `plot.py`. Each `run.py` writes its results + figures to `assets/input_noise/extensions/<name>/`. |
| `submit.sbatch`, `submit_dense.sbatch`, `submit*.sh` | ALICE SLURM submission for the cluster sweep (dense = 50-point `s` grid). |

## Outputs vs. data

- **Figures + accompanying JSON summaries → `assets/input_noise/{pilot,cluster,extensions}/`.**
- **Raw data stays in place:** `input_noise/checkpoints/`, `input_noise/results_cluster/`, `input_noise/results_cluster_all.json` (inputs, regenerable on the cluster).

## Run

```bash
python -m input_noise.run_experiment            # pilot
python -m input_noise.cluster_analyze           # build collapse figures from aggregated JSON
```
