# cfd-aero-pytorch

PyTorch workflows for aerodynamic surrogate modelling on point-cloud and mesh derived datasets (DrivAer, Ahmed, Windsor). The repository contains data loaders, training pipelines, experiment orchestration scripts, and model zoo entries ranging from classic PointNet variants to the PointTransformerV3 integration.

## Environment Setup

- Python 3.10+ is recommended. Install core dependencies with:
  ```bash
  pip install -r requirements.txt
  ```
- Optional extras (required for PointTransformerV3):
  ```bash
  pip install addict timm torch-scatter
  pip install spconv-cu121  # or a build matching your CUDA toolchain
  ```
- Datasets must be staged under the `inputs/` hierarchy referenced in `configs/system_config.yml` (for example `inputs/drivaer_net/stl`, `inputs/drivaer_net/vtk`, `inputs/drivaer_net/targets.csv`).

## Running Single Experiments

Experiments are driven through `source/runner.py`, which bootstraps configuration from `configs/system_config.yml` and applies CLI overrides.

```bash
python source/runner.py \
  --model-arch PointNet \
  --model-name SimplePointNet \
  --dataset-name DrivAerNet
```

Common overrides:

- `--batch-size`, `--epochs`, `--num-points`, `--lr`, `--dropout`
- `--conv-layers` and `--fc-layers` expect colon-delimited lists surrounded by brackets, e.g. `--conv-layers "[3:64:128:256]"`
- `--exp-name` lets you control the experiment folder name; if omitted a descriptive label is generated automatically.

Artifacts (checkpoints, score summaries, logs) are written to the `outputs/` tree configured in `system_config.yml`.

## Batch Execution Scripts

The `scripts/` directory centralises CSV-defined experiment grids and helpers for local runs or SLURM batch queues. CSV columns follow the argument order consumed by each script, so ensure your files mirror the expected schema.

- `submit_local.sh` — Streams experiments sequentially on the local machine using `configs/hyper_parameters.csv`. Optional filters let you scope by dataset, architecture, or model name.
- `submit_cluster_batch.sh` — Submits a filtered slice of `configs/hyper_parameters.csv` to SLURM (`dzagnormal` partition by default). Each row produces a job with dedicated stdout/stderr log files under `scripts/logs/`.
- `submit_cluster_one.sh` — Similar to the batch variant but consumes a user supplied CSV (no conv/fc columns) so you can launch bespoke sweeps.
- `submit_cluster_one_layers.sh` — Cluster launcher for CSVs that include explicit convolution and fully connected layer definitions; uses two nodes by default.
- `predictor.sh` — Fire-and-forget inference job that executes `source/predictor.py` with SLURM logging.

Supporting CSV templates:

- Top-level CSVs (e.g. `pressure.csv`, `shallow_variable_points.csv`, `regpointnet_exp.csv`) bundle curated experiment grids for production runs.
- `scripts/test/` duplicates several CSVs with reduced search spaces for sanity checks or CI-style dry-runs.

All batch scripts assume a Conda environment located at `~/miniconda3/bin/activate python3`; adjust the `source` line if your environment differs. SLURM partition, GPU count, and node settings are also hard-coded and should be tailored to your cluster.

## Configuring Models

`configs/system_config.yml` defines defaults for datasets, environment options, and model hyperparameters. You can either edit the YAML directly or rely on CLI overrides to explore alternative settings. `configs/hyper_parameters.csv` is the canonical grid searched by the helper scripts and must stay in sync with the YAML defaults where applicable.

## PointTransformerV3 Integration

- The `PointTransformerV3Regressor` model wraps the upstream PointTransformerV3 backbone from `/source/model/point_transformer_v3` and supports both drag (scalar) and pressure (per-point) targets.
- Enable it by setting `model_arch: PointNet` and `model_name: PointTransformerV3Regressor` before launching `runner.py` or a batch script.
- Transformer-specific knobs live under `parameters.model.point_transformer` in `configs/system_config.yml` (depth, heads, pooling strategy, etc.).
- Ensure the optional dependencies listed above are installed; the wrapper surfaces a clear error if required CUDA extensions are unavailable.

## Outputs and Monitoring

- Checkpoints: `outputs/models/<exp_name>_best_model.pth`
- Metrics: `outputs/scores/<exp_name>_best_scores.json`
- Logs: `outputs/logs` for training traces, `scripts/logs` for SLURM job stdout/stderr when using the helper scripts.

Review `source/trainer/pointnet_trainer.py` and `source/results/` for deeper insight into evaluation metrics, result aggregation, and saving conventions.
