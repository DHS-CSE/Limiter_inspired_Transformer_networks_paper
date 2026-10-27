# PINN and Transformer experiments

Training and evaluation code for the one-dimensional inviscid Burgers, Allen–Cahn, two-dimensional Burgers, and two-dimensional Guckenheimer equations.

The repository includes conventional PINNs, NTK-applied PINNs, SA-PINNs, alphaSA-PINNs, Causal PINNs, RBA PINNs, BRDR PINNs, and single and multi decoder models. Baseline decoder variants are included to assess the contribution of the limiter-inspired embedding.

## Installation

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Evaluation notebooks also require a Jupyter environment. Figure assembly uses Pillow and IPython:

```bash
python -m pip install jupyterlab pillow ipython
```

The computational-cost measurements were obtained using an NVIDIA GeForce RTX 3080 with PyTorch 2.5.1+cu121 and CUDA 12.1. Install a PyTorch build compatible with your hardware.

## Training

Run the Windows batch files from the repository root containing `main.py`.

| PDE | PINN-based methods | Transformer-based models |
| --- | --- | --- |
| Burgers 1D | `experiments/run_PINN_all_burgers.bat` | `experiments/run_Transformer_all_burgers.bat` |
| Allen–Cahn | `experiments/run_PINN_all_allen_cahn.bat` | `experiments/run_Transformer_all_allen_cahn.bat` |
| Burgers 2D | `experiments/run_PINN_all_burgers2d.bat` | `experiments/run_Transformer_all_burgers2d.bat` |
| Guckenheimer 2D | `experiments/run_PINN_all_guckenheimer2d_dirichlet.bat` | `experiments/run_Transformer_all_gkhm2d_dirichlet.bat` |

For example:

```bat
experiments\run_PINN_all_burgers.bat
```

`experiments/run_ALL_experiments.bat` provides a combined launcher. Use either the combined launcher or the individual scripts to avoid unintended repeated runs.

Model configurations, optimizer settings, loss weights, training lengths, and seeds are specified in the batch files. Check each command's `--seeds` setting before reproducing five-seed statistics.

## Saved training results

Training results are organized under:

```text
runs/
├── burgers/
├── allen_cahn/
├── burgers2d/
└── gkhm2d/
```

Each method directory contains timestamped training runs. Checkpoints are stored under:

```text
<run_directory>/checkpoints/seed<seed>/
```

The evaluation notebooks select run folders using the `*_new` convention. Add `_new` to the intended training-run folder name before evaluation. If several folders match, check the notebook's selection logic to ensure that the intended run is selected.

## Evaluation

| PDE | Notebook |
| --- | --- |
| Burgers 1D | `evaluate_burgers1d_shock_trajectory.ipynb` |
| Allen–Cahn | `evaluate_allen_cahn_transition_metrics.ipynb` |
| Burgers 2D | `evaluate_burgers2d_shock_timewise.ipynb` |
| Guckenheimer 2D | `evaluate_guckenheimer2d_shock_timewise.ipynb` |

Open the notebooks from the repository root and run cells from top to bottom.

Before evaluation:

1. Set the checkpoint paths in `METHODS`.
2. Select the seeds for which checkpoints are available.
3. Match the model configuration to the training settings, including embedding dimension, attention heads, depth, activation functions, and Fourier scale.
4. Check the evaluation grid and target times.

Matching state-dict shapes alone does not guarantee that all model settings match the training configuration.

Use the checkpoint-selection rule specified in each notebook. For SA, NTK, and alphaSA, the final evaluation uses `last_lbfgs.pth`. Additional SA-weight and NTK-history visualizations require the corresponding saved parameter snapshots and analysis files.

The notebooks retain their saved outputs for inspection. These outputs show the recorded evaluations; rerunning a cell replaces its output.

## Evaluation outputs

Evaluation results are saved under `evaluation/<PDE>/`. Depending on the notebook, outputs include:

- Per-seed and aggregated relative solution errors.
- Shock/front-location RMSE and detection rates.
- Transition-width, resolution, and near-front error diagnostics.
- Predicted and reference solutions and absolute-error figures.
- SA-parameter and NTK-coefficient visualizations for Burgers 1D.
- Assembled comparison figures.

Location RMSE is calculated from detected samples and should be interpreted together with detection coverage. Transition width denotes the measured width, not a width error.

Figure assembly is performed in the final notebook cells. Model order, selected seeds, image types, and layout settings can be adjusted there.

## Framework

| Path | Purpose |
| --- | --- |
| `main.py` | Argument handling, data preparation, and experiment dispatch |
| `registry.py` | Model registration and construction |
| `models/` | PINN and Transformer architectures and adapters |
| `embedding/` | Feature embeddings and limiter-inspired modulation |
| `attention/` | Attention modules |
| `losses/` | PDE, boundary, and initial-condition losses |
| `methods/` | Adaptive loss-weighting methods |
| `trainer/` | Training stages, checkpoint handling, and reporting |
| `reference/` | Benchmark reference solutions |
| `utils/` | Data generation, plotting, serialization, and cost measurements |
| `experiments/` | Batch scripts for the numerical experiments |

## Computational-cost records

The `training_cost_seed*.json` files record training wall-clock time, optimizer-stage times, network parameter counts, recorded iteration counts, L-BFGS closure calls, and peak PyTorch GPU memory.

Training time includes in-training evaluation, diagnostics, checkpoint saving, and logging. Initialization and final evaluation and plotting are excluded. Network parameter counts exclude auxiliary adaptive loss weights.

The reported single-seed costs correspond to the specified training procedures. They are not comparisons at equal accuracy or measurements of scaling with collocation-point count.

## Reproducibility

The batch files and evaluation configurations specify the settings used for each experiment. Preserve the model settings, checkpoint-selection rules, evaluation grids, and metric settings when reproducing the comparisons.

Exact numerical results can vary with the software environment, GPU operations, and determinism settings. The `--deterministic` option is available in the training interface; check the supplied commands for the setting used in each experiment.
