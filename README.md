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

## Main command-line arguments

Use the supplied batch files for the experiment configurations. Parser defaults are convenience values and do not replace the settings in those scripts. To list the available arguments, run:

```bat
python main.py --help
```

### Problem and model

| Argument | Purpose |
| --- | --- |
| `--pde` | Select `burgers_1d`, `allen_cahn`, `burgers_2d`, or `guckenheimer_dirichlet_2d`. |
| `--model_name` | Registered model name. Copy the exact name from the corresponding batch file; do not rely on the parser default. |
| `--layers` | Layer widths for PINN models, including the input and output widths; for example, `2 50 50 50 50 50 1` for a standard Burgers 1D PINN. Follow the batch file for models with embeddings. |
| `--activation` | Activation selection for models that expose this argument. Use the experiment's model configuration. |
| `--embed_dim`, `--num_heads`, `--depth` | Embedding dimension, attention head count, and depth for the relevant Transformer models. |
| `--fourier_scale` | Fourier feature scale for models that use Fourier features. |
| `--sharp_const` | Modulation-strength setting passed to the relevant models. |
| `--no_sharp_const` | Pass `None` instead of the numerical sharp constant. This flag is not the baseline embedding ablation; select the baseline model specified in the batch file for that comparison. |

Model-specific arguments are interpreted by the selected model. Match these settings when loading checkpoints, even if `load_state_dict` succeeds.

### Training and evaluation grids

| Argument | Purpose |
| --- | --- |
| `--h_x`, `--h_y`, `--h_t` | Training-grid spacings. `h_y` is used for the two-dimensional problems. |
| `--eval_h_x_learning`, `--eval_h_y_learning`, `--eval_h_t_learning` | Grid spacings used for reference-error monitoring during training. |
| `--eval_h_x`, `--eval_h_y`, `--eval_h_t` | Grid spacings used for final evaluation in the trainers. The evaluation notebooks have their own grid settings. |
| `--point_type` | `uniform` by default. The random-data branch in this entry point is implemented for Burgers 1D; it is not a general random-sampling switch for all PDEs. |
| `--n_domain_x` | Passed as `num_random_x_points` to the Burgers 1D random-data generator. |
| `--guckenheimer_amplitude`, `--guckenheimer_d1`, `--guckenheimer_d2`, `--guckenheimer_t_final` | Curved-front geometry and final time for the Guckenheimer benchmark. Keep the reference and notebook settings consistent with training. |

The benchmark references are prepared internally. The Guckenheimer Dirichlet branch requires `--reference_path` to be omitted. The `--batch_size` argument is a data-generation argument, not a general minibatch-training control.

### Optimization, loss weights, and outputs

| Argument | Purpose |
| --- | --- |
| `--optimizer`, `--lr`, `--weight_decay` | Main network optimizer, learning rate, and weight decay. |
| `--scheduler` | Learning-rate schedule: `fixed`, `none`, `rba`, or `brdr`. Selecting a schedule does not activate a weighting method. |
| `--num_epochs` | Main optimizer-stage epoch budget. Early stopping or causal-stage completion can end training earlier. |
| `--lbfgs_epochs` | Number of outer L-BFGS refinement steps; each step may evaluate the closure multiple times. Set to `0` to skip refinement. |
| `--target_loss`, `--early_stop_patience`, `--min_epochs` | Loss-based stopping controls for the applicable trainer branches. A nonpositive target disables target-loss stopping; zero patience disables patience-based stopping. Causal runs also use their stage-specific stopping rules. |
| `--lambda_pde`, `--lambda_bc`, `--lambda_ini` | Multipliers for the PDE, boundary, and initial losses. |
| `--lambda_bc_grad` | Boundary-derivative loss multiplier for Allen–Cahn. |
| `--lambda_ini1`, `--lambda_ini2` | Burgers 1D PINN initial-loss multipliers for the regions `x <= 0.5` and `x > 0.5`; each falls back to `lambda_ini` if omitted. |
| `--seeds` | Space-separated initialization seeds, for example `1 2 3 4 5`. The parser default is a single seed. |
| `--device` | Execution device, such as `cuda` or `cpu`. |
| `--deterministic` | Request the seed helper's deterministic backend settings. This option alone does not promise bitwise agreement across environments. |
| `--save_dir` | Parent output directory; the entry point creates a timestamped run folder underneath it. |
| `--log_every`, `--record_every`, `--eval_every` | Console-reporting, history-recording, and evaluation intervals. |
| `--analysis_every` | Analysis-saving interval. Diagnostic files can be large. |
| `--no_save_plots` | Disable the trainer's optional final plot saving. |
| `--compute_ntk_ratio` / `--no-compute_ntk_ratio` | Enable or disable NTK-ratio diagnostics in trainers that support them. Disabled by default; independent of NTK loss weighting. |

### Adaptive weighting methods

Select at most one method flag. With none enabled, the adaptive weighting objects are disabled.

| Method | Flag | Main settings |
| --- | --- | --- |
| SA | `--use_sa` | `--sa_weight_mode`, `--sa_optimizer`, `--sa_lr`, `--sa_weight_decay`, `--sa_init_*` |
| NTK | `--use_ntk` | `--ntk_update_every`, `--ntk_probes`, `--ntk_seed` |
| alphaSA | `--use_asa` | Both SA and NTK settings; do not combine `--use_sa` and `--use_ntk`. |
| Causal | `--use_causal` | `--causal_epsilon_schedule`, `--causal_stage_epochs`, `--causal_delta`, `--causal_normalization`, `--causal_include_initial` |
| RBA | `--use_rba` | `--rba_gamma`, `--rba_eta`, `--rba_offset` |
| BRDR | `--use_brdr` | `--brdr_beta_residual`, `--brdr_beta_weight`; follow the batch file for the remaining settings. |

For SA and alphaSA, `--sa_weight_mode linear` uses `mean(w * r**2)`, while `square` uses `mean(w**2 * r**2)`. Allen–Cahn uses `linear` in the reported setup; the other PDE examples use `square`. The parser default is `square`, so specify `linear` explicitly where required. Initial point weights are set with `--sa_init_pde`, `--sa_init_bc`, `--sa_init_bc_grad`, and `--sa_init_ini`. These point weights are separate from the fixed `--lambda_*` multipliers. SA and alphaSA require the fixed uniform-grid path in this implementation.

`--ntk_update_every` sets the update interval. `--ntk_probes` sets the number of randomized trace probes. A value of `0` selects the implementation's residual-count-dependent probe rule; it does not request an exact NTK trace.

For Causal PINNs, explicitly supply the intended `--causal_epsilon_schedule`, even for a fixed parameter (for example, `--causal_epsilon_schedule 1`). The parser supplies a nonempty default schedule, so changing `--causal_epsilon` alone does not replace that schedule. `--causal_stage_epochs` caps each stage, while `--num_epochs` caps the overall main optimizer stage. Use `--no-causal_include_initial` to exclude the initial-condition loss from causal weight computation; this does not remove the initial-condition training loss. `paper` and `separate` are implementation option names; use the configuration specified in the experiment script.

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

## Checkpoint availability

Pretrained checkpoints and large analysis tensors are not included in the repository. Train the models using the supplied scripts before rerunning checkpoint-dependent evaluation cells. Saved notebook outputs remain available for inspection.

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

Evaluation output locations are set in each notebook. Depending on the notebook, outputs include:

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
