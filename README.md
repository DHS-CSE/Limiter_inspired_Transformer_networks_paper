# PINN and Transformer experiments

Training and evaluation for Burgers 1D, Allen–Cahn, Burgers 2D and Guckenheimer 2D Dirichlet.

## Training

Run the supplied Windows batch files from the repository root containing `main.py`. Install the packages in `requirements.txt`; evaluation image assembly additionally uses Pillow and IPython, and a Jupyter environment is required for notebooks. Training scripts retain their original arguments and seed lists. Check those lists when reproducing five-seed statistics.

## Evaluation

The four `evaluate_*.ipynb` files contain evaluation, metric aggregation and figure assembly. Run cells from top to bottom with the required checkpoints available. Keep the original `*_new` run-folder selection convention. Checkpoint paths, selection rules, NTK-history intervals and model settings are retained exactly as supplied; this cleanup does not reconcile them with other files or alter them automatically. Configure local paths deliberately before running.

Notebook source cleanup removes comments and excess blank lines only. All code-cell abstract syntax trees are identical to the supplied original. All cells, saved outputs, execution counts and metadata are preserved. Outputs have not been regenerated.

## Framework

`main.py` dispatches the PDE trainer. `models/` defines networks and adapters; `embedding/` and `attention/` contain representation modules; `losses/` defines physical losses; `methods/` implements weighting; `reference/` supplies references; `utils/` provides data, plots, checkpoint and cost utilities. `experiments/` contains launchers. Evaluation notebooks load checkpoints and generate solution, location, detection and other diagnostics.

## Original dependency gap

The supplied Burgers 1D notebook calls `save_prediction_burgers_plot_test` and `save_error_contour_burgers_plot_test`, but their definitions are absent from the supplied archive. They have NOT been replaced or invented. Supply the original functions before rerunning those plotting calls. Existing notebook outputs remain available.

## Validation

Python and batch files are byte-identical to the supplied originals. Notebook code ASTs and all non-source cell fields were compared with the original archive. No training or evaluation was rerun. See `PRESERVATION_REPORT.json`. Bytecode caches are omitted; the original source files are retained.
