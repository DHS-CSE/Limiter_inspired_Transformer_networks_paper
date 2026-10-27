from __future__ import annotations
from reference.reference_allen_cahn import AllenCahnReference
from trainer.allen_cahn_trainer import AllenCahnTrainer
from typing import Any
from trainer.burgers_trainer import BurgersTrainer
from pathlib import Path
import argparse
from reference.reference_burgers2d import burgers2d_exact_solution
from reference.reference_burgers import burgers_solution
from losses.allen_cahn import calculate_loss_allen_cahn_ghost_sharp_analysis
from losses.burgers import calculate_loss_burgers_ghost_sharp_analysis
from utils.trainer_utils import (
    count_parameters, make_allen_cahn_data, make_burgers2d_data, make_burgers_data,
    make_burgers_lhs_random_data, make_guckenheimer2d_dirichlet_data, save_json, set_seed,
    write_excel_report,
)
import gc
from reference.reference_guckenheimer2d_dirichlet import guckenheimer2d_exact_solution
import math
import time
import torch
from registry import build_model
from build_optimizer import build_optimizer
from build_scheduler import build_scheduler
from functools import partial
from losses.burgers2d import calculate_loss_burgers2d_ghost_sharp_analysis
from trainer.burgers2d_trainer import Burgers2DTrainer
from losses.guckenheimer2d_dirichlet import calculate_loss_guckenheimer2d_dirichlet_ghost_sharp_analysis
from trainer.guckenheimer2d_dirichlet_trainer import Guckenheimer2DDirichletTrainer
from methods.sa_ntk import add_sa_ntk_arguments, validate_sa_ntk_args, require_sa_ntk_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='SLFormerFixedSharp')
    parser.add_argument('--layers', type=int, nargs='+', default=[2, 50, 50, 50, 50, 50, 1])
    parser.add_argument('--fourier_scale', type=float, default=1.0)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--activation', type=str, default='Tanh')
    parser.add_argument('--pde', type=str, default='allen_cahn', choices=['allen_cahn', 'burgers_1d', 'burgers_2d', 'guckenheimer_dirichlet_2d'])
    parser.add_argument('--guckenheimer_amplitude', type=float, default=0.15)
    parser.add_argument('--guckenheimer_d1', type=float, default=0.3)
    parser.add_argument('--guckenheimer_d2', type=float, default=0.8)
    parser.add_argument('--guckenheimer_t_final', type=float, default=0.6)
    parser.add_argument('--h_x', type=float, default=1 / 16)
    parser.add_argument('--h_y', type=float, default=1 / 16)
    parser.add_argument('--h_t', type=float, default=1 / 64)
    parser.add_argument('--reference_path', type=str, default='', help='Optional exact/reference solution for logging relative L2 only.')
    parser.add_argument('--n_domain_x', type=int, default=129)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--point_type', choices=['uniform', 'random'], default='uniform')
    parser.add_argument('--sharp_const', type=float, default=0.01)
    parser.add_argument('--no_sharp_const', action='store_true')
    parser.add_argument('--lambda_pde', type=float, default=1.0)
    parser.add_argument('--lambda_bc', type=float, default=10.0)
    parser.add_argument('--lambda_bc_grad', type=float, default=10.0)
    parser.add_argument('--lambda_ini', type=float, default=100.0)
    parser.add_argument('--lambda_ini1', type=float, default=None, help='Burgers PINN: coefficient for x<=0.5; defaults to lambda_ini.')
    parser.add_argument('--lambda_ini2', type=float, default=None, help='Burgers PINN: coefficient for x>0.5; defaults to lambda_ini.')
    parser.add_argument('--rho', type=float, default=5.0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--scheduler', type=str, default='fixed', choices=['fixed', 'none', 'rba', 'brdr'])
    parser.add_argument('--target_loss', type=float, default=-1.0, help='Stop when total loss <= target_loss. Disabled if <= 0.')
    parser.add_argument('--early_stop_patience', type=int, default=0, help='Stop after this many epochs without total-loss improvement. Disabled if 0.')
    parser.add_argument('--min_epochs', type=int, default=0, help='Do not early stop before this epoch.')
    parser.add_argument('--num_epochs', type=int, default=20000)
    parser.add_argument('--lbfgs_epochs', type=int, default=200)
    parser.add_argument('--scheduler_stop', type=int, default=5000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--record_every', type=int, default=1000)
    parser.add_argument('--analysis_every', type=int, default=1000)
    parser.add_argument('--write_jsonl_live', action='store_true', help='Append history.jsonl during training instead of writing only at the end.')
    parser.add_argument('--eval_every', type=int, default=1000, help='Evaluate relative L2 every N epochs when --reference_path is provided.')
    parser.add_argument('--save_dir', type=str, default='runs')
    parser.add_argument('--save_plots', action='store_true', default=True)
    parser.add_argument('--no_save_plots', dest='save_plots', action='store_false')
    parser.add_argument('--hard_periodic', action='store_true')
    parser.add_argument('--use_causal', action='store_true')
    parser.add_argument('--causal_epsilon', type=float, default=1.0)
    parser.add_argument('--causal_epsilon_schedule', type=float, nargs='+', default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument('--causal_stage_epochs', type=int, default=20000)
    parser.add_argument('--causal_delta', type=float, default=0.99)
    parser.add_argument('--causal_num_time_bins', type=int, default=None)
    parser.add_argument('--causal_include_initial', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--causal_normalization', type=str, choices=['paper', 'separate'], default='paper')
    parser.add_argument('--use_rba', action='store_true')
    parser.add_argument('--rba_gamma', type=float, default=0.999)
    parser.add_argument('--rba_eta', type=float, default=0.01)
    parser.add_argument('--rba_offset', type=float, default=0.0)
    parser.add_argument('--use_brdr', action='store_true')
    parser.add_argument('--brdr_beta_residual', type=float, default=0.999)
    parser.add_argument('--brdr_beta_weight', type=float, default=0.999)
    parser.add_argument('--brdr_beta_scale', type=float, default=0.999)
    parser.add_argument('--brdr_eps', type=float, default=1e-14)
    parser.add_argument('--use_weight_norm', action='store_true')
    parser.add_argument('--best_ckpt_name', type=str, default=f'best_loss.pth')
    parser.add_argument('--last_ckpt_name', type=str, default='last.pth')
    parser.add_argument('--interrupt_ckpt_name', type=str, default='interrupt.pth')
    parser.add_argument('--history_name', type=str, default='history')
    parser.add_argument('--manifest_name', type=str, default='manifest')
    parser.add_argument('--metrics_name', type=str, default='metrics')
    parser.add_argument('--excel_name', type=str, default='results')
    parser.add_argument('--prediction_name', default='prediction')
    parser.add_argument('--eval_h_x', type=float, default=1 / 100)
    parser.add_argument('--eval_h_y', type=float, default=1 / 100)
    parser.add_argument('--eval_h_t', type=float, default=1 / 100)
    parser.add_argument('--eval_h_x_learning', type=float, default=1 / 32)
    parser.add_argument('--eval_h_y_learning', type=float, default=1 / 32)
    parser.add_argument('--eval_h_t_learning', type=float, default=1 / 32)
    parser.add_argument('--compute_ntk_ratio', action=argparse.BooleanOptionalAction, default=False, help='Compute and save diagnostic NTK and limiter ratios.')
    add_sa_ntk_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()
    for name in ('lambda_ini1', 'lambda_ini2'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative.')
    if args.pde == 'guckenheimer_dirichlet_2d':
        if args.reference_path:
            raise ValueError('Dirichlet curved-front case uses its exact reference; omit --reference_path.')
        if args.point_type != 'uniform' or getattr(args, 'use_domain_batch', False):
            raise ValueError('This integration uses the existing fixed full ghost-grid path.')
    validate_sa_ntk_args(args)
    if args.pde == 'guckenheimer_dirichlet_2d' and any((getattr(args, flag, False) for flag in ('use_sa', 'use_ntk', 'use_asa'))):
        require_sa_ntk_loss(calculate_loss_guckenheimer2d_dirichlet_ghost_sharp_analysis)
    device = torch.device(args.device)
    print('Device:', device)
    print('Model:', args.model_name)
    print('Scheduler:', args.scheduler)
    print('Seeds:', args.seeds)
    all_summaries: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = f'{timestamp}_{args.model_name}'
    root_dir = Path(args.save_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    run_dir = root_dir / run_dir
    if args.pde == 'allen_cahn':
        data = make_allen_cahn_data(h_x=args.h_x, h_t=args.h_t, device=device, batch_size=args.batch_size)
        reference = AllenCahnReference(Nx=512, dt=0.001, T=1.0, D=0.0001, beta=5.0, save_every=10)
        u_ref_train = reference.evaluate_tensor(data['domain_coords_inner'])
    elif args.pde == 'burgers_1d':
        data = make_burgers_data(args.h_x, args.h_t, device)
        reference = burgers_solution
        u_ref_train = reference(data['domain_coords_inner'])
    elif args.pde == 'burgers_2d':
        data = make_burgers2d_data(args.h_x, args.h_y, args.h_t, device)
        reference = burgers2d_exact_solution
        u_ref_train = reference(data['domain_coords_inner'])
    elif args.pde == 'guckenheimer_dirichlet_2d':
        reference = partial(guckenheimer2d_exact_solution, amplitude=args.guckenheimer_amplitude, d1=args.guckenheimer_d1, d2=args.guckenheimer_d2)
        data = make_guckenheimer2d_dirichlet_data(args.h_x, args.h_y, args.h_t, device, amplitude=args.guckenheimer_amplitude, d1=args.guckenheimer_d1, d2=args.guckenheimer_d2, batch_size=args.batch_size, t_bounds=(0.0, args.guckenheimer_t_final))
        u_ref_train = reference(data['domain_coords_inner'])
    for seed in args.seeds:
        set_seed(seed, deterministic=args.deterministic)
        if args.pde == 'burgers_1d':
            if args.point_type == 'random':
                data = make_burgers_lhs_random_data(h_x=args.h_x, h_t=args.h_t, device=device, batch_size=args.batch_size, num_random_x_points=args.n_domain_x, seed=seed, dtype=torch.float32)
                reference = burgers_solution
                u_ref_train = reference(data['domain_coords_inner'])
        model = build_model(args).to(device)
        total_params, trainable_params = count_parameters(model)
        print('=' * 60)
        print(f'Model: {args.model_name}')
        print(f'Total parameters     : {total_params:,}')
        print(f'Trainable parameters : {trainable_params:,}')
        print('=' * 60)
        optimizer = build_optimizer(model, args)
        scheduler = build_scheduler(optimizer, args)
        if args.pde == 'allen_cahn':
            trainer = AllenCahnTrainer(args=args, seed=seed, run_dir=run_dir, device=device, model=model, optimizer=optimizer, scheduler=scheduler, data=data, loss_fn=calculate_loss_allen_cahn_ghost_sharp_analysis, reference=reference, u_ref=u_ref_train)
        elif args.pde == 'burgers_1d':
            trainer = BurgersTrainer(args=args, seed=seed, run_dir=run_dir, device=device, model=model, optimizer=optimizer, scheduler=scheduler, data=data, loss_fn=calculate_loss_burgers_ghost_sharp_analysis, reference_function=reference, u_ref=u_ref_train)
        elif args.pde == 'burgers_2d':
            trainer = Burgers2DTrainer(args=args, seed=seed, run_dir=run_dir, device=device, model=model, optimizer=optimizer, scheduler=scheduler, data=data, loss_fn=calculate_loss_burgers2d_ghost_sharp_analysis, reference_function=reference, u_ref=u_ref_train)
        elif args.pde == 'guckenheimer_dirichlet_2d':
            trainer = Guckenheimer2DDirichletTrainer(args=args, seed=seed, run_dir=run_dir, device=device, model=model, optimizer=optimizer, scheduler=scheduler, data=data, loss_fn=calculate_loss_guckenheimer2d_dirichlet_ghost_sharp_analysis, reference_function=reference, u_ref=u_ref_train)
        summary, history = trainer.train()
        all_summaries.append(summary)
        all_history.extend(history)
        del trainer
        del history
        del summary
        gc.collect()
        torch.cuda.empty_cache()
    save_json(run_dir / 'config.json', vars(args))
    write_excel_report(run_dir / 'all_results.xlsx', all_summaries, all_history, args)
    print('\nFinished.')
    for r in all_summaries:
        print(f"seed={r['seed']} best_loss={r['best_loss']:.3e} best_epoch={r['best_epoch']} stop={r['stop_reason']} rel_l2={r['relative_l2_error_of_best_loss_model']} checkpoint={r['best_loss_checkpoint']}")
    print(f"\nExcel summary: {run_dir / 'all_results.xlsx'}")
if __name__ == '__main__':
    main()
