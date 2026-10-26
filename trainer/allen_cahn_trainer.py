from pathlib import Path
from utils.trainer_utils import (
    append_jsonl, current_lr, evaluate_model_on_inner_grid, make_allen_cahn_data, print_training_status,
    relative_l2_error, save_analysis_tensors, save_checkpoint, save_error_contour_plot, save_error_curve,
    save_history_jsonl, save_json, save_loss_plot, save_lr_plot, save_manifest, save_prediction_plot,
    write_excel_report,
)
import math
import time
import torch
from build_scheduler import step_scheduler
from methods.brdr import BRDRWeights, BRDRScale
from methods.rba import RBAWeights
from methods.causal import CausalPINNWeighting
from methods.sa_ntk import init_sa_ntk, restore_sa_ntk
from utils.training_cost import TrainingCostTracker

class AllenCahnTrainer:

    def __init__(
        self, args, seed: int, run_dir, device, model, optimizer, scheduler, data, loss_fn, reference, u_ref
    ):
        self.args = args
        self.seed = seed
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data = data
        self.loss_fn = loss_fn
        self.sa_weight, self.ntk_weight = init_sa_ntk(args, seed, device)
        self.run_dir = Path(run_dir)
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.bad_epochs = 0
        self.stop_reason = 'max_epochs'
        self.history = []
        self.last_record = None
        self.loss_value = float('inf')
        self.epoch = 0
        self.interrupted = False
        self.train_start_time = time.time()
        self.reference = reference
        self.u_ref_train = u_ref
        self.last_rel_l2 = float('nan')
        self.rba_weight = None
        self.brdr_weight = None
        self.brdr_scale = None
        self.causal_weight = None
        self.optimizer_lbfgs = None
        self.lbfgs_ran = False
        self.lbfgs_best_loss = float('inf')
        self.lbfgs_best_step = 0
        self.lbfgs_mode = 'not_run'
        self.adam_optimization_sec = 0.0
        self.lbfgs_optimization_sec = 0.0
        self.eval_data_learning = make_allen_cahn_data(
            h_x=args.eval_h_x_learning, h_t=args.eval_h_t_learning, device=device
        )
        self.eval_reference = self.reference.evaluate_tensor(self.eval_data_learning['domain_coords_inner'])
        if args.use_rba:
            self.rba_weight = RBAWeights(
                shape=(1, self.data['domain_coords_inner'].shape[1], 1), device=device, gamma=args.rba_gamma,
                eta=args.rba_eta, offset=args.rba_offset
            )
        if args.use_brdr:
            self.brdr_weight = BRDRWeights(
                beta_residual=args.brdr_beta_residual, beta_weight=args.brdr_beta_weight, eps=args.brdr_eps
            )
            self.brdr_scale = BRDRScale(beta_scale=args.brdr_beta_scale, eps=args.brdr_eps)
            self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.causal_schedule = []
        self.causal_stage_index = 0
        self.causal_stage_epoch = 0
        self.causal_pending_stage = None
        self.causal_stage_records = []
        self.causal_schedule_complete = False
        self.causal_last_min_weight = None
        if args.use_causal:
            supplied = getattr(args, 'causal_epsilon_schedule', None)
            self.causal_schedule = [float(v) for v in (supplied if supplied is not None else [args.causal_epsilon])]
            if not self.causal_schedule or any((not math.isfinite(v) or v <= 0 for v in self.causal_schedule)) or any((
                b <= a for a, b in zip(self.causal_schedule, self.causal_schedule[1:])
            )):
                raise ValueError('causal_epsilon_schedule must be finite, positive and strictly increasing.')
            self.causal_stage_limit = int(args.causal_stage_epochs)
            self.causal_delta = float(args.causal_delta)
            if self.causal_stage_limit < 1 or not 0 < self.causal_delta < 1:
                raise ValueError('Require causal_stage_epochs >= 1 and 0 < causal_delta < 1.')
            self.causal_weight = CausalPINNWeighting(
                epsilon=self.causal_schedule[0], num_time_bins=getattr(args, 'causal_num_time_bins', None),
                include_initial=args.causal_include_initial, normalization=args.causal_normalization
            )
        self._setup_dirs()

    def _begin_causal_epoch(self):
        if self.causal_weight is None:
            return
        if self.causal_pending_stage is not None:
            self.causal_stage_index = self.causal_pending_stage
            self.causal_pending_stage = None
            self.causal_stage_epoch = 0
            self.causal_weight.set_epsilon(self.causal_schedule[self.causal_stage_index])
            self.best_loss = float('inf')
            self.best_epoch = 0
            self.bad_epochs = 0
        self.causal_stage_epoch += 1
        if self.causal_stage_epoch == 1:
            print(f'[Causal] stage={self.causal_stage_index + 1}/{len(self.causal_schedule)}, epsilon={self.causal_weight.epsilon:g}, stage_limit={self.causal_stage_limit}, delta={self.causal_delta:g}')

    def _finish_causal_epoch(self, aux):
        if self.causal_weight is None:
            return False
        if 'causal_min_weight' not in aux:
            raise KeyError("The Allen–Cahn loss must return aux['causal_min_weight'].")
        value = aux['causal_min_weight']
        minimum = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        if not math.isfinite(minimum):
            raise FloatingPointError('Non-finite causal min weight; a stage transition cannot be assessed.')
        self.causal_last_min_weight = minimum
        criterion_met = minimum > self.causal_delta
        capped = self.causal_stage_epoch >= self.causal_stage_limit
        reason = 'weight_threshold' if criterion_met else 'stage_epoch_limit' if capped else 'global_epoch_limit' if self.epoch == self.args.num_epochs else ''
        record = dict(
            epoch=self.epoch, stage=self.causal_stage_index + 1, stage_epoch=self.causal_stage_epoch,
            epsilon=self.causal_weight.epsilon, min_weight=minimum, delta=self.causal_delta,
            criterion_met=criterion_met, stage_end_reason=reason
        )
        self.causal_stage_records.append(record)
        append_jsonl(self.log_dir / 'causal_schedule.jsonl', record)
        if not (criterion_met or capped):
            return False
        print(f"[Causal] stage={record['stage']} ended at epoch={self.epoch}; epsilon={record['epsilon']:g}, min_weight={minimum:.6e}, reason={reason}")
        stage_path = self.best_loss_path.with_name(f"causal_stage{record['stage']}_best_loss{self.best_loss_path.suffix}")
        import shutil
        if self.best_loss_path.is_file():
            shutil.copy2(self.best_loss_path, stage_path)
        if self.causal_stage_index + 1 < len(self.causal_schedule):
            self.causal_pending_stage = self.causal_stage_index + 1
            return False
        self.causal_schedule_complete = True
        self.stop_reason = 'causal_schedule_converged' if criterion_met else 'causal_schedule_stage_limits'
        return True

    def _restore_causal_checkpoint(self, checkpoint):
        if self.causal_weight is None:
            return
        state = checkpoint.get('extra', {}).get('causal_weight')
        if state is not None:
            self.causal_weight.load_state_dict(state)

    def _setup_dirs(self):
        args = self.args
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.analysis_dir = self.run_dir / 'analysis'
        self.log_dir = self.run_dir / 'logs'
        self.plot_dir = self.run_dir / 'plots'
        self.result_dir = self.run_dir / 'results'
        self.metric_dir = self.run_dir / 'metrics'
        for d in [
            self.checkpoint_dir, self.analysis_dir, self.log_dir, self.plot_dir, self.result_dir,
            self.metric_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)
        self.seed_ckpt_dir = self.checkpoint_dir / f'seed{self.seed}'
        self.seed_analysis_dir = self.analysis_dir / f'seed{self.seed}'
        self.seed_ckpt_dir.mkdir(exist_ok=True)
        self.seed_analysis_dir.mkdir(exist_ok=True)
        self.best_loss_path = self.seed_ckpt_dir / args.best_ckpt_name
        self.interrupt_path = self.seed_ckpt_dir / args.interrupt_ckpt_name
        self.last_path = self.seed_ckpt_dir / args.last_ckpt_name
        self.last_adam_path = self.seed_ckpt_dir / 'last_adam.pth'
        self.best_lbfgs_path = self.seed_ckpt_dir / 'lbfgs_best_loss.pth'
        self.last_lbfgs_path = self.seed_ckpt_dir / 'last_lbfgs.pth'
        self.log_path = self.log_dir / f'{args.history_name}_seed{self.seed}.jsonl'
        self.manifest_path = self.metric_dir / f'{args.manifest_name}_seed{self.seed}.json'
        self.best_metrics_path = self.metric_dir / f'{args.metrics_name}_seed{self.seed}.json'
        self.prediction_path = self.seed_analysis_dir / f'{args.prediction_name}_seed{self.seed}.pt'
        self.analysis_path = self.seed_analysis_dir / 'analysis.pt'

    def _update_ntk_weights(self, data):
        if self.ntk_weight is None or not self.ntk_weight.should_update(self.epoch):
            return
        args = self.args
        sharp_const = None if args.no_sharp_const else args.sharp_const
        with torch.enable_grad():
            _, _, graph_aux = self.loss_fn(
                sa=self.sa_weight, ntk=self.ntk_weight, model=self.model, data=data, sharp_const=sharp_const,
                rba=self.rba_weight, brdr=self.brdr_weight, causal=self.causal_weight, update_weighting=False,
                hard_periodic=args.hard_periodic, lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc,
                lambda_bc_grad=args.lambda_bc_grad, lambda_ini=args.lambda_ini, rho=args.rho,
                return_residual_graph=True
            )
            self.ntk_weight.update(self.model, graph_aux['adaptive_residual_groups'], self.epoch)

    def train_one_epoch(self):
        args = self.args
        self.optimizer.zero_grad(set_to_none=True)
        if self.sa_weight is not None:
            self.sa_weight.zero_grad()
        sharp_const = None if args.no_sharp_const else args.sharp_const
        train_data = self.data
        raw_loss, loss_dict, aux = self.loss_fn(
            sa=self.sa_weight, ntk=self.ntk_weight, model=self.model, data=train_data,
            sharp_const=sharp_const, rba=self.rba_weight, brdr=self.brdr_weight, causal=self.causal_weight,
            update_weighting=True, hard_periodic=args.hard_periodic, lambda_pde=args.lambda_pde,
            lambda_bc=args.lambda_bc, lambda_bc_grad=args.lambda_bc_grad, lambda_ini=args.lambda_ini,
            rho=args.rho
        )
        if self.brdr_scale is not None:
            old_scale = self.brdr_scale.get_scale()
            optimization_loss = raw_loss * old_scale
        else:
            old_scale = None
            optimization_loss = raw_loss
        if torch.isnan(raw_loss).any() or torch.isnan(optimization_loss).any():
            return (raw_loss, optimization_loss, loss_dict, aux)
        optimization_loss.backward()
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    print(name)
                    break
        if self.brdr_scale is not None:
            grads = [p.grad for p in self.trainable_params if p.grad is not None]
            grad_norm_sq = torch.zeros((), device=self.device)
            for g in grads:
                grad_norm_sq.add_(g.square().sum())
            lr = current_lr(self.optimizer)
            new_scale = self.brdr_scale.update(
                loss=optimization_loss.detach(), grad_norm_sq=grad_norm_sq, lr=lr, beta_scale=1.0 - lr
            )
            correction = new_scale / max(old_scale, 1e-14)
            if grads:
                torch._foreach_mul_(grads, correction)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
        self.optimizer.step()
        if self.sa_weight is not None:
            self.sa_weight.step()
        self._update_ntk_weights(train_data)
        if self.sa_weight is not None or self.ntk_weight is not None or self.causal_weight is not None:
            raw_loss, loss_dict, aux = self.loss_fn(
                sa=self.sa_weight, ntk=self.ntk_weight, model=self.model, data=train_data,
                sharp_const=sharp_const, rba=self.rba_weight, brdr=self.brdr_weight,
                causal=self.causal_weight, update_weighting=False, hard_periodic=args.hard_periodic,
                lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc, lambda_bc_grad=args.lambda_bc_grad,
                lambda_ini=args.lambda_ini, rho=args.rho
            )
            optimization_loss = raw_loss
        if torch.isnan(raw_loss).any() or torch.isnan(optimization_loss).any():
            return (raw_loss, optimization_loss, loss_dict, aux)
        step_scheduler(self.scheduler, raw_loss.detach().item())
        return (raw_loss, optimization_loss, loss_dict, aux)

    def evaluate_train_grid(self):
        sharp_const = None if self.args.no_sharp_const else self.args.sharp_const
        coords, pred, _ = evaluate_model_on_inner_grid(
            model=self.model, data=self.data, sharp_const=sharp_const
        )
        rel_l2 = relative_l2_error(pred, self.u_ref_train)
        return (rel_l2, coords, pred)

    def evaluate_reference_grid(self):
        coords, pred, _ = evaluate_model_on_inner_grid(
            model=self.model, data=self.eval_data_learning,
            sharp_const=None if self.args.no_sharp_const else self.args.sharp_const
        )
        rel_l2 = relative_l2_error(pred, self.eval_reference)
        return (rel_l2, coords, pred)

    def train(self):
        self.training_cost = TrainingCostTracker(self.model, self.device)
        args = self.args
        save_manifest(
            self.manifest_path, args=args, seed=self.seed, status='running', epoch=0,
            best_loss=self.best_loss, best_epoch=self.best_epoch, stop_reason=self.stop_reason,
            run_dir=self.run_dir
        )
        adam_nan = False
        try:
            for self.epoch in range(1, args.num_epochs + 1):
                self._begin_causal_epoch()
                self.model.train()
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                start = time.perf_counter()
                self.training_cost.adam_epochs_attempted += 1
                loss, brdr_loss, loss_dict, aux = self.train_one_epoch()
                self.training_cost.adam_epochs_completed += 1
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                self.adam_optimization_sec += time.perf_counter() - start
                self.loss_value = float(loss.detach().cpu())
                if torch.isnan(loss).any() or torch.isnan(brdr_loss).any():
                    adam_nan = True
                    self.stop_reason = 'adam_nan'
                    print(f'Adam loss is NaN at epoch {self.epoch}. Restoring the existing Adam checkpoint for L-BFGS.')
                    break
                elapsed = self.adam_optimization_sec
                do_eval = self.epoch == 1 or (args.eval_every > 0 and self.epoch % args.eval_every == 0) or self.epoch == args.num_epochs
                if do_eval:
                    if args.point_type == 'random':
                        rel_l2, _, _ = self.evaluate_reference_grid()
                    else:
                        rel_l2, _, _ = self.evaluate_train_grid()
                    self.last_rel_l2 = rel_l2
                else:
                    rel_l2 = self.last_rel_l2
                improved = self.loss_value < self.best_loss
                if improved:
                    self.best_loss = self.loss_value
                    self.best_epoch = self.epoch
                    self.bad_epochs = 0
                    self.save_checkpoint(self.best_loss_path, self.best_loss)
                    save_analysis_tensors(
                        self.analysis_path, epoch=self.best_epoch, aux=aux, loss=self.best_loss,
                        rel_l2=rel_l2, lr=current_lr(self.optimizer), mode='best'
                    )
                else:
                    self.bad_epochs += 1
                rec = {
                    'epoch': self.epoch, 'seed': self.seed, 'model_name': args.model_name,
                    'scheduler': args.scheduler, 'optimizer': args.optimizer, 'phase': 'adam',
                    'monitor': 'total_loss', 'total': self.loss_value,
                    'pde': float(loss_dict['pde'].detach().cpu()),
                    'bc': float(loss_dict['bc'].detach().cpu()),
                    'bc_grad': float(loss_dict['bc_grad'].detach().cpu()),
                    'ini': float(loss_dict['ini'].detach().cpu()), 'best_loss': self.best_loss,
                    'best_epoch': self.best_epoch, 'relative_l2_error': rel_l2,
                    'lr': current_lr(self.optimizer), 'elapsed_sec': elapsed
                }
                if self.causal_weight is not None:
                    rec.update(
                        causal_stage=self.causal_stage_index + 1, causal_stage_epoch=self.causal_stage_epoch,
                        causal_epsilon=self.causal_weight.epsilon,
                        causal_min_weight=float(aux['causal_min_weight'].detach().cpu())
                    )
                self.last_record = rec
                if self.epoch % args.analysis_every == 0:
                    save_analysis_tensors(
                        self.analysis_path, epoch=self.epoch, aux=aux, loss=self.loss_value, rel_l2=rel_l2,
                        lr=current_lr(self.optimizer), mode='epoch'
                    )
                if self.epoch % args.record_every == 0 or self.epoch == 1 or do_eval:
                    self.history.append(rec)
                    if args.write_jsonl_live:
                        append_jsonl(self.log_path, rec)
                if self.epoch % args.log_every == 0 or self.epoch == 1:
                    print_training_status(
                        seed=self.seed, epoch=self.epoch, num_epochs=args.num_epochs, loss=self.loss_value,
                        best_loss=self.best_loss, rel_l2=rel_l2, loss_dict=loss_dict,
                        lr=current_lr(self.optimizer), best_epoch=self.best_epoch,
                        start_time=self.train_start_time
                    )
                if self._finish_causal_epoch(aux):
                    break
                if self.causal_weight is None:
                    if args.target_loss > 0 and self.loss_value <= args.target_loss and (self.epoch >= args.min_epochs):
                        self.save_checkpoint(self.best_loss_path, self.best_loss)
                        save_analysis_tensors(
                            self.analysis_path, epoch=self.best_epoch, aux=aux, loss=self.best_loss,
                            rel_l2=rel_l2, lr=current_lr(self.optimizer), mode='best'
                        )
                        self.stop_reason = 'target_loss'
                        break
                    if args.early_stop_patience > 0 and self.bad_epochs >= args.early_stop_patience and (self.epoch >= args.min_epochs):
                        self.stop_reason = 'patience_by_total_loss'
                        break
        except KeyboardInterrupt:
            self.interrupted = True
            self.stop_reason = 'keyboard_interrupt'
            print('\nKeyboardInterrupt detected. Saving results...')
        if adam_nan:
            if not (self.best_loss_path.exists() or self.last_path.exists()):
                raise RuntimeError('Adam loss became NaN before a checkpoint was saved; L-BFGS cannot restore a valid Adam state.')
            if args.lbfgs_epochs <= 0:
                print('Adam stopped on NaN; L-BFGS is disabled.')
        elif self.interrupted:
            self.save_checkpoint(self.interrupt_path, self.loss_value)
        else:
            self.save_checkpoint(self.last_path, self.loss_value)
        if not adam_nan and (not self.interrupted) and (getattr(args, 'use_ntk', False) or getattr(args, 'use_asa', False)):
            self.save_checkpoint(self.last_adam_path, self.loss_value)
        self.adam_nan = adam_nan
        if self.causal_weight is not None and (not self.causal_schedule_complete) and (not self.interrupted):
            print('[Causal] Global epoch budget ended before schedule completion; see causal_schedule.jsonl.')
        if args.lbfgs_epochs > 0 and (not self.interrupted):
            self.run_lbfgs()
        return self.finalize()

    def run_lbfgs(self):
        args = self.args
        self.training_cost.begin_lbfgs()
        use_last_adam = getattr(args, 'use_ntk', False) or getattr(args, 'use_asa', False)
        use_last_lbfgs = any((getattr(args, name, False) for name in ('use_sa', 'use_ntk', 'use_asa')))
        checkpoint_path = self.last_adam_path if use_last_adam else self.best_loss_path
        if self.adam_nan or not checkpoint_path.is_file():
            checkpoint_path = self.best_loss_path if self.best_loss_path.is_file() else self.last_path
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self._restore_causal_checkpoint(checkpoint)
        extra = checkpoint.get('extra', {})
        restore_sa_ntk(extra, self.sa_weight, self.ntk_weight, self.device)
        if self.sa_weight is not None:
            self.sa_weight.freeze()
        if self.ntk_weight is not None:
            self.ntk_weight.freeze()
        if self.brdr_weight is not None and extra.get('brdr_weight') is not None:
            self.brdr_weight.load_state_dict(extra['brdr_weight'], device=self.device)
        if self.brdr_scale is not None and extra.get('brdr_scale') is not None:
            self.brdr_scale.load_state_dict(extra['brdr_scale'])
        original_requires_grad = [p.requires_grad for p in self.model.parameters()]
        if hasattr(self.model, 'final_mlp'):
            for p in self.model.parameters():
                p.requires_grad = False
            lbfgs_params = []
            for p in self.model.final_mlp.parameters():
                p.requires_grad = True
            lbfgs_params.extend(self.model.final_mlp.parameters())
            if hasattr(self.model, 'final_mlp_fusion'):
                for p in self.model.final_mlp_fusion.parameters():
                    p.requires_grad = True
                lbfgs_params.extend(self.model.final_mlp_fusion.parameters())
            if hasattr(self.model, 'fusion_norm'):
                for p in self.model.fusion_norm.parameters():
                    p.requires_grad = True
                lbfgs_params.extend(self.model.fusion_norm.parameters())
            self.optimizer_lbfgs = torch.optim.LBFGS(
                lbfgs_params, lr=0.6, max_iter=20, history_size=100, tolerance_grad=1e-08,
                tolerance_change=1e-10, line_search_fn='strong_wolfe'
            )
        else:
            self.optimizer_lbfgs = torch.optim.LBFGS(
                self.model.parameters(), lr=0.6, max_iter=20, history_size=100, tolerance_grad=1e-08,
                tolerance_change=1e-10, line_search_fn='strong_wolfe'
            )
        fixed_scale = self.brdr_scale.get_scale() if self.brdr_scale is not None else 1.0
        sharp_const = None if args.no_sharp_const else args.sharp_const
        self.lbfgs_ran = True
        self.training_cost.set_lbfgs_parameters(self.model)

        def closure():
            self.training_cost.lbfgs_closure_calls += 1
            self.optimizer_lbfgs.zero_grad(set_to_none=True)
            raw_loss, _, _ = self.loss_fn(
                sa=self.sa_weight, ntk=self.ntk_weight, model=self.model, data=self.data,
                sharp_const=sharp_const, rba=self.rba_weight, brdr=self.brdr_weight,
                causal=self.causal_weight, update_weighting=False, hard_periodic=args.hard_periodic,
                lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc, lambda_bc_grad=args.lambda_bc_grad,
                lambda_ini=args.lambda_ini, rho=args.rho
            )
            optimization_loss = raw_loss * fixed_scale
            optimization_loss.backward()
            return optimization_loss
        print('Starting LBFGS refinement...')
        try:
            for lbfgs_epoch in range(1, args.lbfgs_epochs + 1):
                self.model.train()
                self.training_cost.lbfgs_steps_attempted += 1
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                self.optimizer_lbfgs.step(closure)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                self.lbfgs_optimization_sec += time.perf_counter() - start
                self.training_cost.lbfgs_steps_completed += 1
                self.epoch = args.num_epochs + lbfgs_epoch
                raw_loss, loss_dict, aux = self.loss_fn(
                    sa=self.sa_weight, ntk=self.ntk_weight, model=self.model, data=self.data,
                    sharp_const=sharp_const, rba=self.rba_weight, brdr=self.brdr_weight,
                    causal=self.causal_weight, update_weighting=False, hard_periodic=args.hard_periodic,
                    lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc, lambda_bc_grad=args.lambda_bc_grad,
                    lambda_ini=args.lambda_ini, rho=args.rho
                )
                self.loss_value = float(raw_loss.detach().cpu())
                if self.loss_value < self.lbfgs_best_loss:
                    self.lbfgs_best_loss = self.loss_value
                    self.lbfgs_best_step = lbfgs_epoch
                    if not use_last_lbfgs:
                        self.save_checkpoint(
                            self.best_lbfgs_path, self.loss_value, optimizer=self.optimizer_lbfgs,
                            phase='lbfgs'
                        )
                if self.loss_value < self.best_loss:
                    self.best_loss = self.loss_value
                    self.best_epoch = self.epoch
                do_eval = lbfgs_epoch == 1 or lbfgs_epoch == args.lbfgs_epochs or (args.eval_every > 0 and lbfgs_epoch % args.eval_every == 0)
                if do_eval:
                    rel_l2, _, _ = self.evaluate_reference_grid()
                    self.last_rel_l2 = rel_l2
                rec = dict(
                    epoch=self.epoch, seed=self.seed, model_name=args.model_name, phase='lbfgs',
                    total=self.loss_value, best_loss=self.best_loss, best_epoch=self.best_epoch,
                    relative_l2_error=self.last_rel_l2, lr=0.6,
                    elapsed_sec=self.adam_optimization_sec + self.lbfgs_optimization_sec
                )
                rec.update({k: float(loss_dict[k].detach().cpu()) for k in ('pde', 'bc', 'bc_grad', 'ini')})
                self.last_record = rec
                if lbfgs_epoch % args.record_every == 0 or do_eval:
                    self.history.append(rec)
                    if args.write_jsonl_live:
                        append_jsonl(self.log_path, rec)
                if lbfgs_epoch % args.analysis_every == 0:
                    save_analysis_tensors(
                        self.analysis_path, epoch=self.epoch, aux=aux, loss=self.loss_value,
                        rel_l2=self.last_rel_l2, lr=0.6, mode='epoch'
                    )
                if lbfgs_epoch % args.log_every == 0 or do_eval:
                    print_training_status(
                        seed=self.seed, epoch=self.epoch, num_epochs=args.num_epochs + args.lbfgs_epochs,
                        loss=self.loss_value, best_loss=self.best_loss, rel_l2=self.last_rel_l2,
                        loss_dict=loss_dict, lr=0.6, best_epoch=self.best_epoch,
                        start_time=self.train_start_time
                    )
            self.save_checkpoint(
                self.last_lbfgs_path, self.loss_value, optimizer=self.optimizer_lbfgs, phase='lbfgs'
            )
        finally:
            for p, requires_grad in zip(self.model.parameters(), original_requires_grad):
                p.requires_grad_(requires_grad)

    def save_checkpoint(self, path, metric_value, optimizer=None, phase='adam'):
        save_checkpoint(
            path=path, model=self.model, optimizer=self.optimizer if optimizer is None else optimizer,
            scheduler=self.scheduler if phase == 'adam' else None, epoch=self.epoch,
            metric_value=metric_value, args=self.args,
            extra={
                'monitor': 'total_loss', 'seed': self.seed, 'run_dir': str(self.run_dir),
                'stop_reason': self.stop_reason, 'phase': phase,
                'causal_weight': self.causal_weight.state_dict() if self.causal_weight is not None else None,
                'causal_stage_index': self.causal_stage_index, 'causal_stage_epoch': self.causal_stage_epoch,
                'sa_weight': self.sa_weight.state_dict() if self.sa_weight is not None else None,
                'ntk_weight': self.ntk_weight.state_dict() if self.ntk_weight is not None else None,
                'brdr_weight': self.brdr_weight.state_dict() if self.brdr_weight is not None else None,
                'brdr_scale': self.brdr_scale.state_dict() if self.brdr_scale is not None else None
            }
        )

    def finalize(self):
        cost_metrics = self.training_cost.finish(self.metric_dir, self.seed, self.args.model_name)
        args = self.args
        total_time = time.time() - self.train_start_time
        if self.last_record is not None and (len(self.history) == 0 or self.history[-1].get('epoch') != self.last_record.get('epoch')):
            self.history.append(self.last_record)
        if self.interrupted and (not self.interrupt_path.exists()):
            self.save_checkpoint(self.interrupt_path, self.loss_value)
        if not self.lbfgs_ran and (not getattr(self, 'adam_nan', False)) and (not self.last_path.exists()):
            self.save_checkpoint(self.last_path, self.loss_value)
        save_history_jsonl(self.log_path, self.history)
        use_last_lbfgs = any((getattr(args, name, False) for name in ('use_sa', 'use_ntk', 'use_asa')))
        if self.lbfgs_ran:
            eval_checkpoint_path = self.last_lbfgs_path if use_last_lbfgs else self.best_lbfgs_path
        else:
            eval_checkpoint_path = self.best_loss_path if self.best_loss_path.is_file() else self.last_path
        selection_rule = 'last_lbfgs' if self.lbfgs_ran and use_last_lbfgs else 'minimum_total_training_loss'
        checkpoint = torch.load(eval_checkpoint_path, map_location=self.device)
        extra = checkpoint.get('extra', {})
        self.model.load_state_dict(checkpoint['model_state'])
        self._restore_causal_checkpoint(checkpoint)
        restore_sa_ntk(checkpoint.get('extra', {}), self.sa_weight, self.ntk_weight, self.device)
        if self.brdr_weight is not None and extra.get('brdr_weight') is not None:
            self.brdr_weight.load_state_dict(extra['brdr_weight'], device=self.device)
        if self.brdr_scale is not None and extra.get('brdr_scale') is not None:
            self.brdr_scale.load_state_dict(extra['brdr_scale'])
        eval_data = make_allen_cahn_data(h_x=args.eval_h_x, h_t=args.eval_h_t, device=self.device)
        coords_eval, u_pred, aux = evaluate_model_on_inner_grid(
            model=self.model, data=eval_data, sharp_const=None if args.no_sharp_const else args.sharp_const
        )
        u_ref = self.reference.evaluate_tensor(coords_eval)
        rel_l2 = relative_l2_error(u_pred, u_ref)
        torch.save(
            {
                'coords': coords_eval.cpu(), 'u_pred': u_pred.cpu(), 'relative_l2_error': rel_l2,
                'best_loss': self.best_loss, 'best_epoch': self.best_epoch,
                'best_checkpoint': str(eval_checkpoint_path), 'selection_rule': selection_rule,
                'lbfgs_ran': self.lbfgs_ran, 'lbfgs_mode': self.lbfgs_mode,
                'lbfgs_best_loss': self.lbfgs_best_loss, 'lbfgs_best_step': self.lbfgs_best_step
            },
            self.prediction_path
        )
        if args.save_plots:
            save_loss_plot(
                self.history, self.plot_dir / f'loss_curve_seed{self.seed}.png',
                title=f'Loss curve: seed {self.seed}'
            )
            save_lr_plot(
                self.history, self.plot_dir / f'lr_curve_seed{self.seed}.png',
                title=f'LR curve: seed {self.seed}'
            )
            save_error_curve(
                self.history, self.plot_dir / f'error_curve_seed{self.seed}.png',
                title=f'Relative L2 error curve: seed {self.seed}'
            )
            save_prediction_plot(
                coords_eval, u_pred, eval_data, self.plot_dir / f'prediction_best_loss_seed{self.seed}.png',
                title=f'Best-loss prediction: {args.model_name}, seed {self.seed}'
            )
            save_error_contour_plot(
                coords_eval, u_pred, u_ref, eval_data, self.plot_dir / f'error_best_loss_seed{self.seed}.png',
                title=f'Best-loss absolute error: {args.model_name}, seed {self.seed}'
            )
        summary = {
            'seed': self.seed, 'model_name': args.model_name, 'scheduler': args.scheduler,
            'optimizer': args.optimizer, 'lr_init': args.lr, 'final_lr': current_lr(self.optimizer),
            'best_loss': self.best_loss, 'best_epoch': self.best_epoch, 'final_epoch': self.epoch,
            'stop_reason': self.stop_reason, 'relative_l2_error_of_best_loss_model': rel_l2,
            'total_time_sec': total_time, 'time_per_epoch_sec': total_time / max(self.epoch, 1),
            'h_x': args.h_x, 'h_t': args.h_t,
            'sharp_const': None if args.no_sharp_const else args.sharp_const, 'lambda_pde': args.lambda_pde,
            'lambda_bc': args.lambda_bc, 'lambda_bc_grad': args.lambda_bc_grad, 'lambda_ini': args.lambda_ini,
            'lbfgs_ran': self.lbfgs_ran, 'lbfgs_mode': self.lbfgs_mode,
            'lbfgs_outer_steps': args.lbfgs_epochs, 'lbfgs_lr': 0.6, 'lbfgs_max_iter': 20,
            'lbfgs_history_size': getattr(self, 'lbfgs_history_size', None),
            'lbfgs_best_loss': self.lbfgs_best_loss if self.lbfgs_ran else None,
            'lbfgs_best_step': self.lbfgs_best_step if self.lbfgs_ran else None,
            'best_loss_checkpoint': str(eval_checkpoint_path),
            'best_lbfgs_checkpoint': str(self.best_lbfgs_path) if self.best_lbfgs_path.exists() else '',
            'last_lbfgs_checkpoint': str(self.last_lbfgs_path) if self.last_lbfgs_path.exists() else '',
            'last_checkpoint': str(self.last_path),
            'interrupt_checkpoint': str(self.interrupt_path) if self.interrupt_path.exists() else '',
            'prediction_best_loss': str(self.prediction_path), 'run_dir': str(self.run_dir),
            'adam_optimization_sec': self.adam_optimization_sec,
            'lbfgs_optimization_sec': self.lbfgs_optimization_sec,
            'total_optimization_sec': self.adam_optimization_sec + self.lbfgs_optimization_sec,
            'adam_time_per_epoch_sec': self.adam_optimization_sec / max(
                self.training_cost.adam_epochs_completed, 1
            )
        }
        summary.update(cost_metrics)
        write_excel_report(
            self.result_dir / f'{args.excel_name}_seed{self.seed}.xlsx', [summary], self.history, args
        )
        save_json(self.metric_dir / f'metrics_seed{self.seed}.json', summary)
        save_manifest(
            self.manifest_path, args=args, seed=self.seed,
            status='interrupted' if self.interrupted else 'finished', epoch=self.epoch,
            best_loss=self.best_loss, best_epoch=self.best_epoch, stop_reason=self.stop_reason,
            run_dir=self.run_dir, extra={'metrics': summary}
        )
        save_analysis_tensors(
            self.analysis_path, epoch=self.epoch, aux=aux, loss=self.best_loss, rel_l2=rel_l2,
            lr=0.6 if self.lbfgs_ran else current_lr(self.optimizer), mode='final'
        )
        return (summary, self.history)
