from pathlib import Path
from utils.trainer_utils import (
    append_jsonl, current_lr, evaluate_model_on_inner_grid, make_burgers_data, print_training_status,
    relative_l2_error, save_analysis_tensors, save_checkpoint, save_error_contour_burgers_plot,
    save_error_curve, save_history_jsonl, save_json, save_loss_burgers_plot, save_lr_plot, save_manifest,
    save_prediction_burgers_plot, write_excel_report,
)
import math
import time
import torch
import gc
from build_scheduler import step_scheduler
from reference.reference_burgers import burgers_solution
from methods.brdr import BRDRWeights, BRDRScale
from methods.rba import RBAWeights
from methods.causal import CausalPINNWeighting
_USE_DEFAULT = object()
from methods.sa_ntk import init_sa_ntk, restore_sa_ntk
from utils.training_cost import TrainingCostTracker

class BurgersTrainer:

    def __init__(self, args, seed: int, run_dir, device, model, optimizer, scheduler, data, loss_fn, reference_function, u_ref):
        self.args = args
        self.seed = seed
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data = data
        self.loss_fn = loss_fn
        self.sa_weight, self.ntk_weight = init_sa_ntk(args, seed, device)
        if getattr(model, 'input_type', 'PINN') == 'PINN':
            for owner in (self.sa_weight, self.ntk_weight):
                if owner is not None:
                    owner.split_initial = True
                    owner.initial_split_x = 0.5
        self.run_dir = Path(run_dir)
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.bad_epochs = 0
        self.stop_reason = 'max_epochs'
        self.history = []
        self.ntk_ratio_list = []
        self.ntk_step_list = []
        self.limiter_ratio_list = []
        self.last_record = None
        self.loss_value = float('inf')
        self.epoch = 0
        self.interrupted = False
        self.train_start_time = time.time()
        self.reference = reference_function
        self.u_ref_train = u_ref
        self.brdr_weight = None
        self.brdr_scale = None
        self.rba_weight = None
        self.causal_weight = None
        self.eval_data_learning = make_burgers_data(h_x=args.eval_h_x_learning, h_t=args.eval_h_t_learning, device=device)
        self.eval_reference = self.reference(self.eval_data_learning['domain_coords_inner'])
        if args.use_rba and args.use_brdr:
            raise ValueError('--use_rba and --use_brdr cannot be enabled together.')
        if args.use_rba:
            self.rba_weight = RBAWeights(shape=(1, self.data['domain_coords_inner'].shape[1], 1), device=device, gamma=args.rba_gamma, eta=args.rba_eta, offset=args.rba_offset)
        if args.use_brdr:
            self.brdr_weight = BRDRWeights(beta_residual=args.brdr_beta_residual, beta_weight=args.brdr_beta_weight, eps=args.brdr_eps)
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
            if not self.causal_schedule or any((not math.isfinite(v) or v <= 0 for v in self.causal_schedule)) or any((b <= a for a, b in zip(self.causal_schedule, self.causal_schedule[1:]))):
                raise ValueError('causal_epsilon_schedule must be finite, positive and strictly increasing.')
            self.causal_stage_limit = int(args.causal_stage_epochs)
            self.causal_delta = float(args.causal_delta)
            if self.causal_stage_limit < 1 or not 0 < self.causal_delta < 1:
                raise ValueError('Require causal_stage_epochs >= 1 and 0 < causal_delta < 1.')
            self.causal_weight = CausalPINNWeighting(epsilon=self.causal_schedule[0], num_time_bins=getattr(args, 'causal_num_time_bins', None), include_initial=args.causal_include_initial, normalization=args.causal_normalization)
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
            raise KeyError("The Burgers loss must return aux['causal_min_weight'].")
        value = aux['causal_min_weight']
        minimum = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        if not math.isfinite(minimum):
            raise FloatingPointError('Non-finite causal min weight; a stage transition cannot be assessed.')
        self.causal_last_min_weight = minimum
        criterion_met = minimum > self.causal_delta
        capped = self.causal_stage_epoch >= self.causal_stage_limit
        reason = 'weight_threshold' if criterion_met else 'stage_epoch_limit' if capped else 'global_epoch_limit' if self.epoch == self.args.num_epochs else ''
        record = dict(epoch=self.epoch, stage=self.causal_stage_index + 1, stage_epoch=self.causal_stage_epoch, epsilon=self.causal_weight.epsilon, min_weight=minimum, delta=self.causal_delta, criterion_met=criterion_met, stage_end_reason=reason)
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
        for d in [self.checkpoint_dir, self.analysis_dir, self.log_dir, self.plot_dir, self.result_dir, self.metric_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.seed_ckpt_dir = self.checkpoint_dir / f'seed{self.seed}'
        self.seed_analysis_dir = self.analysis_dir / f'seed{self.seed}'
        self.seed_ckpt_dir.mkdir(exist_ok=True)
        self.seed_analysis_dir.mkdir(exist_ok=True)
        self.best_loss_path = self.seed_ckpt_dir / args.best_ckpt_name
        self.best_loss_lbfgs_path = self.seed_ckpt_dir / f'lbfgs_{args.best_ckpt_name}'
        self.interrupt_path = self.seed_ckpt_dir / args.interrupt_ckpt_name
        self.last_path = self.seed_ckpt_dir / args.last_ckpt_name
        self.last_adam_path = self.seed_ckpt_dir / ('last_adam' + self.best_loss_path.suffix)
        self.last_lbfgs_path = self.seed_ckpt_dir / ('last_lbfgs' + self.best_loss_path.suffix)
        self.log_path = self.log_dir / f'{args.history_name}_seed{self.seed}.jsonl'
        self.manifest_path = self.metric_dir / f'{args.manifest_name}_seed{self.seed}.json'
        self.best_metrics_path = self.metric_dir / f'{args.metrics_name}_seed{self.seed}.json'
        self.prediction_path = self.seed_analysis_dir / f'{args.prediction_name}_seed{self.seed}.pt'
        self.analysis_path = self.seed_analysis_dir / 'analysis.pt'
        self.ntk_ratio_path = self.seed_analysis_dir / 'ntk_ratio.pt'

    def estimate_ntk_trace(self, output, num_samples=10):
        output = output.reshape(-1)
        params = [p for p in self.model.parameters() if p.requires_grad]
        trace_estimate = torch.zeros((), device=output.device, dtype=output.dtype)
        for _ in range(num_samples):
            v = torch.empty_like(output).bernoulli_(0.5)
            v = v.mul_(2.0).sub_(1.0)
            grads = torch.autograd.grad(outputs=output, inputs=params, grad_outputs=v, retain_graph=True, create_graph=False, allow_unused=True)
            sample_trace = torch.zeros((), device=output.device, dtype=output.dtype)
            for g in grads:
                if g is not None:
                    sample_trace += torch.sum(g ** 2)
            trace_estimate += sample_trace
        trace_estimate /= num_samples
        mean_trace = trace_estimate / output.numel()
        return (trace_estimate, mean_trace)

    def analyze_ntk_ratio(self, lhs, limiter):
        residual_mag = torch.abs(lhs.detach())
        threshold = torch.quantile(residual_mag.reshape(-1), 0.9)
        shock_mask = residual_mag > threshold
        smooth_mask = residual_mag <= threshold
        shock_mask_flat = shock_mask.reshape(-1)
        smooth_mask_flat = smooth_mask.reshape(-1)
        lhs_flat = lhs.reshape(-1, 1)
        lhs_shock = lhs_flat[shock_mask_flat]
        lhs_smooth = lhs_flat[smooth_mask_flat]
        if lhs_shock.numel() == 0 or lhs_smooth.numel() == 0:
            return None
        limiter_flat = limiter.reshape(-1, 1)
        limiter_shock = limiter_flat[shock_mask_flat]
        limiter_smooth = limiter_flat[smooth_mask_flat]
        trace_K_shock, mean_trace_K_shock = self.estimate_ntk_trace(lhs_shock, num_samples=min(100, lhs_shock.shape[0]))
        trace_K_smooth, mean_trace_K_smooth = self.estimate_ntk_trace(lhs_smooth, num_samples=min(100, lhs_smooth.shape[0]))
        ntk_ratio = trace_K_shock / (trace_K_smooth + 1e-12)
        limiter_ratio = limiter_shock.detach().mean() / (limiter_smooth.detach().mean() + 1e-12)
        return {'ntk_ratio': float(ntk_ratio.detach().cpu()), 'limiter_ratio': float(limiter_ratio.cpu()), 'trace_shock': float(trace_K_shock.detach().cpu()), 'trace_smooth': float(trace_K_smooth.detach().cpu()), 'threshold': float(threshold.detach().cpu())}

    def _update_ntk_weights(self, data):
        if self.ntk_weight is None or not self.ntk_weight.should_update(self.epoch):
            return
        args = self.args
        sharp_const = None if args.no_sharp_const else args.sharp_const
        with torch.enable_grad():
            _, _, graph_aux = self.loss_fn(sa=self.sa_weight, ntk=self.ntk_weight, model=self.model, data=data, rba=self.rba_weight, brdr=self.brdr_weight, causal=self.causal_weight, update_weighting=False, sharp_const=sharp_const, lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc, lambda_ini=args.lambda_ini, lambda_ini1=getattr(args, 'lambda_ini1', None), lambda_ini2=getattr(args, 'lambda_ini2', None), return_residual_graph=True)
            self.ntk_weight.update(self.model, graph_aux['adaptive_residual_groups'], self.epoch)

    def train_one_epoch(self):
        args = self.args
        optimizer = self.optimizer
        optimizer.zero_grad(set_to_none=True)
        if self.sa_weight is not None:
            self.sa_weight.zero_grad()
        sharp_const = None if args.no_sharp_const else args.sharp_const
        raw_loss, loss_dict, aux = self.loss_fn(sa=self.sa_weight, ntk=self.ntk_weight, model=self.model, data=self.data, rba=self.rba_weight, brdr=self.brdr_weight, causal=self.causal_weight, update_weighting=True, sharp_const=sharp_const, lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc, lambda_ini=args.lambda_ini, lambda_ini1=getattr(args, 'lambda_ini1', None), lambda_ini2=getattr(args, 'lambda_ini2', None))
        optimization_loss = raw_loss
        if self.brdr_scale is not None:
            old_scale = self.brdr_scale.get_scale()
            optimization_loss = raw_loss * old_scale
        else:
            old_scale = None
            optimization_loss = raw_loss
        optimization_loss.backward()
        if self.brdr_scale is not None:
            grads = [p.grad for p in self.trainable_params if p.grad is not None]
            grad_norm_sq = torch.zeros((), device=self.device)
            for g in grads:
                grad_norm_sq.add_(g.square().sum())
            lr = current_lr(optimizer)
            new_scale = self.brdr_scale.update(loss=optimization_loss.detach(), grad_norm_sq=grad_norm_sq, lr=lr, beta_scale=1.0 - lr)
            correction = new_scale / max(old_scale, 1e-14)
            torch._foreach_mul_(grads, correction)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
        optimizer.step()
        if self.sa_weight is not None:
            self.sa_weight.step()
        self._update_ntk_weights(self.data)
        if self.sa_weight is not None or self.ntk_weight is not None or self.causal_weight is not None:
            raw_loss, loss_dict, aux = self.loss_fn(sa=self.sa_weight, ntk=self.ntk_weight, model=self.model, data=self.data, rba=self.rba_weight, brdr=self.brdr_weight, causal=self.causal_weight, update_weighting=False, sharp_const=sharp_const, lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc, lambda_ini=args.lambda_ini, lambda_ini1=getattr(args, 'lambda_ini1', None), lambda_ini2=getattr(args, 'lambda_ini2', None))
            optimization_loss = raw_loss
        if self.epoch < args.scheduler_stop + 1:
            step_scheduler(self.scheduler, raw_loss.detach().item())
        return (raw_loss, optimization_loss, loss_dict, aux)

    def evaluate_train_grid(self):
        sharp_const = None if self.args.no_sharp_const else self.args.sharp_const
        coords, pred, _ = evaluate_model_on_inner_grid(model=self.model, data=self.data, sharp_const=sharp_const)
        rel_l2 = relative_l2_error(pred, self.u_ref_train)
        return (rel_l2, coords, pred)

    def evaluate_reference_grid(self):
        coords, pred, _ = evaluate_model_on_inner_grid(model=self.model, data=self.eval_data_learning, sharp_const=None if self.args.no_sharp_const else self.args.sharp_const)
        rel_l2 = relative_l2_error(pred, self.eval_reference)
        return (rel_l2, coords, pred)

    def train(self):
        self.training_cost = TrainingCostTracker(self.model, self.device)
        args = self.args
        optimizer = self.optimizer
        save_manifest(self.manifest_path, args=args, seed=self.seed, status='running', epoch=0, best_loss=self.best_loss, best_epoch=self.best_epoch, stop_reason=self.stop_reason, run_dir=self.run_dir)
        try:
            for self.epoch in range(1, args.num_epochs + 1):
                self._begin_causal_epoch()
                self.model.train()
                self.training_cost.adam_epochs_attempted += 1
                loss, optimization_loss, loss_dict, aux = self.train_one_epoch()
                self.training_cost.adam_epochs_completed += 1
                if getattr(args, 'compute_ntk_ratio', False) and self.epoch % 1000 == 0 and (getattr(self.model, 'input_type', 'PINN') != 'PINN'):
                    self.model.train()
                    sharp_const = None if args.no_sharp_const else args.sharp_const
                    _, _, ntk_aux = self.loss_fn(sa=self.sa_weight, ntk=self.ntk_weight, model=self.model, data=self.data, rba=self.rba_weight, brdr=self.brdr_weight, causal=self.causal_weight, update_weighting=False, sharp_const=sharp_const, lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc, lambda_ini=args.lambda_ini, lambda_ini1=getattr(args, 'lambda_ini1', None), lambda_ini2=getattr(args, 'lambda_ini2', None))
                    lhs = ntk_aux['pde_residual']
                    limiter = ntk_aux['m_dom_q']
                    ntk_result = self.analyze_ntk_ratio(lhs=lhs, limiter=limiter)
                    if ntk_result is not None:
                        self.ntk_step_list.append(self.epoch)
                        self.ntk_ratio_list.append(ntk_result['ntk_ratio'])
                        self.limiter_ratio_list.append(ntk_result['limiter_ratio'])
                        print(f"[NTK] epoch={self.epoch:6d} | ratio={ntk_result['ntk_ratio']:.6f} | limiter_ratio={ntk_result['limiter_ratio']:.6f} | trace_hard={ntk_result['trace_shock']:.4e} | trace_smooth={ntk_result['trace_smooth']:.4e}")
                        torch.save({'epoch': self.ntk_step_list, 'ntk_ratio': self.ntk_ratio_list, 'limiter_ratio': self.limiter_ratio_list}, self.ntk_ratio_path)
                    del ntk_aux, lhs, limiter
                self.loss_value = float(loss.detach().cpu())
                elapsed = time.time() - self.train_start_time
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
                else:
                    self.bad_epochs += 1
                if self.epoch % args.analysis_every == 0:
                    save_analysis_tensors(self.analysis_path, epoch=self.epoch, aux=aux, loss=self.loss_value, rel_l2=rel_l2, lr=current_lr(optimizer), mode='epoch')
                if self.epoch % args.record_every == 0 or self.epoch == 1 or do_eval:
                    rec = {'epoch': self.epoch, 'seed': self.seed, 'model_name': args.model_name, 'scheduler': args.scheduler, 'optimizer': args.optimizer, 'monitor': 'total_loss', 'total': self.loss_value, 'pde': float(loss_dict['pde'].detach().cpu()), 'bc': float(loss_dict['bc'].detach().cpu()), 'ini': float(loss_dict['ini'].detach().cpu()), 'best_loss': self.best_loss, 'best_epoch': self.best_epoch, 'relative_l2_error': rel_l2, 'lr': current_lr(optimizer), 'elapsed_sec': elapsed}
                    if 'ini1' in loss_dict:
                        rec.update({key: float(loss_dict[key].detach().cpu()) for key in ('ini1', 'ini2', 'ini1_weighted', 'ini2_weighted')})
                    if self.causal_weight is not None:
                        rec.update(causal_stage=self.causal_stage_index + 1, causal_stage_epoch=self.causal_stage_epoch, causal_epsilon=self.causal_weight.epsilon, causal_min_weight=float(aux['causal_min_weight'].detach().cpu()))
                    self.history.append(rec)
                    if args.write_jsonl_live:
                        append_jsonl(self.log_path, rec)
                    self.last_record = rec
                if self.epoch % args.log_every == 0 or self.epoch == 1:
                    print_training_status(seed=self.seed, epoch=self.epoch, num_epochs=args.num_epochs, loss=self.loss_value, best_loss=self.best_loss, rel_l2=rel_l2, loss_dict=loss_dict, lr=current_lr(optimizer), best_epoch=self.best_epoch, start_time=self.train_start_time)
                    self.print_initial_losses(loss_dict)
                if self._finish_causal_epoch(aux):
                    break
                if self.causal_weight is None:
                    if args.target_loss > 0 and self.loss_value <= args.target_loss and (self.epoch >= args.min_epochs):
                        self.stop_reason = 'target_loss'
                        break
                    if args.early_stop_patience > 0 and self.bad_epochs >= args.early_stop_patience and (self.epoch >= args.min_epochs):
                        self.stop_reason = 'patience_by_total_loss'
                        break
        except KeyboardInterrupt:
            self.interrupted = True
            self.stop_reason = 'keyboard_interrupt'
            print('\nKeyboardInterrupt detected. Saving results...')
        if self.causal_weight is not None and (not self.causal_schedule_complete) and (not self.interrupted):
            print(f'[Causal] Global training budget ended before schedule completion; last trained epsilon={self.causal_weight.epsilon:g}. See causal_schedule.jsonl.')
        if self.interrupted:
            return self.finalize()
        if self.stop_reason == 'non_finite_loss':
            return self.finalize()
        if args.lbfgs_epochs <= 0:
            return self.finalize()
        if getattr(args, 'use_ntk', False) or getattr(args, 'use_asa', False):
            self.save_checkpoint(self.last_adam_path, self.loss_value)
        self.last_adam_lr = current_lr(self.optimizer)
        self.optimizer = None
        gc.collect()
        torch.cuda.empty_cache()
        return self.run_lbfgs()

    def run_lbfgs(self):
        self.training_cost.begin_lbfgs()
        use_last_adam = getattr(self.args, 'use_ntk', False) or getattr(self.args, 'use_asa', False)
        use_last_lbfgs = any((getattr(self.args, name, False) for name in ('use_sa', 'use_ntk', 'use_asa')))
        checkpoint_path = self.last_adam_path if use_last_adam else self.best_loss_path
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self._restore_causal_checkpoint(checkpoint)
        restore_sa_ntk(checkpoint.get('extra', {}), self.sa_weight, self.ntk_weight, self.device)
        if self.sa_weight is not None:
            self.sa_weight.freeze()
        if self.ntk_weight is not None:
            self.ntk_weight.freeze()
        extra = checkpoint.get('extra', {})
        brdr_weight_state = extra.get('brdr_weight')
        brdr_scale_state = extra.get('brdr_scale')
        if self.brdr_weight is not None and brdr_weight_state is not None:
            self.brdr_weight.load_state_dict(brdr_weight_state, device=self.device)
        if self.brdr_scale is not None and brdr_scale_state is not None:
            self.brdr_scale.load_state_dict(brdr_scale_state)
        if hasattr(self.model, 'final_mlp'):
            for p in self.model.parameters():
                p.requires_grad = False
            lbfgs_params = []
            if hasattr(self.model, 'final_mlp_fusion'):
                for p in self.model.final_mlp_fusion.parameters():
                    p.requires_grad = True
                lbfgs_params.extend(self.model.final_mlp_fusion.parameters())
            for p in self.model.final_mlp.parameters():
                p.requires_grad = True
            lbfgs_params.extend(self.model.final_mlp.parameters())
            if hasattr(self.model, 'fusion_norm'):
                for p in self.model.fusion_norm.parameters():
                    p.requires_grad = True
                lbfgs_params.extend(self.model.fusion_norm.parameters())
            self.optimizer_lbfgs = torch.optim.LBFGS(lbfgs_params, lr=0.6, max_iter=20, history_size=100, tolerance_grad=1e-08, tolerance_change=1e-10, line_search_fn='strong_wolfe')
        else:
            self.optimizer_lbfgs = torch.optim.LBFGS(self.model.parameters(), lr=0.6, max_iter=20, history_size=100, tolerance_grad=1e-08, tolerance_change=1e-10, line_search_fn='strong_wolfe')
        sharp_const = None if self.args.no_sharp_const else self.args.sharp_const
        fixed_scale = self.brdr_scale.get_scale() if self.brdr_scale is not None else 1.0

        def closure():
            self.training_cost.lbfgs_closure_calls += 1
            self.optimizer_lbfgs.zero_grad(set_to_none=True)
            raw_loss, _, _ = self.loss_fn(sa=self.sa_weight, ntk=self.ntk_weight, model=self.model, data=self.data, rba=self.rba_weight, brdr=self.brdr_weight, causal=self.causal_weight, update_weighting=False, sharp_const=sharp_const, lambda_pde=self.args.lambda_pde, lambda_bc=self.args.lambda_bc, lambda_ini=self.args.lambda_ini, lambda_ini1=getattr(self.args, 'lambda_ini1', None), lambda_ini2=getattr(self.args, 'lambda_ini2', None))
            optimization_loss = raw_loss * fixed_scale
            optimization_loss.backward()
            return optimization_loss
        print('Starting LBFGS refinement...')
        self.training_cost.set_lbfgs_parameters(self.model)
        for lbfgs_epoch in range(1, self.args.lbfgs_epochs + 1):
            self.training_cost.lbfgs_steps_attempted += 1
            self.optimizer_lbfgs.step(closure)
            self.training_cost.lbfgs_steps_completed += 1
            self.epoch = self.args.num_epochs + lbfgs_epoch
            raw_loss, loss_dict, aux = self.loss_fn(sa=self.sa_weight, ntk=self.ntk_weight, model=self.model, data=self.data, rba=self.rba_weight, brdr=self.brdr_weight, causal=self.causal_weight, update_weighting=False, sharp_const=sharp_const, lambda_pde=self.args.lambda_pde, lambda_bc=self.args.lambda_bc, lambda_ini=self.args.lambda_ini, lambda_ini1=getattr(self.args, 'lambda_ini1', None), lambda_ini2=getattr(self.args, 'lambda_ini2', None))
            self.loss_value = float(raw_loss.detach().cpu())
            if self.loss_value < self.best_loss:
                self.best_loss = self.loss_value
                self.best_epoch = self.epoch
                if not use_last_lbfgs:
                    self.save_checkpoint(self.best_loss_lbfgs_path, self.best_loss, optimizer=self.optimizer_lbfgs, scheduler=None)
                if self.args.point_type == 'random':
                    rel_l2, _, _ = self.evaluate_reference_grid()
                else:
                    rel_l2, _, _ = self.evaluate_train_grid()
                save_analysis_tensors(self.analysis_path, epoch=self.best_epoch, aux=aux, loss=self.best_loss, rel_l2=rel_l2, lr=0.6, mode='best')
            if lbfgs_epoch == 1 or lbfgs_epoch % 100 == 0 or lbfgs_epoch == self.args.lbfgs_epochs:
                if self.args.point_type == 'random':
                    rel_l2, _, _ = self.evaluate_reference_grid()
                else:
                    rel_l2, _, _ = self.evaluate_train_grid()
                self.last_rel_l2 = rel_l2
                print(f'[LBFGS] epoch={lbfgs_epoch:5d}, raw_loss={self.loss_value:.4e}, best_loss={self.best_loss:.4e}')
                self.print_initial_losses(loss_dict)
            else:
                rel_l2 = self.last_rel_l2
        if use_last_lbfgs:
            self.save_checkpoint(self.last_lbfgs_path, self.loss_value, optimizer=self.optimizer_lbfgs, scheduler=None)
        self.lbfgs_completed = True
        for p in self.model.parameters():
            p.requires_grad = True
        self.optimizer_lbfgs = None
        gc.collect()
        torch.cuda.empty_cache()
        return self.finalize()

    @staticmethod
    def print_initial_losses(loss_dict):
        if 'ini1' not in loss_dict:
            return
        values = {key: float(loss_dict[key].detach().cpu()) for key in ('ini1', 'ini2', 'ini1_weighted', 'ini2_weighted')}
        print(f"[Initial] ini1={values['ini1']:.4e}, ini2={values['ini2']:.4e}, weighted_ini1={values['ini1_weighted']:.4e}, weighted_ini2={values['ini2_weighted']:.4e}")

    def save_sa_visualization(self, checkpoint_path):
        if self.sa_weight is None or not self.sa_weight.weights:
            return
        sa = self.sa_weight
        coords_by_group = {'pde': self.data['domain_coords_inner'], 'bc': self.data['bnd_coords'], 'ini': self.data['ini_coords_inner']}
        groups = {}
        for key, weight in sa.weights.items():
            coords = coords_by_group['ini' if key.startswith('ini') else key]
            if key in ('ini1', 'ini2'):
                mask = coords[0, :, 0] <= sa.initial_split_x
                coords = coords[:, mask if key == 'ini1' else ~mask, :]
            if coords.shape[:-1] != weight.shape[:-1]:
                raise ValueError(f'SA coordinate/weight mismatch for {key}.')
            raw = weight.detach().cpu().clone()
            coefficient = self.ntk_weight.coefficients.get(key, 1.0) if self.ntk_weight is not None else 1.0
            effective = raw.square() if sa.mode == 'square' else raw.clone()
            groups[key] = {'coords': coords.detach().cpu().clone(), 'parameter': raw, 'effective_weight': effective, 'ntk_coefficient': coefficient, 'combined_weight': coefficient * effective}
        path = Path(checkpoint_path)
        torch.save({'epoch': self.epoch, 'checkpoint': str(path), 'mode': sa.mode, 'split_initial': sa.split_initial, 'initial_split_x': sa.initial_split_x, 'groups': groups}, path.with_name(path.stem + '_sa_weights.pt'))

    def save_checkpoint(self, path, metric_value, optimizer=None, scheduler=_USE_DEFAULT):
        if optimizer is None:
            optimizer = None if self.optimizer is None else self.optimizer
        if scheduler is _USE_DEFAULT:
            scheduler = self.scheduler
        save_checkpoint(path=path, model=self.model, optimizer=optimizer, scheduler=scheduler, epoch=self.epoch, metric_value=metric_value, args=self.args, extra={'monitor': 'raw_total_loss', 'seed': self.seed, 'run_dir': str(self.run_dir), 'stop_reason': self.stop_reason, 'sa_weight': self.sa_weight.state_dict() if self.sa_weight is not None else None, 'ntk_weight': self.ntk_weight.state_dict() if self.ntk_weight is not None else None, 'brdr_weight': self.brdr_weight.state_dict() if self.brdr_weight is not None else None, 'causal_weight': self.causal_weight.state_dict() if self.causal_weight is not None else None, 'causal_schedule_state': dict(schedule=self.causal_schedule, stage=self.causal_stage_index + 1, stage_epoch=self.causal_stage_epoch, schedule_complete=self.causal_schedule_complete) if self.causal_weight is not None else None, 'brdr_scale': self.brdr_scale.state_dict() if self.brdr_scale is not None else None})
        self.save_sa_visualization(path)

    def finalize(self):
        cost_metrics = self.training_cost.finish(self.metric_dir, self.seed, self.args.model_name)
        args = self.args
        total_time = time.time() - self.train_start_time
        if self.last_record is not None and (len(self.history) == 0 or self.history[-1].get('epoch') != self.last_record.get('epoch')):
            self.history.append(self.last_record)
        if self.interrupted:
            self.save_checkpoint(self.interrupt_path, self.loss_value)
        self.save_checkpoint(self.last_path, self.loss_value, optimizer=None)
        save_history_jsonl(self.log_path, self.history)
        use_last_lbfgs = any((getattr(args, name, False) for name in ('use_sa', 'use_ntk', 'use_asa')))
        if use_last_lbfgs and self.lbfgs_completed:
            eval_checkpoint_path = self.last_lbfgs_path
            selection_rule = 'last_lbfgs_model'
        elif not use_last_lbfgs and self.best_loss_lbfgs_path.exists():
            eval_checkpoint_path = self.best_loss_lbfgs_path
            selection_rule = 'minimum_total_training_loss'
        else:
            eval_checkpoint_path = self.best_loss_path if self.best_loss_path.exists() else self.last_path
            selection_rule = 'minimum_total_training_loss'
        checkpoint = torch.load(eval_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self._restore_causal_checkpoint(checkpoint)
        restore_sa_ntk(checkpoint.get('extra', {}), self.sa_weight, self.ntk_weight, self.device)
        eval_data = make_burgers_data(h_x=args.eval_h_x, h_t=args.eval_h_t, device=self.device)
        coords_eval_ghost, u_pred, aux = evaluate_model_on_inner_grid(model=self.model, data=eval_data, sharp_const=None if args.no_sharp_const else args.sharp_const)
        coords_eval = eval_data['domain_coords_inner']
        u_ref_eval = self.reference(eval_data['domain_coords_inner'])
        rel_l2 = relative_l2_error(u_pred, u_ref_eval)
        torch.save({'coords': coords_eval.cpu(), 'u_pred': u_pred.cpu(), 'relative_l2_error': rel_l2, 'best_loss': self.best_loss, 'best_epoch': self.best_epoch, 'best_checkpoint': str(eval_checkpoint_path), 'selection_rule': selection_rule}, self.prediction_path)
        if args.save_plots:
            save_loss_burgers_plot(self.history, self.plot_dir / f'loss_curve_seed{self.seed}.png', title=f'Loss curve: seed {self.seed}')
            save_lr_plot(self.history, self.plot_dir / f'lr_curve_seed{self.seed}.png', title=f'LR curve: seed {self.seed}')
            save_error_curve(self.history, self.plot_dir / f'error_curve_seed{self.seed}.png', title=f'Relative L2 error curve: seed {self.seed}')
            save_prediction_burgers_plot(coords_eval, u_pred, eval_data, self.plot_dir / f'prediction_best_loss_seed{self.seed}.png', title=f'{selection_rule} Best-loss prediction: {args.model_name}, seed {self.seed}')
            save_error_contour_burgers_plot(coords_eval, u_pred, u_ref_eval, eval_data, self.plot_dir / f'error_best_loss_seed{self.seed}.png', title=f'{selection_rule} Best-loss absolute error: {args.model_name}, seed {self.seed}')
        summary = {'seed': self.seed, 'model_name': args.model_name, 'scheduler': args.scheduler, 'optimizer': f'{args.optimizer} + lbfgs', 'lr_init': args.lr, 'adam_lr': self.last_adam_lr, 'lbfgs_lr': 0.6, 'best_loss': self.best_loss, 'best_epoch': self.best_epoch, 'final_epoch': self.epoch, 'selection_rule': selection_rule, 'stop_reason': self.stop_reason, 'relative_l2_error_of_best_loss_model': rel_l2, 'total_time_sec': total_time, 'time_per_epoch_sec': total_time / max(self.epoch, 1), 'h_x': args.h_x, 'h_t': args.h_t, 'sharp_const': None if args.no_sharp_const else args.sharp_const, 'lambda_pde': args.lambda_pde, 'lambda_bc': args.lambda_bc, 'lambda_ini': args.lambda_ini, 'lambda_ini1': args.lambda_ini if getattr(args, 'lambda_ini1', None) is None else args.lambda_ini1, 'lambda_ini2': args.lambda_ini if getattr(args, 'lambda_ini2', None) is None else args.lambda_ini2, 'best_loss_checkpoint': str(eval_checkpoint_path), 'last_checkpoint': str(self.last_path), 'interrupt_checkpoint': str(self.interrupt_path) if self.interrupt_path.exists() else '', 'prediction_best_loss': str(self.prediction_path), 'run_dir': str(self.run_dir)}
        summary.update(cost_metrics)
        write_excel_report(self.result_dir / f'{args.excel_name}_seed{self.seed}.xlsx', [summary], self.history, args)
        save_json(self.metric_dir / f'metrics_seed{self.seed}.json', summary)
        save_manifest(self.manifest_path, args=args, seed=self.seed, status='interrupted' if self.interrupted else 'finished', epoch=self.epoch, best_loss=self.best_loss, best_epoch=self.best_epoch, stop_reason=self.stop_reason, run_dir=self.run_dir, extra={'metrics': summary})
        save_analysis_tensors(self.analysis_path, epoch=self.epoch, aux=aux, loss=self.loss_value, rel_l2=rel_l2, lr=self.last_adam_lr, mode='final')
        return (summary, self.history)
