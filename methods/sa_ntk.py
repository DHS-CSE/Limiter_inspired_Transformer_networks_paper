from __future__ import annotations
import argparse
import math
import torch
SUPPORTED_PDES = {'allen_cahn', 'burgers_1d', 'burgers_2d', 'guckenheimer_dirichlet_2d'}

def add_sa_ntk_arguments(parser):
    method = parser.add_mutually_exclusive_group()
    method.add_argument('--use_sa', action='store_true')
    method.add_argument('--use_ntk', action='store_true')
    method.add_argument('--use_asa', '--use_aSA', dest='use_asa', action='store_true')
    parser.add_argument('--sa_weight_mode', choices=['linear', 'square'], default='square')
    parser.add_argument('--sa_optimizer', choices=['auto', 'sgd', 'adam', 'adamw'], default='auto')
    parser.add_argument('--sa_lr', type=float, default=None)
    parser.add_argument('--sa_weight_decay', type=float, default=None)
    for group in ('pde', 'bc', 'bc_grad', 'ini'):
        parser.add_argument('--sa_init_' + group, type=float, default=None)
    parser.add_argument('--sa_min_weight', type=float, default=None)
    parser.add_argument('--ntk_update_every', type=int, default=None)
    parser.add_argument('--ntk_probes', type=int, default=10, help='0 uses notebook probe counts.')
    parser.add_argument('--ntk_trace_eps', type=float, default=1e-12)
    parser.add_argument('--ntk_seed', type=int, default=12345)
    parser.add_argument('--sa_ntk_split_initial', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--sa_ntk_initial_split_x', type=float, default=0.5)

def validate_sa_ntk_args(args):
    flags = ('use_rba', 'use_brdr', 'use_causal', 'use_sa', 'use_ntk', 'use_asa')
    if sum((bool(getattr(args, k, False)) for k in flags)) > 1:
        raise ValueError('Choose only one of RBA, BRDR, Causal, SA, NTK or aSA; use --use_asa for SA+NTK.')
    active = any((getattr(args, k, False) for k in ('use_sa', 'use_ntk', 'use_asa')))
    if not active:
        return
    if args.pde not in SUPPORTED_PDES:
        raise ValueError(f'SA/NTK/aSA supports {sorted(SUPPORTED_PDES)}.')
    if getattr(args, 'use_sa', False) or getattr(args, 'use_asa', False):
        if getattr(args, 'point_type', 'uniform') != 'uniform' or getattr(args, 'use_domain_batch', False):
            raise ValueError('SA/aSA point weights require fixed full-grid coordinates; disable resampling/domain batches.')
    if getattr(args, 'ntk_update_every', None) is not None and args.ntk_update_every < 1:
        raise ValueError('ntk_update_every must be positive.')
    if getattr(args, 'ntk_probes', 10) < 0 or not math.isfinite(getattr(
        args, 'ntk_trace_eps', 1e-12
    )) or getattr(args, 'ntk_trace_eps', 1e-12) <= 0:
        raise ValueError('ntk_probes must be nonnegative and trace eps finite/positive.')
    for name in ('sa_lr', 'sa_init_pde', 'sa_init_bc', 'sa_init_bc_grad', 'sa_init_ini'):
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(name + ' must be finite and positive.')
    if getattr(args, 'sa_weight_decay', None) is not None and args.sa_weight_decay < 0:
        raise ValueError('sa_weight_decay must be nonnegative.')

def validate_loss_methods(rba=None, brdr=None, causal=None, sa=None, ntk=None):
    if sum((rba is not None, brdr is not None, causal is not None, sa is not None or ntk is not None)) > 1:
        raise ValueError('Only one loss weighting method may be enabled.')

def init_sa_ntk(args, seed, device):
    validate_sa_ntk_args(args)
    sa = SAWeights(args, device) if getattr(args, 'use_sa', False) or getattr(args, 'use_asa', False) else None
    ntk = NTKWeights(args, seed, device) if getattr(args, 'use_ntk', False) or getattr(args, 'use_asa', False) else None
    return (sa, ntk)

def residual_groups_for_weights(groups, data, sa=None, ntk=None):
    owner = sa if sa is not None else ntk
    groups = {k: v for k, v in groups.items() if v is not None}
    if owner is not None and owner.pde == 'burgers_1d' and owner.split_initial and ('ini' in groups):
        coords = data.get('ini_coords_inner', data.get('initial_coords_inner'))
        if coords is None:
            raise KeyError('Physical IC coordinates needed for split initial loss.')
        mask = coords[0, :, 0] <= owner.initial_split_x
        if not mask.any() or mask.all():
            raise ValueError('IC split needs two nonempty regions.')
        residual = groups.pop('ini')
        groups.update(ini1=residual[:, mask, :], ini2=residual[:, ~mask, :])
    return groups

def adaptive_group_losses(groups, data, sa=None, ntk=None, update_weighting=True, merge_initial=True):
    groups = residual_groups_for_weights(groups, data, sa, ntk)
    losses = sa.losses(
        groups, trainable=update_weighting
    ) if sa is not None else {k: r.square().mean() for k, r in groups.items()}
    if ntk is not None:
        losses = {k: ntk.coefficients.get(k, 1.0) * value for k, value in losses.items()}
    aux = {}
    if sa is not None:
        aux.update({f'sa_{k}_weight': w.detach() for k, w in sa.weights.items()})
    if ntk is not None:
        ref = next(iter(groups.values()))
        aux.update({f'ntk_{k}_coefficient': ref.new_tensor(ntk.coefficients.get(k, 1.0)) for k in groups})
    if merge_initial and 'ini1' in losses:
        losses['ini'] = losses.pop('ini1') + losses.pop('ini2')
    return (losses, aux)

class SAWeights:

    def __init__(self, args, device):
        self.pde = args.pde
        self.device = torch.device(device)
        ac = self.pde == 'allen_cahn'
        self.mode = getattr(args, 'sa_weight_mode', 'square')
        self.optimizer_name = getattr(args, 'sa_optimizer', 'auto')
        if self.optimizer_name == 'auto':
            self.optimizer_name = 'sgd' if ac else 'adamw'
        lr = getattr(args, 'sa_lr', None)
        self.lr = lr if lr is not None else 0.001 if ac else 0.0001
        decay = getattr(args, 'sa_weight_decay', None)
        self.decay = decay if decay is not None else 0.01 if self.optimizer_name == 'adamw' else 0.0
        self.initial = {k: getattr(args, 'sa_init_' + k, None) for k in ('pde', 'bc', 'bc_grad', 'ini')}
        defaults = {
            'pde': 1.0, 'bc': 3.0, 'bc_grad': 3.0, 'ini': 10.0
        } if ac else {'pde': 10.0, 'bc': 1.0, 'bc_grad': 1.0, 'ini': 1.0}
        self.initial = {k: defaults[k] if v is None else v for k, v in self.initial.items()}
        self.min_weight = getattr(args, 'sa_min_weight', None)
        self.split_initial = getattr(args, 'sa_ntk_split_initial', False)
        self.initial_split_x = getattr(args, 'sa_ntk_initial_split_x', 0.5)
        self.weights = {}
        self.optimizer = None
        self.frozen = False

    def _make_optimizer(self):
        cls = {
            'sgd': torch.optim.SGD, 'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW
        }[self.optimizer_name]
        self.optimizer = cls(list(self.weights.values()), lr=self.lr, weight_decay=self.decay)

    def initialize(self, groups):
        if not self.weights:
            for key, r in groups.items():
                initial = self.initial['ini' if key.startswith('ini') else key]
                self.weights[key] = torch.nn.Parameter(
                    torch.full_like(r, initial), requires_grad=not self.frozen
                )
            self._make_optimizer()
        if set(groups) != set(self.weights) or any((
            r.shape != self.weights[k].shape for k, r in groups.items()
        )):
            raise ValueError('SA residual group shapes changed: point weights require a fixed grid.')

    def losses(self, groups, trainable=True):
        self.initialize(groups)
        losses = {}
        for key, r in groups.items():
            w = self.weights[key] if trainable and (not self.frozen) else self.weights[key].detach()
            losses[key] = (w * r).square().mean() if self.mode == 'square' else (w * r.square()).mean()
        return losses

    def zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        if self.frozen or self.optimizer is None:
            return
        for w in self.weights.values():
            if w.grad is not None:
                w.grad.neg_()
        self.optimizer.step()
        if self.min_weight is not None:
            with torch.no_grad():
                for w in self.weights.values():
                    w.clamp_(min=self.min_weight)

    def freeze(self):
        self.frozen = True
        for w in self.weights.values():
            w.requires_grad_(False)

    def state_dict(self):
        return {
            'pde': self.pde, 'mode': self.mode, 'optimizer_name': self.optimizer_name,
            'split_initial': self.split_initial, 'initial_split_x': self.initial_split_x,
            'weights': {k: v.detach().cpu().clone() for k, v in self.weights.items()},
            'optimizer': self.optimizer.state_dict() if self.optimizer else None, 'frozen': self.frozen
        }

    def load_state_dict(self, state, device=None):
        for key in ('pde', 'mode', 'optimizer_name', 'split_initial', 'initial_split_x'):
            if state[key] != getattr(self, key):
                raise ValueError('SA checkpoint configuration mismatch: ' + key)
        self.frozen = state['frozen']
        self.weights = {
            k: torch.nn.Parameter(v.to(device or self.device), requires_grad=not self.frozen) for k,
            v in state['weights'].items()
        }
        if self.weights:
            self._make_optimizer()
            if state['optimizer'] is not None:
                self.optimizer.load_state_dict(state['optimizer'])

class NTKWeights:

    def __init__(self, args, seed, device):
        self.pde = args.pde
        self.device = torch.device(device)
        self.update_every = getattr(
            args, 'ntk_update_every', None
        ) or (1000 if self.pde == 'allen_cahn' else 2500)
        self.probes = getattr(args, 'ntk_probes', 10)
        self.eps = getattr(args, 'ntk_trace_eps', 1e-12)
        self.generator = torch.Generator(device=self.device).manual_seed(getattr(args, 'ntk_seed', 12345) + seed)
        self.split_initial = getattr(args, 'sa_ntk_split_initial', False)
        self.initial_split_x = getattr(args, 'sa_ntk_initial_split_x', 0.5)
        self.coefficients = {}
        self.traces = {}
        self.last_update_epoch = None
        self.frozen = False

    def should_update(self, epoch):
        return not self.frozen and (epoch - 1) % self.update_every == 0 and (self.last_update_epoch != epoch)

    def estimate_trace(self, output, parameters, probes):
        params = tuple((p for p in parameters if p.requires_grad))
        if not params or not output.requires_grad:
            return 0.0
        trace = output.new_zeros(())
        for _ in range(probes):
            v = torch.randint(
                0, 2, output.shape, device=output.device, generator=self.generator
            ).to(output.dtype) * 2 - 1
            grads = torch.autograd.grad(output, params, grad_outputs=v, retain_graph=True, allow_unused=True)
            trace += sum((g.detach().square().sum() for g in grads if g is not None), output.new_zeros(()))
        return float(trace / probes)

    def update(self, model, groups, epoch):
        if not self.should_update(epoch):
            return False
        traces = {}
        for key, r in groups.items():
            m = self.probes or (max(1, r.numel() // 10) if self.pde != 'allen_cahn' and key == 'pde' else max(1, r.numel()))
            traces[key] = self.estimate_trace(r, model.parameters(), m)
        total = sum(traces.values())
        total_tensor = torch.tensor(total, device=self.device)
        self.coefficients = {
            k: (total_tensor / torch.tensor(t, device=self.device)).item() for k, t in traces.items()
        }
        self.traces = traces
        self.last_update_epoch = epoch
        return True

    def freeze(self):
        self.frozen = True

    def state_dict(self):
        return {
            'pde': self.pde, 'split_initial': self.split_initial, 'initial_split_x': self.initial_split_x,
            'coefficients': dict(self.coefficients), 'traces': dict(self.traces),
            'last_update_epoch': self.last_update_epoch, 'frozen': self.frozen,
            'generator_state': self.generator.get_state().cpu()
        }

    def load_state_dict(self, state):
        for key in ('pde', 'split_initial', 'initial_split_x'):
            if state[key] != getattr(self, key):
                raise ValueError('NTK checkpoint configuration mismatch: ' + key)
        self.coefficients = dict(state['coefficients'])
        self.traces = dict(state['traces'])
        self.last_update_epoch = state['last_update_epoch']
        self.frozen = state['frozen']
        self.generator.set_state(state['generator_state'].cpu())

def restore_sa_ntk(extra, sa=None, ntk=None, device=None):
    for key, obj in (('sa_weight', sa), ('ntk_weight', ntk)):
        state = extra.get(key)
        if obj is not None and state is not None:
            if key == 'sa_weight':
                obj.load_state_dict(state, device=device)
            else:
                obj.load_state_dict(state)
        elif obj is not None and state is None:
            raise ValueError('Checkpoint is missing ' + key + ' state for the enabled method.')

def require_sa_ntk_loss(loss_fn):
    import inspect
    required = {'sa', 'ntk', 'return_residual_graph'}
    if not required.issubset(inspect.signature(loss_fn).parameters):
        raise RuntimeError('The current periodic loss has not been integrated with SA/NTK. Provide losses/guckenheimer2d_periodic.py to complete the inline update.')
