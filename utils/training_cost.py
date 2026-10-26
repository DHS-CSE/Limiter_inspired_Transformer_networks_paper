import json
import time
from pathlib import Path
import torch

class TrainingCostTracker:

    def __init__(self, model, device):
        self.device = torch.device(device)
        self.is_cuda = self.device.type == 'cuda'
        self.model_parameter_count = sum((p.numel() for p in model.parameters()))
        self.adam_trainable_parameter_count = sum((p.numel() for p in model.parameters() if p.requires_grad))
        self.lbfgs_trainable_parameter_count = 0
        self.adam_epochs_attempted = 0
        self.adam_epochs_completed = 0
        self.lbfgs_steps_attempted = 0
        self.lbfgs_steps_completed = 0
        self.lbfgs_closure_calls = 0
        self.phase_seconds = {'adam': 0.0, 'lbfgs': 0.0}
        self._phase = None
        self._phase_start = None
        self._report = None
        self.metadata = {'cost_device': str(self.device), 'cost_gpu_name': torch.cuda.get_device_name(self.device) if self.is_cuda else None, 'cost_torch_version': str(torch.__version__), 'cost_cuda_version': torch.version.cuda if self.is_cuda else None, 'cost_scope': 'train entry to finalize entry; includes in-training evaluation, diagnostics, checkpoint I/O and logging; excludes initialization and final evaluation/plots', 'cost_memory_scope': 'peak PyTorch allocation/reservation over the same training interval; includes allocations already live at interval start', 'cost_memory_unit': 'GiB (2**30 bytes)'}
        self._synchronize()
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._start = time.perf_counter()
        self._phase_start = self._start
        self._phase = 'adam'

    def _synchronize(self):
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

    def begin_lbfgs(self):
        if self._report is not None or self._phase == 'lbfgs':
            return
        self._synchronize()
        now = time.perf_counter()
        self.phase_seconds[self._phase] += now - self._phase_start
        self._phase = 'lbfgs'
        self._phase_start = now

    def set_lbfgs_parameters(self, model):
        self.lbfgs_trainable_parameter_count = sum((p.numel() for p in model.parameters() if p.requires_grad))

    def finish(self, metric_dir, seed, model_name):
        if self._report is not None:
            return dict(self._report)
        self._synchronize()
        now = time.perf_counter()
        self.phase_seconds[self._phase] += now - self._phase_start
        self._report = {'seed': seed, 'model_name': model_name, 'training_wall_time_sec': now - self._start, 'adam_stage_wall_time_sec': self.phase_seconds['adam'], 'lbfgs_stage_wall_time_sec': self.phase_seconds['lbfgs'], 'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated(self.device) / 2 ** 30 if self.is_cuda else None, 'peak_gpu_reserved_gib': torch.cuda.max_memory_reserved(self.device) / 2 ** 30 if self.is_cuda else None, 'model_parameter_count': self.model_parameter_count, 'adam_trainable_parameter_count': self.adam_trainable_parameter_count, 'lbfgs_trainable_parameter_count': self.lbfgs_trainable_parameter_count, 'adam_epochs_attempted': self.adam_epochs_attempted, 'adam_epochs_completed': self.adam_epochs_completed, 'lbfgs_steps_attempted': self.lbfgs_steps_attempted, 'lbfgs_steps_completed': self.lbfgs_steps_completed, 'lbfgs_closure_calls': self.lbfgs_closure_calls, **self.metadata}
        path = Path(metric_dir) / f'training_cost_seed{seed}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._report, indent=2, ensure_ascii=False), encoding='utf-8')
        memory = self._report['peak_gpu_allocated_gib']
        memory_text = f'{memory:.3f} GiB' if memory is not None else 'N/A (CPU)'
        print(f"[Training cost | seed{seed}] Adam={self.phase_seconds['adam']:.2f}s, L-BFGS={self.phase_seconds['lbfgs']:.2f}s, total={now - self._start:.2f}s, peak allocated={memory_text}")
        return dict(self._report)
