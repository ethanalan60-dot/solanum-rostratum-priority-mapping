"""Independent CPU audit of accumulation/skip wiring; no CUDA AMP claim.

The controlled scaler below imitates only the documented step/backoff contract.
Actual CUDA overflow verification is separately owned by proposal_review.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import copy
import json
import random

import torch
from torch import nn

import train_common as tc


class ControlledCPUScaler:
    def __init__(self, skip_attempts=()):
        self.value = 32.
        self.skip_attempts = set(skip_attempts)
        self.attempt = 0
        self.pending = False
        self.found_inf = False
        self.calls = []

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.value

    def scale(self, loss):
        return loss * self.value

    def unscale_(self, optimizer):
        assert not self.pending, 'unscale called twice for one accumulated group'
        self.pending = True
        self.attempt += 1
        self.calls.append(('unscale', self.attempt))
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    parameter.grad.div_(self.value)
        self.found_inf = self.attempt in self.skip_attempts
        if self.found_inf:
            optimizer.param_groups[0]['params'][0].grad.flatten()[0] = float('inf')

    def step(self, optimizer):
        assert self.pending
        self.calls.append(('step', self.attempt))
        if not self.found_inf:
            optimizer.step()

    def update(self):
        assert self.pending
        self.calls.append(('update', self.attempt))
        if self.found_inf:
            self.value /= 2
        self.pending = False


class ToyModel(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.variant = variant

    def forward(self, x, valid):
        logits = self.conv(x)
        return logits.mean((1, 2, 3)) if self.variant == 'regression' else logits


def main(output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.manual_seed(20260916)
    report = dict(test_accessed=False, R6_accessed=False, datasets_accessed=False,
                  gpu_used=False, threads=2, torch=str(torch.__version__),
                  scope='CPU train_epoch wiring with controlled scaler; not CUDA AMP verification',
                  source_sha256={name:tc.sha256_file(tc.ROOT/name) for name in
                                 ['train_common.py', 'conservative_resample.py', Path(__file__).name]},
                  cases=[])
    x = torch.rand(9, 3, 256, 256)
    valid = torch.rand(9, 1, 256, 256)
    valid *= torch.linspace(.2, 1., 9)[:, None, None, None]
    soft = torch.rand_like(valid)
    q = tc.coverage(soft, valid)
    batches = [(x[i:i+2], soft[i:i+2], valid[i:i+2], q[i:i+2]) for i in range(0, 9, 2)]
    for variant in ('regression', 'hard_seg', 'soft_seg'):
        model = ToyModel(variant)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = ControlledCPUScaler({2})
        events = []
        guard = tc.OptimizerStepGuard(optimizer, scaler, events.append)
        args = SimpleNamespace(freeze_batchnorm=False, accumulation_steps=2, variant=variant,
                               label_grid=256, scale_augmentation=False, amp=False, step_pause=0)
        metrics = tc.train_epoch(model, batches, optimizer, scaler, torch.device('cpu'), args,
                                 random.Random(17), guard, epoch=1)
        groups = list(tc.microbatch_groups(batches, 2))
        for index in (0, 2):
            group = groups[index]
            denominator = (sum(len(b[0]) for b in group) if variant == 'regression'
                           else sum(float(b[2].sum(dtype=torch.float64)) for b in group))
            reference_optimizer.zero_grad(set_to_none=True)
            for xx, ss, vv, qq in group:
                logits = reference(tc.degrade_rgb(xx, 1, vv), vv)
                (tc.loss_numerator(logits, ss, vv, qq, variant, 256)/denominator).backward()
            nn.utils.clip_grad_norm_(reference.parameters(), 5., error_if_nonfinite=True)
            reference_optimizer.step()
        maximum_error = max(float((a-b).abs().max()) for a,b in zip(model.parameters(), reference.parameters()))
        assert maximum_error <= 1e-7, maximum_error
        assert metrics['optimizer_updates'] == 2
        assert metrics['optimizer_update_attempts'] == 3
        assert metrics['optimizer_skipped_overflows'] == 1
        assert metrics['train_factor_sample_counts'] == {'1':9, '2':0, '4':0}
        assert [e['group_samples'] for e in events] == [4, 4, 1]
        assert [e['status'] for e in events] == ['updated', 'overflow_skipped', 'updated']
        assert all(p.grad is None for p in model.parameters())
        assert all(float(s['step']) == 2 for s in optimizer.state.values())
        assert scaler.calls == [(call, i) for i in (1,2,3) for call in ('unscale','step','update')]
        # Epoch scheduling remains one transition per completed epoch, not per
        # attempted or actual update; trainer caller implements this policy.
        report['cases'].append(dict(variant=variant, parameter_maximum_error=maximum_error,
                                   metrics=metrics, group_denominators=[e['group_denominator'] for e in events],
                                   groups=[4,4,1], real_adamw_steps=2, cleared_gradients=True,
                                   scaler_calls=scaler.calls))
    # A completed epoch containing only skipped steps must fail before the caller
    # can evaluate, save a checkpoint or advance its epoch scheduler.
    model = ToyModel('regression')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = ControlledCPUScaler({1,2,3})
    guard = tc.OptimizerStepGuard(optimizer, scaler)
    args.variant = 'regression'
    try:
        tc.train_epoch(model,batches,optimizer,scaler,torch.device('cpu'),args,random.Random(17),guard,1)
        raise AssertionError('Zero-update epoch accepted')
    except FloatingPointError as error:
        assert 'zero actual optimizer updates' in str(error)
        assert guard.actual == 0 and guard.skipped == 3
    report.update(zero_update_epoch_rejected=True,passed=True,completed_utc=tc.utc_now())
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'passed':True, 'cases':len(report['cases']), 'output':str(output), 'gpu_used':False}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
