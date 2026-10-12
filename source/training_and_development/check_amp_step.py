"""Targeted CPU/CUDA checks of finite updates, AMP skip/backoff and hard guards."""
from pathlib import Path
import argparse
import copy
import json
import torch
from torch import nn
import train_common as tc


def equal_state(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def run(output):
    if output.exists():
        raise FileExistsError(output)
    tc.configure_runtime('cpu', threads=2)
    report = {'started_utc':tc.utc_now(),'test_accessed':False,'datasets_accessed':False,
              'source_sha256':{'train_common.py':tc.sha256_file(tc.ROOT/'train_common.py')}}
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    guard = tc.OptimizerStepGuard(optimizer, torch.cuda.amp.GradScaler(enabled=False))
    before = copy.deepcopy(model.state_dict())
    for p in model.parameters():
        p.grad = torch.full_like(p, float('inf'))
    try:
        guard.finish(model)
        raise AssertionError('CPU nonfinite gradients accepted')
    except FloatingPointError:
        assert equal_state(before, model.state_dict()) and guard.actual == 0
        assert all(p.grad is None for p in model.parameters())
    report['cpu_nonfinite_gradients_rejected'] = True
    for p in model.parameters():
        p.grad = torch.full_like(p, 1e30)
    try:
        guard.finish(model)
        raise AssertionError('Finite-gradient norm overflow accepted')
    except RuntimeError:
        assert guard.actual == 0 and guard.last_event['all_unscaled_gradients_finite']
    report['finite_gradient_norm_overflow_rejected'] = True
    try:
        tc.require_finite_forward(torch.tensor([float('inf')]), model)
        raise AssertionError('Inf logits accepted')
    except FloatingPointError:
        report['nonfinite_logits_rejected'] = True
    model.register_buffer('bad_buffer', torch.tensor(float('nan')))
    try:
        tc.require_finite_forward(torch.zeros(1), model)
        raise AssertionError('NaN model buffer accepted')
    except FloatingPointError:
        report['nonfinite_buffers_rejected'] = True

    device = tc.configure_runtime('cuda', threads=2, gpu_memory_fraction=.16)
    model = nn.Linear(3, 2).to(device)
    with torch.no_grad():
        model.weight.fill_(.01)
        model.bias.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.1)
    scaler = torch.cuda.amp.GradScaler(init_scale=65536.)
    events = []
    guard = tc.OptimizerStepGuard(optimizer, scaler, events.append)
    x = torch.ones(3, 3, device=device)

    def backward_group(multiplier=1.):
        for batch in (x[:2], x[2:]):
            with torch.autocast(device_type='cuda'):
                logits = model(batch)
            tc.require_finite_forward(logits, model)
            loss = logits.float().square().sum()/x.numel()*multiplier
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()

    backward_group()
    guard.finish(model, {'test_group':'finite_2plus1'})
    assert guard.actual == 1 and guard.skipped == 0
    parameters_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    backward_group(1e8)  # Finite forward/loss; scaled FP16 backward overflows.
    overflow = guard.finish(model, {'test_group':'finite_loss_overflow_2plus1'})
    assert overflow['skipped'] and overflow['scale_after'] == overflow['scale_before']/2
    assert guard.actual == 1 and guard.skipped == 1
    assert equal_state(parameters_before, model.state_dict())
    assert equal_state(optimizer_before, optimizer.state_dict())
    assert all(p.grad is None for p in model.parameters())
    backward_group()
    recovered = guard.finish(model, {'test_group':'finite_recovery_2plus1'})
    assert not recovered['skipped'] and guard.actual == 2 and guard.attempted == 3
    assert not equal_state(parameters_before, model.state_dict())
    assert all(float(state['step']) == 2 for state in optimizer.state.values())
    assert all(p.grad is None for p in model.parameters())
    report.update(gpu_events=events, overflow_preserves_parameters_and_adamw_state=True,
                  recovered_finite_update=True, all_groups_zeroed=True)

    persistent = tc.OptimizerStepGuard(optimizer, scaler, max_consecutive_overflows=2)
    for index in range(2):
        backward_group(1e8)
        try:
            persistent.finish(model)
            assert index == 0
        except FloatingPointError:
            assert index == 1 and persistent.consecutive_overflows == 2
    report.update(persistent_overflow_stops=True,passed=True,completed_utc=tc.utc_now(),
                  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    tc.write_json(output, report)
    print(json.dumps({'passed':True,'output':str(output),'overflow_scale':overflow['scale_after']}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
