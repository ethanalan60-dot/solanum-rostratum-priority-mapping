"""Bounded correctness checks; optional unique-ID small GPU smoke, never a score.

CPU uses synthetic arrays and both official pretrained encoders. GPU smoke reads
only sealed local_v2 train/val, 9 train and 2 val samples, one epoch per case.
"""
from pathlib import Path
import argparse
import gc
import json
import subprocess
import sys
import uuid

import torch

import train_common as tc


def close(a, b, tolerance=2e-6):
    error = float((a-b).abs().max())
    if error > tolerance:
        raise AssertionError(f'Error {error} > {tolerance}')
    return error


def run_checks(output, gpu_smoke=False):
    if output.exists():
        raise FileExistsError(output)
    device = tc.configure_runtime('cpu', threads=2)
    torch.manual_seed(20260916)
    report = {'started_utc': tc.utc_now(), 'test_accessed': False,
              'source_sha256': {name: tc.sha256_file(tc.ROOT/name) for name in
                                ('train_common.py', 'models_common.py', 'conservative_resample.py', 'train_local.py')},
              'cpu_checks': {}, 'gpu_smoke_runs': []}
    x = torch.rand(3, 3, 256, 256)
    valid = torch.rand(3, 1, 256, 256)
    valid[:, :, :35, :65] = 0
    valid[1] *= .37
    soft = torch.rand_like(valid)
    q = tc.coverage(soft, valid)
    changed = x.clone()
    changed.masked_fill_((valid == 0).expand_as(x), 1000.)
    report['cpu_checks']['invalid_rgb_invariance'] = {
        str(r): close(tc.degrade_rgb(x, r, valid), tc.degrade_rgb(changed, r, valid)) for r in tc.VALIDATION_R}
    mass_errors = {}
    for grid in (64, 128, 256):
        coarse, weights = tc.normalized_area_resize(soft, valid, (grid, grid))
        mass_errors[str(grid)] = close(tc.coverage(coarse, weights), q)
    report['cpu_checks']['target_area_conservation'] = mass_errors

    # Unequal valid mass and a one-sample tail expose incorrect /accum_steps loss.
    accumulation = {}
    for variant in ('hard_seg', 'soft_seg', 'regression'):
        for grid in ((64, 128, 256) if variant != 'regression' else (256,)):
            initial = torch.randn(3 if variant == 'regression' else (3, 1, 256, 256), dtype=torch.float32)
            joint = initial.clone().requires_grad_()
            denominator = 3 if variant == 'regression' else float(valid.sum(dtype=torch.float64))
            joint_loss = tc.loss_numerator(joint, soft, valid, q, variant, grid)/denominator
            joint_loss.backward()
            accumulated = initial.clone().requires_grad_()
            for subset in (slice(0, 2), slice(2, 3)):
                loss = tc.loss_numerator(accumulated[subset], soft[subset], valid[subset], q[subset], variant, grid)/denominator
                loss.backward()
            accumulation[f'{variant}_L{grid}'] = close(joint.grad, accumulated.grad, 1e-7)
    report['cpu_checks']['effective_batch_loss_gradient_equivalence'] = accumulation

    # Every encoder/task path must reach all of its trainable parameters.
    shapes = []
    for encoder in ('resnet18', 'convnextv2_tiny'):
        for variant in ('soft_seg', 'regression'):
            model = tc.make_model(encoder, variant, pretrained=True).eval()
            z = model(tc.degrade_rgb(x[:1], 2, valid[:1]), valid[:1])
            expected = (1,) if variant == 'regression' else (1, 1, 256, 256)
            if tuple(z.shape) != expected:
                raise AssertionError((encoder, variant, z.shape))
            denominator = 1 if variant == 'regression' else valid[:1].sum()
            loss = tc.loss_numerator(z, soft[:1], valid[:1], q[:1], variant, 64 if variant == 'soft_seg' else 256)/denominator
            loss.backward()
            missing = [name for name, p in model.encoder.named_parameters()
                       if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum() > 0)]
            if missing:
                raise AssertionError(f'Encoder gradients missing/nonfinite/zero: {missing}')
            shapes.append({'encoder': encoder, 'variant': variant, 'output_shape': list(z.shape),
                           'encoder_gradients_passed': True, 'strict_pretrained_report': model.pretrained_report})
            del model, z, loss
            gc.collect()
    report['cpu_checks']['models'] = shapes

    # Refusing an existing run must not even append failure.json to that run.
    sentinel_id = 'smoke_common_guard_'+uuid.uuid4().hex[:12]
    sentinel = tc.ROOT/'runs'/sentinel_id
    sentinel.mkdir()
    marker = sentinel/'ownership_marker.txt'
    marker.write_text('immutable run ownership probe', encoding='utf-8')
    args = tc.parser().parse_args(['--run-id', sentinel_id, '--variant', 'soft_seg', '--device', 'cpu'])
    try:
        tc.run(args)
        raise AssertionError('Existing run unexpectedly accepted')
    except FileExistsError:
        if list(sentinel.iterdir()) != [marker]:
            raise AssertionError('Existing run was mutated on failure')
    report['cpu_checks']['existing_run_refusal'] = {'passed': True, 'sentinel_directory': str(sentinel)}
    report['cpu_checks']['passed'] = True
    tc.write_json(output, report)

    if gpu_smoke:
        cases = [('regression', 256), ('hard_seg', 128), ('soft_seg', 64)]
        for encoder in ('resnet18', 'convnextv2_tiny'):
            for variant, grid in cases:
                run_id = 'smoke_common_'+encoder+'_'+variant+'_'+uuid.uuid4().hex[:10]
                command = [sys.executable, '-X', 'utf8', '-B', '-s', str(tc.ROOT/'train_common.py'), '--run-id', run_id,
                           '--encoder', encoder, '--variant', variant, '--label-grid', str(grid), '--smoke',
                           '--epochs', '1', '--max-train', '9', '--max-val', '2', '--device', 'cuda',
                           '--batch-size', '4', '--accumulation-steps', '2', '--gpu-memory-fraction', '.16']
                print(json.dumps({'starting_smoke': run_id}), flush=True)
                result = subprocess.run(command, cwd=tc.ROOT, text=True, encoding='utf-8', errors='replace', capture_output=True)
                entry = {'run_id': run_id, 'command': command, 'returncode': result.returncode,
                         'stdout_tail': result.stdout[-5000:], 'stderr_tail': result.stderr[-3000:]}
                status_path = tc.ROOT/'runs'/run_id/'status.json'
                if status_path.exists():
                    entry['status'] = json.loads(status_path.read_text(encoding='utf-8'))
                report['gpu_smoke_runs'].append(entry)
                tc.write_json(output, report)
                if result.returncode:
                    raise RuntimeError(f'Smoke failed: {run_id}; see {output}')
                print(json.dumps({'completed_smoke': run_id, 'peak_allocated_bytes': entry['status']['gpu_peak_allocated_bytes']}), flush=True)
    if tc.sha256_file(tc.ROOT/'train_local.py') != report['source_sha256']['train_local.py']:
        raise AssertionError('Legacy trainer changed during checks')
    report.update(completed_utc=tc.utc_now(), passed=True)
    tc.write_json(output, report)
    print(json.dumps({'passed': True, 'report': str(output), 'gpu_smoke_cases': len(report['gpu_smoke_runs'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-smoke', action='store_true')
    args = parser.parse_args()
    run_checks(args.output, args.gpu_smoke)
