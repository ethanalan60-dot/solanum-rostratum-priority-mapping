"""CPU-only supplement: finite/nonzero gradients for encoder, decoder and head."""
from pathlib import Path
import argparse
import gc
import json
import torch
import train_common as tc


def run(output):
    if output.exists():
        raise FileExistsError(output)
    tc.configure_runtime('cpu', seed=20260916, threads=2)
    x = torch.rand(2, 3, 256, 256)
    valid = torch.rand(2, 1, 256, 256)
    valid[:, :, :30, :50] = 0
    soft = torch.rand_like(valid)
    q = tc.coverage(soft, valid)
    report = {'started_utc': tc.utc_now(), 'device': 'cpu', 'gpu_used': False,
              'datasets_accessed': False, 'test_accessed': False, 'cases': [],
              'source_sha256': {name: tc.sha256_file(tc.ROOT/name) for name in
                                ('train_common.py', 'models_common.py', 'conservative_resample.py', Path(__file__).name)}}
    for encoder in ('resnet18', 'convnextv2_tiny'):
        for variant in ('regression', 'hard_seg', 'soft_seg'):
            model = tc.make_model(encoder, variant, pretrained=True).train()
            logits = model(tc.degrade_rgb(x, 2, valid), valid)
            denominator = len(x) if variant == 'regression' else valid.sum()
            loss = tc.loss_numerator(logits, soft, valid, q, variant, 256)/denominator
            loss.backward()
            groups = {'encoder': [], 'decoder': [], 'head': []}
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if parameter.grad is None or not torch.isfinite(parameter.grad).all() or not parameter.grad.abs().sum() > 0:
                    raise AssertionError(f'{encoder}/{variant}: invalid or zero gradient for {name}')
                group = name.split('.')[0]
                groups[group].append(name)
            report['cases'].append({'encoder': encoder, 'variant': variant, 'loss_finite': bool(torch.isfinite(loss)),
                                    'output_shape': list(logits.shape), 'all_trainable_parameters_passed': True,
                                    'parameter_tensor_counts': {group: len(names) for group, names in groups.items()},
                                    'checked_parameter_names': groups})
            del model, logits, loss
            gc.collect()
    report.update(passed=True, completed_utc=tc.utc_now())
    tc.write_json(output, report)
    print(json.dumps({'passed': True, 'cases': len(report['cases']), 'report': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
