"""Development-only common-encoder baselines; never reads a sealed-test manifest.

Segmentation: the same models_common decoder for both encoders. Regression:
the same encoder with valid-area-pooled last-stage features and a linear head.
R degrades RGB independently of L, the supervision grid. Predictions are always
256x256; this is a matched supervision adaptation, not low-resolution inference.
"""
from pathlib import Path
import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from conservative_resample import normalized_area_resize
from models_common import FeatureEncoder, build_model, sha256_file

ROOT = Path(__file__).resolve().parent
PRETRAINED_SHA256 = {
    'resnet18': 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec',
    'convnextv2_tiny': 'b23c89e2e601d2459bc35f8fef1bf6a3e9c90ab44b2f5b2d1117b984ac7f89a4',
}
VALIDATION_R = (1, 2, 4)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    """Replace only files inside an already exclusively owned run directory."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def configure_runtime(device_name, seed=17, threads=2, gpu_memory_fraction=.16):
    # Runtime caches are kept under the campaign directory.
    for key, sub in [('TORCH_HOME', 'cache/torch'), ('HF_HOME', 'cache/huggingface'),
                     ('TEMP', 'tmp'), ('TMP', 'tmp')]:
        folder = ROOT/sub
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    device = torch.device('cuda:0' if device_name == 'cuda' else device_name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback.')
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device


def load_development_manifests():
    """Only the fixed local_v2 train, val and lock files are opened."""
    data_dir = ROOT/'local_data_v2'
    lock_path = data_dir/'DATASET_LOCK.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    rows, hashes = {}, {}
    for split in ('train', 'val'):
        path = data_dir/(split+'.json')
        digest = sha256_file(path)
        if digest != lock['manifest_sha256'][path.name]:
            raise RuntimeError(f'{split} manifest changed after sealing')
        records = json.loads(path.read_text(encoding='utf-8'))
        ids = [r['id'] for r in records]
        coordinates = [(r['source_id'], r['x'], r['y']) for r in records]
        if len(set(ids)) != len(ids) or len(set(coordinates)) != len(coordinates):
            raise RuntimeError(f'Duplicate source coordinates or IDs in {split}')
        if not records or any(r.get('split') != split for r in records):
            raise RuntimeError(f'Unexpected split content in {split}')
        for r in records:
            cache = Path(r['cache']).resolve()
            if not cache.is_relative_to((data_dir/'cache').resolve()):
                raise RuntimeError('Cache path outside the sealed local_v2 cache directory')
            if not r.get('cache_sha256'):
                raise RuntimeError('Every cache needs a sealed SHA256')
        rows[split], hashes[split+'.json'] = records, digest
    train_keys = {(r['source_id'], r['x'], r['y']) for r in rows['train']}
    val_keys = {(r['source_id'], r['x'], r['y']) for r in rows['val']}
    if train_keys & val_keys or {r['id'] for r in rows['train']} & {r['id'] for r in rows['val']}:
        raise RuntimeError('Train/validation source overlap')
    hashes['DATASET_LOCK.json'] = sha256_file(lock_path)
    return rows, hashes


def bounded_subset(records, count):
    if not count:
        return records
    positives = [r for r in records if r['coverage'] > 0]
    zeros = [r for r in records if r['coverage'] == 0]
    selected = positives[:count//2] + zeros[:count-count//2]
    if len(selected) < count:
        used = {r['id'] for r in selected}
        selected += [r for r in records if r['id'] not in used][:count-len(selected)]
    return selected


class DevelopmentTiles(Dataset):
    def __init__(self, records, augment=False, seed=17):
        self.records, self.augment = records, augment
        self.rng = random.Random(seed)
        self.verified = set()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        if r['id'] not in self.verified:
            if sha256_file(r['cache']) != r['cache_sha256']:
                raise RuntimeError(f"Cache changed: {r['id']}")
            self.verified.add(r['id'])
        with np.load(r['cache'], allow_pickle=False) as cache:
            x = cache['image'].copy()
            soft, valid = cache['soft'].astype(np.float32), cache['valid'].astype(np.float32)
        if x.shape != (256, 256, 3) or soft.shape != (256, 256) or valid.shape != soft.shape:
            raise ValueError(f"Unexpected cache shape: {r['id']}")
        if not np.isfinite(soft).all() or not np.isfinite(valid).all() or np.any((valid < 0) | (valid > 1)) or np.any((soft < 0) | (soft > 1)) or valid.sum() <= 0:
            raise ValueError(f"Invalid cache values: {r['id']}")
        if self.augment:
            if self.rng.random() < .5:
                x, soft, valid = x[:, ::-1], soft[:, ::-1], valid[:, ::-1]
            if self.rng.random() < .5:
                x, soft, valid = x[::-1], soft[::-1], valid[::-1]
            x = np.clip(x.astype(np.float32)*self.rng.uniform(.9, 1.1)+self.rng.uniform(-8, 8), 0, 255)
        x[valid == 0] = 0
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()/255,
                torch.from_numpy(soft.copy())[None], torch.from_numpy(valid.copy())[None],
                torch.tensor(r['coverage'], dtype=torch.float32))


def degrade_rgb(x, factor, valid):
    """Same-footprint valid-normalized RGB degradation then ImageNet normalization."""
    if factor not in VALIDATION_R:
        raise ValueError('Development RGB factors are restricted to 1,2,4')
    if factor != 1:
        small, small_valid = normalized_area_resize(x, valid, (x.shape[-2]//factor, x.shape[-1]//factor))
        numerator = F.interpolate(small*small_valid, size=x.shape[-2:], mode='bilinear', align_corners=False)
        denominator = F.interpolate(small_valid, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.where(denominator > 0, numerator/torch.where(denominator > 0, denominator, torch.ones_like(denominator)), torch.zeros_like(numerator))
    x = torch.where(valid > 0, x, torch.zeros_like(x))
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    return (x-mean)/std


def coverage(probability, valid):
    denominator = valid.sum((1, 2, 3))
    numerator = (torch.where(valid > 0, probability, torch.zeros_like(probability))*valid).sum((1, 2, 3))
    return torch.where(denominator > 0, numerator/torch.where(denominator > 0, denominator, torch.ones_like(denominator)), torch.zeros_like(numerator))


def require_finite_forward(logits, model):
    # Sigmoid can hide +/-Inf logits; GradScaler does not protect BN buffers.
    values = [logits] + [buffer for buffer in model.buffers() if buffer.is_floating_point()]
    if not bool(torch.stack([torch.isfinite(value).all() for value in values]).all()):
        raise FloatingPointError('Nonfinite forward logits or model buffers; AMP skip is not applicable')


class OptimizerStepGuard:
    """Public GradScaler workflow with observable, bounded overflow handling.

    Finite gradients still use error_if_nonfinite=True clipping. If unscale_
    detects Inf/NaN, do not clip: let scaler.step skip and scaler.update back off.
    A public optimizer post-step hook proves whether the optimizer really ran.
    """
    def __init__(self, optimizer, scaler, event_writer=None, max_consecutive_overflows=8, min_scale=1.):
        self.optimizer, self.scaler = optimizer, scaler
        self.event_writer = event_writer
        self.max_consecutive_overflows, self.min_scale = max_consecutive_overflows, min_scale
        self.attempted = self.actual = self.skipped = self.consecutive_overflows = 0
        self.last_event = None
        self.hook = optimizer.register_step_post_hook(self._after_optimizer_step)

    def _after_optimizer_step(self, optimizer, args, kwargs):
        self.actual += 1

    def finish(self, model, context=None):
        self.attempted += 1
        event = dict(context or {})
        event.update(attempt=self.attempted, amp_enabled=self.scaler.is_enabled(),
                     scale_before=float(self.scaler.get_scale()), optimizer_step_executed=False)
        before = self.actual
        try:
            # Exactly once, after the complete accumulated group.
            self.scaler.unscale_(self.optimizer)
            named = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
            if not named:
                raise RuntimeError('No gradients for optimizer attempt')
            finite_flags = torch.stack([torch.isfinite(p.grad).all() for _, p in named])
            finite = bool(finite_flags.all())
            event['all_unscaled_gradients_finite'] = finite
            if finite:
                # A finite-element gradient norm overflow is a separate hard failure.
                event['unscaled_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5., error_if_nonfinite=True))
            else:
                event['nonfinite_gradient_tensors'] = [name for (name, _), ok in zip(named, finite_flags.tolist()) if not ok]
                if not self.scaler.is_enabled():
                    raise FloatingPointError('Nonfinite gradients with AMP disabled')
            self.scaler.step(self.optimizer)
            self.scaler.update()
            event['scale_after'] = float(self.scaler.get_scale())
            event['optimizer_step_executed'] = self.actual == before+1
            event['skipped'] = not finite
            if self.actual-before != int(finite):
                raise RuntimeError('GradScaler optimizer skip disagrees with unscaled finite-gradient inspection')
            if not math.isfinite(event['scale_after']):
                raise FloatingPointError('Nonfinite AMP scale')
            if not finite:
                self.skipped += 1
                self.consecutive_overflows += 1
                if not event['scale_after'] < event['scale_before']:
                    raise RuntimeError('AMP overflow did not reduce scale')
                if self.consecutive_overflows >= self.max_consecutive_overflows or event['scale_after'] < self.min_scale:
                    raise FloatingPointError('Persistent AMP overflow or scale below safety floor')
            else:
                self.consecutive_overflows = 0
            event.update(status='overflow_skipped' if not finite else 'updated',
                         actual_updates_total=self.actual, skipped_updates_total=self.skipped,
                         consecutive_overflows=self.consecutive_overflows)
            return event
        except BaseException as error:
            event.update(status='failed', error=str(error), actual_updates_total=self.actual,
                         skipped_updates_total=self.skipped, consecutive_overflows=self.consecutive_overflows)
            raise
        finally:
            # Clear the entire accumulated group after success, skip or hard failure.
            self.optimizer.zero_grad(set_to_none=True)
            self.last_event = event
            if self.event_writer is not None:
                self.event_writer(event)


class CommonRegressionModel(nn.Module):
    """Direct window regression; intentionally no spatial decoder or inferred map."""
    def __init__(self, encoder, pretrained=True):
        super().__init__()
        self.encoder = FeatureEncoder(encoder, pretrained=pretrained,
                                      expected_sha256=PRETRAINED_SHA256[encoder] if pretrained else None)
        self.head = nn.Linear(self.encoder.channels[-1], 1)
        nn.init.constant_(self.head.bias, -2.5)

    @property
    def pretrained_report(self):
        return self.encoder.pretrained_report

    def forward(self, x, valid):
        features = self.encoder(x)[-1]
        # Features occupy the same nominal footprint; valid is an area weight,
        # not a claim that convolutional receptive fields exclude invalid pixels.
        _, weights = normalized_area_resize(valid, valid, features.shape[-2:])
        denominator = weights.sum((2, 3))
        pooled = (features.float()*weights).sum((2, 3))/denominator.clamp_min(torch.finfo(torch.float32).tiny)
        return self.head(pooled).squeeze(1)


def make_model(encoder, variant, pretrained=True):
    if variant == 'regression':
        return CommonRegressionModel(encoder, pretrained)
    return build_model(encoder, pretrained=pretrained,
                       expected_sha256=PRETRAINED_SHA256[encoder] if pretrained else None)


def loss_numerator(logits, soft, valid, q, variant, label_grid):
    """Return loss sum in 256-grid area units (regression: sample units).

    Aggregates sigmoid probabilities, never logits. Hard targets use majority
    valid area >=0.5 after exact soft-label aggregation; this is deliberately
    distinct from local_v2's cached nearest-neighbor hard mask.
    """
    if variant == 'regression':
        return F.smooth_l1_loss(logits.float().sigmoid(), q, beta=.1, reduction='sum')
    if variant not in ('hard_seg', 'soft_seg') or label_grid not in (64, 128, 256):
        raise ValueError('Unknown segmentation variant or supervision grid')
    if (label_grid, label_grid) == soft.shape[-2:]:
        probability, target, grid_valid = logits.float().sigmoid(), soft, valid
    else:
        probability, grid_valid = normalized_area_resize(logits.float().sigmoid(), valid, (label_grid, label_grid))
        target, _ = normalized_area_resize(soft, valid, (label_grid, label_grid))
    if variant == 'hard_seg':
        target = (target >= .5).float()
    # Explicit common clipping rule makes probability-space BCE finite under AMP.
    bce = F.binary_cross_entropy(probability.clamp(1e-7, 1-1e-7), target, reduction='none')
    cell_area = valid.shape[-2]*valid.shape[-1]/(label_grid*label_grid)
    return (bce*grid_valid).sum()*cell_area


def summarize(predicted, true):
    p, t = np.asarray(predicted, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if len(t) == 0 or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Empty or nonfinite evaluation')
    error, positive = p-t, t > 0
    zero_mae = float(np.abs(error[~positive]).mean()) if (~positive).any() else None
    positive_mae = float(np.abs(error[positive]).mean()) if positive.any() else None
    return {'n': len(t), 'n_positive': int(positive.sum()), 'mae': float(np.abs(error).mean()),
            'bias': float(error.mean()), 'mae_zero': zero_mae, 'mae_positive': positive_mae,
            'balanced_mae': float(np.mean([v for v in (zero_mae, positive_mae) if v is not None]))}


@torch.inference_mode()
def evaluate(model, loader, device, variant, amp):
    model.eval()
    results = {}
    for factor in VALIDATION_R:
        predicted, true = [], []
        tp = fp = fn = 0.
        for x, soft, valid, q in loader:
            x, soft, valid = x.to(device), soft.to(device), valid.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(degrade_rgb(x, factor, valid), valid)
            require_finite_forward(logits, model)
            p = logits.float().sigmoid()
            estimated = p if variant == 'regression' else coverage(p, valid)
            predicted.extend(estimated.cpu().tolist())
            true.extend(q.tolist())
            if variant != 'regression':
                binary = (p > .5).float()
                tp += float((binary*soft*valid).sum())
                fp += float((binary*(1-soft)*valid).sum())
                fn += float(((1-binary)*soft*valid).sum())
        result = summarize(predicted, true)
        if variant != 'regression':
            result.update(target_iou=tp/max(tp+fp+fn, 1), dice=2*tp/max(2*tp+fp+fn, 1))
        results[str(factor)] = result
    return {'selection_balanced_mae': float(np.mean([v['balanced_mae'] for v in results.values()])),
            'scales': results}


def microbatch_groups(loader, accumulation_steps):
    group = []
    for batch in loader:
        group.append(batch)
        if len(group) == accumulation_steps:
            yield group
            group = []
    if group:
        yield group


def train_epoch(model, loader, optimizer, scaler, device, args, factor_rng, step_guard, epoch):
    model.train()
    if args.freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()  # Affine parameters stay trainable; running statistics freeze.
    total_numerator = total_denominator = 0.
    before_actual, before_skipped, before_attempted = step_guard.actual, step_guard.skipped, step_guard.attempted
    factor_counts = {str(r): 0 for r in VALIDATION_R}
    for group_index, group in enumerate(microbatch_groups(loader, args.accumulation_steps), 1):
        denominator = (sum(len(batch[0]) for batch in group) if args.variant == 'regression'
                       else sum(float(batch[2].sum(dtype=torch.float64)) for batch in group))
        if denominator <= 0:
            raise ValueError('Empty effective batch')
        optimizer.zero_grad(set_to_none=True)
        group_numerator = 0.
        for x, soft, valid, q in group:
            x, soft, valid, q = x.to(device), soft.to(device), valid.to(device), q.to(device)
            factor = factor_rng.choice(VALIDATION_R) if args.scale_augmentation else 1
            factor_counts[str(factor)] += len(x)
            with torch.autocast(device_type=device.type, enabled=args.amp):
                logits = model(degrade_rgb(x, factor, valid), valid)
            require_finite_forward(logits, model)
            numerator = loss_numerator(logits, soft, valid, q, args.variant, args.label_grid)
            loss = numerator/denominator
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward()
            total_numerator += float(numerator.detach())
            group_numerator += float(numerator.detach())
        step_guard.finish(model, {'epoch': epoch, 'group': group_index, 'group_loss': group_numerator/denominator,
                                  'group_denominator': denominator, 'group_samples': sum(len(batch[0]) for batch in group)})
        total_denominator += denominator
        if args.step_pause:
            time.sleep(args.step_pause)
    actual = step_guard.actual-before_actual
    if actual == 0:
        raise FloatingPointError('Epoch had zero actual optimizer updates')
    return {'train_loss': total_numerator/total_denominator, 'optimizer_updates': actual,
            'optimizer_update_attempts': step_guard.attempted-before_attempted,
            'optimizer_skipped_overflows': step_guard.skipped-before_skipped,
            'amp_scale_end': float(scaler.get_scale()), 'consecutive_overflows': step_guard.consecutive_overflows,
            'train_factor_sample_counts': factor_counts}


def validate_args(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', args.run_id) or '..' in args.run_id:
        raise ValueError('run-id must be a simple unique directory name')
    for key in ('epochs', 'batch_size', 'accumulation_steps', 'threads'):
        if getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive')
    if not 0 < args.gpu_memory_fraction <= .16:
        raise ValueError('Per-process GPU fraction must be in (0,.16]')
    if args.step_pause < 0 or args.weight_decay < 0:
        raise ValueError('Negative pause or weight decay')
    if args.device == 'cpu' and args.amp:
        args.amp = False
    if args.smoke:
        if not args.run_id.startswith('smoke_common_') or not 2 <= args.max_train <= 16 or not 2 <= args.max_val <= 8 or args.epochs != 1:
            raise ValueError('Smoke requires smoke_common_ ID, one epoch, 2..16 train and 2..8 val samples')
    elif args.diagnostic:
        if not args.run_id.startswith('diag_common_amp_v2_') or not 2 <= args.diagnostic_epochs <= 6 or args.max_train or args.max_val or args.no_pretrained:
            raise ValueError('AMP diagnostic requires a unique diag_common_amp_v2_ ID, full train/val, and 2..6 epochs')
    elif args.max_train or args.max_val or args.no_pretrained or args.diagnostic_epochs:
        raise ValueError('Subsets and random initialization are smoke-only')
    if args.variant == 'regression' and args.label_grid != 256:
        raise ValueError('Regression has no supervision grid; use the neutral label-grid256')


def run(args):
    validate_args(args)
    run_dir = ROOT/'runs'/args.run_id
    # Atomic ownership: failure before mkdir never writes into another run.
    run_dir.mkdir(parents=False, exist_ok=False)
    start = time.time()
    try:
        write_json(run_dir/'status.json', {'status': 'starting', 'started_utc': utc_now(), 'pid': os.getpid(), 'test_accessed': False})
        device = configure_runtime(args.device, args.seed, args.threads, args.gpu_memory_fraction)
        rows, manifest_hashes = load_development_manifests()
        train = bounded_subset(rows['train'], args.max_train)
        val = bounded_subset(rows['val'], args.max_val)
        source_dir = run_dir/'source'
        source_dir.mkdir()
        source_hashes = {}
        for name in ('train_common.py', 'models_common.py', 'conservative_resample.py'):
            shutil.copy2(ROOT/name, source_dir/name)
            source_hashes[name] = sha256_file(source_dir/name)
        for split in ('train', 'val'):
            shutil.copy2(ROOT/'local_data_v2'/(split+'.json'), source_dir/(split+'.json'))
        cfg = vars(args).copy()
        cfg.update(python=sys.version, python_executable=sys.executable, torch=str(torch.__version__), numpy=np.__version__,
                   device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   manifest_sha256=manifest_hashes, source_sha256=source_hashes, stage='smoke' if args.smoke else ('diagnostic' if args.diagnostic else 'development'),
                   n_train=len(train), n_val=len(val), selected_train_ids=[r['id'] for r in train], selected_val_ids=[r['id'] for r in val],
                   effective_batch_size=args.batch_size*args.accumulation_steps, validation_R=list(VALIDATION_R),
                   batchnorm_policy='running statistics use each microbatch unless explicitly frozen; accumulation is not physical batch8 equivalence',
                   checkpoint_selection='unweighted mean of zero/positive-stratum MAE over R1/2/4; strictly smaller; earliest tie',
                   decision_metrics_used_for_selection=False, test_accessed=False,
                   hard_target_definition='valid-area majority >=0.5 after conservative soft-target aggregation; differs from v2 nearest',
                   supervision='probability BCE clipped [1e-7,1-1e-7]; valid-area weighted; fine-grid mass units',
                   regression='valid-area pooled final-stage features; linear head; SmoothL1 beta0.1; sample-weighted',
                   tail_batch_policy='keep; normalize accumulated gradient by actual whole-group area or sample count',
                   scheduler_policy='cosine over fixed full epoch budget; no early termination; choose best validation checkpoint',
                   amp_step_policy='v2: once/group unscale; finite gradients strict clip; overflow scaler-skip/backoff; public actual-step hook; all-group zero_grad',
                   overflow_guard={'max_consecutive':8, 'min_scale':1.0, 'zero_update_epoch':'fail', 'nonfinite_logits_loss_buffers':'fail'},
                   fixed_output_grid=256, started_utc=utc_now())
        write_json(run_dir/'config.json', cfg)
        truth = [r['coverage'] for r in val]
        train_mean = float(np.mean([r['coverage'] for r in train]))
        write_json(run_dir/'constant_baselines.json', {'zero': summarize([0.]*len(val), truth), 'train_mean': summarize([train_mean]*len(val), truth)})
        generator = torch.Generator().manual_seed(args.seed)
        if args.sampler == 'balanced':
            positives = sum(r['coverage'] > 0 for r in train)
            negatives = len(train)-positives
            weights = [.5/max(positives, 1) if r['coverage'] > 0 else .5/max(negatives, 1) for r in train]
            sampler = WeightedRandomSampler(weights, len(train), replacement=True, generator=generator)
        else:
            sampler = None
        train_loader = DataLoader(DevelopmentTiles(train, True, args.seed), batch_size=args.batch_size, sampler=sampler,
                                  shuffle=sampler is None, generator=generator, num_workers=0, pin_memory=device.type == 'cuda')
        val_loader = DataLoader(DevelopmentTiles(val), batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=device.type == 'cuda')
        model = make_model(args.encoder, args.variant, not args.no_pretrained).to(device)
        cfg['pretrained_report'] = model.pretrained_report
        cfg['parameter_count'] = sum(p.numel() for p in model.parameters())
        write_json(run_dir/'config.json', cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        def record_optimizer_event(event):
            with (run_dir/'optimizer_events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')
        step_guard = OptimizerStepGuard(optimizer, scaler, record_optimizer_event)
        factor_rng = random.Random(args.seed+1000)
        best, best_epoch = math.inf, None
        write_json(run_dir/'status.json', {'status': 'running', 'pid': os.getpid(), 'started_utc': cfg['started_utc'], 'test_accessed': False})
        executed_epochs = args.diagnostic_epochs if args.diagnostic else args.epochs
        for epoch in range(1, executed_epochs+1):
            training = train_epoch(model, train_loader, optimizer, scaler, device, args, factor_rng, step_guard, epoch)
            metrics = evaluate(model, val_loader, device, args.variant, args.amp)
            metrics.update(training, epoch=epoch, elapsed_seconds=time.time()-start, lr=optimizer.param_groups[0]['lr'])
            if metrics['selection_balanced_mae'] < best:
                best, best_epoch = metrics['selection_balanced_mae'], epoch
                checkpoint = {'model': model.state_dict(), 'config': cfg, 'epoch': epoch, 'validation': metrics,
                              'grad_scaler': scaler.state_dict(), 'optimizer_step_counts': {'attempted':step_guard.attempted,'actual':step_guard.actual,'skipped':step_guard.skipped}}
                torch.save(checkpoint, run_dir/'best.pt.tmp')
                (run_dir/'best.pt.tmp').replace(run_dir/'best.pt')
                write_json(run_dir/'best_validation.json', metrics)
            with (run_dir/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+'\n')
            write_json(run_dir/'status.json', {'status': 'running', 'pid': os.getpid(), 'epoch': epoch,
                                              'best_epoch': best_epoch, 'updated_utc': utc_now(), 'test_accessed': False,
                                              'optimizer_attempts_total':step_guard.attempted,'optimizer_updates_total':step_guard.actual,
                                              'optimizer_skips_total':step_guard.skipped,'amp_scale':float(scaler.get_scale())})
            print(json.dumps({'run_id': args.run_id, 'epoch': epoch, 'train_loss': training['train_loss'],
                              'selection_balanced_mae': metrics['selection_balanced_mae'], 'updates': training['optimizer_updates'],
                              'attempts':training['optimizer_update_attempts'],'skipped':training['optimizer_skipped_overflows'],
                              'amp_scale':training['amp_scale_end']}), flush=True)
            scheduler.step()
        final = {'status': 'completed', 'stage': cfg['stage'], 'completed_utc': utc_now(),
                 'best_epoch': best_epoch, 'best_validation_balanced_mae': best,
                 'elapsed_seconds': time.time()-start, 'test_accessed': False,
                 'executed_epochs':executed_epochs, 'optimizer_attempts_total':step_guard.attempted,
                 'optimizer_updates_total':step_guard.actual,'optimizer_skips_total':step_guard.skipped,
                 'amp_scale':float(scaler.get_scale()),
                 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                 'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0}
        write_json(run_dir/'status.json', final)
        print(json.dumps(final), flush=True)
        return final
    except BaseException as error:
        failure = {'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc(),
                   'failed_utc': utc_now(), 'test_accessed': False}
        write_json(run_dir/'failure.json', failure)
        write_json(run_dir/'status.json', failure)
        raise


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--encoder', choices=tuple(PRETRAINED_SHA256), default='resnet18')
    ap.add_argument('--variant', choices=['regression', 'hard_seg', 'soft_seg'], required=True)
    ap.add_argument('--label-grid', type=int, choices=[64, 128, 256], default=256)
    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--accumulation-steps', type=int, default=2)
    ap.add_argument('--lr', type=float, choices=[1e-4, 3e-4], default=1e-4)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--gpu-memory-fraction', type=float, default=.16)
    ap.add_argument('--step-pause', type=float, default=.05)
    ap.add_argument('--scale-augmentation', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--sampler', choices=['balanced', 'natural'], default='balanced')
    ap.add_argument('--freeze-batchnorm', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--diagnostic', action='store_true')
    ap.add_argument('--diagnostic-epochs', type=int, default=0)
    ap.add_argument('--max-train', type=int, default=0)
    ap.add_argument('--max-val', type=int, default=0)
    ap.add_argument('--no-pretrained', action='store_true')
    return ap


if __name__ == '__main__':
    run(parser().parse_args())
